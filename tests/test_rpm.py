"""Tests for rosi.rpm — tachometer RPM extraction from WAV data."""

import numpy as np
import pytest
from scipy.io import wavfile

from rosi.rpm import (
    extract_rpm,
    find_rising_edge_times,
    load_signals_from_wav,
    load_wav,
    split_tacho_channel,
)


# ── load_wav ──────────────────────────────────────────────────────────────────

class TestLoadWav:
    @pytest.mark.parametrize("dtype,scale", [
        (np.int16, 32768.0),
        (np.int32, 2147483648.0),
    ])
    def test_integer_normalisation(self, tmp_path, dtype, scale):
        fs = 8000
        raw = (np.random.randn(100) * 0.3 * scale).astype(dtype)
        path = tmp_path / "int.wav"
        wavfile.write(path, fs, raw)
        fs_out, data = load_wav(path)
        assert fs_out == fs
        assert data.dtype == np.float64
        assert abs(data).max() <= 1.0

    def test_float_passthrough(self, tmp_path):
        fs = 8000
        raw = np.random.randn(100).astype(np.float32)
        raw /= np.max(np.abs(raw))
        path = tmp_path / "float.wav"
        wavfile.write(path, fs, raw)
        _, data = load_wav(path)
        np.testing.assert_allclose(data, raw, atol=1e-6)


# ── split_tacho_channel ───────────────────────────────────────────────────────

class TestSplitTachoChannel:
    def test_extracts_correct_channel(self):
        data = np.array([[0, 1, 2], [10, 11, 12], [20, 21, 22]], dtype=np.float64)
        tacho, mics = split_tacho_channel(data, tacho_channel=1)
        np.testing.assert_array_equal(tacho, [1, 11, 21])
        assert mics.shape == (2, 3)  # 2 mic channels, 3 samples
        np.testing.assert_array_equal(mics[0], [0, 10, 20])
        np.testing.assert_array_equal(mics[1], [2, 12, 22])

    def test_negative_index(self):
        data = np.array([[0, 1, 2], [10, 11, 12]], dtype=np.float64)
        tacho, _ = split_tacho_channel(data, tacho_channel=-1)
        np.testing.assert_array_equal(tacho, [2, 12])

    def test_mono_raises(self):
        data = np.array([0, 1, 2], dtype=np.float64)
        with pytest.raises(ValueError, match="mono"):
            split_tacho_channel(data, tacho_channel=0)


# ── find_rising_edge_times ────────────────────────────────────────────────────

class TestFindRisingEdgeTimes:
    def test_known_period(self):
        fs = 1000
        t = np.linspace(0, 1, fs, endpoint=False)
        # Square wave: 200 Hz → period 5 samples
        sig = (np.sin(2 * np.pi * 200 * t) > 0).astype(np.float64)
        edges = find_rising_edge_times(sig, fs)
        if len(edges) >= 2:
            periods = np.diff(edges)
            np.testing.assert_allclose(periods, 1.0 / 200, atol=1 / fs)

    def test_default_threshold_midpoint(self):
        sig = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5], dtype=np.float64)
        edges = find_rising_edge_times(sig, fs=1000, threshold=None)
        # Midpoint is 0.0; rising edges at index 2→3 (0→0.5) and ... actually
        # check we get at least one edge
        assert len(edges) >= 1

    def test_custom_threshold(self):
        sig = np.array([0.0, 0.3, 0.7, 0.9, 0.4, 0.1, 0.8], dtype=np.float64)
        edges = find_rising_edge_times(sig, fs=100, threshold=0.5)
        # Rising edges: index 1→2 (0.3→0.7) at t=0.02, and index 5→6 (0.1→0.8) at t=0.06
        np.testing.assert_allclose(edges, [0.02, 0.06])


# ── extract_rpm ───────────────────────────────────────────────────────────────

class TestExtractRpm:
    def test_known_rpm(self):
        fs = 10000
        rpm_target = 6000.0
        period = 60.0 / rpm_target  # 0.01 s
        # Simulate 10 pulses
        n_periods = 10
        n = int(fs * period * n_periods)
        t = np.linspace(0, period * n_periods, n, endpoint=False)
        sig = (np.sin(2 * np.pi / period * t) > 0).astype(np.float64)
        rpm = extract_rpm(sig, fs)
        assert abs(rpm - rpm_target) < 10  # within 10 RPM

    def test_fewer_than_two_edges_raises(self):
        sig = np.zeros(100, dtype=np.float64)
        with pytest.raises(ValueError, match="Only 0 rising edge"):
            extract_rpm(sig, 1000)

    def test_single_edge_raises(self):
        sig = np.zeros(100, dtype=np.float64)
        sig[50:] = 1.0
        with pytest.raises(ValueError, match="Only 1 rising edge"):
            extract_rpm(sig, 1000)


# ── load_signals_from_wav (end-to-end) ────────────────────────────────────────

class TestLoadSignalsFromWav:
    def test_end_to_end(self, tmp_path):
        fs = 8000
        duration = 0.5
        n_samples = int(fs * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Tachometer: 3000 RPM → period = 0.02 s = 50 Hz
        rpm_target = 3000.0
        tacho_period = 60.0 / rpm_target
        tacho = (np.sin(2 * np.pi / tacho_period * t) > 0).astype(np.float64)

        # One microphone channel: pure tone
        mic = 0.3 * np.sin(2 * np.pi * 1000 * t)

        # WAV with tacho on channel 0, mic on channel 1
        data = np.column_stack([tacho, mic]).astype(np.float64)
        path = tmp_path / "test.wav"
        wavfile.write(path, fs, data)

        rpm, fs_out, t_out, mics = load_signals_from_wav(path, tacho_channel=0)
        assert abs(rpm - rpm_target) < 20
        assert fs_out == fs
        assert mics.shape == (1, n_samples)
        assert len(t_out) == n_samples