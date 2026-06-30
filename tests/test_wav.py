"""Tests for rosi.wav — loading multi-channel WAV files."""

import numpy as np
import pytest
from scipy.io import wavfile

from rosi.wav import load_wav_signals


def _write_int16_wav(path, fs, signals):
    """Write an int16 WAV.  signals shape (n_ch, n_samples)."""
    data = (signals.T * 32767).astype(np.int16)
    wavfile.write(path, fs, data)


class TestLoadWavSignals:
    def test_shapes_and_fs(self, tmp_path):
        fs = 16000
        n_ch, n_samples = 3, 400
        t_sig = np.arange(n_samples) / fs
        sigs = 0.5 * np.sin(2 * np.pi * 440 * t_sig[np.newaxis, :] + np.arange(n_ch)[:, np.newaxis])
        path = tmp_path / "test.wav"
        _write_int16_wav(path, fs, sigs)

        fs_out, t_out, signals_out = load_wav_signals(path)
        assert fs_out == fs
        assert signals_out.shape == (n_ch, n_samples)
        assert t_out.shape == (n_samples,)

    def test_values_in_range(self, tmp_path):
        fs = 8000
        sigs = np.array([[0.0, 0.5, -0.5, 1.0, -1.0]], dtype=np.float64)
        path = tmp_path / "test.wav"
        _write_int16_wav(path, fs, sigs)

        _, _, out = load_wav_signals(path)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_mono_wav(self, tmp_path):
        fs = 8000
        mono = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(fs * 0.1)))
        path = tmp_path / "mono.wav"
        wavfile.write(path, fs, (mono * 32767).astype(np.int16))

        _, _, out = load_wav_signals(path)
        assert out.shape[0] == 1

    def test_float32_wav_passthrough(self, tmp_path):
        fs = 16000
        n_samples = 200
        sigs = np.random.randn(2, n_samples).astype(np.float32)
        sigs /= np.max(np.abs(sigs))  # ensure in [-1, 1]
        path = tmp_path / "float.wav"
        wavfile.write(path, fs, sigs.T)

        _, _, out = load_wav_signals(path)
        np.testing.assert_allclose(out, sigs, atol=1e-6)

    def test_t_axis(self, tmp_path):
        fs = 5000
        n_samples = 150
        sigs = np.zeros((1, n_samples))
        path = tmp_path / "t_test.wav"
        wavfile.write(path, fs, sigs.T.astype(np.int16))

        _, t_out, _ = load_wav_signals(path)
        np.testing.assert_allclose(t_out, np.arange(n_samples) / fs)