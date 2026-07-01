"""Tests for rosi.sim — signal simulation functions."""

import numpy as np

from rosi.sim import (
    make_mic_array,
    retarded_time,
    retarded_time_vec,
    simulate_signals,
    source_position,
)

# ── make_mic_array ────────────────────────────────────────────────────────────


class TestMakeMicArray:
    def test_shape(self):
        arr = make_mic_array(10, 2.0, 1.5)
        assert arr.shape == (10, 3)

    def test_radius(self):
        arr = make_mic_array(8, 3.0, 0.5)
        radii = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
        np.testing.assert_allclose(radii, 3.0, atol=1e-12)

    def test_z(self):
        arr = make_mic_array(5, 1.0, 2.5)
        np.testing.assert_allclose(arr[:, 2], 2.5)


# ── source_position ───────────────────────────────────────────────────────────


class TestSourcePosition:
    def test_static_source(self):
        t = np.array([0.0, 0.5, 1.0])
        pos = source_position(t, R=2.0, omega=0.0, phi0=0.0)
        expected = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        np.testing.assert_allclose(pos, expected)

    def test_rotating_source_at_t0(self):
        t = np.array([0.0])
        pos = source_position(t, R=1.0, omega=1.0, phi0=0.0)
        np.testing.assert_allclose(pos[0], [1.0, 0.0, 0.0])

    def test_z_always_zero(self):
        t = np.linspace(0, 1, 100)
        pos = source_position(t, R=0.5, omega=2.0, phi0=0.3)
        np.testing.assert_allclose(pos[:, 2], 0.0)


# ── retarded_time convergence ─────────────────────────────────────────────────


class TestRetardedTime:
    def test_static_source_scalar(self):
        c = 343.0
        x_mic = np.array([1.0, 0.0, 1.0])
        R = 0.5
        t_e = retarded_time(0.1, x_mic, R, omega=0.0, phi0=0.0, c=c)
        dist = np.linalg.norm(x_mic - np.array([R, 0.0, 0.0]))
        expected = 0.1 - dist / c
        np.testing.assert_allclose(t_e, expected, atol=1e-9)

    def test_vec_agrees_with_scalar(self):
        c = 343.0
        x_mic = np.array([1.0, 0.5, 1.0])
        t_arr = np.linspace(0.0, 0.1, 20)
        t_e_vec = retarded_time_vec(t_arr, x_mic, R=0.3, omega=0.0, phi0=0.0, c=c)
        for i, t in enumerate(t_arr):
            t_e_scalar = retarded_time(t, x_mic, R=0.3, omega=0.0, phi0=0.0, c=c)
            np.testing.assert_allclose(t_e_vec[i], t_e_scalar, atol=1e-12)


# ── simulate_signals ──────────────────────────────────────────────────────────


class TestSimulateSignals:
    def test_shapes(self):
        fs = 8000
        duration = 0.1
        mics = make_mic_array(4, 1.0, 1.5)
        sources = [
            {
                "R": 0.3,
                "omega": 1.0,
                "phi0": 0.0,
                "freq": 500,
                "amplitude": 1.0,
                "phase": 0.0,
            }
        ]
        t, signals = simulate_signals(sources, mics, fs, duration, 343.0)
        assert t.shape == (int(fs * duration),)
        assert signals.shape == (4, int(fs * duration))

    def test_zero_sources_returns_zeros(self):
        mics = make_mic_array(3, 1.0, 1.0)
        t, signals = simulate_signals([], mics, 8000, 0.1, 343.0)
        np.testing.assert_allclose(signals, 0.0)

    def test_fft_peak_at_source_freq(self):
        fs = 8000
        duration = 0.2
        mics = make_mic_array(4, 1.0, 1.5)
        freq = 1000
        sources = [
            {
                "R": 0.3,
                "omega": 0.0,
                "phi0": 0.0,
                "freq": freq,
                "amplitude": 1.0,
                "phase": 0.0,
            }
        ]
        t, signals = simulate_signals(sources, mics, fs, duration, 343.0)
        fft = np.fft.rfft(signals[0])
        freqs = np.fft.rfftfreq(len(t), d=1.0 / fs)
        peak_idx = np.argmax(np.abs(fft))
        assert abs(freqs[peak_idx] - freq) < fs / len(t) * 2
