"""Tests for rosi.beamform — scan grid, CSM, beamformer correctness."""

import numpy as np
import pytest

from rosi.beamform import (
    compute_global_csm,
    make_scan_grid,
    power_map_to_grid,
    rosi_beamform_freq,
)
from rosi.sim import make_mic_array, simulate_signals

# ── make_scan_grid ────────────────────────────────────────────────────────────


class TestMakeScanGrid:
    def test_shape(self):
        grid = make_scan_grid(1.0, 8, 10)
        assert grid.shape == (80, 2)

    def test_r_range(self):
        grid = make_scan_grid(2.0, 10, 5)
        r_vals = grid[:, 0]
        assert r_vals.min() > 0
        assert abs(r_vals.max() - 2.0) < 1e-12

    def test_theta_range(self):
        grid = make_scan_grid(1.0, 4, 8)
        theta_vals = grid[:, 1]
        assert theta_vals.min() >= 0
        assert theta_vals.max() < 2 * np.pi


# ── power_map_to_grid ─────────────────────────────────────────────────────────


class TestPowerMapToGrid:
    def test_roundtrip(self):
        n_r, n_theta = 8, 10
        grid = make_scan_grid(1.0, n_r, n_theta)
        n_scan = len(grid)
        power_map = np.random.rand(n_scan, 5)
        r_vals, theta_vals, power_2d = power_map_to_grid(
            power_map, grid, n_r, n_theta, freq_idx=2
        )
        np.testing.assert_allclose(power_2d.ravel(), power_map[:, 2])

    def test_shape(self):
        n_r, n_theta = 6, 8
        grid = make_scan_grid(1.0, n_r, n_theta)
        power_map = np.random.rand(len(grid), 3)
        _, _, power_2d = power_map_to_grid(power_map, grid, n_r, n_theta)
        assert power_2d.shape == (n_theta, n_r)


# ── compute_global_csm ────────────────────────────────────────────────────────


class TestGlobalCSM:
    def test_hermitian(self):
        fs = 8000
        duration = 0.2
        mics = make_mic_array(4, 1.0, 1.5)
        sources = [
            {
                "R": 0.3,
                "omega": 0.0,
                "phi0": 0.0,
                "freq": 500,
                "amplitude": 1.0,
                "phase": 0.0,
            }
        ]
        _, signals = simulate_signals(sources, mics, fs, duration, 343.0)
        freqs, C = compute_global_csm(signals, fs, fft_size=256, overlap=0.5)
        for fi in range(len(freqs)):
            np.testing.assert_allclose(C[fi], C[fi].conj().T, atol=1e-10)

    def test_diagonal_peak_at_source_freq(self):
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
        _, signals = simulate_signals(sources, mics, fs, duration, 343.0)
        freqs, C = compute_global_csm(signals, fs, fft_size=256, overlap=0.5)
        # C is (N_freq, N_mics, N_mics); pick the frequency slice closest to source freq
        fi = np.argmin(np.abs(freqs - freq))
        diag = np.abs(np.diag(C[fi]))
        peak_mic = np.argmax(diag)
        assert diag[peak_mic] > 0


# ── rosi_beamform_freq ────────────────────────────────────────────────────────


class TestBeamformFreq:
    def test_freq_mask(self):
        fs = 8000
        duration = 0.2
        mics = make_mic_array(4, 1.0, 1.5)
        grid = make_scan_grid(0.5, 4, 4)
        sources = [
            {
                "R": 0.3,
                "omega": 0.0,
                "phi0": 0.0,
                "freq": 500,
                "amplitude": 1.0,
                "phase": 0.0,
            }
        ]
        _, signals = simulate_signals(sources, mics, fs, duration, 343.0)
        t = np.arange(len(signals[0])) / fs
        freqs_out, power_map = rosi_beamform_freq(
            signals,
            t,
            mics,
            grid,
            0.0,
            343.0,
            fft_size=128,
            overlap=0.5,
            f_min=400,
            f_max=600,
            n_jobs=1,
            verbose=False,
        )
        assert np.all(freqs_out >= 400)
        assert np.all(freqs_out <= 600)

    def test_localisation_smoke_test(self):
        """Simulate one source, beamform, assert power at true location exceeds background."""
        fs = 8000
        duration = 0.3
        R_true = 0.3
        phi0_true = 0.0
        mics = make_mic_array(6, 1.0, 1.5)
        omega = 0.0
        freq = 1000
        sources = [
            {
                "R": R_true,
                "omega": omega,
                "phi0": phi0_true,
                "freq": freq,
                "amplitude": 1.0,
                "phase": 0.0,
            }
        ]
        _, signals = simulate_signals(sources, mics, fs, duration, 343.0)
        t = np.arange(len(signals[0])) / fs
        grid = make_scan_grid(0.6, 12, 12)
        freqs_out, power_map = rosi_beamform_freq(
            signals,
            t,
            mics,
            grid,
            omega,
            343.0,
            fft_size=256,
            overlap=0.5,
            f_min=800,
            f_max=1200,
            n_jobs=1,
            verbose=False,
        )
        f_idx = np.argmin(np.abs(freqs_out - freq))
        col = power_map[:, f_idx]

        # Find the scan point closest to the true source
        dists = np.sqrt((grid[:, 0] - R_true) ** 2 + (grid[:, 1] - phi0_true) ** 2)
        true_idx = np.argmin(dists)
        assert col[true_idx] > col.min() * 2  # true-source power exceeds background


# ── backend parity ────────────────────────────────────────────────────────────


class TestBackendParity:
    def test_numpy_vs_numba(self):
        pytest.importorskip("numba")
        from rosi.beamform_numba import rosi_beamform_freq_numba

        fs = 8000
        duration = 0.15
        mics = make_mic_array(4, 1.0, 1.5)
        grid = make_scan_grid(0.5, 4, 4)
        sources = [
            {
                "R": 0.3,
                "omega": 0.0,
                "phi0": 0.0,
                "freq": 800,
                "amplitude": 1.0,
                "phase": 0.0,
            }
        ]
        _, signals = simulate_signals(sources, mics, fs, duration, 343.0)
        t = np.arange(len(signals[0])) / fs
        _, p_numpy = rosi_beamform_freq(
            signals,
            t,
            mics,
            grid,
            0.0,
            343.0,
            fft_size=128,
            overlap=0.5,
            f_min=600,
            f_max=1000,
            n_jobs=1,
            verbose=False,
        )
        _, p_numba = rosi_beamform_freq_numba(
            signals,
            t,
            mics,
            grid,
            0.0,
            343.0,
            fft_size=128,
            overlap=0.5,
            f_min=600,
            f_max=1000,
            verbose=False,
        )
        np.testing.assert_allclose(p_numpy, p_numba, rtol=1e-5)
