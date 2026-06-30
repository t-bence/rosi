"""Tests for rosi.beamform_numba — Numba backend specific checks."""

import numpy as np
import pytest

from rosi.beamform import make_scan_grid
from rosi.sim import make_mic_array, simulate_signals

numba = pytest.importorskip("numba")

from rosi.beamform_numba import rosi_beamform_freq_numba  # noqa: E402


class TestBeamformNumba:
    def test_output_shapes(self):
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
        freqs_out, power_map = rosi_beamform_freq_numba(
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
        assert power_map.shape[0] == len(grid)
        assert power_map.shape[1] == len(freqs_out)

    def test_localisation(self):
        """Numba backend should also locate the source correctly."""
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
        freqs_out, power_map = rosi_beamform_freq_numba(
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
            verbose=False,
        )
        f_idx = np.argmin(np.abs(freqs_out - freq))
        col = power_map[:, f_idx]

        dists = np.sqrt((grid[:, 0] - R_true) ** 2 + (grid[:, 1] - phi0_true) ** 2)
        true_idx = np.argmin(dists)
        assert col[true_idx] > col.min() * 2  # true-source power exceeds background
