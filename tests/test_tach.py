"""Tests for rosi.tach — per-revolution rotor phase from a tachometer channel."""

import numpy as np
import pytest

from rosi.tach import extract_rpm_from_tach, rotor_phase_from_tach


def _pulse_train(fs, rpm, n_revs):
    """Build a square-wave tach signal with one rising edge per revolution."""
    period = 60.0 / rpm
    n = int(fs * period * n_revs)
    t = np.linspace(0, period * n_revs, n, endpoint=False)
    sig = (np.sin(2 * np.pi / period * t) > 0).astype(np.float64)
    return t, sig


# ── extract_rpm_from_tach ───────────────────────────────────────────────────────


class TestExtractRpmFromTach:
    def test_known_rpm(self):
        fs = 10000
        _, sig = _pulse_train(fs, rpm=6000.0, n_revs=10)
        rpm, pulse_times = extract_rpm_from_tach(sig, fs)
        assert abs(rpm - 6000.0) < 10
        assert len(pulse_times) >= 9

    def test_fewer_than_two_pulses_raises(self):
        sig = np.zeros(100, dtype=np.float64)
        with pytest.raises(ValueError, match="fewer than 2 pulses"):
            extract_rpm_from_tach(sig, fs=1000)


# ── rotor_phase_from_tach ────────────────────────────────────────────────────────


class TestRotorPhaseFromTach:
    def test_phase_advances_2pi_per_pulse(self):
        pulse_times = np.array([0.0, 0.1, 0.2, 0.3])
        t = pulse_times.copy()
        theta = rotor_phase_from_tach(pulse_times, t)
        np.testing.assert_allclose(theta, 2 * np.pi * np.arange(4))

    def test_monotonically_increasing(self):
        fs = 5000
        _, sig = _pulse_train(fs, rpm=3000.0, n_revs=8)
        t = np.arange(len(sig)) / fs
        _, pulse_times = extract_rpm_from_tach(sig, fs)
        theta = rotor_phase_from_tach(pulse_times, t)
        assert np.all(np.diff(theta) >= 0)

    def test_extrapolates_before_and_after_pulses(self):
        pulse_times = np.array([1.0, 1.1, 1.2])
        t = np.array([0.9, 1.0, 1.1, 1.2, 1.3])
        theta = rotor_phase_from_tach(pulse_times, t)
        # Before the first pulse: linear extrapolation using the first-revolution rate.
        omega_start = 2 * np.pi / 0.1
        assert theta[0] == pytest.approx(0.0 - omega_start * 0.1)
        # After the last pulse: linear extrapolation using the last-revolution rate.
        omega_end = 2 * np.pi / 0.1
        assert theta[-1] == pytest.approx(4 * np.pi + omega_end * 0.1)

    def test_fewer_than_two_pulses_raises(self):
        with pytest.raises(ValueError, match="at least 2 tach pulses"):
            rotor_phase_from_tach(np.array([1.0]), np.array([1.0, 2.0]))

    def test_phase0_offset(self):
        pulse_times = np.array([0.0, 0.1])
        t = pulse_times.copy()
        theta = rotor_phase_from_tach(pulse_times, t, phase0=1.5)
        np.testing.assert_allclose(theta, [1.5, 1.5 + 2 * np.pi])
