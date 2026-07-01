"""rosi_tach.py — Derive rotor RPM from a tachometer pulse-train channel."""

import numpy as np


def extract_rpm_from_tach(
    tach_signal: np.ndarray,
    fs: float,
    threshold_frac: float = 0.5,
) -> tuple[float, np.ndarray]:
    """
    Detect rising-edge pulse times in a tach channel and return the mean RPM
    (assuming one pulse per revolution) plus the pulse timestamps.

    tach_signal : (N_samples,) raw tach channel
    fs          : sample rate [Hz]
    """
    threshold = threshold_frac * tach_signal.max()
    above = tach_signal > threshold
    rising = np.where(np.diff(above.astype(int)) == 1)[0] + 1
    pulse_times = rising / fs

    if len(pulse_times) < 2:
        raise ValueError("Tach channel: fewer than 2 pulses detected — cannot estimate RPM")

    periods = np.diff(pulse_times)
    rpm = 60.0 / periods.mean()
    return rpm, pulse_times


def rotor_phase_from_tach(
    pulse_times: np.ndarray,
    t: np.ndarray,
    phase0: float = 0.0,
) -> np.ndarray:
    """
    Build the unwrapped rotor phase theta(t) [rad] at each sample in `t`,
    treating each tach pulse as exactly one revolution (2*pi phase advance).

    Phase is linearly interpolated between consecutive pulses, so the angular
    velocity is piecewise-constant per revolution rather than a single average
    over the whole recording — this tracks RPM drift within the measurement.
    Samples before the first pulse / after the last use the edge revolution's
    angular velocity for linear extrapolation.
    """
    if len(pulse_times) < 2:
        raise ValueError("Need at least 2 tach pulses to build a rotor phase curve")

    theta_pulses = 2 * np.pi * np.arange(len(pulse_times)) + phase0
    theta = np.interp(t, pulse_times, theta_pulses)

    omega_start = (theta_pulses[1] - theta_pulses[0]) / (pulse_times[1] - pulse_times[0])
    before = t < pulse_times[0]
    theta[before] = theta_pulses[0] + omega_start * (t[before] - pulse_times[0])

    omega_end = (theta_pulses[-1] - theta_pulses[-2]) / (pulse_times[-1] - pulse_times[-2])
    after = t > pulse_times[-1]
    theta[after] = theta_pulses[-1] + omega_end * (t[after] - pulse_times[-1])

    return theta
