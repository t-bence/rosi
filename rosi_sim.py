"""
rosi_sim.py — Simulation of rotating acoustic point sources recorded by a microphone array.
"""

import numpy as np
from scipy.interpolate import interp1d


def make_mic_array(n_mics: int, radius: float, z: float = 1.5) -> np.ndarray:
    """Return (n_mics, 3) mic positions on a circle at height z."""
    angles = np.linspace(0, 2 * np.pi, n_mics, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n_mics, z)])


def source_position(t: np.ndarray, R: float, omega: float, phi0: float) -> np.ndarray:
    """
    Return (len(t), 3) lab-frame positions for a source rotating at angular velocity omega.
    Source lies in z=0 plane at radius R with initial phase phi0.
    """
    theta = omega * t + phi0
    return np.column_stack([R * np.cos(theta), R * np.sin(theta), np.zeros_like(t)])


def retarded_time(
    t: float,
    x_mic: np.ndarray,
    R: float,
    omega: float,
    phi0: float,
    c: float,
    n_iter: int = 8,
) -> float:
    """
    Solve retarded time t_e for a single (mic, observation-time) pair via fixed-point iteration:
        t_e <- t - |x_mic - x_source(t_e)| / c
    Returns scalar t_e.
    """
    t_e = t  # initial guess
    for _ in range(n_iter):
        theta_e = omega * t_e + phi0
        xs = np.array([R * np.cos(theta_e), R * np.sin(theta_e), 0.0])
        dist = np.linalg.norm(x_mic - xs)
        t_e = t - dist / c
    return t_e


def retarded_time_vec(
    t_arr: np.ndarray,
    x_mic: np.ndarray,
    R: float,
    omega: float,
    phi0: float,
    c: float,
    n_iter: int = 8,
) -> np.ndarray:
    """
    Vectorised retarded time solver over a time array t_arr.
    Returns array of same shape as t_arr.
    """
    t_e = t_arr.copy()
    for _ in range(n_iter):
        theta_e = omega * t_e + phi0
        xs = np.column_stack([R * np.cos(theta_e), R * np.sin(theta_e), np.zeros_like(t_e)])
        dist = np.linalg.norm(x_mic[np.newaxis, :] - xs, axis=1)
        t_e = t_arr - dist / c
    return t_e


def simulate_signals(
    sources: list[dict],
    mic_positions: np.ndarray,
    fs: float,
    t_total: float,
    c: float,
    n_iter: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate microphone signals for a set of rotating point sources.

    sources: list of dicts with keys: R, omega, phi0, freq, amplitude, phase
    mic_positions: (N_mics, 3)
    Returns: (t, signals) where signals is (N_mics, N_samples)
    """
    n_samples = int(fs * t_total)
    t = np.arange(n_samples) / fs
    n_mics = mic_positions.shape[0]
    signals = np.zeros((n_mics, n_samples))

    for src in sources:
        R, omega, phi0 = src["R"], src["omega"], src["phi0"]
        freq, amp, psi = src["freq"], src["amplitude"], src["phase"]

        for m in range(n_mics):
            t_e = retarded_time_vec(t, mic_positions[m], R, omega, phi0, c, n_iter)
            theta_e = omega * t_e + phi0
            xs = np.column_stack([R * np.cos(theta_e), R * np.sin(theta_e), np.zeros_like(t_e)])
            r_e = np.linalg.norm(mic_positions[m][np.newaxis, :] - xs, axis=1)
            r_e = np.maximum(r_e, 1e-6)  # avoid division by zero
            signals[m] += amp / (4 * np.pi * r_e) * np.sin(2 * np.pi * freq * t_e + psi)

    return t, signals


def make_signal_interpolators(t: np.ndarray, signals: np.ndarray):
    """Return list of cubic interpolators, one per microphone."""
    return [
        interp1d(t, signals[m], kind="cubic", bounds_error=False, fill_value=0.0)
        for m in range(signals.shape[0])
    ]
