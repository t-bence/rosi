"""
rpm_from_wav.py — Load WAV recordings and extract RPM from an optical tachometer channel.

In a typical ROSI measurement the WAV file contains one channel per microphone
*plus* one tachometer channel stored in the slot of one of those microphone
positions.  The tachometer (optical sensor + reflective patch) produces exactly
one pulse per rotor revolution:

    ___         ___
   |   |       |   |
___|   |_______|   |___    ← one pulse = one revolution

The time between consecutive rising edges equals the rotor period T [s], giving
RPM = 60 / T.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_wav(wav_path: str | Path) -> tuple[int, np.ndarray]:
    """
    Load a WAV file and normalise samples to float64 in [-1, 1].

    Returns
    -------
    fs : int
        Sample rate in Hz.
    data : np.ndarray, shape (N_samples, N_channels) or (N_samples,) for mono
        Normalised sample data.
    """
    from scipy.io import wavfile

    fs, raw = wavfile.read(wav_path)
    data = raw.astype(np.float64)
    if raw.dtype == np.int16:
        data /= 32_768.0
    elif raw.dtype == np.int32:
        data /= 2_147_483_648.0
    elif raw.dtype == np.uint8:
        data = (data - 128.0) / 128.0
    # float32 / float64 WAVs are already in [-1, 1] by convention
    return fs, data


def split_tacho_channel(
    data: np.ndarray,
    tacho_channel: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate the tachometer channel from the microphone channels.

    Parameters
    ----------
    data : np.ndarray
        WAV data — (N_samples, N_channels) for multi-channel or (N_samples,) mono.
    tacho_channel : int
        0-based index (negative indices accepted, e.g. -1 = last channel).

    Returns
    -------
    tacho : np.ndarray, shape (N_samples,)
    mic_signals : np.ndarray, shape (N_mics, N_samples)
        Remaining channels in their original order.
    """
    if data.ndim == 1:
        raise ValueError(
            "WAV file is mono — tachometer and microphone channels cannot be separated."
        )
    n_channels = data.shape[1]
    idx = tacho_channel % n_channels  # normalise negative index

    tacho = data[:, idx]
    mic_cols = [i for i in range(n_channels) if i != idx]
    mic_signals = data[:, mic_cols].T  # (N_mics, N_samples)
    return tacho, mic_signals


def find_rising_edge_times(
    signal: np.ndarray,
    fs: int,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Return the times [s] of all rising edges in a tachometer signal.

    A rising edge is a sample-to-sample transition from ≤ threshold to > threshold.
    The threshold defaults to the midpoint of the signal's amplitude range.
    """
    if threshold is None:
        threshold = 0.5 * (float(signal.min()) + float(signal.max()))

    above = signal > threshold
    # np.diff is True where the sign changes; +1 means 0→1 (rising edge)
    edge_indices = np.where(np.diff(above.astype(np.int8)) > 0)[0] + 1
    return edge_indices.astype(np.float64) / fs


def extract_rpm(
    tacho_signal: np.ndarray,
    fs: int,
    threshold: float | None = None,
) -> float:
    """
    Estimate rotor speed [RPM] from an optical tachometer signal.

    One reflective patch per revolution → one rising edge per revolution.
    The median inter-edge interval gives a robust period estimate that
    rejects partial revolutions at the start / end of the recording.

    Raises
    ------
    ValueError
        If fewer than 2 rising edges are detected.
    """
    edge_times = find_rising_edge_times(tacho_signal, fs, threshold)

    if len(edge_times) < 2:
        thr = threshold if threshold is not None else 0.5 * (float(tacho_signal.min()) + float(tacho_signal.max()))
        raise ValueError(
            f"Only {len(edge_times)} rising edge(s) detected in tachometer signal "
            f"(threshold = {thr:.4g}). "
            "Check the tacho_channel index and signal quality."
        )

    periods = np.diff(edge_times)
    period = float(np.median(periods))
    rpm = 60.0 / period

    n_revs = len(edge_times) - 1
    print(
        f"  Tachometer: {n_revs} complete revolution(s) detected, "
        f"median period = {period * 1e3:.2f} ms  →  RPM = {rpm:.2f}"
    )
    return rpm


def load_signals_from_wav(
    wav_path: str | Path,
    tacho_channel: int,
    threshold: float | None = None,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """
    Load a multi-channel WAV file containing mic signals and one tachometer
    channel, extract RPM from the tachometer, and return everything needed
    for beamforming.

    Parameters
    ----------
    wav_path : str or Path
        Path to the WAV file.
    tacho_channel : int
        0-based channel index of the tachometer signal.
    threshold : float, optional
        Detection threshold for rising-edge detection.  Auto-detected if None.

    Returns
    -------
    rpm : float
        Estimated rotor speed in revolutions per minute.
    fs : int
        Sample rate in Hz (taken from the WAV file header).
    t : np.ndarray, shape (N_samples,)
        Time axis in seconds.
    mic_signals : np.ndarray, shape (N_mics, N_samples)
        Normalised microphone signals (tachometer channel removed).
    """
    wav_path = Path(wav_path)
    print(f"Loading WAV: {wav_path}")

    fs, data = load_wav(wav_path)
    n_samples = data.shape[0]
    n_channels = data.shape[1] if data.ndim > 1 else 1
    print(
        f"  {n_channels} channel(s), {n_samples} samples @ {fs} Hz "
        f"({n_samples / fs:.2f} s)"
    )

    tacho, mic_signals = split_tacho_channel(data, tacho_channel)
    rpm = extract_rpm(tacho, fs, threshold)

    t = np.arange(n_samples, dtype=np.float64) / fs
    return rpm, fs, t, mic_signals
