"""
rosi_wav.py — Load a multi-channel WAV file as microphone signals.
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def load_wav_signals(wav_path: str | Path) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Read a multi-channel WAV file and return (sample_rate, t, signals).

    signals shape: (n_channels, n_samples), float64, normalised to [-1, 1]
    t shape:       (n_samples,), seconds from zero
    """
    fs, data = wavfile.read(wav_path)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    signals = data.T.astype(np.float64)

    if np.issubdtype(data.dtype, np.integer):
        signals /= np.iinfo(data.dtype).max

    n_samples = signals.shape[1]
    t = np.arange(n_samples) / fs

    return fs, t, signals
