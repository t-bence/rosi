"""
rosi_beamform.py — Frequency-domain ROSI beamformer with joblib parallelism.

Algorithm (Sijtsma):
  For each scan point k in the rotating frame:
    1. Compute forward propagation delay to each mic: τ_km(t_emit) = |x_m − y_k(t_emit)| / c
    2. Evaluate p_m at arrival time t_emit + τ_km  (de-rotate / align signals)
    3. DAS = Σ_m  p_m(t_emit + τ_km) · dist_km   (compensate 1/r spreading)
    4. Welch FFT of DAS  →  per-frequency power at scan point k

Interpolation: numpy.interp (linear, uniform grid) — ~2× faster than cubic splines;
accuracy loss is negligible at 44.1 kHz for sources below ~10 kHz.
Parallelism: joblib process pool over the scan-point axis (embarrassingly parallel).
"""

import numpy as np
from scipy.signal import get_window
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm


# ── Scan grid ─────────────────────────────────────────────────────────────────

def make_scan_grid(r_max: float, n_r: int, n_theta: int) -> np.ndarray:
    """Return (n_r * n_theta, 2) scan points (r, theta_offset) in the rotating frame."""
    r_vals = np.linspace(r_max / n_r, r_max, n_r)
    theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rr, tt = np.meshgrid(r_vals, theta_vals)
    return np.column_stack([rr.ravel(), tt.ravel()])


# ── Global CSM (informational) ────────────────────────────────────────────────

def compute_global_csm(
    signals: np.ndarray,
    fs: float,
    fft_size: int = 4096,
    overlap: float = 0.5,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-averaged CSM from raw (non-de-rotated) signals.
    Useful for checking active frequencies; NOT usable for spatial localisation
    of rotating sources (the source smears across all mic pairs).

    Returns: freqs (N_freq,), C (N_freq, N_mics, N_mics)
    """
    n_mics, n_samples = signals.shape
    hop = int(fft_size * (1 - overlap))
    win = get_window(window, fft_size)
    win /= win.sum()
    starts = np.arange(0, n_samples - fft_size + 1, hop)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / fs)
    C = np.zeros((len(freqs), n_mics, n_mics), dtype=np.complex128)
    for s in starts:
        P = np.array([np.fft.rfft(signals[m, s:s + fft_size] * win) for m in range(n_mics)])
        C += np.einsum("mf,nf->fmn", P, np.conj(P))
    C /= len(starts)
    return freqs, C


# ── Per-scan-point worker (module-level so joblib can pickle it) ──────────────

def _beamform_one(
    k: int,
    scan_grid: np.ndarray,
    signals: np.ndarray,
    t_emit: np.ndarray,
    t_sig: np.ndarray,
    mic_positions: np.ndarray,
    omega: float,
    c: float,
    fft_size: int,
    win: np.ndarray,
    block_starts: np.ndarray,
    freq_mask: np.ndarray,
) -> np.ndarray:
    """Compute the de-rotated DAS spectrum for one scan point. Returns (N_f,) power."""
    scan_r, scan_theta = scan_grid[k]
    n_emit = len(t_emit)

    theta_s = omega * t_emit + scan_theta
    ys = np.empty((n_emit, 3))
    ys[:, 0] = scan_r * np.cos(theta_s)
    ys[:, 1] = scan_r * np.sin(theta_s)
    ys[:, 2] = 0.0

    das = np.zeros(n_emit)
    for m in range(signals.shape[0]):
        diff = mic_positions[m] - ys          # (n_emit, 3)
        dist = np.linalg.norm(diff, axis=1)   # (n_emit,)
        t_arr = t_emit + dist / c
        das += np.interp(t_arr, t_sig, signals[m]) * dist

    # Welch FFT of the de-rotated DAS time series
    psd = np.zeros(len(freq_mask))
    for s in block_starts:
        block_fft = np.fft.rfft(das[s:s + fft_size] * win)
        psd += np.abs(block_fft) ** 2
    psd /= len(block_starts)

    return psd[freq_mask]


# ── Main beamformer ───────────────────────────────────────────────────────────

def rosi_beamform_freq(
    signals: np.ndarray,
    t_sig: np.ndarray,
    mic_positions: np.ndarray,
    scan_grid: np.ndarray,
    omega: float,
    c: float,
    fft_size: int = 512,
    overlap: float = 0.5,
    f_min: float = 0.0,
    f_max: float = np.inf,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Correct ROSI frequency-domain beamformer.

    signals       : (N_mics, N_samples) raw mic signals
    t_sig         : (N_samples,)        time axis matching signals
    mic_positions : (N_mics, 3)
    scan_grid     : (N_scan, 2)         (r, theta_offset) in rotating frame
    omega         : rotation angular velocity [rad/s]
    c             : speed of sound [m/s]
    fft_size      : Welch block length applied to the de-rotated DAS series
    overlap       : Welch overlap fraction
    f_min, f_max  : frequency band of interest [Hz]
    n_jobs        : joblib workers (-1 = all cores)

    Returns
    -------
    freqs_out  : (N_f,)        frequency bins in [f_min, f_max]
    power_map  : (N_scan, N_f) beamformer power
    """
    # Trim emission times so all arrival times stay inside the signal window
    max_dist = np.max(np.linalg.norm(mic_positions, axis=1)) + np.max(scan_grid[:, 0])
    t_emit = t_sig[t_sig < (t_sig[-1] - max_dist / c)]
    n_emit = len(t_emit)

    # Welch setup
    hop = max(1, int(fft_size * (1 - overlap)))
    win = get_window("hann", fft_size).astype(np.float64)
    win /= win.sum()
    block_starts = np.arange(0, n_emit - fft_size + 1, hop)

    freqs_das = np.fft.rfftfreq(fft_size, d=t_sig[1] - t_sig[0])
    freq_mask = (freqs_das >= f_min) & (freqs_das <= f_max)
    freqs_out = freqs_das[freq_mask]
    n_scan = scan_grid.shape[0]

    n_workers = cpu_count() if n_jobs == -1 else n_jobs
    if verbose:
        print(f"  Workers: {n_workers}  |  scan points: {n_scan}  |  "
              f"N_emit: {n_emit:,}  |  Welch blocks/pt: {len(block_starts)}")

    # Use tqdm to wrap the generator for progress bar
    jobs_gen = (
        delayed(_beamform_one)(
            k, scan_grid, signals, t_emit, t_sig,
            mic_positions, omega, c,
            fft_size, win, block_starts, freq_mask,
        )
        for k in range(n_scan)
    )

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        tqdm(jobs_gen, total=n_scan, desc="Beamforming", unit="point")
    )

    power_map = np.array(results)   # (N_scan, N_f)
    return freqs_out, power_map


# ── Reshape helpers ───────────────────────────────────────────────────────────

def power_map_to_grid(
    power_map: np.ndarray,
    scan_grid: np.ndarray,
    n_r: int,
    n_theta: int,
    freq_idx: int = 0,
) -> tuple:
    """Reshape power_map[:, freq_idx] to (n_theta, n_r). Returns (r_vals, theta_vals, power_2d)."""
    r_vals = np.unique(scan_grid[:, 0])
    theta_vals = np.unique(scan_grid[:, 1])
    power_2d = power_map[:, freq_idx].reshape(n_theta, n_r)
    return r_vals, theta_vals, power_2d
