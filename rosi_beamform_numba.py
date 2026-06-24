"""
rosi_beamform_numba.py — Numba-JIT ROSI beamformer.

Uses @njit(parallel=True) to compile the innermost (n_emit × n_mics) loops to
native ARM SIMD, running across all performance cores on the M2 Pro.

Key optimisations
─────────────────
1. Trig precomputation (emission times):
   cos(ω·t_emit) and sin(ω·t_emit) computed once for all N_emit times.

2. Distance formula factorisation (works for any mic array geometry):
   dist²(k,n,m) = C_m + r_k² − 2·(cos_ot[n]·P[k,m] + sin_ot[n]·Q[k,m])

   where  C_m = ‖mic_pos[m]‖²                         (per mic, static)
          P[k,m] = r_k·(mic_x[m]·cos θ_k + mic_y[m]·sin θ_k)  (per scan×mic)
          Q[k,m] = r_k·(mic_y[m]·cos θ_k − mic_x[m]·sin θ_k)  (per scan×mic)

   Derivation: ys(k,n) = r_k·[cos(ω·t+θ_k), sin(ω·t+θ_k), 0], expand dist²,
   apply angle-addition identity using cos_ot/sin_ot already in registers.
   For a circular array at uniform radius/height all C_m are equal, giving a
   further simplification — but the code is general.

3. The 16 P and Q values per scan point fit in L1 during the mic inner loop
   → zero mic_pos memory traffic in the hot path.

Memory: das buffer is (chunk_size, N_emit) float64.  Default chunk_size=500:
  2 s @ 22 kHz  (N_emit≈44 k):  500 × 44k × 8 = 176 MB  ✓
  30 s @ 44 kHz (N_emit≈1.3 M): 500 × 1.3M × 8 = 5.2 GB  (reduce if OOM)
"""

import numpy as np
from numba import njit, prange
from scipy.signal import get_window


@njit(parallel=True, cache=True, fastmath=True)
def _compute_das_chunk(
    P: np.ndarray,           # (K, N_mics)  precomputed dot-product helper
    Q: np.ndarray,           # (K, N_mics)  precomputed dot-product helper
    C_mic: np.ndarray,       # (N_mics,)    ‖mic_pos[m]‖²
    scan_r_sq: np.ndarray,   # (K,)         r_k²
    cos_ot: np.ndarray,      # (N_emit,)    cos(omega * t_emit)
    sin_ot: np.ndarray,      # (N_emit,)    sin(omega * t_emit)
    t_emit: np.ndarray,      # (N_emit,)
    signals: np.ndarray,     # (N_mics, N_t)
    t_sig_0: float,
    dt: float,
    n_t: int,
    c: float,
) -> np.ndarray:             # (K, N_emit)
    """
    De-rotated DAS for a chunk of K scan points.

    Inner loop arithmetic (per mic, per emit time):
        dot   = cos_ot * P[k,m]  +  sin_ot * Q[k,m]     # 2 mul + 1 add
        dist  = sqrt(C_m + r_k² − 2·dot)                 # 1 mul + 1 sub + 1 sqrt
        t_arr = t_emit + dist/c                           # 1 div + 1 add
        [interpolate + accumulate]

    No mic_pos loads inside the hot loops.
    """
    K = P.shape[0]
    n_emit = len(t_emit)
    n_mics = P.shape[1]
    das = np.zeros((K, n_emit), np.float64)

    for k in prange(K):
        rk_sq = scan_r_sq[k]

        for n in range(n_emit):
            cos_n = cos_ot[n]
            sin_n = sin_ot[n]
            te    = t_emit[n]

            acc = 0.0
            for m in range(n_mics):
                dot  = cos_n * P[k, m] + sin_n * Q[k, m]
                dist = np.sqrt(C_mic[m] + rk_sq - 2.0 * dot)
                t_arr = te + dist / c
                fi = (t_arr - t_sig_0) / dt
                if fi < 0.0:
                    fi = 0.0
                elif fi > n_t - 2:
                    fi = float(n_t - 2)
                i0 = int(fi)
                a  = fi - i0
                acc += (signals[m, i0] + a * (signals[m, i0 + 1] - signals[m, i0])) * dist
            das[k, n] = acc
    return das


def rosi_beamform_freq_numba(
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
    chunk_size: int = 500,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-JIT ROSI frequency-domain beamformer.

    Same interface as rosi_beamform_freq().  Processes scan_grid in chunks of
    `chunk_size` to keep the DAS buffer under (chunk_size × N_emit × 8) bytes.

    The JIT kernel is compiled on first call (~1–2 s, cached to disk for reuse).
    """
    max_dist = np.max(np.linalg.norm(mic_positions, axis=1)) + np.max(scan_grid[:, 0])
    t_emit   = t_sig[t_sig < t_sig[-1] - max_dist / c]
    n_emit   = len(t_emit)
    dt       = float(t_sig[1] - t_sig[0])
    t_sig_0  = float(t_sig[0])

    # ── Precompute trig for all emission times (used by every scan point) ────
    cos_ot = np.ascontiguousarray(np.cos(omega * t_emit), dtype=np.float64)
    sin_ot = np.ascontiguousarray(np.sin(omega * t_emit), dtype=np.float64)

    # ── Precompute per-mic constant: ‖mic_pos[m]‖² ──────────────────────────
    mic_x  = mic_positions[:, 0].astype(np.float64)
    mic_y  = mic_positions[:, 1].astype(np.float64)
    C_mic  = np.ascontiguousarray(
        mic_x**2 + mic_y**2 + mic_positions[:, 2].astype(np.float64)**2,
        dtype=np.float64
    )  # (N_mics,)

    # ── Precompute P and Q for all scan points ───────────────────────────────
    # P[k,m] = r_k * (mic_x[m]*cos θ_k + mic_y[m]*sin θ_k)
    # Q[k,m] = r_k * (mic_y[m]*cos θ_k − mic_x[m]*sin θ_k)
    r_all     = scan_grid[:, 0].astype(np.float64)          # (N_scan,)
    cos_th_all = np.cos(scan_grid[:, 1]).astype(np.float64)  # (N_scan,)
    sin_th_all = np.sin(scan_grid[:, 1]).astype(np.float64)  # (N_scan,)

    # Shape: (N_scan, N_mics) — a few MB at most
    P_all = (r_all[:, None] *
             (mic_x[None, :] * cos_th_all[:, None]
              + mic_y[None, :] * sin_th_all[:, None])).astype(np.float64)
    Q_all = (r_all[:, None] *
             (mic_y[None, :] * cos_th_all[:, None]
              - mic_x[None, :] * sin_th_all[:, None])).astype(np.float64)
    scan_r_sq_all = (r_all ** 2).astype(np.float64)         # (N_scan,)

    # ── Welch setup ──────────────────────────────────────────────────────────
    hop       = max(1, int(fft_size * (1 - overlap)))
    win       = get_window("hann", fft_size).astype(np.float64)
    win      /= win.sum()
    starts    = np.arange(0, n_emit - fft_size + 1, hop)

    freqs_das = np.fft.rfftfreq(fft_size, d=dt)
    freq_mask = (freqs_das >= f_min) & (freqs_das <= f_max)
    freqs_out = freqs_das[freq_mask]
    n_scan    = scan_grid.shape[0]

    if verbose:
        mem_mb = chunk_size * n_emit * 8 / 1e6
        print(f"  Numba JIT  |  scan: {n_scan}  |  N_emit: {n_emit:,}  |  "
              f"blocks/pt: {len(starts)}  |  chunk: {chunk_size} ({mem_mb:.0f} MB DAS buf)")
        print(f"  (first call triggers JIT compilation, cached to disk afterwards)")

    signals_c = np.ascontiguousarray(signals, dtype=np.float64)
    t_emit_c  = np.ascontiguousarray(t_emit,  dtype=np.float64)

    power_map = np.zeros((n_scan, int(freq_mask.sum())), dtype=np.float32)

    for chunk_start in range(0, n_scan, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_scan)
        if verbose:
            print(f"  {100*chunk_start/n_scan:5.1f}%", end="\r", flush=True)

        das = _compute_das_chunk(
            np.ascontiguousarray(P_all[chunk_start:chunk_end]),
            np.ascontiguousarray(Q_all[chunk_start:chunk_end]),
            C_mic,
            np.ascontiguousarray(scan_r_sq_all[chunk_start:chunk_end]),
            cos_ot, sin_ot, t_emit_c,
            signals_c, t_sig_0, dt, int(signals_c.shape[1]), float(c),
        )  # (K, N_emit)

        psd = np.zeros((chunk_end - chunk_start, len(freqs_das)), dtype=np.float64)
        for s in starts:
            blk  = das[:, s:s + fft_size] * win[None, :]
            psd += np.abs(np.fft.rfft(blk, axis=1)) ** 2
        psd /= len(starts)

        power_map[chunk_start:chunk_end] = psd[:, freq_mask].astype(np.float32)

    if verbose:
        print(f"  100.0%  done")

    return freqs_out, power_map
