"""
ROSI — Rotating Source Identification (Sijtsma's method)

Simulates rotating acoustic point sources, records them on a microphone array,
then reconstructs where the sources are using frequency-domain beamforming.

Run with:
    uv run main.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from scipy.interpolate import griddata

from rosi_sim import simulate_signals
from rosi_beamform import make_scan_grid, compute_global_csm, power_map_to_grid

try:
    from rosi_beamform_numba import rosi_beamform_freq_numba as beamform
    _BACKEND = "Numba JIT (fast)"
except ImportError:
    from rosi_beamform import rosi_beamform_freq as beamform
    _BACKEND = "numpy + joblib (install numba for ~7× speedup)"

# ── Load config ───────────────────────────────────────────────────────────────

def load_mic_positions(csv_path: str | Path) -> np.ndarray:
    """
    Load mic positions from a CSV file.  Returns (N, 3) float64 array.
    Columns must be x, y, z (metres).  An optional header row is skipped,
    as are blank lines and lines starting with #.
    """
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue  # skip header
    if not rows:
        raise ValueError(f"{csv_path}: no numeric rows found")
    arr = np.array(rows, dtype=np.float64)
    if arr.shape[1] != 3:
        raise ValueError(f"{csv_path}: expected 3 columns (x, y, z), got {arr.shape[1]}")
    return arr


cfg_path = Path("config.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

FS      = int(cfg["sample_rate"])
T_TOTAL = float(cfg["duration"])
C       = float(cfg["speed_of_sound"])
OMEGA   = 2 * np.pi * float(cfg["rpm"]) / 60

mic_csv = Path(cfg["mic_positions_csv"])
mic_positions = load_mic_positions(mic_csv)
print(f"Loaded {len(mic_positions)} mic positions from {mic_csv}")

sg_cfg       = cfg["scan_grid"]
SCAN_R_MAX   = float(sg_cfg["r_max"])
N_SCAN_R     = int(sg_cfg["n_r"])
N_SCAN_THETA = int(sg_cfg["n_theta"])

FFT_SIZE = int(cfg["fft_size"])
OVERLAP  = float(cfg["overlap"])
F_MIN    = float(cfg["f_min"])
F_MAX    = float(cfg["f_max"])

SOURCES = []
for s in cfg.get("sources", []):
    SOURCES.append({
        "R":         float(s["R"]),
        "phi0":      float(s["phi0"]),
        "freq":      float(s["freq"]),
        "amplitude": float(s["amplitude"]),
        "phase":     float(s.get("phase", 0.0)),
        "omega":     OMEGA,
    })

# ── Simulate ──────────────────────────────────────────────────────────────────

print(f"Simulating {len(SOURCES)} source(s) on {len(mic_positions)} mics "
      f"({FS} Hz, {T_TOTAL:.1f} s)...")
t0 = time.time()
t, signals = simulate_signals(SOURCES, mic_positions, FS, T_TOTAL, C)
print(f"  Done in {time.time()-t0:.2f} s")

# ── Global CSM (informational only) ──────────────────────────────────────────

print("Computing global CSM...")
freqs_csm, C_csm = compute_global_csm(signals, FS, FFT_SIZE, OVERLAP)

# ── Beamform ──────────────────────────────────────────────────────────────────

scan_grid = make_scan_grid(SCAN_R_MAX, N_SCAN_R, N_SCAN_THETA)
print(f"Running ROSI beamformer on {len(scan_grid)} scan points [{_BACKEND}]...")
t0 = time.time()
freqs_out, power_map = beamform(
    signals, t, mic_positions, scan_grid, OMEGA, C,
    fft_size=FFT_SIZE, overlap=OVERLAP, f_min=F_MIN, f_max=F_MAX,
)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.2f} s")

# ── Timing extrapolation ──────────────────────────────────────────────────────

n_emit_now = int(np.sum(t < t[-1] - (np.max(np.linalg.norm(mic_positions, axis=1)) + SCAN_R_MAX) / C))
n_emit_tgt = int(30 * 44100)
n_scan_tgt = 161 * 161
cost_ratio = (n_emit_tgt / n_emit_now) * (n_scan_tgt / len(scan_grid))
t_tgt = elapsed * cost_ratio
print(f"\n  Extrapolation to 161×161 grid, 30 s @ 44100 Hz:")
print(f"    Estimated time: {t_tgt:,.0f} s = {t_tgt/3600:.1f} h")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 5))
fig.suptitle("ROSI — Rotating Source Identification", fontsize=13)

# Panel 1: global CSM at source frequency (shows smearing — expected)
ax1 = fig.add_subplot(1, 3, 1)
f_idx_csm = np.argmin(np.abs(freqs_csm - SOURCES[0]["freq"])) if SOURCES else 0
im = ax1.imshow(20 * np.log10(np.abs(C_csm[f_idx_csm]) + 1e-12),
                cmap="viridis", origin="upper", aspect="equal")
fig.colorbar(im, ax=ax1, label="dB")
ax1.set_title(f"Raw CSM @ {freqs_csm[f_idx_csm]:.0f} Hz\n"
              f"(spatial smearing expected for rotating sources)")
ax1.set_xlabel("Mic index"); ax1.set_ylabel("Mic index")

# Panel 2: ROSI source map
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_aspect("equal")

f_plot_idx = np.argmin(np.abs(freqs_out - SOURCES[0]["freq"])) if SOURCES else 0
r_vals, theta_vals, power_2d = power_map_to_grid(
    power_map, scan_grid, N_SCAN_R, N_SCAN_THETA, freq_idx=f_plot_idx)
power_db = 10 * np.log10(power_2d / power_2d.max() + 1e-12)

TT, RR = np.meshgrid(theta_vals, r_vals)
pts_x = RR.ravel() * np.cos(TT.ravel())
pts_y = RR.ravel() * np.sin(TT.ravel())
pts_v = power_db.T.ravel()

lim = SCAN_R_MAX * 1.05
gx, gy = np.linspace(-lim, lim, 200), np.linspace(-lim, lim, 200)
GX, GY = np.meshgrid(gx, gy)
GV = griddata((pts_x, pts_y), pts_v, (GX, GY), method="linear", fill_value=np.nan)
GV[GX**2 + GY**2 > SCAN_R_MAX**2] = np.nan

pcm = ax2.pcolormesh(GX, GY, GV, cmap="hot_r", vmin=-6, vmax=0, shading="auto")
fig.colorbar(pcm, ax=ax2, label="dB re peak", fraction=0.046)

bin_half = (freqs_out[1] - freqs_out[0]) / 2
for i, src in enumerate(SOURCES):
    if abs(src["freq"] - freqs_out[f_plot_idx]) <= bin_half:
        ax2.scatter([src["R"] * np.cos(src["phi0"])],
                    [src["R"] * np.sin(src["phi0"])],
                    marker="o", s=150, color="none", edgecolors="cyan",
                    linewidths=2.5, zorder=5, label=f"Src {i+1} ({src['freq']:.0f} Hz)")

for r_ring in np.linspace(SCAN_R_MAX / 4, SCAN_R_MAX, 4):
    ax2.add_patch(plt.Circle((0, 0), r_ring, fill=False,
                              color="white", alpha=0.3, linewidth=0.8))
ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
ax2.set_title(f"ROSI map @ {freqs_out[f_plot_idx]:.0f} Hz\n(rotating frame, 6 dB range)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2, color="white")

# Panel 3: DAS spectrum at peak vs background
ax3 = fig.add_subplot(1, 3, 3)
peak_idx = np.argmax(power_map[:, f_plot_idx])
bg_idx   = np.argmin(power_map[:, f_plot_idx])
ax3.semilogy(freqs_out / 1e3, power_map[peak_idx], label="Peak scan point", linewidth=1.5)
ax3.semilogy(freqs_out / 1e3, power_map[bg_idx],   label="Min scan point",  linewidth=1.0, alpha=0.7)
for src in SOURCES:
    ax3.axvline(src["freq"] / 1e3, color="red", linestyle="--", alpha=0.5,
                label=f"{src['freq']:.0f} Hz")
ax3.set_xlabel("Frequency [kHz]"); ax3.set_ylabel("DAS power")
ax3.set_title("Per-scan-point DAS spectrum")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

plt.tight_layout()
out = "rosi_result.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.show()
