"""
ROSI — Rotating Source Identification (Sijtsma's method)

Simulates rotating acoustic point sources, records them on a microphone array,
then reconstructs where the sources are using frequency-domain beamforming.

Run with:
    rosi run
"""

import sys
import time
import argparse
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

# ── Config loading and validation ──────────────────────────────────────────────


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


def load_config(config_path: str | Path) -> dict:
    """Load and parse config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Config file is empty: {config_path}")
    return cfg


def validate_config(config: dict) -> list[str]:
    """
    Validate config structure and values.
    Returns list of error messages (empty = valid).
    """
    errors = []

    # Required top-level keys
    required_keys = [
        "sample_rate", "duration", "speed_of_sound", "rpm",
        "mic_positions_csv", "scan_grid", "fft_size", "overlap",
        "f_min", "f_max", "output_image"
    ]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    # Check for unknown keys (strict)
    allowed_keys = set(required_keys + ["sources"])
    unknown = set(config.keys()) - allowed_keys
    if unknown:
        errors.append(f"Unknown keys in config: {', '.join(sorted(unknown))}")

    if errors:
        return errors

    # Type and range validation
    try:
        sr = int(config["sample_rate"])
        if sr <= 0:
            errors.append("sample_rate must be > 0")
    except (TypeError, ValueError):
        errors.append("sample_rate must be an integer")

    try:
        dur = float(config["duration"])
        if dur <= 0:
            errors.append("duration must be > 0")
    except (TypeError, ValueError):
        errors.append("duration must be a number")

    try:
        c = float(config["speed_of_sound"])
        if c <= 0:
            errors.append("speed_of_sound must be > 0")
    except (TypeError, ValueError):
        errors.append("speed_of_sound must be a number")

    try:
        rpm = float(config["rpm"])
        if rpm < 0:
            errors.append("rpm must be >= 0")
    except (TypeError, ValueError):
        errors.append("rpm must be a number")

    # Mic CSV
    try:
        mic_csv = Path(config["mic_positions_csv"])
        if not mic_csv.exists():
            errors.append(f"mic_positions_csv not found: {mic_csv}")
    except Exception as e:
        errors.append(f"Error checking mic_positions_csv: {e}")

    # Scan grid
    try:
        sg = config["scan_grid"]
        r_max = float(sg["r_max"])
        if r_max <= 0:
            errors.append("scan_grid.r_max must be > 0")
        n_r = int(sg["n_r"])
        if n_r < 1:
            errors.append("scan_grid.n_r must be >= 1")
        n_theta = int(sg["n_theta"])
        if n_theta < 1:
            errors.append("scan_grid.n_theta must be >= 1")
    except Exception as e:
        errors.append(f"Invalid scan_grid: {e}")

    # FFT size (should be power of 2)
    try:
        fft = int(config["fft_size"])
        if fft < 64:
            errors.append("fft_size must be >= 64")
        if (fft & (fft - 1)) != 0:
            errors.append("fft_size should be a power of 2")
    except (TypeError, ValueError):
        errors.append("fft_size must be an integer")

    # Overlap
    try:
        ovlp = float(config["overlap"])
        if not (0 <= ovlp < 1):
            errors.append("overlap must be in [0, 1)")
    except (TypeError, ValueError):
        errors.append("overlap must be a number")

    # Frequencies
    try:
        f_min = float(config["f_min"])
        f_max = float(config["f_max"])
        if f_min < 0:
            errors.append("f_min must be >= 0")
        if f_max <= f_min:
            errors.append("f_max must be > f_min")
    except (TypeError, ValueError):
        errors.append("f_min and f_max must be numbers")

    # Output image
    try:
        out_path = Path(config["output_image"])
        # Check parent directory is writable
        parent = out_path.parent if out_path.parent != Path() else Path(".")
        if not parent.exists():
            errors.append(f"Output directory does not exist: {parent}")
    except Exception as e:
        errors.append(f"Error checking output_image path: {e}")

    return errors


def merge_config_with_args(config: dict, args) -> dict:
    """
    Merge CLI arguments into config dict.
    CLI args override YAML values.
    """
    # Sample rate
    if args.sample_rate is not None:
        config["sample_rate"] = args.sample_rate

    # Duration
    if args.duration is not None:
        config["duration"] = args.duration

    # Speed of sound
    if args.speed_of_sound is not None:
        config["speed_of_sound"] = args.speed_of_sound

    # RPM
    if args.rpm is not None:
        config["rpm"] = args.rpm

    # Mic CSV
    if args.mic_positions_csv is not None:
        config["mic_positions_csv"] = args.mic_positions_csv

    # Scan grid
    if args.r_max is not None:
        config["scan_grid"]["r_max"] = args.r_max
    if args.n_r is not None:
        config["scan_grid"]["n_r"] = args.n_r
    if args.n_theta is not None:
        config["scan_grid"]["n_theta"] = args.n_theta

    # FFT size
    if args.fft_size is not None:
        config["fft_size"] = args.fft_size

    # Overlap
    if args.overlap is not None:
        config["overlap"] = args.overlap

    # Frequencies
    if args.f_min is not None:
        config["f_min"] = args.f_min
    if args.f_max is not None:
        config["f_max"] = args.f_max

    # Output image
    if args.output is not None:
        config["output_image"] = args.output

    return config


# ── Main simulation ────────────────────────────────────────────────────────────


def main_with_args(args):
    """Run ROSI with parsed CLI arguments."""
    
    # Load config
    config_path = Path(args.config)
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return 1

    # Merge CLI args into config
    config = merge_config_with_args(config, args)

    # Validate config
    errors = validate_config(config)
    if errors:
        print("CONFIG VALIDATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    # Print config if --dry-run
    if args.dry_run:
        print("Merged configuration:")
        print(yaml.dump(config, default_flow_style=False))
        print("\nDry run complete. To execute, remove --dry-run flag.")
        return 0

    # Extract config values
    FS = int(config["sample_rate"])
    T_TOTAL = float(config["duration"])
    C = float(config["speed_of_sound"])
    OMEGA = 2 * np.pi * float(config["rpm"]) / 60

    mic_csv = Path(config["mic_positions_csv"])
    mic_positions = load_mic_positions(mic_csv)
    print(f"Loaded {len(mic_positions)} mic positions from {mic_csv}")

    sg_cfg = config["scan_grid"]
    SCAN_R_MAX = float(sg_cfg["r_max"])
    N_SCAN_R = int(sg_cfg["n_r"])
    N_SCAN_THETA = int(sg_cfg["n_theta"])

    FFT_SIZE = int(config["fft_size"])
    OVERLAP = float(config["overlap"])
    F_MIN = float(config["f_min"])
    F_MAX = float(config["f_max"])

    OUTPUT_IMAGE = str(config["output_image"])

    SOURCES = []
    for s in config.get("sources", []):
        SOURCES.append({
            "R": float(s["R"]),
            "phi0": float(s["phi0"]),
            "freq": float(s["freq"]),
            "amplitude": float(s["amplitude"]),
            "phase": float(s.get("phase", 0.0)),
            "omega": OMEGA,
        })

    # ── Simulate ──────────────────────────────────────────────────────────────

    print(f"Simulating {len(SOURCES)} source(s) on {len(mic_positions)} mics "
          f"({FS} Hz, {T_TOTAL:.1f} s)...")
    t0 = time.time()
    t, signals = simulate_signals(SOURCES, mic_positions, FS, T_TOTAL, C)
    print(f"  Done in {time.time()-t0:.2f} s")

    # ── Global CSM (informational only) ────────────────────────────────────

    print("Computing global CSM...")
    freqs_csm, C_csm = compute_global_csm(signals, FS, FFT_SIZE, OVERLAP)

    # ── Beamform ───────────────────────────────────────────────────────────

    scan_grid = make_scan_grid(SCAN_R_MAX, N_SCAN_R, N_SCAN_THETA)
    print(f"Running ROSI beamformer on {len(scan_grid)} scan points [{_BACKEND}]...")
    t0 = time.time()
    freqs_out, power_map = beamform(
        signals, t, mic_positions, scan_grid, OMEGA, C,
        fft_size=FFT_SIZE, overlap=OVERLAP, f_min=F_MIN, f_max=F_MAX,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f} s")

    # ── Timing extrapolation ───────────────────────────────────────────────

    n_emit_now = int(np.sum(t < t[-1] - (np.max(np.linalg.norm(mic_positions, axis=1)) + SCAN_R_MAX) / C))
    n_emit_tgt = int(30 * 44100)
    n_scan_tgt = 161 * 161
    cost_ratio = (n_emit_tgt / n_emit_now) * (n_scan_tgt / len(scan_grid))
    t_tgt = elapsed * cost_ratio
    print(f"\n  Extrapolation to 161×161 grid, 30 s @ 44100 Hz:")
    print(f"    Estimated time: {t_tgt:,.0f} s = {t_tgt/3600:.1f} h")

    # ── Plot ───────────────────────────────────────────────────────────────

    if not args.no_plot:
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
        ax1.set_xlabel("Mic index")
        ax1.set_ylabel("Mic index")

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
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.set_title(f"ROSI map @ {freqs_out[f_plot_idx]:.0f} Hz\n(rotating frame, 6 dB range)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2, color="white")

        # Panel 3: DAS spectrum at peak vs background
        ax3 = fig.add_subplot(1, 3, 3)
        peak_idx = np.argmax(power_map[:, f_plot_idx])
        bg_idx = np.argmin(power_map[:, f_plot_idx])
        ax3.semilogy(freqs_out / 1e3, power_map[peak_idx], label="Peak scan point", linewidth=1.5)
        ax3.semilogy(freqs_out / 1e3, power_map[bg_idx], label="Min scan point", linewidth=1.0, alpha=0.7)
        for src in SOURCES:
            ax3.axvline(src["freq"] / 1e3, color="red", linestyle="--", alpha=0.5,
                        label=f"{src['freq']:.0f} Hz")
        ax3.set_xlabel("Frequency [kHz]")
        ax3.set_ylabel("DAS power")
        ax3.set_title("Per-scan-point DAS spectrum")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {OUTPUT_IMAGE}")
        plt.show()
    else:
        print(f"(Skipping plot, would save to {OUTPUT_IMAGE})")

    return 0


def main():
    """Backward compatibility: allow running as script."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ROSI — Rotating Source Identification (backward compat entry point)"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--speed-of-sound", type=float)
    parser.add_argument("--rpm", type=float)
    parser.add_argument("--mic-positions-csv", type=str)
    parser.add_argument("--r-max", type=float)
    parser.add_argument("--n-r", type=int)
    parser.add_argument("--n-theta", type=int)
    parser.add_argument("--fft-size", type=int)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--f-min", type=float)
    parser.add_argument("--f-max", type=float)

    args = parser.parse_args()
    sys.exit(main_with_args(args))


if __name__ == "__main__":
    main()
