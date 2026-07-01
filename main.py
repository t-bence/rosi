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
from pathlib import Path
from scipy.interpolate import griddata
from scipy.io import savemat
from pydantic import ValidationError

from config_schema import load_config_from_yaml, merge_config_with_overrides, ROSIConfig
from rosi_sim import simulate_signals
from rosi_wav import load_wav_signals
from rosi_tach import extract_rpm_from_tach, rotor_phase_from_tach
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
            parts = line.replace(",", " ").split()
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


def load_and_validate_config(config_path: str, cli_args) -> ROSIConfig:
    """
    Load config from YAML and merge with CLI overrides.
    Raises ValidationError if config is invalid.
    """
    # Load YAML config
    config = load_config_from_yaml(config_path)
    
    # Prepare CLI overrides
    overrides = {
        "sample_rate": cli_args.sample_rate,
        "duration": cli_args.duration,
        "speed_of_sound": cli_args.speed_of_sound,
        "rpm": cli_args.rpm,
        "mic_positions_csv": cli_args.mic_positions_csv,
        "array_distance": cli_args.array_distance,
        "tach_channel": cli_args.tach_channel,
        "rotation_direction": cli_args.rotation_direction,
        "r_max": cli_args.r_max,
        "n_r": cli_args.n_r,
        "n_theta": cli_args.n_theta,
        "fft_size": cli_args.fft_size,
        "overlap": cli_args.overlap,
        "f_min": cli_args.f_min,
        "f_max": cli_args.f_max,
        "output_image": cli_args.output,
    }
    
    # Remove None values
    overrides = {k: v for k, v in overrides.items() if v is not None}
    
    # Merge and re-validate
    if overrides:
        config = merge_config_with_overrides(config, overrides)
    
    return config


# ── Main simulation ────────────────────────────────────────────────────────────


def main_with_args(args):
    """Run ROSI with parsed CLI arguments."""
    
    # Load and validate config with Pydantic
    try:
        config = load_and_validate_config(args.config, args)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except ValidationError as e:
        print("CONFIG VALIDATION FAILED:")
        for error in e.errors():
            loc = " → ".join(str(x) for x in error["loc"])
            print(f"  {loc}: {error['msg']}")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    # Print config if --dry-run
    if args.dry_run:
        print("Merged configuration:")
        print(config.model_dump_json(indent=2))
        print("\nDry run complete. To execute, remove --dry-run flag.")
        return 0

    # Extract config values
    C = float(config.speed_of_sound)
    ROT_DIR = int(config.rotation_direction)
    OMEGA = ROT_DIR * 2 * np.pi * float(config.rpm) / 60 if config.rpm is not None else None

    mic_csv = Path(config.mic_positions_csv)
    mic_positions = load_mic_positions(mic_csv)
    mic_positions[:, 2] += float(config.array_distance)
    print(f"Loaded {len(mic_positions)} mic positions from {mic_csv} "
          f"(array_distance={config.array_distance:.3f} m)")

    SCAN_R_MAX = float(config.scan_grid.r_max)
    N_SCAN_R = int(config.scan_grid.n_r)
    N_SCAN_THETA = int(config.scan_grid.n_theta)

    FFT_SIZE = int(config.fft_size)
    OVERLAP = float(config.overlap)
    F_MIN = float(config.f_min)
    F_MAX = float(config.f_max)

    OUTPUT_IMAGE = str(config.output_image)

    SOURCES = []
    for s in config.sources:
        SOURCES.append({
            "R": float(s.R),
            "phi0": float(s.phi0),
            "freq": float(s.freq),
            "amplitude": float(s.amplitude),
            "phase": float(s.phase),
            "omega": OMEGA,
        })

    # ── Acquire signals ───────────────────────────────────────────────────────

    if config.wav_file:
        print(f"Loading signals from WAV: {config.wav_file}")
        FS, t, signals = load_wav_signals(config.wav_file)
        n_wav_ch = signals.shape[0]
        n_mics = len(mic_positions)
        if n_wav_ch != n_mics:
            print(f"ERROR: WAV has {n_wav_ch} channels but mic array has {n_mics} positions "
                  f"(mic_positions_csv rows must correspond 1:1 to WAV channels, including "
                  f"a placeholder row for tach_channel if set)")
            return 1

        pulse_times = None
        if config.tach_channel is not None:
            rpm_measured, pulse_times = extract_rpm_from_tach(signals[config.tach_channel], FS)
            rpm_per_rev = 60.0 / np.diff(pulse_times)
            print(f"  Tach channel {config.tach_channel}: {len(pulse_times)} pulses, "
                  f"RPM mean={rpm_measured:.2f} min={rpm_per_rev.min():.2f} "
                  f"max={rpm_per_rev.max():.2f} (1 pulse/rev, per-revolution omega used)")
            signals = np.delete(signals, config.tach_channel, axis=0)
            mic_positions = np.delete(mic_positions, config.tach_channel, axis=0)

        if config.duration is not None:
            n_samples = min(len(t), int(round(float(config.duration) * FS)))
            t, signals = t[:n_samples], signals[:, :n_samples]
        T_TOTAL = float(t[-1])
        print(f"  {signals.shape[0]} mic channels, {FS} Hz, {T_TOTAL:.2f} s")

        THETA = ROT_DIR * rotor_phase_from_tach(pulse_times, t) if pulse_times is not None else OMEGA * t
    else:
        FS = int(config.sample_rate)
        T_TOTAL = float(config.duration)
        print(f"Simulating {len(SOURCES)} source(s) on {len(mic_positions)} mics "
              f"({FS} Hz, {T_TOTAL:.1f} s)...")
        t0 = time.time()
        t, signals = simulate_signals(SOURCES, mic_positions, FS, T_TOTAL, C)
        print(f"  Done in {time.time()-t0:.2f} s")
        THETA = OMEGA * t

    # ── Global CSM (informational only) ────────────────────────────────────

    print("Computing global CSM...")
    freqs_csm, C_csm = compute_global_csm(signals, FS, FFT_SIZE, OVERLAP)

    # ── Beamform ───────────────────────────────────────────────────────────

    scan_grid = make_scan_grid(SCAN_R_MAX, N_SCAN_R, N_SCAN_THETA)
    print(f"Running ROSI beamformer on {len(scan_grid)} scan points [{_BACKEND}]...")
    t0 = time.time()
    freqs_out, power_map = beamform(
        signals, t, mic_positions, scan_grid, THETA, C,
        fft_size=FFT_SIZE, overlap=OVERLAP, f_min=F_MIN, f_max=F_MAX,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f} s")

    # ── Save results ──────────────────────────────────────────────────────

    mat_path = Path(OUTPUT_IMAGE).with_suffix(".mat")
    savemat(str(mat_path), {
        "freqs_csm":      freqs_csm,
        "C_csm":          C_csm,
        "freqs_beamform": freqs_out,
        "power_map":      power_map,
        "scan_grid":      scan_grid,
    })
    print(f"Saved results → {mat_path}")

    # ── Timing extrapolation ───────────────────────────────────────────────

    n_emit_now = int(np.sum(t < t[-1] - (np.max(np.linalg.norm(mic_positions, axis=1)) + SCAN_R_MAX) / C))
    n_emit_tgt = int(30 * 44100)
    n_scan_tgt = 161 * 161
    cost_ratio = (n_emit_tgt / n_emit_now) * (n_scan_tgt / len(scan_grid))
    t_tgt = elapsed * cost_ratio
    print("\n  Extrapolation to 161×161 grid, 30 s @ 44100 Hz:")
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
    parser.add_argument("--array-distance", type=float)
    parser.add_argument("--tach-channel", type=int)
    parser.add_argument("--rotation-direction", type=int, choices=[1, -1])
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
