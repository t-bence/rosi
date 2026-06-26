"""
ROSI CLI — Unified entry point for all commands.
"""

import sys
import argparse
from pathlib import Path


def cmd_generate_array(args):
    """Handler for 'rosi generate-array' subcommand."""
    from utils.generate_array import generate_cylindrical_csv
    
    output_path = Path(args.output)
    
    # Prompt before overwrite unless --force
    if output_path.exists() and not args.force:
        response = input(f"File '{args.output}' already exists. Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Cancelled.")
            return 0
    
    generate_cylindrical_csv(args.output, args.N, args.R, args.Z)
    return 0


def cmd_validate(args):
    """Handler for 'rosi validate' subcommand."""
    import yaml
    from pathlib import Path
    
    config_path = Path(args.config)
    
    # Check file exists
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1
    
    # Load YAML
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to parse {config_path}: {e}")
        return 1
    
    if not config:
        print(f"ERROR: Config file is empty: {config_path}")
        return 1
    
    # Basic structure checks
    errors = []
    
    # Check required top-level keys
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
    
    # Check mic CSV exists and is readable
    mic_csv = Path(config.get("mic_positions_csv", ""))
    if mic_csv and not mic_csv.exists():
        errors.append(f"Mic CSV not found: {mic_csv}")
    
    if errors:
        print("Config validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1
    
    # Check Numba availability
    try:
        import numba
        print("✓ Numba available (will use fast JIT backend)")
    except ImportError:
        print("⚠ Numba not available (will use slower numpy+joblib backend)")
    
    print("✓ Config is valid")
    return 0


def cmd_run(args):
    """Handler for 'rosi run' subcommand."""
    # Import main() function and set up sys.argv to pass parsed args
    # We'll call main() directly with the args object
    from main import main_with_args
    
    return main_with_args(args)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="rosi",
        description="ROSI — Rotating Source Identification beamforming"
    )
    
    sub = parser.add_subparsers(dest="cmd", required=True, help="Command to run")
    
    # ─ rosi generate-array ─────────────────────────────────────────────────────
    gen = sub.add_parser(
        "generate-array",
        help="Generate a circular microphone array CSV"
    )
    gen.add_argument(
        "-N", type=int, default=100,
        help="Number of microphones (default: 100)"
    )
    gen.add_argument(
        "-R", type=float, default=5.0,
        help="Radius in metres (default: 5.0)"
    )
    gen.add_argument(
        "-Z", type=float, default=10.0,
        help="Height (z-coordinate) in metres (default: 10.0)"
    )
    gen.add_argument(
        "-o", "--output", type=str, default="mics.csv",
        help="Output CSV filename (default: mics.csv)"
    )
    gen.add_argument(
        "-f", "--force", action="store_true",
        help="Overwrite output file without prompting"
    )
    gen.set_defaults(func=cmd_generate_array)
    
    # ─ rosi validate ───────────────────────────────────────────────────────────
    val = sub.add_parser(
        "validate",
        help="Validate config and check dependencies"
    )
    val.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    val.set_defaults(func=cmd_validate)
    
    # ─ rosi run ────────────────────────────────────────────────────────────────
    run = sub.add_parser(
        "run",
        help="Run ROSI simulation and beamforming"
    )
    run.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    run.add_argument(
        "--output", type=str, default=None,
        help="Output image filename (overrides config.yaml)"
    )
    run.add_argument(
        "--dry-run", action="store_true",
        help="Validate and show config without running simulation"
    )
    run.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib plotting (headless mode)"
    )
    
    # Config overrides
    run.add_argument("--sample-rate", type=int, help="Override sample_rate")
    run.add_argument("--duration", type=float, help="Override duration")
    run.add_argument("--speed-of-sound", type=float, help="Override speed_of_sound")
    run.add_argument("--rpm", type=float, help="Override rpm")
    run.add_argument("--mic-positions-csv", type=str, help="Override mic_positions_csv")
    run.add_argument("--r-max", type=float, help="Override scan_grid.r_max")
    run.add_argument("--n-r", type=int, help="Override scan_grid.n_r")
    run.add_argument("--n-theta", type=int, help="Override scan_grid.n_theta")
    run.add_argument("--fft-size", type=int, help="Override fft_size")
    run.add_argument("--overlap", type=float, help="Override overlap")
    run.add_argument("--f-min", type=float, help="Override f_min")
    run.add_argument("--f-max", type=float, help="Override f_max")
    
    run.set_defaults(func=cmd_run)
    
    # Parse and dispatch
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
