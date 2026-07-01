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
    from config_schema import load_config_from_yaml
    from pydantic import ValidationError
    
    config_path = args.config
    
    # Try to load and validate config
    try:
        load_config_from_yaml(config_path)
        print("✓ Config is valid")
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
    
    # Check Numba availability
    try:
        import importlib.util
        if importlib.util.find_spec("numba") is not None:
            print("✓ Numba available (will use fast JIT backend)")
        else:
            print("⚠ Numba not available (will use slower numpy+joblib backend)")
    except Exception:
        print("⚠ Numba not available (will use slower numpy+joblib backend)")
    
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
        "--config", type=str, default="data/input/config.yaml",
        help="Path to config file (default: data/input/config.yaml)"
    )
    val.set_defaults(func=cmd_validate)
    
    # ─ rosi run ────────────────────────────────────────────────────────────────
    run = sub.add_parser(
        "run",
        help="Run ROSI simulation and beamforming"
    )
    run.add_argument(
        "--config", type=str, default="data/input/config.yaml",
        help="Path to config file (default: data/input/config.yaml)"
    )
    run.add_argument(
        "--output", type=str, default=None,
        help="Output image filename (default: data/output/rosi_result.png)"
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
    run.add_argument("--array-distance", type=float, help="Override array_distance")
    run.add_argument("--tach-channel", type=int, help="Override tach_channel")
    run.add_argument("--rotation-direction", type=int, choices=[1, -1],
                      help="Override rotation_direction (1=CCW, -1=CW)")
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
