# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run all commands via the `./rosi` wrapper (uses `uv` to manage Python env automatically):

```bash
./rosi validate                                      # validate config + check Numba
./rosi run                                           # simulate + beamform → PNG
./rosi run --dry-run                                 # print merged config, skip compute
./rosi run --no-plot --output /tmp/result.png        # headless mode
./rosi generate-array -N 24 -R 1.5 -Z 2.0 -o data/input/mics.csv
```

To run a Python file directly (bypassing the CLI):

```bash
uv run python main.py
uv run python -c "from config_schema import load_config_from_yaml; ..."
```

There are no automated tests; validate by running `./rosi validate` then `./rosi run --dry-run`.

## Architecture

The pipeline has three stages, each in its own module:

1. **`rosi_sim.py`** — simulates rotating point sources, computes Doppler-shifted arrival times at each microphone, and returns time-domain pressure signals (numpy arrays).

2. **`rosi_beamform.py` / `rosi_beamform_numba.py`** — delay-and-sum beamformer operating in the co-rotating frame. `main.py` auto-selects the Numba variant if available (~7× faster); both expose the same public API. The key operation: for each scan point, interpolate mic signals at their Doppler-corrected timestamps, sum across mics, then Welch-average the FFT to get power vs. frequency.

3. **`main.py`** (`main_with_args`) — orchestrates the two stages, generates the 3-panel result figure (raw CSM / ROSI map / DAS spectrum), and saves to PNG.

**Config layer:** `config_schema.py` defines Pydantic models (`ROSIConfig`, `ScanGridConfig`, `SourceConfig`). Config is loaded from YAML, validated, and CLI overrides are merged before any computation. The CLI (`rosi_cli.py`) handles argument parsing and dispatches to handlers that call `main_with_args`.

**Data layout:**
- `data/input/config.yaml` — all simulation/beamforming parameters (edit to customize)
- `data/input/mics.csv` — microphone x,y,z positions in metres
- `data/output/` — generated results (git-ignored)

## Key constraints

- `fft_size` must be a power of 2 and ≥ 64 (validated by Pydantic).
- `overlap` must be in `[0, 1)`.
- The output directory must exist before running — `data/output/` is created on first use but not auto-created for custom paths.
- Numba JIT cache is stored in `__pycache__`; first run after code changes incurs ~2 s recompilation.
- Runtime scales roughly as `O(n_scan_points × duration × sample_rate)` — keep `duration` short (≤ 5 s) during development.
