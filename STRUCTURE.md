# ROSI Project Structure

## Overview

The ROSI project is organized with **code** and **data** cleanly separated.

## Directory Structure

```
rosi/
├── Code (Python modules)
│   ├── rosi                    # CLI entry point script
│   ├── config_schema.py        # Pydantic config models
│   ├── main.py                 # Main entry point
│   ├── rosi_cli.py             # CLI command handlers
│   ├── rosi_sim.py             # Signal simulation
│   ├── rosi_beamform.py        # Beamformer (numpy)
│   ├── rosi_beamform_numba.py  # Beamformer (numba JIT)
│   └── utils/
│       └── generate_array.py   # Array generator utility
│
├── Data
│   ├── input/
│   │   ├── config.yaml         # Configuration file
│   │   └── mics.csv            # Microphone array positions
│   │
│   └── output/
│       └── rosi_result.png     # Generated beamforming results
│
├── Config
│   ├── pyproject.toml          # Project dependencies
│   ├── .gitignore              # Git ignore rules
│   └── README.md               # Documentation
│
└── Lock
    └── uv.lock                 # Dependency lock file
```

## Data Folders

### `data/input/`
**Purpose:** User-provided input files  
**Contains:**
- `config.yaml` — Simulation configuration (sample rate, rotor speed, grid resolution, etc.)
- `mics.csv` — Microphone array layout (x, y, z positions)

**What you do:** Edit these files to customize experiments

### `data/output/`
**Purpose:** Generated results  
**Contains:**
- `rosi_result.png` — Beamforming visualization (3-panel figure)
- Other output files as generated

**What you do:** View results, backup interesting runs, etc.

## Why Separate?

✓ **Cleaner codebase** — No data files cluttering the repo  
✓ **Git-friendly** — Output files aren't committed  
✓ **Reproducible** — Input config is tracked in version control  
✓ **Easy to backup** — Keep `data/input/` for important experiments  
✓ **Scalable** — Generate many output files without repo bloat  

## Usage

Everything works as before, but files are in their designated folders:

```bash
# Run with default paths (data/input/config.yaml → data/output/rosi_result.png)
./rosi run

# Use custom config
./rosi run --config experiments/exp1.yaml

# Generate array to data/input/
./rosi generate-array -N 24 -R 1.5 -Z 2.0 -o data/input/my_array.csv

# Override output path
./rosi run --output results/experiment_1.png
```

## What's Tracked by Git

- ✓ `data/input/` — Configuration and baseline array
- ✓ Code files — All Python modules
- ✗ `data/output/` — Generated results (ignored)
- ✗ `__pycache__/` — Python cache
- ✗ `.venv/` — Virtual environment

This keeps the repository small while preserving all reproducibility information.
