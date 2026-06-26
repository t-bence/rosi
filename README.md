# ROSI — Rotating Source Identification

A Python implementation of the ROSI beamforming method by Sijtsma (AIAA 2001-2167) for localising acoustic sources on a rotating structure (propellers, fans, wind turbines) using a fixed microphone array.

## What it does

1. **Simulates** rotating tonal point sources radiating sound into the air
2. **Records** the pressure at each microphone, accounting for the Doppler-shifted travel time from the moving source to each fixed mic
3. **Reconstructs** the source distribution using delay-and-sum beamforming in the rotating reference frame, then Welch-averaged FFT to get a power map per frequency

The key insight of ROSI: before taking any FFT, the signals are first aligned in time to a co-rotating scan point. This undoes the Doppler smearing that would otherwise make it impossible to localise sources at typical rotor speeds.

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) — a fast Python package manager. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then use the `rosi` command (uv handles the Python version and all dependencies automatically):

```bash
# Validate everything is ready
./rosi validate

# Run the simulation + beamforming
./rosi run
```

That's it. On first run uv will download Python and install packages if needed. Subsequent runs are instant.

## Output

Every run writes two files to `data/output/` (same stem, different extension):

| File | Contents |
|---|---|
| `rosi_result.png` | Three-panel figure (see below) |
| `rosi_result.mat` | Numeric results in MATLAB format (see below) |

### Figure panels

| Panel | What it shows |
|---|---|
| Raw CSM | The cross-spectral matrix from unprocessed mic signals — rotating sources appear smeared, which is *expected* |
| ROSI map | Power in the rotating frame at the source frequency — peaks should align with the true source positions (cyan circles) |
| DAS spectrum | Frequency spectrum at the peak scan point vs the quietest one — confirms which frequency is dominant |

### MATLAB data file

Load with `d = load('data/output/rosi_result.mat')`. Variables:

| Variable | Shape | Description |
|---|---|---|
| `freqs_csm` | `(N_freq × 1)` | Frequency axis for `C_csm` — full spectrum, 0 to fs/2 [Hz] |
| `C_csm` | `(N_freq × N_mics × N_mics)` | Global cross-spectral matrix, complex double |
| `freqs_beamform` | `(N_f × 1)` | Frequency axis for `power_map` — band-limited to `[f_min, f_max]` [Hz] |
| `power_map` | `(N_scan × N_f)` | Beamformer power at each scan point and frequency |
| `scan_grid` | `(N_scan × 2)` | Scan point coordinates: column 1 = r [m], column 2 = θ [rad] |

`freqs_csm` and `freqs_beamform` share the same bin spacing (`fs / fft_size`) but different coverage: `freqs_csm` spans the full FFT range while `freqs_beamform` is the subset inside your configured `[f_min, f_max]` band. Use `freqs_csm` to index into `C_csm` and `freqs_beamform` to index into `power_map`.

## Configuration

All settings live in `data/input/config.yaml` — no need to touch any Python code.

```yaml
sample_rate:    22050    # Hz
duration:       2.0      # seconds
speed_of_sound: 343.0    # m/s
rpm:            600      # rotor speed [rev/min]

mic_positions_csv: data/input/mics.csv   # CSV with x,y,z columns (metres)

scan_grid:
  r_max:   0.80   # outer radius of the scan area [m]
  n_r:     20     # radial resolution
  n_theta: 36     # angular resolution (increase for finer maps)

fft_size: 512     # Welch block length
overlap:  0.5     # Welch overlap fraction (0–1)
f_min:    2000    # frequency band to compute [Hz]
f_max:    4000

output_image: data/output/rosi_result.png  # output PNG path

sources:          # simulated rotating tonal sources
  - R: 0.50  phi0: 0.0     freq: 3000  amplitude: 1.0
  - R: 0.30  phi0: 2.094   freq: 3300  amplitude: 0.7
```

### Microphone positions

`data/input/mics.csv` is a plain CSV file with one mic per row and columns `x, y, z` (metres).
An optional header row is allowed; lines starting with `#` are ignored.

```
x,y,z
1.500000,0.000000,1.500000
1.060660,1.060660,1.500000
...
```

Replace with your own array layout, or point `mic_positions_csv` to a different file in `data/input/`.

## Command-line interface

All ROSI tasks use the `rosi` command:

```bash
./rosi --help                                    # Show all commands
./rosi validate                                  # Validate config and check dependencies
./rosi generate-array -N 24 -R 1.5 -Z 2.0      # Create microphone array
./rosi run                                       # Run simulation + beamforming
```

### Run simulation with custom parameters

Override any config value without editing `config.yaml`:

```bash
./rosi run --rpm 600 --duration 5               # Change rotor speed and duration
./rosi run --f-min 1000 --f-max 5000            # Change frequency band
./rosi run --config my_experiment.yaml          # Use a different config file
./rosi run --output my_results.png              # Custom output filename
```

### Validation and dry-run

Validate your configuration before running a long simulation:

```bash
./rosi validate                                  # Check config, deps, and numba
./rosi run --dry-run                             # Show merged config, don't compute
./rosi run --dry-run --rpm 1200 --duration 30   # Preview what will run
```

### Headless mode

Skip matplotlib and save directly to file:

```bash
./rosi run --no-plot --output /tmp/result.png
```

## Generate a microphone array

Generate a uniform circular microphone array into `data/input/`:

```bash
./rosi generate-array -N 24 -R 1.5 -Z 2.0 -o data/input/mics.csv
```

This creates an array with 24 microphones (`-N`) at 1.5 meter radius (`-R`), at 2.0 m height (`-Z`).

The command will **prompt before overwriting** the output file for safety. Use `-f` / `--force` to skip the prompt:

```bash
./rosi generate-array -N 32 -R 2.0 -Z 1.5 -o data/input/my_array.csv      # Creates custom array
./rosi generate-array -N 32 -R 2.0 -Z 1.5 -o data/input/my_array.csv -f   # Force overwrite
```

You can also manually edit CSV files in `data/input/` (plain CSV with columns `x, y, z` in metres).

## Performance

For a realistic case (161×161 scan grid, 30 s signal at 44100 Hz), the script automatically uses Numba's JIT compiler if available, giving a ~7× speedup over the pure-numpy fallback:

| Backend | Small test (720 pts, 2 s) | 161×161 grid, 30 s |
|---|---|---|
| numpy + joblib | ~3 s | ~0.9 h |
| Numba JIT | ~0.4 s | ~0.1 h |

Numba is included in the default dependencies, so you get the fast path automatically. The first run compiles the kernel (~2 s extra); every subsequent run uses the cached result.

It seems that 1 s of signal takes around 80 s, 10 s of signal takes 800 s to simulate with parameters comparable to my measurements on a MacBook Pro with M2 Pro and 16 GB memory.


## Files and Directories

**Project structure:**
```
rosi/                           — Project root
├── Code
│   ├── rosi                    — CLI entry point (run commands via this script)
│   ├── config_schema.py        — Pydantic config validation models
│   ├── main.py                 — Entry point and plots
│   ├── rosi_cli.py             — CLI command definitions
│   ├── rosi_sim.py             — Simulate rotating sources → mic signals
│   ├── rosi_beamform.py        — Frequency-domain ROSI beamformer (numpy + joblib)
│   ├── rosi_beamform_numba.py  — Same algorithm, Numba JIT (faster)
│   └── utils/generate_array.py — Generate circular microphone arrays
│
└── Data
    ├── data/input/
    │   ├── config.yaml         — All settings (edit this)
    │   └── mics.csv            — Microphone positions (x, y, z in metres)
    │
    └── data/output/
        └── rosi_result.png     — Output beamforming results
```

**Edit these to customize:**
- `data/input/config.yaml` — Signal, scan grid, beamforming parameters
- `data/input/mics.csv` — Microphone array layout

## Method reference

> Sijtsma, P. (2001). *Experimental Techniques for Identification and Characterisation of Noise Sources.* NLR Technical Publication NLR-TP-2001-170. National Aerospace Laboratory NLR.

The forward-propagation (emission-time) formulation used here follows the same principle: for each scan point co-rotating with the rotor, compute the arrival time at each microphone, interpolate the recorded signal, and accumulate the delay-and-sum output. Taking the Welch-averaged power spectrum of this de-rotated signal gives the source power at each frequency without Doppler smearing at any rotor speed.

