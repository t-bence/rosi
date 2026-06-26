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

Then run (uv handles the Python version and all dependencies automatically):

```bash
uv run main.py
```

That's it. On first run uv will download Python and install packages if needed. Subsequent runs are instant.

## Output

The script produces `rosi_result.png` with three panels:

| Panel | What it shows |
|---|---|
| Raw CSM | The cross-spectral matrix from unprocessed mic signals — rotating sources appear smeared, which is *expected* |
| ROSI map | Power in the rotating frame at the source frequency — peaks should align with the true source positions (cyan circles) |
| DAS spectrum | Frequency spectrum at the peak scan point vs the quietest one — confirms which frequency is dominant |

## Configuration

All settings live in `config.yaml` — no need to touch any Python code.

```yaml
sample_rate:    22050    # Hz
duration:       2.0      # seconds
speed_of_sound: 343.0    # m/s
rpm:            600      # rotor speed [rev/min]

mic_positions_csv: mics.csv   # CSV with x,y,z columns (metres)

scan_grid:
  r_max:   0.80   # outer radius of the scan area [m]
  n_r:     20     # radial resolution
  n_theta: 36     # angular resolution (increase for finer maps)

fft_size: 512     # Welch block length
overlap:  0.5     # Welch overlap fraction (0–1)
f_min:    2000    # frequency band to compute [Hz]
f_max:    4000

sources:          # simulated rotating tonal sources
  - R: 0.50  phi0: 0.0     freq: 3000  amplitude: 1.0
  - R: 0.30  phi0: 2.094   freq: 3300  amplitude: 0.7
```

### Microphone positions

`mics.csv` is a plain CSV file with one mic per row and columns `x, y, z` (metres).
An optional header row is allowed; lines starting with `#` are ignored.

```
x,y,z
1.500000,0.000000,1.500000
1.060660,1.060660,1.500000
...
```

Replace `mics.csv` with your own array layout, or point `mic_positions_csv` at a different file.

## Generate a microphone array

You can add a custom array to `mics.csv` as described above. You can also generate a uniform circular array by running the following:

`uv run utils/generate_array.py -N 24 -R 1 -Z 1.5`

This will create an array with 24 microphones (`N`) at 1 meter radius `R`, at 1.5 m distance from the source plane (`Z`).

Warning: this will overwrite the existing `mics.py`, unless you specify an output microphone file name too with `-o filename.csv`.

## Performance

For a realistic case (161×161 scan grid, 30 s signal at 44100 Hz), the script automatically uses Numba's JIT compiler if available, giving a ~7× speedup over the pure-numpy fallback:

| Backend | Small test (720 pts, 2 s) | 161×161 grid, 30 s |
|---|---|---|
| numpy + joblib | ~3 s | ~0.9 h |
| Numba JIT | ~0.4 s | ~0.1 h |

Numba is included in the default dependencies, so you get the fast path automatically. The first run compiles the kernel (~2 s extra); every subsequent run uses the cached result.

It seems that 1 s of signal takes around 80 s, 10 s of signal takes 800 s to simulate with parameters comparable to my measurements on a MacBook Pro with M2 Pro and 16 GB memory.


## Files

```
config.yaml               — all settings (edit this, not the Python files)
mics.csv                  — microphone positions (x, y, z in metres)
main.py                   — entry point and plots
rosi_sim.py               — simulate rotating sources → mic signals
rosi_beamform.py          — frequency-domain ROSI beamformer (numpy + joblib)
rosi_beamform_numba.py    — same algorithm, Numba JIT (faster)
```

## Method reference

> Sijtsma, P. (2001). *Experimental Techniques for Identification and Characterisation of Noise Sources.* NLR Technical Publication NLR-TP-2001-170. National Aerospace Laboratory NLR.

The forward-propagation (emission-time) formulation used here follows the same principle: for each scan point co-rotating with the rotor, compute the arrival time at each microphone, interpolate the recorded signal, and accumulate the delay-and-sum output. Taking the Welch-averaged power spectrum of this de-rotated signal gives the source power at each frequency without Doppler smearing at any rotor speed.

