# PDHproject

Final project: numerical simulations for laser locking to a Fabry-Perot cavity using the
Pound-Drever-Hall (PDH) technique.

## Simulations

1. **Gaussian beam propagation (mode matching with Thorlabs collimators).**
   A 1550 nm beam leaves an FC/APC collimator (F220APC / F260APC / F280APC, waist from the datasheet
   beam diameter), travels a distance d to a focusing lens f, and is refocused. Using the q parameter
   and ABCD matrices, the script computes the new waist position z_out and radius w_out.
2. **Fabry-Perot stability and characterization.**
   From the five system parameters (ROC1, ROC2, R1, R2, L): g1, g2, TEM00 waist w0 and Rayleigh
   range z_R, FSR, finesse and linewidth, plus transmission/reflection spectra from the Airy function.
3. **Homodyne measurement of random phase noise.**
   A field with random-walk phase noise (exponential autocorrelation → Lorentzian spectrum) is beaten
   with a delayed copy of itself without a frequency shift. The Lorentzian FWHM of the beat spectrum is
   extracted and plotted vs the delay τ, showing it settles at 2Δν once τ exceeds the coherence time τc.

## Project structure

```
PDHproject/
├── src/                        # Physics/math core only (no plotting)
│   ├── optics_utils.py         # q parameter, ABCD matrices, collimator model, focusing a beam with a thin lens
│   ├── fabry_perot.py          # FabryPerot class: g1/g2, w0, zR, FSR, finesse, linewidth, Airy T/R
│   └── signal_dsp.py           # Random-walk phase noise, homodyne beat, PSD, Lorentzian fit, analytic beat spectrum
├── simulations/                # Runnable scripts
│   ├── sim1_mode_matching.py   # Simulation 1: z_out and w_out vs focal length and distance
│   ├── sim2_cavity_stability.py# Simulation 2: cavity parameters and Airy transmission/reflection
│   └── sim3_homodyne_noise.py  # Simulation 3: measured linewidth vs delay
├── data_out/                   # Simulation outputs (png/csv) — not tracked in git
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

From the project root:

```bash
python simulations/sim1_mode_matching.py
python simulations/sim2_cavity_stability.py
python simulations/sim3_homodyne_noise.py --show   # --show opens the plot window
```

Plots and CSV files are saved to `data_out/`. Each simulation's parameters are defined at the top of its file.
