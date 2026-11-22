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
   The mirror curvatures enter the spectrum through the Gouy phase: the TEM_mn resonances sit at
   ν_{q,m,n} = (c/2L)[q + (m+n+1)/π · arccos(±√(g1 g2))], so ROC1/ROC2 set the transverse-mode spacing
   while the FSR stays c/2L. The lab cavity is half-symmetric (plano-concave): M1 flat, M2 with
   ROC = 15 mm, L = 10 mm → g1 = 1, g2 = 1/3, w0 ≈ 59 µm on the flat mirror, FSR ≈ 15 GHz.
3. **Homodyne measurement of random phase noise.**
   A field with random-walk phase noise (exponential autocorrelation → Lorentzian spectrum) is beaten
   with a delayed copy of itself without a frequency shift. The Lorentzian FWHM of the beat spectrum is
   extracted and plotted vs the delay τ, showing it settles at 2Δν once τ exceeds the coherence time τc.

## Project structure

```
PDHproject/
├── src/                        # Physics/math core only (no plotting)
│   ├── optics_utils.py         # q parameter, ABCD matrices, collimator model, focusing a beam with a thin lens
│   ├── fabry_perot.py          # FabryPerot class: g1/g2, w0, zR, FSR, finesse, linewidth, Gouy phase, TEM_mn resonances, Airy T/R
│   └── signal_dsp.py           # Random-walk phase noise, homodyne beat, PSD, Lorentzian fit, analytic beat spectrum
├── simulations/                # Runnable scripts
│   ├── sim1_mode_matching.py   # Simulation 1: z_out and w_out vs focal length and distance
│   ├── sim2_cavity_stability.py# Simulation 2: cavity parameters and Airy transmission/reflection
│   └── sim3_homodyne_noise.py  # Simulation 3: measured linewidth vs delay
├── data_out/                   # Simulation outputs (png/csv)
├── self_heterodyne/            # Delayed self-heterodyne linewidth simulation (separate from the above)
│   ├── dsh/                    # Physics/analysis only (no plotting)
│   │   ├── model.py            # tau_d, tau_c, coherent fraction, DSH beat spectrum, ESA trace (RBW, floor, averaging)
│   │   ├── analysis.py         # -3/-20 dB widths, Lorentzian fit, coherent (full DSH) fit, regime, flags, dnu_best
│   │   └── runs.py             # Find and load run folders (lab or simulation): results, ESA traces, scope traces
│   ├── simulations/
│   │   ├── dsh_run.py          # Shared runner + all parameters (linewidth, beat power, ESA acquisitions, repeats)
│   │   ├── sim_fiber_<L>m.py   # One script per fiber length: 500, 2000, 5000, 7000 m
│   │   ├── PPCL550_laser_data.md             # PPCL550 laser settings and state (text) + screenshot
│   │   └── ppcl550_settings_2026-11-15.png   # Pure Photonics GUI screenshot, 2026-11-15
│   ├── sh_linewidth.py         # LAB measurement: drives the ESA + scope, writes runs into data_measured/
│   ├── sh_data.py              # Browse runs: list / show / open folder / copy older runs from the PyCharm folder
│   └── data_measured/          # ALL data: lab runs delay_XXXXXm/<timestamp>/ + simulations <timestamp>_sim/
│                               #   summary.csv = lab, summary_sim.csv = simulations
└── requirements.txt
```

## Self-heterodyne simulation

Simulates the delayed self-heterodyne (DSH) linewidth measurement (80 MHz AOM, SMF-28 delay fiber,
Keysight ESA) and analyzes the simulated traces exactly like the lab data. The analysis includes the
direct −3 dB and −20 dB widths, a Lorentzian fit, and a fit to the full partially coherent DSH spectrum
(Richter/Mercer). The ESA model convolves the spectrum with the RBW filter and adds the noise floor and
averaging noise. Each fiber length shows a different regime: 500 m and 2 km are partially coherent, with a
central coherent peak and fringes, so only the coherent fit is valid. 5 km and 7 km are incoherent, giving a
Lorentzian beat of FWHM 2Δν.

```bash
python self_heterodyne/simulations/sim_fiber_500m.py
python self_heterodyne/simulations/sim_fiber_2000m.py
python self_heterodyne/simulations/sim_fiber_5000m.py
python self_heterodyne/simulations/sim_fiber_7000m.py --note "..."
```

All data is in one place, `self_heterodyne/data_measured/`. Each simulation run creates a new folder
`delay_XXXXXm/<YYYY-MM-DD_HH-MM-SS>_sim/` next to the lab runs of the same fiber length. Every run
folder holds the CSV and PNG of each ESA trace, plus `results.csv`, `metadata.json` and `run.log`.

### Lab measurement

`self_heterodyne/sh_linewidth.py` drives the Keysight N9010B ESA and DSOX1204G scope (needs `pyvisa`
and the instruments on USB). Each run writes `data_measured/delay_XXXXXm/<YYYY-MM-DD_HH-MM-SS>/` and
appends to `data_measured/summary.csv`. Runs made earlier with the PyCharm copy of the script were
copied in with `sh_data.py copy --all`.

```bash
cd self_heterodyne
python sh_linewidth.py --check                       # verify instrument connections
python sh_linewidth.py --monitor                     # live beat power, for polarization alignment
python sh_linewidth.py --delay 5000 --note "5000 m spool, laser PPCL550, Plaser 14dBm, 10 dB att, whisper on"
python sh_linewidth.py --delay 2000 --simulate       # offline test, no instruments (writes a run folder too)
```

### Accessing run data

```bash
python self_heterodyne/sh_data.py                       # interactive: pick a run -> results, open folder
python self_heterodyne/sh_data.py list                  # all runs (measured + sim, best linewidth, note)
python self_heterodyne/sh_data.py list --kind lab       # only measurements (--kind sim: only simulations)
python self_heterodyne/sh_data.py show 5000 --kind lab  # latest 5 km measurement: results + file list
python self_heterodyne/sh_data.py show 5000 --run 2026-11-12_16-51-43
python self_heterodyne/sh_data.py open 5000             # open the folder in Explorer
python self_heterodyne/sh_data.py copy --all            # bring runs from the old PyCharm lab folder (merges summary.csv)
```

In Python (from `self_heterodyne/`): `from dsh import runs; run = runs.find_run(runs.DATA_ROOT, 5000, kind="lab")`,
then `run.results`, `run.trace("wide", 1)` and `run.scope("before")`.

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
python simulations/sim3_homodyne_noise.py
```

Each run opens the plot window and saves plots and CSV files to `data_out/` (add `--no-show` to only save). Each simulation's parameters are defined at the top of its file.
