"""Delayed self-heterodyne simulation: 7000 m delay fiber.

Expected physics (laser linewidth dnu = 52 kHz, SMF-28 n_g = 1.4682):
    tau_d = 34.3 us, fringe spacing 1/tau_d = 29.2 kHz
    tau_d / tau_c = 5.6, coherent fraction exp(-2 pi dnu tau_d) = 0.001 %
    max RBW for fringes (0.1 / tau_d) = 2.9 kHz
    regime: incoherent
    Clean Lorentzian beat of FWHM 2 dnu ~ 104 kHz: -3 dB, Lorentzian and coherent fits agree.
    The wide RBW (5.1 kHz) is above 0.1/tau_d, so RBW_too_coarse_for_fringes is expected.

Parameters (linewidth, beat power, ESA acquisitions, repeats) are set in dsh_run.py.
Each run creates a new folder self_heterodyne/data_measured/delay_07000m/<YYYY-MM-DD_HH-MM-SS>_sim/
next to the lab runs, with every ESA trace (CSV + PNG), overview.png, results.csv, metadata.json
and run.log. Browse it with:  python self_heterodyne/sh_data.py show 7000 --kind sim

Run from the project root:  python self_heterodyne/simulations/sim_fiber_7000m.py [--no-show] [--note "..."]
"""

import argparse

from dsh_run import run_fiber

FIBER_LENGTH_M = 7000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-show", action="store_true", help="only save the figures, do not open a window")
    parser.add_argument("--note", default="", help="free-text note stored with the run")
    args = parser.parse_args()
    run_fiber(FIBER_LENGTH_M, show=not args.no_show, note=args.note)
