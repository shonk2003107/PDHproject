"""Simulation 1: mode matching a laser beam into the Fabry-Perot cavity with two lenses.

For every pair of stock lenses, finds the lens positions that maximize the TEM00
overlap with the cavity eigenmode, then plots the best solution and its tolerance.

Run from the project root:  python simulations/sim1_mode_matching.py [--show]
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.fabry_perot import FabryPerot  # noqa: E402
from src.optics_utils import (  # noqa: E402
    beam_radius, mode_overlap, optimize_two_lens, q_from_waist, trace_beam, waist_from_q,
)

OUT_DIR = ROOT / "data_out"

# ----------------------------- parameters ---------------------------------
WAVELENGTH = 1064e-9
LASER_WAIST = 0.5e-3          # laser waist radius [m], located at z = 0
CAVITY_DISTANCE = 1.0         # distance from laser waist to cavity input mirror [m]
CAVITY = FabryPerot(length=0.10, R1=0.99, R2=0.999, roc1=np.inf, roc2=0.5, wavelength=WAVELENGTH)
STOCK_LENSES = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, -0.05, -0.10]  # [m]
MIN_GAP = 0.03                # minimal spacing between elements [m]


def main(show: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    q_in = q_from_waist(LASER_WAIST, WAVELENGTH)
    q_cav = CAVITY.eigenmode_q()            # cavity mode at the input mirror
    w0_cav, z_waist_cav = waist_from_q(q_cav, WAVELENGTH)
    print(f"Cavity waist: {w0_cav * 1e6:.1f} um at {z_waist_cav * 1e3:.1f} mm from M1")
    print(f"No lenses: overlap = {mode_overlap(q_in + CAVITY_DISTANCE, q_cav):.4f}")

    # --- search all lens pairs ------------------------------------------------
    results = []
    for f1, f2 in itertools.product(STOCK_LENSES, repeat=2):
        sol = optimize_two_lens(q_in, q_cav, CAVITY_DISTANCE, f1, f2, min_gap=MIN_GAP)
        results.append({"f1_m": f1, "f2_m": f2, "x1_m": sol["x1"], "x2_m": sol["x2"], "eta": sol["eta"]})
    results.sort(key=lambda r: r["eta"], reverse=True)

    with open(OUT_DIR / "sim1_lens_pairs.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("\nTop 5 lens pairs:")
    for r in results[:5]:
        print(f"  f1={r['f1_m']*1e3:6.0f} mm @ {r['x1_m']*1e3:7.1f} mm | "
              f"f2={r['f2_m']*1e3:6.0f} mm @ {r['x2_m']*1e3:7.1f} mm | eta={r['eta']:.6f}")
    best = results[0]

    # --- beam trace of the best solution -------------------------------------
    elements = [("space", best["x1_m"]), ("lens", best["f1_m"]),
                ("space", best["x2_m"] - best["x1_m"]), ("lens", best["f2_m"]),
                ("space", CAVITY_DISTANCE - best["x2_m"])]
    z, w, _ = trace_beam(q_in, WAVELENGTH, elements)
    z_c = np.linspace(0, CAVITY.length, 200)
    w_c = beam_radius(q_cav + z_c, WAVELENGTH)

    # --- tolerance: overlap vs lens displacement ------------------------------
    dx = np.linspace(-5e-3, 5e-3, 201)
    def eta_for(x1, x2):
        _, _, q_out = trace_beam(q_in, WAVELENGTH, [("space", x1), ("lens", best["f1_m"]),
                                                    ("space", x2 - x1), ("lens", best["f2_m"]),
                                                    ("space", CAVITY_DISTANCE - x2)], points_per_meter=2)
        return mode_overlap(q_out, q_cav)
    eta_l1 = [eta_for(best["x1_m"] + d, best["x2_m"]) for d in dx]
    eta_l2 = [eta_for(best["x1_m"], best["x2_m"] + d) for d in dx]

    # --- plots -------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    ax1.plot(z * 1e3, w * 1e3, label="input beam")
    ax1.plot(z * 1e3, -w * 1e3, color="C0")
    ax1.plot((CAVITY_DISTANCE + z_c) * 1e3, w_c * 1e3, "C3--", label="cavity eigenmode")
    ax1.plot((CAVITY_DISTANCE + z_c) * 1e3, -w_c * 1e3, "C3--")
    for x, f, name in [(best["x1_m"], best["f1_m"], "L1"), (best["x2_m"], best["f2_m"], "L2")]:
        ax1.axvline(x * 1e3, color="k", lw=1)
        ax1.annotate(f"{name}: f={f*1e3:.0f} mm", (x * 1e3, 0),
                     xytext=(4, 0), textcoords="offset points", rotation=90, va="center")
    ax1.axvspan(CAVITY_DISTANCE * 1e3, (CAVITY_DISTANCE + CAVITY.length) * 1e3, color="0.9", label="cavity")
    ax1.set_xlabel("z [mm]")
    ax1.set_ylabel("beam radius w [mm]")
    ax1.set_title(f"Best two-lens mode matching (overlap = {best['eta']:.5f})")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(dx * 1e3, eta_l1, label="move L1")
    ax2.plot(dx * 1e3, eta_l2, label="move L2")
    ax2.set_xlabel("lens displacement [mm]")
    ax2.set_ylabel("mode overlap")
    ax2.set_title("Alignment tolerance")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim1_mode_matching.png", dpi=150)
    print(f"\nSaved results to {OUT_DIR}")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the figure window")
    main(parser.parse_args().show)
