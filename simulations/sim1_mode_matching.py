"""Simulation 1: Gaussian beam propagation (mode matching based on Thorlabs collimators).

A 1550 nm beam leaves a Thorlabs FC/APC fiber collimator (waist w_in = D/2 taken from
the datasheet beam diameter D), propagates a distance d to a focusing lens of focal
length f, and is refocused. Using the complex beam parameter q and the ABCD matrices
of free space and a thin lens, we compute:
  * z_out - distance of the new waist from the lens
  * w_out - radius of the new waist (at the focus)

The ABCD result is cross-checked against the closed-form thin-lens waist formula.

Run from the project root:  python simulations/sim1_mode_matching.py [--no-show]
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.optics_utils import (  # noqa: E402
    Collimator, collimated_beam_diameter, focus_beam, focus_beam_analytic, rayleigh_range, trace_beam,
)

OUT_DIR = ROOT / "data_out"

# ----------------------------- parameters ---------------------------------
WAVELENGTH = 1550e-9            # [m]
FIBER_MFD = 10.4e-6             # SMF-28 mode field diameter at 1550 nm [m] (used only for a cross-check)

# Thorlabs FC/APC collimators, 1550 nm versions.
# TODO: verify focal length and 1/e^2 beam diameter against the Thorlabs datasheets
#       (thorlabs.com product pages F220APC-1550, F260APC-1550, F280APC-1550).
COLLIMATORS = [
    Collimator("F220APC-1550", focal_length=11.0e-3, beam_diameter=2.1e-3),
    Collimator("F260APC-1550", focal_length=15.4e-3, beam_diameter=2.9e-3),
    Collimator("F280APC-1550", focal_length=18.4e-3, beam_diameter=3.4e-3),
]

FOCAL_LENGTHS_MM = np.array([25, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300])  # lens catalogue [mm]
LENS_DISTANCE = 0.10            # distance collimator -> focusing lens d [m]
EXAMPLE_FOCAL_MM = 100          # focal length used for the beam-profile plot [mm]


def main(show: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Wavelength: {WAVELENGTH * 1e9:.0f} nm, collimator-lens distance d = {LENS_DISTANCE * 1e3:.0f} mm\n")
    print("Input beams (waist at the collimator output):")
    for col in COLLIMATORS:
        d_theory = collimated_beam_diameter(WAVELENGTH, col.focal_length, FIBER_MFD)
        print(f"  {col.name}: D = {col.beam_diameter * 1e3:.2f} mm (datasheet), "
              f"{d_theory * 1e3:.2f} mm (4 lambda f / pi MFD) -> w_in = {col.waist * 1e3:.3f} mm, "
              f"z_R = {rayleigh_range(col.waist, WAVELENGTH):.2f} m")

    # --- z_out and w_out for every collimator / lens combination ------------
    f_m = FOCAL_LENGTHS_MM * 1e-3
    rows = []
    print("\nOutput waist (ABCD):")
    print(f"  {'collimator':<14}{'f [mm]':>8}{'z_out [mm]':>12}{'w_out [um]':>12}")
    for col in COLLIMATORS:
        z_out, w_out = focus_beam(col.waist, WAVELENGTH, LENS_DISTANCE, f_m)
        z_chk, w_chk = focus_beam_analytic(col.waist, WAVELENGTH, LENS_DISTANCE, f_m)
        assert np.allclose(z_out, z_chk) and np.allclose(w_out, w_chk), "ABCD and analytic results differ"
        for f, z, w in zip(FOCAL_LENGTHS_MM, z_out, w_out):
            rows.append({"collimator": col.name, "w_in_mm": col.waist * 1e3, "d_mm": LENS_DISTANCE * 1e3,
                         "f_mm": f, "z_out_mm": z * 1e3, "w_out_um": w * 1e6})
            print(f"  {col.name:<14}{f:>8.0f}{z * 1e3:>12.2f}{w * 1e6:>12.2f}")

    with open(OUT_DIR / "sim1_focus_table.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # --- plots -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    f_dense = np.linspace(FOCAL_LENGTHS_MM.min(), FOCAL_LENGTHS_MM.max(), 400) * 1e-3

    ax = axes[0, 0]
    for i, col in enumerate(COLLIMATORS):
        _, w = focus_beam(col.waist, WAVELENGTH, LENS_DISTANCE, f_dense)
        _, w_pts = focus_beam(col.waist, WAVELENGTH, LENS_DISTANCE, f_m)
        ax.plot(f_dense * 1e3, w * 1e6, color=f"C{i}", label=col.name)
        ax.plot(FOCAL_LENGTHS_MM, w_pts * 1e6, "o", color=f"C{i}", ms=4)
    ax.set_xlabel("focal length f [mm]")
    ax.set_ylabel("$w_{out}$ [um]")
    ax.set_title("Output waist radius vs focal length")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for i, col in enumerate(COLLIMATORS):
        z, _ = focus_beam(col.waist, WAVELENGTH, LENS_DISTANCE, f_dense)
        ax.plot(f_dense * 1e3, z * 1e3, color=f"C{i}", label=col.name)
    ax.plot(f_dense * 1e3, f_dense * 1e3, "k:", lw=1, label="$z_{out} = f$")
    ax.set_xlabel("focal length f [mm]")
    ax.set_ylabel("$z_{out}$ [mm]")
    ax.set_title("Output waist position (from the lens) vs focal length")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    distances = np.linspace(0.0, 2.0, 400)
    f_ex = EXAMPLE_FOCAL_MM * 1e-3
    for i, col in enumerate(COLLIMATORS):
        z, w = focus_beam(col.waist, WAVELENGTH, distances, f_ex)
        ax.plot(distances * 1e3, (z - f_ex) * 1e3, color=f"C{i}", label=col.name)
    ax.axvline(LENS_DISTANCE * 1e3, color="k", ls="--", lw=1, label=f"d = {LENS_DISTANCE * 1e3:.0f} mm")
    ax.set_xlabel("collimator - lens distance d [mm]")
    ax.set_ylabel("$z_{out} - f$ [mm]")
    ax.set_title(f"Focus shift from the focal plane vs d (f = {EXAMPLE_FOCAL_MM} mm)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for i, col in enumerate(COLLIMATORS):
        z_out, w_out = focus_beam(col.waist, WAVELENGTH, LENS_DISTANCE, f_ex)
        z, w = trace_beam(col.waist, WAVELENGTH, LENS_DISTANCE, f_ex, z_after=2.0 * z_out)
        ax.plot(z * 1e3, w * 1e3, color=f"C{i}", label=f"{col.name}: $w_{{out}}$ = {w_out * 1e6:.1f} um")
        ax.plot(z * 1e3, -w * 1e3, color=f"C{i}")
        ax.plot((LENS_DISTANCE + z_out) * 1e3, 0, "x", color=f"C{i}")
    ax.axvline(LENS_DISTANCE * 1e3, color="k", lw=1.5)
    ax.annotate(f"lens f = {EXAMPLE_FOCAL_MM} mm", (LENS_DISTANCE * 1e3, 0), xytext=(4, 30),
                textcoords="offset points", rotation=90)
    ax.set_xlabel("z from collimator [mm]")
    ax.set_ylabel("beam radius w(z) [mm]")
    ax.set_title("Beam profile (x = new waist)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim1_mode_matching.png", dpi=150)
    print(f"\nSaved results to {OUT_DIR.relative_to(ROOT)}/")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="only save the figure, do not open a window")
    main(show=not parser.parse_args().no_show)
