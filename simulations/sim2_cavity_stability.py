"""Simulation 2: stability analysis and characterization of a Fabry-Perot cavity.

Inputs are the five system parameters ROC1, ROC2, R1, R2 and L. Everything is computed
from closed-form expressions:
  * spatial  - g1, g2, TEM00 waist w0 and its Rayleigh range z_R
  * spectral - free spectral range FSR, finesse and resonance linewidth (FWHM)
  * plots    - transmission and reflection vs frequency from the Airy function
               T = T_max / (1 + F sin^2(delta / 2)),  R = 1 - T

Run from the project root:  python simulations/sim2_cavity_stability.py [--no-show]
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.fabry_perot import FabryPerot  # noqa: E402

OUT_DIR = ROOT / "data_out"

# ----------------------------- parameters ---------------------------------
ROC1 = 0.5          # mirror 1 radius of curvature [m] (np.inf = flat)
ROC2 = 0.5          # mirror 2 radius of curvature [m]
R1 = 0.99           # mirror 1 power reflectivity
R2 = 0.99           # mirror 2 power reflectivity
L = 0.10            # mirror separation [m]
WAVELENGTH = 1550e-9


def main(show: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cav = FabryPerot(roc1=ROC1, roc2=ROC2, R1=R1, R2=R2, length=L, wavelength=WAVELENGTH)

    print(f"Inputs: ROC1 = {ROC1} m, ROC2 = {ROC2} m, R1 = {R1}, R2 = {R2}, L = {L} m\n")
    print("Spatial (TEM00):")
    print(f"  g1 = {cav.g1:.4f}, g2 = {cav.g2:.4f}, g1*g2 = {cav.g1 * cav.g2:.4f} -> "
          f"{'stable' if cav.is_stable else 'UNSTABLE'}")
    if cav.is_stable:
        print(f"  waist w0          = {cav.waist() * 1e6:.2f} um "
              f"(at {cav.waist_position() * 1e3:.2f} mm from M1)")
        print(f"  Rayleigh range zR = {cav.rayleigh_range() * 1e3:.2f} mm")
    print("Spectral:")
    print(f"  FSR               = {cav.fsr / 1e9:.4f} GHz")
    print(f"  coefficient F     = {cav.coefficient_of_finesse:.1f}")
    print(f"  finesse           = {cav.finesse:.1f}")
    print(f"  linewidth (FWHM)  = {cav.linewidth / 1e6:.3f} MHz")

    # --- Airy transmission / reflection --------------------------------------
    df = np.linspace(-1.5 * cav.fsr, 1.5 * cav.fsr, 60001)
    trans = cav.transmission(df)
    refl = cav.reflection(df)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    ax = axes[0]
    ax.plot(df / 1e9, trans, label="transmission T")
    ax.plot(df / 1e9, refl, label="reflection R = 1 - T")
    for k in (-1, 0, 1):
        ax.axvline(k * cav.fsr / 1e9, color="k", ls=":", lw=0.8)
    ax.annotate("", xy=(cav.fsr / 1e9, 0.5), xytext=(0, 0.5),
                arrowprops={"arrowstyle": "<->", "color": "k"})
    ax.text(cav.fsr / 2e9, 0.53, f"FSR = {cav.fsr / 1e9:.3f} GHz", ha="center")
    ax.set_xlabel("frequency detuning [GHz]")
    ax.set_ylabel("power fraction")
    ax.set_title(f"Airy function: R1 = {R1}, R2 = {R2}, L = {L * 1e3:.0f} mm, finesse = {cav.finesse:.0f}")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    zoom = np.linspace(-5 * cav.linewidth, 5 * cav.linewidth, 4001)
    t_zoom = cav.transmission(zoom)
    ax.plot(zoom / 1e6, t_zoom, label="transmission T")
    ax.plot(zoom / 1e6, cav.reflection(zoom), label="reflection R")
    half = t_zoom.max() / 2
    ax.annotate("", xy=(cav.linewidth / 2e6, half), xytext=(-cav.linewidth / 2e6, half),
                arrowprops={"arrowstyle": "<->", "color": "k"})
    ax.text(0, half + 0.03, f"FWHM = {cav.linewidth / 1e6:.2f} MHz", ha="center")
    ax.set_xlabel("frequency detuning [MHz]")
    ax.set_ylabel("power fraction")
    ax.set_title("Zoom on one resonance")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim2_cavity_stability.png", dpi=150)

    with open(OUT_DIR / "sim2_airy_spectrum.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["detuning_Hz", "transmission", "reflection"])
        writer.writerows(zip(df, trans, refl))

    print(f"\nSaved results to {OUT_DIR.relative_to(ROOT)}/")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="only save the figure, do not open a window")
    main(show=not parser.parse_args().no_show)
