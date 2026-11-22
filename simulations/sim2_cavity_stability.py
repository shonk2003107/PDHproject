"""Simulation 2: stability analysis and characterization of a Fabry-Perot cavity.

Inputs are the five system parameters ROC1, ROC2, R1, R2 and L. Everything is computed
from closed-form expressions:
  * spatial  - g1, g2, TEM00 waist w0 and its Rayleigh range z_R
  * spectral - free spectral range FSR, finesse and resonance linewidth (FWHM)
  * modes    - resonance frequencies of the transverse modes TEM_mn,
               nu_{q,m,n} = (c / 2L) [ q + (m + n + 1) / pi * arccos(+-sqrt(g1 g2)) ]
               (plano-concave ROC1 = inf, ROC2 = R: arccos(sqrt(1 - L/R)); symmetric ROC1 = ROC2 = R:
               arccos(1 - L/R)). The Gouy term cancels between q and
               q + 1, so FSR = c / 2L for any ROC; the ROCs set where TEM_mn sits inside the FSR.
  * plots    - transmission and reflection vs frequency from the Airy function
               T = T_max / (1 + F sin^2(delta / 2)),  R = 1 - T, and the TEM_mn resonances

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
# Half-symmetric (plano-concave) cavity: M1 flat, M2 concave with ROC = 15 mm, L = 10 mm.
# g1 = 1, g2 = 1 - L/ROC2 = 1/3 -> stable; the TEM00 waist sits on the flat mirror M1.
ROC1 = np.inf       # mirror 1 radius of curvature [m] (np.inf = flat)
ROC2 = 15e-3        # mirror 2 radius of curvature [m]
R1 = 0.99           # mirror 1 power reflectivity
R2 = 0.99           # mirror 2 power reflectivity
L = 10e-3           # mirror separation [m]
WAVELENGTH = 1550e-9
MAX_MODE_ORDER = 4  # highest transverse-mode order m + n shown in the mode plot

# (ROC1, ROC2) pairs printed for comparison with the chosen cavity (same L = 10 mm)
ROC_COMPARISON = [(np.inf, 15e-3), (np.inf, 25e-3), (np.inf, 50e-3), (15e-3, 15e-3), (np.inf, 10e-3)]


def fmt_roc(roc: float) -> str:
    """Radius of curvature as text in mm ('flat' for np.inf)."""
    return "flat" if np.isinf(roc) else f"{roc * 1e3:g} mm"


def main(show: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cav = FabryPerot(roc1=ROC1, roc2=ROC2, R1=R1, R2=R2, length=L, wavelength=WAVELENGTH)

    print(f"Inputs: ROC1 = {fmt_roc(ROC1)}, ROC2 = {fmt_roc(ROC2)}, R1 = {R1}, R2 = {R2}, "
          f"L = {L * 1e3:g} mm\n")
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

    if cav.is_stable:
        q0 = int(round(2 * L / WAVELENGTH))  # longitudinal index of the TEM00 mode nearest the laser
        nu00 = cav.resonance_frequency(q0)
        print("Transverse modes (Gouy phase):")
        print(f"  Gouy phase zeta   = arccos(+-sqrt(g1 g2)) = {cav.gouy_phase:.4f} rad")
        print(f"  mode spacing      = FSR * zeta / pi = {cav.transverse_mode_spacing / 1e6:.2f} MHz")
        print(f"  nu(q={q0}, 0, 0) = {nu00 / 1e12:.6f} THz")
        print("  m+n | nu - nu(q,0,0) [MHz] | offset folded into one FSR [MHz]")
        for order in range(MAX_MODE_ORDER + 1):
            shift = cav.resonance_frequency(q0, order, 0) - nu00
            print(f"  {order:3d} | {shift / 1e6:20.2f} | {cav.mode_offset(order) / 1e6:12.2f}")
        print(f"  check: nu(q+1,0,0) - nu(q,0,0) = "
              f"{(cav.resonance_frequency(q0 + 1) - nu00) / 1e9:.4f} GHz = FSR")

    print("\nEffect of the mirror curvatures (same R1, R2, L):")
    print("      ROC1 |      ROC2 |  g1*g2 |  FSR [GHz] | zeta [rad] | mode spacing [MHz]")
    for roc1, roc2 in ROC_COMPARISON:
        c = FabryPerot(roc1=roc1, roc2=roc2, R1=R1, R2=R2, length=L, wavelength=WAVELENGTH)
        if c.is_stable:
            print(f"  {fmt_roc(roc1):>8} | {fmt_roc(roc2):>9} | {c.g1 * c.g2:6.3f} | {c.fsr / 1e9:10.4f} | "
                  f"{c.gouy_phase:10.4f} | {c.transverse_mode_spacing / 1e6:18.2f}")
        else:
            print(f"  {fmt_roc(roc1):>8} | {fmt_roc(roc2):>9} | {c.g1 * c.g2:6.3f} | {c.fsr / 1e9:10.4f} | "
                  "unstable (needs 0 < g1*g2 < 1)")

    # --- Airy transmission / reflection --------------------------------------
    df = np.linspace(-1.5 * cav.fsr, 1.5 * cav.fsr, 60001)
    trans = cav.transmission(df)
    refl = cav.reflection(df)

    n_rows = 3 if cav.is_stable else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4.5 * n_rows))

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

    # --- transverse modes TEM_mn inside one FSR --------------------------------
    if cav.is_stable:
        ax = axes[2]
        span = np.linspace(-0.05 * cav.fsr, 1.05 * cav.fsr, 60001)
        for order in range(MAX_MODE_ORDER + 1):
            ax.plot(span / 1e9, cav.transmission(span, mode_order=order),
                    label=f"m+n = {order}  (+{cav.mode_offset(order) / 1e6:.0f} MHz)")
        ax.axvline(cav.fsr / 1e9, color="k", ls=":", lw=0.8)
        ax.set_xlabel("frequency detuning from TEM00 resonance [GHz]")
        ax.set_ylabel("transmission")
        ax.set_title(f"TEM_mn resonances: ROC1 = {fmt_roc(ROC1)}, ROC2 = {fmt_roc(ROC2)}, "
                     f"Gouy phase = {cav.gouy_phase:.3f} rad, "
                     f"mode spacing = {cav.transverse_mode_spacing / 1e6:.1f} MHz\n"
                     "(each mode drawn at full height; real peak heights depend on mode matching)")
        ax.legend(loc="center right", fontsize=8)
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
