"""Simulation 4: the full optical path, fiber -> collimator -> lens -> Fabry-Perot cavity.

A Gaussian beam leaves the fiber collimator (waist w_c at its output), travels a distance d to
the focusing lens f and is focused onto the flat mirror M1 of the plano-concave cavity, where
the cavity TEM00 mode has its waist. The figure shows
  * the beam radius w(z) along the whole path (ABCD / q parameter, as in simulation 1),
  * a zoom on the cavity: the focused input beam vs the cavity TEM00 mode, whose wavefront
    radius at M2 equals ROC2,
  * a table of all parameters, including the mode-matching efficiency into TEM00.

Collimator, lens distance and wavelength come from simulation 1; the cavity and the assumed
residual misalignment from simulation 2, so the three figures always agree.

Run from the project root:  python simulations/sim4_system_layout.py [--no-show]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from simulations import sim1_mode_matching as sim1  # noqa: E402
from simulations import sim2_cavity_stability as sim2  # noqa: E402
from src.fabry_perot import FabryPerot  # noqa: E402
from src.optics_utils import focus_beam, rayleigh_range, trace_beam  # noqa: E402

OUT_DIR = ROOT / "data_out"

# ----------------------------- parameters ---------------------------------
COLLIMATOR = sim2.COLLIMATOR    # same collimator as the simulation 2 mode matching
LENS_FOCAL = sim1.EXAMPLE_FOCAL_MM * 1e-3   # focusing lens of the lab setup [m]
LENS_DISTANCE = sim1.LENS_DISTANCE
WAVELENGTH = sim1.WAVELENGTH


def main(show: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    col = next(c for c in sim1.COLLIMATORS if c.name == COLLIMATOR)
    cav = FabryPerot(roc1=sim2.ROC1, roc2=sim2.ROC2, R1=sim2.R1, R2=sim2.R2, length=sim2.L,
                     wavelength=WAVELENGTH)

    # --- input beam: collimator -> lens -> focus ---------------------------------
    z_focus, w_focus = (float(v) for v in focus_beam(col.waist, WAVELENGTH, LENS_DISTANCE, LENS_FOCAL))
    z_m1 = LENS_DISTANCE + z_focus          # M1 (flat) sits at the focus = cavity waist
    z_m2 = z_m1 + cav.length
    z, w = trace_beam(col.waist, WAVELENGTH, LENS_DISTANCE, LENS_FOCAL, z_after=z_focus + 0.05)

    # --- cavity TEM00 mode -------------------------------------------------------
    w0, zr = cav.waist(), cav.rayleigh_range()
    z_cav = np.linspace(0.0, cav.length, 400)
    w_cav = w0 * np.sqrt(1.0 + (z_cav / zr) ** 2)
    r_m2 = cav.length + zr ** 2 / cav.length            # wavefront radius at M2 (= ROC2)
    zr_focus = rayleigh_range(w_focus, WAVELENGTH)

    p_ideal = cav.mode_coupling(w_focus, max_order=0)[0]            # waist on M1, perfect alignment
    p_sim2 = cav.mode_coupling(sim2.INPUT_WAIST, sim2.INPUT_WAIST_OFFSET, sim2.INPUT_OFFSET_X,
                               sim2.INPUT_OFFSET_Y, sim2.INPUT_TILT_X, sim2.INPUT_TILT_Y,
                               max_order=0)[0]

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.85], hspace=0.35)

    # --- panel 1: whole path -----------------------------------------------------
    ax = fig.add_subplot(gs[0])
    zmm = z * 1e3
    ax.fill_between(zmm, -w * 1e3, w * 1e3, color="C3", alpha=0.25, lw=0)
    ax.plot(zmm, w * 1e3, color="C3", lw=1.2)
    ax.plot(zmm, -w * 1e3, color="C3", lw=1.2)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    y_top = 2.6
    # fiber + collimator
    ax.plot([-60, -18], [0, 0], color="goldenrod", lw=3)
    ax.text(-40, 0.25, "SMF-28\nfiber", ha="center", va="bottom", fontsize=9)
    ax.add_patch(Rectangle((-18, -1.9), 18, 3.8, fc="0.75", ec="k"))
    ax.text(-9, -2.3, f"collimator\n{col.name}", ha="center", va="top", fontsize=9)
    # lens
    lens_z = LENS_DISTANCE * 1e3
    ax.annotate("", xy=(lens_z, 2.2), xytext=(lens_z, -2.2),
                arrowprops={"arrowstyle": "<->", "color": "C0", "lw": 2.5})
    ax.text(lens_z, -2.3, f"lens f = {LENS_FOCAL * 1e3:.0f} mm", ha="center", va="top", fontsize=9,
            color="C0")
    # cavity
    for zm, name in ((z_m1, "M1 flat"), (z_m2, f"M2 ROC {sim2.ROC2 * 1e3:g} mm")):
        ax.plot([zm * 1e3] * 2, [-1.6, 1.6], color="k", lw=2.5)
    ax.text((z_m1 + z_m2) / 2 * 1e3, -2.3, f"cavity\nL = {cav.length * 1e3:g} mm", ha="center",
            va="top", fontsize=9)
    # dimensions
    for z0, z1, label in ((0.0, lens_z, f"d = {lens_z:.0f} mm"),
                          (lens_z, z_m1 * 1e3, f"z_focus = {z_focus * 1e3:.1f} mm")):
        ax.annotate("", xy=(z1, y_top), xytext=(z0, y_top), arrowprops={"arrowstyle": "<->"})
        ax.text((z0 + z1) / 2, y_top + 0.12, label, ha="center", fontsize=9)
    ax.text(5, col.waist * 1e3 + 0.12, f"w = {col.waist * 1e3:.2f} mm", fontsize=9, color="C3")
    ax.text(z_m1 * 1e3 - 4, 0.35, f"w = {w_focus * 1e6:.0f} µm\non M1", ha="right", fontsize=9,
            color="C3")
    ax.set_xlim(-65, z[-1] * 1e3 + 5)
    ax.set_ylim(-3.4, 3.2)
    ax.set_xlabel("z from the collimator output [mm]")
    ax.set_ylabel("beam radius w(z) [mm]")
    ax.set_title(f"Full optical path at {WAVELENGTH * 1e9:.0f} nm (1/e² radius, ABCD propagation)")
    ax.grid(alpha=0.3)

    # --- panel 2: zoom on the cavity -------------------------------------------
    ax = fig.add_subplot(gs[1])
    pad = 6e-3
    in_mask = (z > z_m1 - pad) & (z < z_m2 + pad)
    z_rel_in = (z[in_mask] - z_m1) * 1e3
    w_in_zoom = w[in_mask] * 1e6
    ax.fill_between(z_rel_in, -w_in_zoom, w_in_zoom, color="C3", alpha=0.15, lw=0)
    ax.plot(z_rel_in, w_in_zoom, color="C3", lw=1.5, label=f"input beam (w = {w_focus * 1e6:.1f} µm at M1)")
    ax.plot(z_rel_in, -w_in_zoom, color="C3", lw=1.5)
    ax.plot(z_cav * 1e3, w_cav * 1e6, color="C0", lw=2, ls="--",
            label=f"cavity TEM00 (w0 = {w0 * 1e6:.1f} µm at M1)")
    ax.plot(z_cav * 1e3, -w_cav * 1e6, color="C0", lw=2, ls="--")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    h = 230
    ax.plot([0, 0], [-h, h], color="k", lw=3)
    ax.text(0, h + 10, "M1 (flat)", ha="center", va="bottom", fontsize=9)
    # M2: concave toward M1. The real sag over this height is ~2 um, so it is exaggerated here.
    y_m2 = np.linspace(-h, h, 200)
    ax.plot(cav.length * 1e3 - 0.6 * (y_m2 / h) ** 2, y_m2, color="k", lw=3)
    ax.text(cav.length * 1e3, h + 10, f"M2 (ROC = {sim2.ROC2 * 1e3:g} mm)", ha="center", va="bottom",
            fontsize=9)
    ax.annotate("", xy=(cav.length * 1e3 - 0.1, 190), xytext=(0.1, 190), arrowprops={"arrowstyle": "<->"})
    ax.text(cav.length * 1e3 / 2, 195, f"L = {cav.length * 1e3:g} mm", ha="center", va="bottom", fontsize=9)
    # w0: the cavity waist, a radius measured from the axis, sitting on the flat mirror M1
    ax.annotate("", xy=(0.35, -w0 * 1e6), xytext=(0.35, 0),
                arrowprops={"arrowstyle": "<->", "color": "C0", "lw": 1.8, "shrinkA": 0, "shrinkB": 0})
    ax.plot([0, 1.0], [-w0 * 1e6] * 2, color="C0", lw=1.0)
    ax.text(0.6, -w0 * 1e6 / 2, f"w0 = {w0 * 1e6:.1f} µm\n(cavity waist, on M1)", ha="left", va="center",
            fontsize=10, color="C0", fontweight="bold")
    w_m2 = w0 * np.sqrt(1.0 + (cav.length / zr) ** 2)
    ax.text(cav.length * 1e3 - 0.9, w_m2 * 1e6 + 8, f"w = {w_m2 * 1e6:.0f} µm\nR = {r_m2 * 1e3:.1f} mm",
            ha="right", va="bottom", fontsize=9, color="C0")
    ax.set_xlim(-pad * 1e3, (cav.length + pad) * 1e3)
    ax.set_ylim(-h - 60, h + 60)
    ax.set_xlabel("z from M1 [mm]")
    ax.set_ylabel("beam radius [µm]")
    ax.set_title("Zoom on the cavity: focused input beam vs cavity TEM00 mode "
                 "(mirror curvature exaggerated)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # --- panel 3: parameter table ----------------------------------------------
    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    left = [
        ("Source", ""),
        ("wavelength", f"{WAVELENGTH * 1e9:.0f} nm"),
        ("fiber", f"SMF-28, MFD {sim1.FIBER_MFD * 1e6:.1f} µm"),
        ("Collimator", col.name),
        ("focal length", f"{col.focal_length * 1e3:.1f} mm"),
        ("beam diameter / radius", f"{col.beam_diameter * 1e3:.2f} mm / {col.waist * 1e3:.3f} mm"),
        ("Rayleigh range", f"{rayleigh_range(col.waist, WAVELENGTH):.2f} m"),
        ("Focusing lens", f"f = {LENS_FOCAL * 1e3:.0f} mm, d = {LENS_DISTANCE * 1e3:.0f} mm"),
        ("focus after the lens", f"{z_focus * 1e3:.1f} mm"),
        ("focused waist / z_R", f"{w_focus * 1e6:.1f} µm / {zr_focus * 1e3:.2f} mm"),
    ]
    right = [
        ("Cavity (plano-concave)", ""),
        ("M1 / M2", f"flat / ROC {sim2.ROC2 * 1e3:g} mm,  R1 = R2 = {sim2.R1}"),
        ("length L", f"{cav.length * 1e3:g} mm,  g1 = {cav.g1:.3f}, g2 = {cav.g2:.3f}"),
        ("TEM00 waist w0 / z_R", f"{w0 * 1e6:.1f} µm on M1 / {zr * 1e3:.2f} mm"),
        ("w on M2", f"{w_m2 * 1e6:.1f} µm"),
        ("FSR / finesse / FWHM", f"{cav.fsr / 1e9:.2f} GHz / {cav.finesse:.0f} / {cav.linewidth / 1e6:.1f} MHz"),
        ("transverse mode spacing", f"{cav.transverse_mode_spacing / 1e9:.3f} GHz"),
        ("Mode matching into TEM00", ""),
        ("focus exactly on M1", f"{100 * p_ideal:.2f} %  (waist size {w_focus * 1e6:.0f} vs {w0 * 1e6:.0f} µm only)"),
        ("+ misalignment (sim 2)", f"{100 * p_sim2:.1f} %  ({sim2.INPUT_WAIST_OFFSET * 1e3:g} mm, "
                                   f"{sim2.INPUT_OFFSET_X * 1e6:g} µm, {sim2.INPUT_TILT_X * 1e3:g} mrad)"),
    ]
    for x0, rows in ((0.0, left), (0.5, right)):
        for i, (key, val) in enumerate(rows):
            y = 1.0 - i * 0.1
            if not val:
                ax.text(x0, y, key, fontsize=10.5, fontweight="bold", va="top", transform=ax.transAxes)
            else:
                ax.text(x0 + 0.015, y, key, fontsize=9.5, va="top", transform=ax.transAxes)
                ax.text(x0 + 0.2, y, val, fontsize=9.5, va="top", transform=ax.transAxes)
    ax.set_title("System parameters", loc="left", fontsize=12)

    fig.savefig(OUT_DIR / "sim4_system_layout.png", dpi=150, bbox_inches="tight")
    print(f"Collimator {col.name}: w = {col.waist * 1e3:.3f} mm; lens f = {LENS_FOCAL * 1e3:.0f} mm "
          f"at d = {LENS_DISTANCE * 1e3:.0f} mm")
    print(f"Focus {z_focus * 1e3:.1f} mm after the lens, w = {w_focus * 1e6:.1f} um")
    print(f"Cavity w0 = {w0 * 1e6:.1f} um on M1, w(M2) = {w_m2 * 1e6:.1f} um, R(M2) = {r_m2 * 1e3:.2f} mm")
    print(f"Mode matching into TEM00: {100 * p_ideal:.2f} % (ideal placement), "
          f"{100 * p_sim2:.1f} % (with the simulation 2 misalignment)")
    print(f"Saved {(OUT_DIR / 'sim4_system_layout.png').relative_to(ROOT)}")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="only save the figure, do not open a window")
    main(show=not parser.parse_args().no_show)
