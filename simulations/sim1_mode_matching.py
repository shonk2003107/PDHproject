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
EXAMPLE_FOCAL_MM = 200          # lab focusing lens, used for the beam-profile plot and the schematic [mm]


def draw_schematic():
    """Schematic (NOT to scale) of the collimator -> lens -> new waist system with every parameter
    of this simulation and the q-parameter formulas. Coordinates are arbitrary drawing units."""
    from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

    red, purple, blue, green = "#d62728", "#9467bd", "#1f77b4", "#2ca02c"
    fig, ax = plt.subplots(figsize=(17, 7.2))
    ax.set_xlim(-6, 124)
    ax.set_ylim(-13.6, 12.6)
    ax.axis("off")
    ax.add_patch(Rectangle((0, -12.6), 122, 24.4, fill=False, ec="0.3", lw=1))
    ax.set_title("System schematic with all parameters — NOT to scale", loc="left", fontsize=13)
    ax.text(-3.5, 0, "r  (transverse radius)", rotation=90, ha="center", va="center", fontsize=11)
    ax.text(61, -13.4, "z  (propagation direction)", ha="center", va="top", fontsize=11)

    # geometry in drawing units
    x_col, x_lens, x_waist, x_end = 16.0, 48.0, 80.0, 116.0
    w_in, w_out = 4.5, 0.9
    w_lens = 4.6                                        # input beam grows very slightly up to the lens
    zr2 = (x_waist - x_lens) / np.sqrt((w_lens / w_out) ** 2 - 1.0)   # z'_R so that w(lens) = w_lens

    # beam envelope
    x1 = np.linspace(x_col, x_lens, 100)
    w1 = w_in + (w_lens - w_in) * ((x1 - x_col) / (x_lens - x_col)) ** 2
    x2 = np.linspace(x_lens, x_end, 400)
    w2 = w_out * np.sqrt(1.0 + ((x2 - x_waist) / zr2) ** 2)
    xb, wb = np.concatenate([x1, x2]), np.concatenate([w1, w2])
    ax.fill_between(xb, -wb, wb, color=red, alpha=0.13, lw=0)
    ax.plot(xb, wb, color=red, lw=2.2)
    ax.plot(xb, -wb, color=red, lw=2.2)
    ax.plot([x_col, 120], [0, 0], "k--", lw=0.9)
    ax.text(120.5, 0.4, "optical axis z", ha="right", va="bottom", fontsize=10)

    # fiber + collimator
    ax.plot([-1.5, 4], [0, 0], color="0.35", lw=3)
    ax.add_patch(Rectangle((4, -5.7), 12, 11.4, fc="0.45", ec="0.3"))
    ax.text(10, 0, "fiber\nMFD", color="white", ha="center", va="center", fontsize=10)
    ax.text(9.3, 6.2, "FC/APC collimator\n($f_c$ , $D$ from datasheet)", ha="center", va="bottom", fontsize=10.5)

    # w_in, D, R = inf, w(z) formula
    ax.annotate("", xy=(19.5, w_in), xytext=(19.5, 0), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(20.2, 2.4, r"$w_{in} = D/2$", fontsize=12, va="center")
    ax.plot([25.5, 25.5], [-w_in, w_in], color=purple, lw=1.5)
    ax.text(26.0, 3.5, r"$R=\infty$", color=purple, fontsize=11)
    ax.annotate("", xy=(30.5, w_in), xytext=(30.5, -w_in), arrowprops={"arrowstyle": "<->", "lw": 1.4})
    ax.text(31.2, -0.6, r"$D$ (1/e$^2$ diameter)", fontsize=11, va="center")
    ax.text(31.5, -6.6, r"$w(z) = w_{in}\sqrt{1+(z/z_R)^2} \approx w_{in}$", color=red, fontsize=12,
            ha="center", va="center")

    # lens
    ax.add_patch(Ellipse((x_lens, 0), 2.6, 13.6, fc="#9ecae1", ec="#08519c", lw=1.5, alpha=0.9))
    ax.text(x_lens + 2.0, 7.4, rf"thin lens, focal length $f$ = {EXAMPLE_FOCAL_MM} mm", color=blue, fontsize=11.5)

    # wavefront curvature arcs
    yy = np.linspace(-1.0, 1.0, 50)
    w_c = w_out * np.sqrt(1.0 + ((62.0 - x_waist) / zr2) ** 2)
    ax.plot(62.0 + 1.4 * yy ** 2, yy * w_c, color=purple, lw=1.6)          # converging: "(" shape
    ax.text(62.6, -w_c - 0.6, "$R(z)<0$\nconverging", color=purple, ha="center", va="top", fontsize=10.5)
    w_d = w_out * np.sqrt(1.0 + ((101.0 - x_waist) / zr2) ** 2)
    ax.plot(101.0 - 1.4 * yy ** 2, yy * w_d, color=purple, lw=1.6)         # diverging: ")" shape
    ax.text(100.0, w_d + 0.4, "$R(z)>0$\ndiverging", color=purple, ha="center", va="bottom", fontsize=10.5)
    ax.text(116.0, -2.6, r"$R(z) = \frac{1}{\mathrm{Re}(1/q)}$", color=purple, ha="right", fontsize=11.5)

    # new waist = focal plane (z_R >> |d - f| makes z_out - f negligible)
    ax.plot([x_waist, x_waist], [-11.3, 8.6], color=red, ls="-.", lw=1.4)
    ax.text(x_waist, 8.8, "new waist = focal plane", color=red, ha="center",
            va="bottom", fontsize=10)

    # w_out, sqrt(2) w_out, depth of focus, divergence
    ax.annotate("", xy=(x_waist, w_out), xytext=(x_waist, 0),
                arrowprops={"arrowstyle": "->", "lw": 1.3, "shrinkA": 0, "shrinkB": 0})
    ax.annotate(r"$w_{out}$", xy=(x_waist + 0.2, w_out * 0.6), xytext=(x_waist + 8, 3.0), fontsize=13,
                arrowprops={"arrowstyle": "->", "lw": 1.1})
    for xp in (x_waist - zr2, x_waist + zr2):
        ax.plot(xp, -np.sqrt(2) * w_out, "ko", ms=4.5)
    ax.text(x_waist + zr2 - 3.5, -0.4, r"$\sqrt{2}\,w_{out}$", fontsize=12, va="center")
    ax.annotate("", xy=(x_waist + zr2, -4.6), xytext=(x_waist - zr2, -4.6), arrowprops={"arrowstyle": "<->"})
    for xp in (x_waist - zr2, x_waist + zr2):
        ax.plot([xp, xp], [-np.sqrt(2) * w_out, -4.9], color="k", lw=0.8)
    ax.text(x_waist, -5.0, r"$2z_R'$ (depth of focus)", ha="center", va="top", fontsize=11,
            bbox={"fc": "white", "ec": "none", "pad": 1.0}, zorder=3)
    xs = np.array([x_waist, x_end])
    ax.plot(xs, (xs - x_waist) * w_out / zr2, color="0.4", ls="--", lw=1.0)
    ax.annotate(r"$\theta' = \lambda/(\pi w_{out}) \approx w_{in}/f$", xy=(108, (108 - x_waist) * w_out / zr2),
                xytext=(88, 9.6), fontsize=12, arrowprops={"arrowstyle": "->", "lw": 1.1})

    # dimension arrows
    for y, x0, x1_, label, color in (
            (-8.6, x_col, x_lens, rf"$d$ = {LENS_DISTANCE * 1e3:.0f} mm  (collimator → lens)", "k"),
            (-10.6, x_lens, x_waist, rf"$f$ = {EXAMPLE_FOCAL_MM} mm  (lens → new waist)", red)):
        ax.annotate("", xy=(x1_, y), xytext=(x0, y), arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.3})
        ax.text((x0 + x1_) / 2, y + 0.25, label, color=color, ha="center", va="bottom", fontsize=11.5)
    ax.plot([x_col, x_col], [-w_in, -8.9], color="0.5", lw=0.8)

    # input Rayleigh range, from the collimators of this simulation
    zrs = [rayleigh_range(c.waist, WAVELENGTH) for c in COLLIMATORS]
    ax.annotate("", xy=(121.5, 10.7), xytext=(x_col, 10.7),
                arrowprops={"arrowstyle": "->", "color": green, "lw": 1.4})
    ax.text(68, 11.0, rf"$z_R = \pi w_{{in}}^2/\lambda \approx {min(zrs):.1f}-{max(zrs):.1f}$ m"
            r"   (far beyond this frame: $z_R \gg d$, so the input beam is ~flat)",
            color=green, ha="center", va="bottom", fontsize=11.5)
    ax.plot([x_col, x_col], [w_in, 10.7], color=green, ls=":", lw=1.0)

    # formula box
    ax.add_patch(FancyBboxPatch((92.5, -12.2), 28.8, 7.0, boxstyle="round,pad=0.3", fc="#fffbe6",
                                ec="0.4", lw=1.0, zorder=3))
    formulas = [
        r"$q_{in} = i\,z_R$",
        r"$q_2 = \left(\frac{1}{d + i z_R} - \frac{1}{f}\right)^{-1}$",
        r"$z_R' = \mathrm{Im}(q_2)$",
        r"$w_{out} = \sqrt{\lambda z_R'/\pi} = \frac{w_{in} f}{\sqrt{(d-f)^2 + z_R^2}} \approx \frac{\lambda f}{\pi w_{in}}$",
    ]
    for i, text in enumerate(formulas):
        ax.text(120.8, -5.9 - 1.6 * i, text, ha="right", va="center", fontsize=11.5, zorder=4)

    fig.tight_layout()
    return fig


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
    # z_R of the collimated beam (2-6 m) >> |d - f| (0.1 m), so z_out - f = f^2 (d - f) / ((d - f)^2 + z_R^2)
    # is below 1 mm, far inside the depth of focus: in practice the new waist sits on the focal plane.
    f_ex = EXAMPLE_FOCAL_MM * 1e-3
    print(f"\nz_out - f over the lens catalogue (z_R >> |d - f| -> z_out = f in practice):")
    print(f"  {'collimator':<14}{'f [mm]':>8}{'z_out - f [mm]':>16}{'2 z_R_out [mm]':>16}")
    for col in COLLIMATORS:
        z_out, w_out = focus_beam(col.waist, WAVELENGTH, LENS_DISTANCE, f_m)
        for f, dz, w in zip(FOCAL_LENGTHS_MM, z_out - f_m, w_out):
            if f in (100, 200, 300):
                print(f"  {col.name:<14}{f:>8.0f}{dz * 1e3:>+16.3f}"
                      f"{2 * rayleigh_range(w, WAVELENGTH) * 1e3:>16.1f}")

    fig, axes = plt.subplot_mosaic([["w"], ["profile"]], figsize=(12, 9))
    f_dense = np.linspace(FOCAL_LENGTHS_MM.min(), FOCAL_LENGTHS_MM.max(), 400) * 1e-3

    ax = axes["w"]
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

    ax = axes["profile"]
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
    draw_schematic().savefig(OUT_DIR / "sim1_schematic.png", dpi=150)
    print(f"\nSaved results to {OUT_DIR.relative_to(ROOT)}/")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="only save the figure, do not open a window")
    main(show=not parser.parse_args().no_show)
