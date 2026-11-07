"""Simulation 2: Fabry-Perot cavity stability, resonance response and PDH error signal.

Panels:
  1. g1-g2 stability diagram with the chosen cavity marked
  2. Waist size and transverse-mode spacing vs cavity length
  3. Reflected / transmitted power vs laser detuning
  4. PDH error signal vs detuning

Run from the project root:  python simulations/sim2_cavity_stability.py [--show]
"""

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.fabry_perot import FabryPerot  # noqa: E402
from src.optics_utils import waist_from_q  # noqa: E402

OUT_DIR = ROOT / "data_out"

# ----------------------------- parameters ---------------------------------
CAVITY = FabryPerot(length=0.10, R1=0.99, R2=0.999, roc1=np.inf, roc2=0.5, wavelength=1064e-9)
MOD_FREQ = 20e6        # EOM modulation frequency Omega/2pi [Hz]
BETA = 1.08            # modulation depth (maximizes J0*J1)
INPUT_POWER = 1e-3     # [W]


def main(show: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    info = CAVITY.summary()
    print("Cavity parameters:")
    for k, v in info.items():
        print(f"  {k:>18}: {v:.6g}" if isinstance(v, float) else f"  {k:>18}: {v}")

    # Cross-check: ABCD round-trip eigenmode vs analytic formula
    w0_abcd, z_abcd = waist_from_q(CAVITY.eigenmode_q(), CAVITY.wavelength)
    w0_an, z_an = CAVITY.waist()
    print(f"\nWaist (ABCD)    : {w0_abcd*1e6:.3f} um @ {z_abcd*1e3:.3f} mm")
    print(f"Waist (analytic): {w0_an*1e6:.3f} um @ {z_an*1e3:.3f} mm")

    slope = CAVITY.pdh_slope(MOD_FREQ, BETA, INPUT_POWER)
    print(f"PDH slope at resonance: {slope:.4e} W/Hz")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- 1. stability diagram ----------------------------------------------
    ax = axes[0, 0]
    g = np.linspace(-2, 2, 600)
    G1, G2 = np.meshgrid(g, g)
    stable = (G1 * G2 > 0) & (G1 * G2 < 1)
    ax.contourf(G1, G2, stable, levels=[0.5, 1.5], colors=["#9ecae1"])
    gg = np.linspace(0.5, 2, 200)
    for s in (1, -1):
        ax.plot(s * gg, s / gg, "k", lw=1)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    for name, (a, b) in {"planar": (1, 1), "confocal": (0, 0), "concentric": (-1, -1)}.items():
        ax.plot(a, b, "ko", ms=4)
        ax.annotate(name, (a, b), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.plot(CAVITY.g1, CAVITY.g2, "r*", ms=14, label=f"this cavity (g1g2={CAVITY.g_product:.2f})")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xlabel("$g_1 = 1 - L/R_1$")
    ax.set_ylabel("$g_2 = 1 - L/R_2$")
    ax.set_title("Stability diagram (shaded = stable)")
    ax.legend(loc="lower right", fontsize=8)

    # --- 2. waist & TMS vs length --------------------------------------------
    ax = axes[0, 1]
    lengths = np.linspace(1e-3, CAVITY.roc2 * 0.999, 400) if np.isinf(CAVITY.roc1) else \
        np.linspace(1e-3, (CAVITY.roc1 + CAVITY.roc2) * 0.999, 400)
    w0s, tms = [], []
    for L in lengths:
        cav = replace(CAVITY, length=L)
        if cav.is_stable:
            w0s.append(cav.waist()[0])
            tms.append(cav.transverse_mode_spacing() / cav.fsr)
        else:
            w0s.append(np.nan)
            tms.append(np.nan)
    ax.plot(lengths * 1e3, np.array(w0s) * 1e6, "C0")
    ax.axvline(CAVITY.length * 1e3, color="r", ls="--", lw=1)
    ax.set_xlabel("cavity length L [mm]")
    ax.set_ylabel("waist $w_0$ [um]", color="C0")
    ax2 = ax.twinx()
    ax2.plot(lengths * 1e3, tms, "C1")
    ax2.set_ylabel("TMS / FSR", color="C1")
    ax.set_title("Waist and transverse-mode spacing vs L")
    ax.grid(alpha=0.3)

    # --- 3. reflection / transmission ---------------------------------------
    ax = axes[1, 0]
    df = np.linspace(-1.2 * CAVITY.fsr, 1.2 * CAVITY.fsr, 40001)
    refl = np.abs(CAVITY.reflection(df)) ** 2
    trans = np.abs(CAVITY.transmission(df)) ** 2
    ax.semilogy(df / 1e6, refl, label="reflected  $|F|^2$")
    ax.semilogy(df / 1e6, trans, label="transmitted $|T|^2$")
    ax.set_xlabel("detuning [MHz]")
    ax.set_ylabel("P / P_in")
    ax.set_title(f"FSR = {CAVITY.fsr/1e9:.3f} GHz, F = {CAVITY.finesse:.0f}, "
                 f"FWHM = {CAVITY.linewidth/1e6:.2f} MHz")
    ax.set_ylim(1e-4, 2)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, which="both")

    # --- 4. PDH error signal --------------------------------------------------
    ax = axes[1, 1]
    span = 2.5 * MOD_FREQ
    df_pdh = np.linspace(-span, span, 20001)
    err = CAVITY.pdh_error_signal(df_pdh, MOD_FREQ, BETA, INPUT_POWER)
    ax.plot(df_pdh / 1e6, err * 1e3)
    ax.set_xlabel("detuning [MHz]")
    ax.set_ylabel("error signal [mW]")
    ax.set_title(f"PDH error signal ($\\Omega/2\\pi$ = {MOD_FREQ/1e6:.0f} MHz, $\\beta$ = {BETA})")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim2_cavity_stability.png", dpi=150)

    with open(OUT_DIR / "sim2_pdh_error_signal.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["detuning_Hz", "error_signal_W", "reflected_power_frac"])
        writer.writerows(zip(df_pdh, err, np.abs(CAVITY.reflection(df_pdh)) ** 2))

    print(f"\nSaved results to {OUT_DIR}")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the figure window")
    main(parser.parse_args().show)
