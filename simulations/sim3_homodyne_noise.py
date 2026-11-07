"""Simulation 3: homodyne measurement of random laser phase noise (linewidth vs delay).

Signal model:  E(t) = E0 exp(i (w t + phi(t))), with phi(t) a random walk whose field
autocorrelation decays exponentially (coherence time tau_c = 1 / (pi dnu)). By the
Wiener-Khinchin theorem the laser spectrum is then a Lorentzian of FWHM dnu.

Measurement (delayed self-homodyne):
  1. Make two copies of the field: the original and a copy delayed by tau.
  2. Beat them without a frequency shift (no AOM): b(t) = E(t) E*(t - tau).
     The beat spectrum is centered on 0 Hz instead of the AOM frequency (e.g. 80 MHz).
  3. Remove the DC (coherent) part, compute the spectrum and fit a Lorentzian -> FWHM.
  4. Repeat for tau from ~0 to several tau_c: the measured FWHM settles at 2 dnu
     once tau exceeds the coherence time (the two copies become uncorrelated).

Run from the project root:  python simulations/sim3_homodyne_noise.py [--no-show]
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import signal_dsp as dsp  # noqa: E402

OUT_DIR = ROOT / "data_out"

# ----------------------------- parameters ---------------------------------
LINEWIDTH = 100e3                  # true laser Lorentzian FWHM dnu [Hz]
FS = 20e6                          # sample rate [Hz] (>> linewidth)
N_SAMPLES = 2**22                  # record length (~0.2 s at 20 MHz)
NPERSEG = 2**12                    # Welch segment length -> resolution FS/NPERSEG
FIT_SPAN = 20                      # fit the spectrum over |f| < FIT_SPAN * dnu
TAU_OVER_TAUC = np.linspace(0.1, 5.0, 40)   # delays to scan, in units of tau_c
EXAMPLE_DELAYS = [0.25, 1.0, 4.0]  # delays shown in the spectrum panel [tau_c]
SEED = 1


def measure(field, tau_over_tauc, tau_c):
    """Beat the field with its delayed copy and fit a Lorentzian. Returns (f, psd, amp, fwhm, n_delay)."""
    n_delay = int(round(tau_over_tauc * tau_c * FS))
    beat = dsp.homodyne_beat(field, n_delay)
    f, p = dsp.two_sided_psd(beat - beat.mean(), FS, NPERSEG)
    band = np.abs(f) < FIT_SPAN * LINEWIDTH
    amp, fwhm = dsp.fit_lorentzian(f[band], p[band], fwhm_guess=2.0 * LINEWIDTH)
    return f, p, amp, fwhm, n_delay


def main(show: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    tau_c = dsp.coherence_time(LINEWIDTH)

    print(f"Laser linewidth dnu = {LINEWIDTH / 1e3:.1f} kHz -> coherence time tau_c = {tau_c * 1e6:.3f} us")
    print(f"Record: {N_SAMPLES / FS * 1e3:.1f} ms, spectral resolution {FS / NPERSEG / 1e3:.2f} kHz\n")

    phi = dsp.random_walk_phase(N_SAMPLES, FS, LINEWIDTH, rng)
    field = dsp.optical_field(phi)

    # --- check the noise model: exponential field autocorrelation -------------
    max_lag = int(4 * tau_c * FS)
    lags = np.arange(max_lag) / FS
    g1 = dsp.field_autocorrelation(field[:2**20], max_lag)

    # --- scan the delay ----------------------------------------------------------
    rows = []
    print(f"  {'tau/tau_c':>9}{'tau [us]':>10}{'FWHM sim [kHz]':>16}{'FWHM theory [kHz]':>19}")
    for r in TAU_OVER_TAUC:
        f, _, _, fwhm, n_delay = measure(field, r, tau_c)
        band = np.abs(f) < FIT_SPAN * LINEWIDTH
        theory = dsp.self_homodyne_spectrum_theory(f[band], LINEWIDTH, n_delay / FS)
        _, fwhm_theory = dsp.fit_lorentzian(f[band], theory, fwhm_guess=2.0 * LINEWIDTH)
        rows.append({"tau_over_tauc": n_delay / FS / tau_c, "tau_s": n_delay / FS,
                     "fwhm_sim_Hz": fwhm, "fwhm_theory_Hz": fwhm_theory})
        print(f"  {n_delay / FS / tau_c:>9.2f}{n_delay / FS * 1e6:>10.2f}"
              f"{fwhm / 1e3:>16.1f}{fwhm_theory / 1e3:>19.1f}")

    final = rows[-1]["fwhm_sim_Hz"]
    print(f"\nLong-delay beat FWHM = {final / 1e3:.1f} kHz -> laser linewidth = FWHM / 2 = "
          f"{final / 2e3:.1f} kHz (true {LINEWIDTH / 1e3:.1f} kHz)")

    with open(OUT_DIR / "sim3_linewidth_vs_delay.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # --- plots ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    t_show = np.arange(int(20 * tau_c * FS)) / FS
    ax.plot(t_show * 1e6, phi[:len(t_show)], lw=0.8)
    ax.set_xlabel("time [us]")
    ax.set_ylabel("$\\phi(t)$ [rad]")
    ax.set_title("Random-walk phase noise $\\phi(t)$")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(lags / tau_c, g1, label="simulated $|\\langle E(t+s)E^*(t)\\rangle|$")
    ax.plot(lags / tau_c, np.exp(-lags / tau_c), "k--", label="$e^{-|s|/\\tau_c}$")
    ax.set_xlabel("lag s / $\\tau_c$")
    ax.set_ylabel("normalized autocorrelation")
    ax.set_title("Field autocorrelation (exponential -> Lorentzian spectrum)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for i, r in enumerate(EXAMPLE_DELAYS):
        f, p, amp, fwhm, _ = measure(field, r, tau_c)
        band = np.abs(f) < FIT_SPAN * LINEWIDTH
        ax.plot(f[band] / 1e3, p[band] / amp, color=f"C{i}", lw=0.8,
                label=f"$\\tau$ = {r:g} $\\tau_c$: FWHM = {fwhm / 1e3:.0f} kHz")
        ax.plot(f[band] / 1e3, dsp.lorentzian(f[band], 1.0, fwhm), "--", color=f"C{i}", lw=1)
    ax.set_xlabel("frequency [kHz]")
    ax.set_ylabel("beat PSD (normalized to fit peak)")
    ax.set_title("Homodyne beat spectra (solid) and Lorentzian fits (dashed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    x = np.array([row["tau_over_tauc"] for row in rows])
    ax.plot(x, [row["fwhm_sim_Hz"] / 1e3 for row in rows], "o", ms=4, label="simulation")
    ax.plot(x, [row["fwhm_theory_Hz"] / 1e3 for row in rows], "k-", lw=1, label="analytic spectrum")
    ax.axhline(2 * LINEWIDTH / 1e3, color="C3", ls="--", label=f"$2\\Delta\\nu$ = {2 * LINEWIDTH / 1e3:.0f} kHz")
    ax.axvline(1.0, color="0.5", ls=":", label="$\\tau = \\tau_c$")
    ax.set_yscale("log")
    ax.set_xlabel("delay $\\tau$ / $\\tau_c$")
    ax.set_ylabel("measured FWHM [kHz]")
    ax.set_title("Measured linewidth vs delay")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim3_homodyne_noise.png", dpi=150)
    print(f"\nSaved results to {OUT_DIR.relative_to(ROOT)}/")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="only save the figure, do not open a window")
    main(show=not parser.parse_args().no_show)
