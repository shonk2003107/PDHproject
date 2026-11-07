"""Simulation 3: balanced homodyne detection with laser phase noise, mixing and LPF.

Model (complex envelopes at optical baseband):
  * One laser with Wiener phase noise (Lorentzian linewidth) feeds both arms.
  * Signal arm: delayed by tau and phase-modulated at f_m (EOM, depth beta).
  * LO arm: held at the quadrature point (theta = pi/2) -> linear phase readout.
  * Balanced detector with shot noise -> RF mixer at f_m -> Butterworth LPF.

Checks: measured current PSD vs. theory (delayed self-homodyne phase noise + shot
noise) and the demodulated DC level vs. the expected 2 K J1(beta).

Run from the project root:  python simulations/sim3_homodyne_noise.py [--show]
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import jv  # noqa: E402

from src import signal_dsp as dsp  # noqa: E402

OUT_DIR = ROOT / "data_out"

# ----------------------------- parameters ---------------------------------
FS = 10e6              # sample rate [Hz]
DURATION = 20e-3       # [s]
LINEWIDTH = 1e3        # laser Lorentzian FWHM [Hz]
DELAY = 1e-6           # path difference between arms [s]
P_SIGNAL = 1e-3        # [W]
P_LO = 10e-3           # [W]
RESPONSIVITY = 0.8     # [A/W]
MOD_FREQ = 1e6         # EOM frequency [Hz]
BETA = 0.05            # modulation depth [rad]
LPF_CUTOFF = 20e3      # [Hz]
LO_PHASE = np.pi / 2   # optical LO phase -> phase quadrature
SEED = 1


def main(show: bool) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    n = int(FS * DURATION)
    t = np.arange(n) / FS
    n_delay = int(round(DELAY * FS))

    # --- optical fields ------------------------------------------------------
    phi_laser = dsp.laser_phase_noise(n, FS, LINEWIDTH, rng)
    e_sig = dsp.optical_field(P_SIGNAL, dsp.delay_samples(phi_laser, n_delay))
    e_sig = dsp.phase_modulate(e_sig, t, MOD_FREQ, BETA)
    e_lo = dsp.optical_field(P_LO, phi_laser + LO_PHASE)

    # --- detection -----------------------------------------------------------
    current = dsp.balanced_homodyne(e_sig, e_lo, RESPONSIVITY, FS, shot_noise=True, rng=rng)
    gain = 2.0 * RESPONSIVITY * np.sqrt(P_SIGNAL * P_LO)   # K: A per rad of phase

    # --- mixing + LPF (demodulate the f_m tone in the sine quadrature) -------
    demod = dsp.demodulate(current, t, FS, MOD_FREQ, lo_phase=-np.pi / 2, cutoff=LPF_CUTOFF)
    # i = K sin(beta sin(w t) + dphi); mixing with 2 sin(w t) and LPF -> 2 K J1(beta)
    expected_dc = 2.0 * gain * jv(1, BETA)
    skip = int(5 / LPF_CUTOFF * FS)           # drop filter edge transients
    measured_dc = np.mean(demod[skip:-skip])
    print(f"Phase-to-current gain K       : {gain*1e3:.4f} mA/rad")
    print(f"Demodulated level (measured)  : {measured_dc*1e6:.3f} uA")
    print(f"Demodulated level (expected)  : {expected_dc*1e6:.3f} uA")

    # --- spectra -------------------------------------------------------------
    f, s_i = dsp.psd(current, FS, nperseg=2**16)
    f = f[1:]
    s_i = s_i[1:]
    s_phase_theory = gain**2 * dsp.delayed_self_homodyne_psd_theory(f, LINEWIDTH, n_delay / FS)
    s_shot = dsp.shot_noise_psd(RESPONSIVITY * (P_SIGNAL + P_LO))

    # --- plots ---------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    ax = axes[0]
    win = slice(0, int(5 / MOD_FREQ * FS))
    ax.plot(t[win] * 1e6, current[win] * 1e3, lw=0.8)
    ax.set_xlabel("time [us]")
    ax.set_ylabel("homodyne current [mA]")
    ax.set_title("Balanced homodyne output (first 5 modulation periods)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.loglog(f, s_i, lw=0.8, label="simulated")
    ax.loglog(f, s_phase_theory + s_shot, "k--", lw=1, label="theory: $K^2 S_{\\Delta\\phi}$ + shot")
    ax.axhline(s_shot, color="C3", ls=":", label="shot-noise floor")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("current PSD [A$^2$/Hz]")
    ax.set_title(f"Current noise spectrum (linewidth {LINEWIDTH:.0f} Hz, delay {DELAY*1e6:.1f} us)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.plot(t[skip:-skip] * 1e3, demod[skip:-skip] * 1e6, lw=0.8, label="mixer + LPF output")
    ax.axhline(expected_dc * 1e6, color="k", ls="--", label="expected $2K J_1(\\beta)$")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("demodulated [uA]")
    ax.set_title(f"Demodulation at {MOD_FREQ/1e6:.1f} MHz, LPF cutoff {LPF_CUTOFF/1e3:.0f} kHz")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim3_homodyne_noise.png", dpi=150)

    with open(OUT_DIR / "sim3_current_psd.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["freq_Hz", "psd_sim_A2_per_Hz", "psd_theory_A2_per_Hz"])
        writer.writerows(zip(f, s_i, s_phase_theory + s_shot))

    print(f"\nSaved results to {OUT_DIR}")
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the figure window")
    main(parser.parse_args().show)
