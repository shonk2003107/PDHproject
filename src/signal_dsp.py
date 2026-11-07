"""Laser phase noise and delayed self-homodyne linewidth measurement.

Field model
-----------
E(t) = E0 exp(i (w0 t + phi(t))). The optical carrier w0 only adds a constant phase
w0*tau to the homodyne beat, so the simulation works with the complex envelope
E0 exp(i phi(t)).

phi(t) is a random walk (Wiener process). Its increments are Gaussian with
variance 2 pi dnu dt, which gives an exponentially decaying field autocorrelation
    <E(t + s) E*(t)> = E0^2 exp(-|s| / tau_c),   tau_c = 1 / (pi dnu),
and therefore (Wiener-Khinchin) a Lorentzian optical spectrum with FWHM dnu.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Phase noise and fields
# ---------------------------------------------------------------------------


def coherence_time(linewidth: float) -> float:
    """tau_c = 1 / (pi dnu) for a Lorentzian line of FWHM dnu [s]."""
    return 1.0 / (np.pi * linewidth)


def random_walk_phase(n_samples: int, fs: float, linewidth: float,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Random-walk phase phi[k] with Gaussian steps of variance 2 pi dnu / fs."""
    rng = rng or np.random.default_rng()
    steps = rng.normal(0.0, np.sqrt(2.0 * np.pi * linewidth / fs), n_samples)
    steps[0] = 0.0
    return np.cumsum(steps)


def optical_field(phase, amplitude: float = 1.0) -> np.ndarray:
    """Complex envelope E0 exp(i phi)."""
    return amplitude * np.exp(1j * np.asarray(phase))


def field_autocorrelation(field, max_lag: int) -> np.ndarray:
    """Normalized |<E(t + s) E*(t)>| for lags s = 0 .. max_lag - 1 samples."""
    field = np.asarray(field)
    n = len(field)
    g = np.array([np.mean(field[k:] * np.conj(field[:n - k])) for k in range(max_lag)])
    return np.abs(g) / np.abs(g[0])


# ---------------------------------------------------------------------------
# Delayed self-homodyne
# ---------------------------------------------------------------------------


def homodyne_beat(field, n_delay: int) -> np.ndarray:
    """Beat of the field with a copy delayed by n_delay samples (no frequency shift).

    b(t) = E(t) E*(t - tau): the interference term detected in a homodyne setup.
    Without an AOM shift its spectrum is centered on 0 Hz.
    """
    field = np.asarray(field)
    if n_delay == 0:
        return np.abs(field) ** 2 + 0j
    return field[n_delay:] * np.conj(field[:-n_delay])


def two_sided_psd(x, fs: float, nperseg: int):
    """Two-sided Welch PSD of a complex signal, with frequencies sorted from -fs/2 to fs/2."""
    f, p = signal.welch(x, fs=fs, nperseg=nperseg, window="hann",
                        detrend=False, return_onesided=False)
    return np.fft.fftshift(f), np.fft.fftshift(p)


def lorentzian(f, amplitude, fwhm):
    """Lorentzian centered on 0: A / (1 + (2 f / FWHM)^2)."""
    return amplitude / (1.0 + (2.0 * f / fwhm) ** 2)


def fit_lorentzian(f, p, fwhm_guess: float):
    """Least-squares Lorentzian fit to a spectrum centered on 0 Hz. Returns (amplitude, fwhm).

    The spectrum is normalized to its peak before fitting so the optimizer works with O(1) numbers.
    """
    p = np.asarray(p)
    scale = p.max()
    (amp, fwhm), _ = curve_fit(lorentzian, f, p / scale, p0=[1.0, fwhm_guess],
                               bounds=([0.0, 0.0], [np.inf, np.inf]))
    return amp * scale, abs(fwhm)


def self_homodyne_spectrum_theory(f, linewidth: float, delay: float):
    """Analytic beat spectrum (continuous part) for a Lorentzian laser of FWHM dnu.

    S(f) = (dnu / pi) / (f^2 + dnu^2) * {1 - exp(-2 pi dnu tau)
            [cos(2 pi f tau) + (dnu / f) sin(2 pi f tau)]}
    plus a coherent delta at f = 0 of weight exp(-2 pi dnu tau) (not included).
    For tau >> tau_c it becomes a Lorentzian with FWHM 2 dnu.
    (Richter et al., IEEE J. Quantum Electron. 22, 2070 (1986).)
    """
    f = np.asarray(f, dtype=float)
    a = 2.0 * np.pi * linewidth * delay
    x = 2.0 * np.pi * f * delay
    sinc_term = np.where(f == 0, 2.0 * np.pi * linewidth * delay, linewidth * np.sin(x) / np.where(f == 0, 1, f))
    bracket = 1.0 - np.exp(-a) * (np.cos(x) + sinc_term)
    return (linewidth / np.pi) / (f**2 + linewidth**2) * bracket
