"""Signal-processing models for homodyne / PDH detection.

Phase noise generation, optical fields, balanced homodyne detection,
electronic mixing (demodulation), low-pass filtering and spectral estimation.
All simulations are done at optical baseband (complex envelopes).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

E_CHARGE = 1.602176634e-19  # C


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------


def laser_phase_noise(n_samples: int, fs: float, linewidth: float,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Wiener (random-walk) phase of a laser with Lorentzian FWHM `linewidth` [Hz].

    Increments are Gaussian with variance 2*pi*linewidth/fs. This corresponds to
    white frequency noise and a one-sided phase PSD S_phi(f) = linewidth / (pi f^2).
    """
    rng = rng or np.random.default_rng()
    steps = rng.normal(0.0, np.sqrt(2.0 * np.pi * linewidth / fs), n_samples)
    steps[0] = 0.0
    return np.cumsum(steps)


def phase_noise_psd_theory(f, linewidth: float):
    """One-sided phase PSD [rad^2/Hz] of a Wiener phase process."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore"):
        return linewidth / (np.pi * f**2)


def delayed_self_homodyne_psd_theory(f, linewidth: float, delay: float):
    """One-sided PSD of phi(t - tau) - phi(t) for a Wiener phase process [rad^2/Hz]."""
    f = np.asarray(f, dtype=float)
    return phase_noise_psd_theory(f, linewidth) * 4.0 * np.sin(np.pi * f * delay) ** 2


def white_noise(n_samples: int, psd_one_sided: float, fs: float,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Real white noise with a given one-sided PSD [units^2/Hz]."""
    rng = rng or np.random.default_rng()
    return rng.normal(0.0, np.sqrt(psd_one_sided * fs / 2.0), n_samples)


def shot_noise_psd(mean_current: float) -> float:
    """One-sided shot-noise current PSD 2 e I [A^2/Hz]."""
    return 2.0 * E_CHARGE * mean_current


# ---------------------------------------------------------------------------
# Optical fields and detection
# ---------------------------------------------------------------------------


def optical_field(power: float, phase) -> np.ndarray:
    """Complex envelope sqrt(P) * exp(i*phase), normalized so |E|^2 = P [W]."""
    return np.sqrt(power) * np.exp(1j * np.asarray(phase))


def phase_modulate(field, t, mod_freq: float, beta: float):
    """Apply sinusoidal phase modulation beta*sin(2 pi f_m t) (EOM)."""
    return field * np.exp(1j * beta * np.sin(2.0 * np.pi * mod_freq * np.asarray(t)))


def photodetect(field, responsivity: float = 0.8, fs: float | None = None,
                shot_noise: bool = False, rng: np.random.Generator | None = None):
    """Single photodiode current R*|E|^2, optionally with shot noise (bandwidth fs/2)."""
    current = responsivity * np.abs(field) ** 2
    if shot_noise:
        if fs is None:
            raise ValueError("fs is required to add shot noise")
        rng = rng or np.random.default_rng()
        sigma = np.sqrt(shot_noise_psd(np.mean(current)) * fs / 2.0)
        current = current + rng.normal(0.0, sigma, current.shape)
    return current


def balanced_homodyne(e_signal, e_lo, responsivity: float = 0.8, fs: float | None = None,
                      shot_noise: bool = False, rng: np.random.Generator | None = None):
    """Difference current of a 50/50 balanced homodyne receiver.

    E_pm = (E_s +/- E_LO)/sqrt(2)  ->  i = I+ - I- = 2 R Re(E_s E_LO*).
    """
    e_plus = (e_signal + e_lo) / np.sqrt(2.0)
    e_minus = (e_signal - e_lo) / np.sqrt(2.0)
    rng = rng or np.random.default_rng()
    i_plus = photodetect(e_plus, responsivity, fs, shot_noise, rng)
    i_minus = photodetect(e_minus, responsivity, fs, shot_noise, rng)
    return i_plus - i_minus


# ---------------------------------------------------------------------------
# Electronic processing
# ---------------------------------------------------------------------------


def mix(x, t, lo_freq: float, lo_phase: float = 0.0):
    """Multiply by a local oscillator 2*cos(2 pi f t + phase).

    The factor 2 makes the low-passed output equal to the amplitude of the
    quadrature of x that is in phase with the LO.
    """
    return 2.0 * np.asarray(x) * np.cos(2.0 * np.pi * lo_freq * np.asarray(t) + lo_phase)


def lowpass_filter(x, fs: float, cutoff: float, order: int = 4, zero_phase: bool = True):
    """Butterworth low-pass filter (second-order sections)."""
    sos = signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x) if zero_phase else signal.sosfilt(sos, x)


def demodulate(x, t, fs: float, lo_freq: float, lo_phase: float, cutoff: float, order: int = 4):
    """Mix with the LO and low-pass filter: returns the baseband quadrature."""
    return lowpass_filter(mix(x, t, lo_freq, lo_phase), fs, cutoff, order)


def psd(x, fs: float, nperseg: int | None = None):
    """One-sided Welch PSD estimate. Returns (f, Pxx) with units^2/Hz."""
    x = np.asarray(x)
    nperseg = nperseg or min(len(x), 2**14)
    return signal.welch(x, fs=fs, nperseg=nperseg, window="hann", detrend="constant")


def delay_samples(x, n: int):
    """Delay a sequence by n samples (first n samples repeat x[0])."""
    if n <= 0:
        return np.asarray(x).copy()
    x = np.asarray(x)
    return np.concatenate([np.full(n, x[0]), x[:-n]])
