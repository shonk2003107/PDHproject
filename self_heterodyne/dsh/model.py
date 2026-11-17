"""Delayed self-heterodyne (DSH) beat spectrum and a model of the ESA that measures it.

Setup
-----
The laser light is split into two arms. One arm goes through a delay fiber of length L,
the other through an AOM that shifts the frequency by f_AOM (80 MHz). The arms are recombined
on a photodiode and the beat around f_AOM is measured on an electrical spectrum analyzer (ESA).

Basic quantities
----------------
    delay time        tau_d = n_g L / c          (n_g = 1.4682 for SMF-28 at 1550 nm, ~4.9 us/km)
    coherence time    tau_c = 1 / (pi dnu)       (dnu = laser linewidth, FWHM)
    fringe spacing    1 / tau_d
    coherent fraction exp(-2 pi dnu tau_d)       (power left in the delta peak at the center)

Beat spectrum for a laser with white frequency noise (Richter et al., IEEE JQE 22, 2070 (1986);
Mercer, J. Lightwave Technol. 9, 485 (1991)), with delta = f - f_AOM:

    S(delta) = A dnu / (dnu^2 + delta^2) * [1 - e^(-2 pi dnu tau_d)
               (cos(2 pi delta tau_d) + (dnu / delta) sin(2 pi delta tau_d))]
               + delta peak of weight e^(-2 pi dnu tau_d) at delta = 0

With A = P_beat / pi the continuum integrates to P_beat (1 - e^(-2 pi dnu tau_d)) and the delta
peak carries the rest. For tau_d -> infinity the spectrum becomes a Lorentzian of FWHM 2 dnu.

ESA model
---------
Each displayed point is the power that passes the resolution-bandwidth (RBW) filter centered on
that frequency: the true spectrum convolved with the filter (Gaussian, -3 dB width = RBW, peak 1,
so a CW tone reads its own power). On top of that: the instrument noise floor (proportional to
the RBW) and the residual scatter of an RMS (power) average over N sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

C_LIGHT = 299_792_458.0            # [m/s]
N_GROUP_SMF28 = 1.4682             # group index of SMF-28 at 1550 nm
GAUSS_ENBW = np.sqrt(np.pi / (4.0 * np.log(2.0)))   # noise bandwidth / RBW of a Gaussian filter (1.0645)


# ---------------------------------------------------------------------------
# Basic quantities
# ---------------------------------------------------------------------------


def delay_time(length_m: float, group_index: float = N_GROUP_SMF28) -> float:
    """tau_d = n_g L / c [s]."""
    return length_m * group_index / C_LIGHT


def fiber_length(tau: float, group_index: float = N_GROUP_SMF28) -> float:
    """Inverse of delay_time: L = c tau_d / n_g [m]."""
    return tau * C_LIGHT / group_index


def coherence_time(linewidth: float) -> float:
    """tau_c = 1 / (pi dnu) [s]."""
    return 1.0 / (np.pi * linewidth)


def coherent_fraction(linewidth: float, tau: float) -> float:
    """Fraction of the beat power left in the central delta peak: exp(-2 pi dnu tau_d)."""
    return float(np.exp(-2.0 * np.pi * linewidth * tau))


def max_rbw_for_fringes(tau: float) -> float:
    """Rule of thumb: RBW < 0.1 / tau_d (a tenth of the fringe spacing)."""
    return 0.1 / tau


# ---------------------------------------------------------------------------
# Beat spectrum
# ---------------------------------------------------------------------------


def dsh_continuum(delta, linewidth: float, tau: float, amplitude: float):
    """Continuum part of the DSH beat spectrum (the delta peak at delta = 0 is not included).

    delta = f - f_AOM [Hz], linewidth = laser FWHM dnu [Hz], tau = fiber delay [s].
    amplitude A: A = P / pi gives a continuum that integrates to P (1 - exp(-2 pi dnu tau)).
    """
    d = np.asarray(delta, dtype=float)
    e = np.exp(-2.0 * np.pi * linewidth * tau)
    x = 2.0 * np.pi * d * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        # (dnu / delta) sin(2 pi delta tau) -> 2 pi dnu tau at delta = 0
        sin_term = np.where(d == 0.0, 2.0 * np.pi * linewidth * tau, linewidth * np.sin(x) / d)
    return amplitude * linewidth / (linewidth**2 + d**2) * (1.0 - e * (np.cos(x) + sin_term))


# ---------------------------------------------------------------------------
# ESA measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Acquisition:
    """One ESA setting: span, RBW, number of trace points and number of averaged sweeps."""
    name: str
    span_hz: float
    rbw_hz: float
    points: int = 1001
    averages: int = 20


def rbw_response(offset, rbw: float):
    """Gaussian RBW filter, peak 1 and -3 dB full width = rbw."""
    return np.exp(-4.0 * np.log(2.0) * (np.asarray(offset, dtype=float) / rbw) ** 2)


def esa_trace(acq: Acquisition, center_hz: float, linewidth: float, tau: float, *,
              beat_power_dbm: float, noise_floor_dbm_hz: float, sweep_noise: float,
              rng: np.random.Generator, beat_offset_hz: float = 0.0):
    """Simulate one averaged ESA trace of the DSH beat.

    Returns (f [Hz], power_dBm with averaging noise, expected power_dBm without noise).

    beat_power_dbm      total beat power (continuum + coherent peak)
    noise_floor_dbm_hz  instrument noise density (displayed floor = this + 10 log10(ENBW))
    sweep_noise         relative standard deviation of one sweep (per point); averaging N
                        sweeps reduces it to sweep_noise / sqrt(N)
    beat_offset_hz      offset of the true beat center from center_hz
    """
    f = np.linspace(center_hz - acq.span_hz / 2, center_hz + acq.span_hz / 2, acq.points)
    x = f - center_hz - beat_offset_hz

    # fine grid: resolve both the RBW filter and the fringes (spacing 1 / tau)
    df = acq.rbw_hz / 10.0
    if tau > 0:
        df = min(df, 0.05 / tau)
    half = acq.span_hz / 2 + abs(beat_offset_hz) + 5.0 * acq.rbw_hz
    xf = np.arange(-int(np.ceil(half / df)), int(np.ceil(half / df)) + 1) * df
    kernel_x = np.arange(-int(np.ceil(4 * acq.rbw_hz / df)), int(np.ceil(4 * acq.rbw_hz / df)) + 1) * df

    p_mw = 10.0 ** (beat_power_dbm / 10.0)
    density = p_mw * dsh_continuum(xf, linewidth, tau, 1.0 / np.pi)             # [mW/Hz]
    continuum = fftconvolve(density, rbw_response(kernel_x, acq.rbw_hz), mode="same") * df
    coherent = p_mw * coherent_fraction(linewidth, tau) * rbw_response(x, acq.rbw_hz)
    floor = 10.0 ** (noise_floor_dbm_hz / 10.0) * GAUSS_ENBW * acq.rbw_hz

    expected = np.clip(np.interp(x, xf, continuum), 0.0, None) + coherent + floor
    # RMS average of N sweeps: Gamma-distributed power with relative std sweep_noise / sqrt(N)
    shape = acq.averages / sweep_noise**2
    measured = expected * rng.gamma(shape, 1.0 / shape, size=f.size)
    return f, 10.0 * np.log10(measured), 10.0 * np.log10(expected)
