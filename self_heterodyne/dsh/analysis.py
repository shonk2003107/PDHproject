"""Laser linewidth extraction from a DSH beat measured on an ESA.

All estimates are the LASER linewidth dnu (FWHM); the beat is twice as wide.

    dnu_3dB      = W_3dB / 2                  valid only for tau_d >> tau_c (incoherent regime)
    dnu_20dB     = W_20dB / (2 sqrt(99))      valid only for tau_d >> tau_c; less sensitive to the
                                              1/f noise that broadens the center of the line
    dnu_lorentz  = Lorentzian-fit FWHM / 2    valid only for tau_d >> tau_c
    dnu_coherent = fit to the full DSH model  valid in every regime (see model.dsh_continuum)

dnu_best:
    coherent fit if it converged with a relative uncertainty <= 30 %,
    no result if the uncertainty is larger (delay too short to constrain the linewidth),
    otherwise the Lorentzian fit, and if that failed too, the direct -3 dB width.
beat_3dB_FWHM_equiv_Hz = 2 dnu_best: the -3 dB width the beat WOULD have with an infinitely long
delay - the classical "3 dB point", extracted from the fit.

Regime: incoherent if exp(-2 pi dnu tau_d) < 0.01, otherwise partially coherent. In the partially
coherent regime the -3 dB point falls on the central coherent peak and measures the RBW, not the
laser, and the envelope around it is not a Lorentzian.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit

from . import model

REL_ERR_MAX = 0.30          # coherent fit: above this the linewidth is "not constrained"
REL_ERR_WARN = 0.10         # coherent fit: flag dnu_uncertain_>10% above this
INCOHERENT_FRACTION = 0.01  # regime boundary on exp(-2 pi dnu tau)
SNR_MIN_20DB = 25.0         # [dB] below this the -20 dB width is unreliable
COHERENT_STARTS = (1.0, 0.3, 3.0, 0.1, 10.0)   # multiples of the linewidth guess


# ---------------------------------------------------------------------------
# Widths
# ---------------------------------------------------------------------------


def _crossing(f, y, i_pk: int, level: float, step: int) -> float:
    """Frequency where y first drops below level, walking from the peak in direction step."""
    i = i_pk
    while 0 < i < len(y) - 1 and y[i] > level:
        i += step
    if y[i] > level:
        return np.nan
    j = i - step
    return float(np.interp(level, [y[i], y[j]], [f[i], f[j]]))


def width_at(f, y, i_pk: int, drop_db: float) -> float:
    """Full width of the line drop_db below the peak (linear interpolation), NaN if out of span."""
    level = y[i_pk] - drop_db
    return _crossing(f, y, i_pk, level, +1) - _crossing(f, y, i_pk, level, -1)


# ---------------------------------------------------------------------------
# Fits (both in the dB domain, so the tails weigh as much as the peak)
# ---------------------------------------------------------------------------


def fit_lorentzian_db(x, y, rbw: float, fwhm_guess: float, peak_db: float, floor_db: float):
    """Lorentzian + constant background, fitted to the trace in dB.

    Parameters: center, log10 FWHM, log10 amplitude, log10 background.
    Returns (beat FWHM [Hz], callable offset -> dB).
    """
    def f_model(xx, x0, lg, la, lb):
        g = 10.0**lg
        return 10.0 * np.log10(10.0**la * (g / 2) ** 2 / ((xx - x0) ** 2 + (g / 2) ** 2) + 10.0**lb)

    span = x.max() - x.min()
    lo = [-5 * max(rbw, fwhm_guess), np.log10(rbw / 5), peak_db / 10 - 3, floor_db / 10 - 3]
    hi = [5 * max(rbw, fwhm_guess), np.log10(span), peak_db / 10 + 3, floor_db / 10 + 3]
    p0 = np.clip([0.0, np.log10(fwhm_guess), peak_db / 10, floor_db / 10], lo, hi)
    popt, _ = curve_fit(f_model, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
    return 10.0 ** popt[1], (lambda xx: f_model(xx, *popt))


def fit_coherent_db(x, y, rbw: float, tau0: float, dnu_guess: float, env_peak_db: float,
                    floor_db: float, mask_rbw: float = 4.0) -> dict:
    """Fit the full DSH continuum (partial coherence + fringes) + background, in dB.

    - +/- mask_rbw RBWs around the center are ignored, so the coherent delta peak does not pull the fit.
    - tau is a free parameter within +/- 10 % of the nominal tau0.
    - Several starting linewidths (the fringes create local minima); the lowest residual wins.

    Returns dict(dnu, tau, tau_at_bound, rel_err, fn).
    """
    def f_model(xx, x0, ldnu, tau_us, la, lb):
        s = model.dsh_continuum(xx - x0, 10.0**ldnu, tau_us * 1e-6, 10.0**la)
        return 10.0 * np.log10(np.clip(s, 0.0, None) + 10.0**lb)

    use = np.abs(x) > mask_rbw * rbw
    x0_lim = max(3 * rbw, 0.2 * dnu_guess)
    tau_lo, tau_hi = tau0 * 1e6 * 0.9, tau0 * 1e6 * 1.1
    best = None
    for k in COHERENT_STARTS:
        dn = float(np.clip(dnu_guess * k, 10.0, 5e6))
        la0 = env_peak_db / 10 + np.log10(dn)          # Lorentzian peak A / dnu = envelope peak
        lo = [-x0_lim, 1.0, tau_lo, la0 - 4, floor_db / 10 - 3]
        hi = [x0_lim, 7.0, tau_hi, la0 + 4, floor_db / 10 + 3]
        p0 = np.clip([0.0, np.log10(dn), tau0 * 1e6, la0, floor_db / 10], lo, hi)
        try:
            popt, pcov = curve_fit(f_model, x[use], y[use], p0=p0, bounds=(lo, hi), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((f_model(x[use], *popt) - y[use]) ** 2))
        if best is None or sse < best[0]:
            best = (sse, popt, pcov)
    if best is None:
        raise RuntimeError("coherent fit did not converge")

    _, popt, pcov = best
    sig = float(np.sqrt(pcov[1, 1])) if np.all(np.isfinite(pcov)) else math.inf
    return {
        "dnu": 10.0 ** popt[1],
        "tau": popt[2] * 1e-6,
        "tau_at_bound": bool(np.isclose(popt[2], tau_lo) or np.isclose(popt[2], tau_hi)),
        "rel_err": (10.0**sig - 1.0) if sig < 5 else math.inf,   # from the log10 parameter
        "fn": (lambda xx: f_model(xx, *popt)),
    }


# ---------------------------------------------------------------------------
# Full analysis of one trace
# ---------------------------------------------------------------------------


def analyze_trace(f, p_db, rbw: float, tau: float, *, center_hz: float, peak_window_hz: float,
                  mask_rbw: float = 4.0, group_index: float = model.N_GROUP_SMF28):
    """Analyze one ESA trace (f [Hz], p_db [dBm]) of a DSH beat with nominal fiber delay tau [s].

    Returns (results dict, fits dict {label: callable(offset from peak [Hz]) -> dB}).
    """
    f = np.asarray(f, dtype=float)
    p_db = np.asarray(p_db, dtype=float)
    res: dict = {"rbw_Hz": rbw, "tau_us": tau * 1e6}
    flags: list[str] = []
    fits: dict = {}

    # --- peak and floor ---------------------------------------------------------------
    sel = np.abs(f - center_hz) < peak_window_hz
    idx = np.flatnonzero(sel) if sel.any() else np.arange(len(f))
    i_pk = int(idx[np.argmax(p_db[idx])])
    f_pk, pk = float(f[i_pk]), float(p_db[i_pk])
    floor = float(np.percentile(p_db, 5))
    res.update(f_peak_Hz=f_pk, peak_dBm=pk, floor_dBm=floor, snr_dB=pk - floor)

    # --- direct -3 dB and -20 dB widths -----------------------------------------------
    w3, w20 = width_at(f, p_db, i_pk, 3.0), width_at(f, p_db, i_pk, 20.0)
    res.update(w3dB_Hz=w3, dnu_3dB_Hz=w3 / 2, w20dB_Hz=w20, dnu_20dB_Hz=w20 / (2 * math.sqrt(99)))
    if np.isfinite(w3) and w3 < 5 * rbw:
        flags.append("3dB_width_RBW_limited")
    if pk - floor < SNR_MIN_20DB:
        flags.append("SNR<25dB_20dB_width_unreliable")

    span = f.max() - f.min()
    if tau <= 0:
        res.update(regime="reference_no_delay", best_method="none", flags=";".join(flags))
        return res, fits
    if span < 0.5 / tau:
        # not even half a fringe fits in the span: the trace looks like the zero-delay reference
        flags.append("delay_too_short_for_this_span")
        res.update(regime="too_short_acts_as_reference", best_method="none", flags=";".join(flags))
        return res, fits
    if rbw > model.max_rbw_for_fringes(tau):
        flags.append("RBW_too_coarse_for_fringes")
    if span < 2 / tau:
        flags.append("span_<2_fringes_prefer_wider_span")

    x = f - f_pk

    # --- Lorentzian fit (valid when tau_d >> tau_c) -----------------------------------
    g0 = w3 if (np.isfinite(w3) and w3 > rbw) else 10 * rbw
    try:
        g, fn = fit_lorentzian_db(x, p_db, rbw, g0, pk, floor)
        res["dnu_lorentz_Hz"] = g / 2
        fits["Lorentzian fit"] = fn
    except (RuntimeError, ValueError):
        flags.append("lorentz_fit_failed")

    # --- full DSH fit with partial coherence ------------------------------------------
    guesses = [res.get("dnu_lorentz_Hz"), res["dnu_20dB_Hz"], res["dnu_3dB_Hz"], 5e3]
    dnu0 = next(v for v in guesses if v is not None and np.isfinite(v) and v > 0)
    env_pk = float(np.max(p_db[np.abs(x) > mask_rbw * rbw]))
    try:
        c = fit_coherent_db(x, p_db, rbw, tau, dnu0, env_pk, floor, mask_rbw)
        res.update(dnu_coherent_Hz=c["dnu"], dnu_coherent_rel_err=c["rel_err"],
                   tau_fit_us=c["tau"] * 1e6, fiber_len_fit_m=model.fiber_length(c["tau"], group_index))
        fits["DSH coherent fit"] = c["fn"]
        if c["tau_at_bound"]:
            flags.append("tau_fit_at_bound_check_fiber_length")
        res["tau_over_tauc"] = math.pi * c["dnu"] * c["tau"]
        incoherent = model.coherent_fraction(c["dnu"], c["tau"]) < INCOHERENT_FRACTION
        res["regime"] = "incoherent" if incoherent else "partially_coherent"
        if not incoherent:
            flags.append("fringes_present_use_dnu_coherent")
    except RuntimeError:
        flags.append("coherent_fit_failed")

    # --- best estimate ------------------------------------------------------------------
    rel_err = res.get("dnu_coherent_rel_err", math.inf)
    if "dnu_coherent_Hz" in res and rel_err <= REL_ERR_MAX:
        res.update(dnu_best_Hz=res["dnu_coherent_Hz"], best_method="coherent_fit")
        if rel_err > REL_ERR_WARN:
            flags.append("dnu_uncertain_>10%")
    elif "dnu_coherent_Hz" in res:
        # converged, but the data do not constrain dnu (typically the delay is too short)
        flags.append("linewidth_not_constrained_delay_too_short")
        res["best_method"] = "none"
    elif "dnu_lorentz_Hz" in res:
        res.update(dnu_best_Hz=res["dnu_lorentz_Hz"], best_method="lorentz_fit")
    else:
        res.update(dnu_best_Hz=res["dnu_3dB_Hz"], best_method="minus3dB")
    if "dnu_best_Hz" in res:
        res["beat_3dB_FWHM_equiv_Hz"] = 2 * res["dnu_best_Hz"]
    res["flags"] = ";".join(flags)
    return res, fits
