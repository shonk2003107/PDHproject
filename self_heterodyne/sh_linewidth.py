#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sh_linewidth.py - Delayed self-heterodyne (DSH) laser linewidth measurement

Instruments
    ESA   : Keysight N9010B EXA   - main spectral measurement of the 80 MHz beat
    Scope : Keysight DSOX1204G    - time-domain beat monitor (power / polarization drift check)
    (opt) : Agilent N5173B        - state is only logged, never controlled (off by default)

Typical session (one run per fiber length):
    python sh_linewidth.py --check                      # verify connections
    python sh_linewidth.py --monitor                    # live beat power -> optimize polarization
    python sh_linewidth.py --delay 0    --note "no fiber, instrument reference"
    python sh_linewidth.py --delay 500
    python sh_linewidth.py --delay 2000
    python sh_linewidth.py --delay 5000
    python sh_linewidth.py --analyze data_measured/delay_02000m/2026-11-12_14-30-05   # re-analyze only
    python sh_linewidth.py --delay 2000 --simulate      # offline test, no instruments needed

Output (every CSV has exactly one header row -> pandas.read_csv / MATLAB readtable):
    data_measured/delay_XXXXXm/<timestamp>/
        esa_<span>_rNN.csv      freq_Hz, power_dBm
        esa_<span>_rNN.png      data + fits
        scope_before.csv        time_s, voltage_V
        scope_after.csv         time_s, voltage_V
        results.csv             one row per ESA trace, all linewidth estimates
        metadata.json           instrument IDs, actual instrument settings, notes
        run.log
    data_measured/summary.csv          one row per ESA trace, appended across ALL runs

Linewidth estimates (all are LASER linewidth, FWHM):
    dnu_3dB      = (-3 dB width of beat) / 2              valid only if tau_d >> tau_c
    dnu_20dB     = (-20 dB width of beat) / (2*sqrt(99))  valid only if tau_d >> tau_c
    dnu_lorentz  = (Lorentzian-fit FWHM) / 2              valid only if tau_d >> tau_c
    dnu_coherent = fit to full DSH spectrum incl. partial coherence (fringes) - valid in all regimes
    dnu_best     = coherent fit when it converged with <30 % uncertainty, else Lorentzian, else -3 dB
                   (left empty when the delay is too short for the data to constrain the linewidth)
    beat_3dB_FWHM_equiv_Hz = 2 * dnu_best : the -3 dB FWHM the beat WOULD have with an
                   infinitely long delay (fully incoherent Lorentzian) - the classical
                   "3 dB width" definition, extracted from the fit
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ================================ CONFIGURATION ================================
CONFIG = {
    # Addresses confirmed by *IDN?: MY65280160 answers as N5173B, so the ESA is the other one.
    "esa_address":    "USB0::0x2A8D::0x1B0B::MY66040483::0::INSTR",   # N9010B
    "scope_address":  "USB0::0x2A8D::0x0396::MY65350063::0::INSTR",   # DSOX1204G
    "siggen_address": "USB0::0x0957::0x1F01::MY65280160::0::INSTR",   # N5173B
    "log_siggen": False,            # N5173B is not in the chain (AOM is driven by the 33600A)

    "laser": "Pure Photonics PPCL550",
    "aom_freq_hz": 80e6,            # AOM shift = beat-note center frequency
    "fiber_group_index": 1.4682,    # SMF-28 @ 1550 nm
    "data_root": str(Path(__file__).resolve().parent / "data_measured"),   # self_heterodyne/data_measured (with the copied runs and simulations)

    # ESA acquisitions; each one is taken once per repeat.
    # Rules: RBW << beat width  AND  RBW << fringe spacing 1/tau_d (41 kHz for 5 km).
    "esa_spans": [
        {"name": "wide",   "span_hz": 5e6,   "rbw_hz": 5e3, "points": 1001, "averages": 20},
        {"name": "narrow", "span_hz": 200e3, "rbw_hz": 200, "points": 1001, "averages": 20},
        {"name": "ultra_wide", "span_hz": 20e6, "rbw_hz": 20e3, "points": 1001, "averages": 20}
    ],
    "esa_sweep_timeout_s": 900,
    "peak_search_window_hz": 1e6,   # search the beat within +/- this around aom_freq_hz
    "center_mask_rbw": 4,           # coherent fit ignores +/- this many RBWs around the delta peak

    "scope_channel": 1,
    "scope_points": 100000,
    "beat_drift_warn": 0.15,        # warn if beat RMS on the scope changes > 15 % during a run

    "monitor_span_hz": 2e6,
    "monitor_rbw_hz": 3e3,
}

C_LIGHT = 299_792_458.0
log = logging.getLogger("dsh")

RESULT_FIELDS = [
    "timestamp", "run_dir", "delay_m", "tau_us", "span_name", "repeat", "file", "rbw_Hz",
    "f_peak_Hz", "peak_dBm", "floor_dBm", "snr_dB",
    "w3dB_Hz", "dnu_3dB_Hz", "w20dB_Hz", "dnu_20dB_Hz",
    "dnu_lorentz_Hz", "dnu_coherent_Hz", "dnu_coherent_rel_err", "tau_fit_us", "fiber_len_fit_m",
    "tau_over_tauc", "regime", "dnu_best_Hz", "beat_3dB_FWHM_equiv_Hz", "best_method", "flags",
    "beat_rms_drift", "note",
]


# ================================= INSTRUMENTS =================================
def open_rm():
    try:
        import pyvisa
    except ImportError:
        sys.exit("pyvisa is not installed:  pip install pyvisa")
    return pyvisa.ResourceManager()


class ScpiInstrument:
    MODEL = ""          # expected text in *IDN?; checked on connect so commands never go to the wrong box

    def __init__(self, rm, address: str, name: str, timeout_ms: int = 20000):
        self.name = name
        self.address = address
        self.inst = rm.open_resource(address)
        self.inst.timeout = timeout_ms
        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"
        try:
            self.inst.clear()
        except Exception:
            pass
        if self.MODEL:
            idn = self.idn()
            norm = lambda t: t.upper().replace(" ", "").replace("-", "")   # "DSO-X 1204G" == "DSOX1204G"
            if norm(self.MODEL) not in norm(idn):
                self.close()
                raise RuntimeError(f"{name}: address {address} answers as '{idn}', expected {self.MODEL}. "
                                   "Fix the address in CONFIG (run --check to see who is where).")

    def write(self, cmd: str) -> None:
        self.inst.write(cmd)

    def query(self, cmd: str) -> str:
        return self.inst.query(cmd).strip()

    def qfloat(self, cmd: str) -> float:
        return float(self.query(cmd))

    def idn(self) -> str:
        return self.query("*IDN?")

    def wait(self, timeout_s: float | None = None) -> None:
        """Block until all pending operations are complete (*OPC?)."""
        old = self.inst.timeout
        if timeout_s:
            self.inst.timeout = int(timeout_s * 1000)
        try:
            self.query("*OPC?")
        finally:
            self.inst.timeout = old

    def errors(self, quiet: bool = False) -> list[str]:
        errs = []
        for _ in range(30):
            e = self.query(":SYST:ERR?")
            try:
                if int(e.split(",")[0]) == 0:
                    break
            except ValueError:
                break
            errs.append(e)
        if not quiet:
            for e in errs:
                log.warning("%s reported: %s", self.name, e)
        return errs

    def write_checked(self, cmd: str, timeout_s: float = 30) -> list[str]:
        """Send one command, wait for it, and return the errors it caused (if any)."""
        self.write(cmd)
        self.wait(timeout_s)
        errs = self.errors(quiet=True)
        for e in errs:
            log.warning("%s rejected '%s': %s", self.name, cmd, e)
        return errs

    def safe_query(self, cmd: str, timeout_s: float = 3.0):
        """Query that returns None instead of hanging on an unsupported command."""
        old = self.inst.timeout
        self.inst.timeout = int(timeout_s * 1000)
        try:
            return self.query(cmd)
        except Exception:
            try:
                self.inst.clear()
            except Exception:
                pass
            self.errors(quiet=True)
            return None
        finally:
            self.inst.timeout = old

    def safe_float(self, cmd: str) -> float:
        r = self.safe_query(cmd)
        try:
            return float(r)
        except (TypeError, ValueError):
            return float("nan")

    def close(self) -> None:
        try:
            self.inst.close()
        except Exception:
            pass


class N9010B(ScpiInstrument):
    """Keysight EXA signal analyzer, Swept SA mode."""
    MODEL = "N9010B"

    # (command, critical) - a critical command that fails stops the run with a clear message
    SETUP_CMDS = [
        (":INST:SEL SA", True),
        (":CONF:SAN", True),
        (":INIT:CONT OFF", True),
        (":UNIT:POW DBM", False),
        (":POW:ATT:AUTO ON", False),
        (":SWE:TIME:AUTO ON", False),
        (":SWE:TYPE AUTO", False),
        (":BAND:VID:AUTO ON", False),
        (":DET:TRAC1 AVER", True),          # average (RMS) detector
        (":AVER:TYPE RMS", True),           # power averaging -> correct lineshape for fitting
        (":FORM:TRAC:DATA REAL,32", True),
        (":FORM:BORD SWAP", True),
    ]

    def setup_common(self) -> None:
        self.write("*CLS")
        mode = self.safe_query(":INST:SEL?")
        log.info("ESA mode before setup: %s", mode)
        failed = []
        for cmd, critical in self.SETUP_CMDS:
            errs = self.write_checked(cmd, timeout_s=60)
            if errs and critical:
                failed.append(f"{cmd}  ->  {errs[0]}")
            if cmd == ":INST:SEL SA":
                time.sleep(2)               # mode switch: let the SA application come up
        if failed:
            raise RuntimeError("ESA rejected essential setup commands:\n    " + "\n    ".join(failed)
                               + "\nRun:  python sh_linewidth.py --diag   and send the output.")

    def configure(self, center, span, rbw, points, averages) -> dict:
        cmds = [f":FREQ:CENT {center:.6f}", f":FREQ:SPAN {span:.6f}",
                f":BAND:RES {rbw:.6f}", f":SWE:POIN {int(points)}"]
        if averages > 1:
            cmds += [":TRAC1:TYPE AVER", f":AVER:COUN {int(averages)}"]
        else:
            cmds += [":TRAC1:TYPE WRIT"]
        failed = [f"{c}  ->  {e[0]}" for c in cmds if (e := self.write_checked(c))]
        if failed:
            raise RuntimeError("ESA rejected configuration commands:\n    " + "\n    ".join(failed)
                               + "\nRun:  python sh_linewidth.py --diag   and send the output.")
        # read back what the instrument actually uses (it may round RBW etc.)
        return {
            "center_hz": self.safe_float(":FREQ:CENT?"),
            "span_hz": self.safe_float(":FREQ:SPAN?"),
            "rbw_hz": rbw_actual if np.isfinite(rbw_actual := self.safe_float(":BAND:RES?")) else rbw,
            "vbw_hz": self.safe_float(":BAND:VID?"),
            "points": int(points),
            "averages": int(averages),
            "sweep_time_s": self.safe_float(":SWE:TIME?"),
            "atten_db": self.safe_float(":POW:ATT?"),
            "ref_level_dbm": self.safe_float(":DISP:WIND:TRAC:Y:RLEV?"),
        }

    def acquire(self, timeout_s: float):
        """Single measurement: N averaged sweeps, then read trace 1."""
        self.write(":INIT:IMM")
        self.wait(timeout_s)
        data = self.inst.query_binary_values(":TRAC:DATA? TRACE1", datatype="f",
                                             is_big_endian=False, container=np.array)
        f = np.linspace(self.qfloat(":FREQ:STAR?"), self.qfloat(":FREQ:STOP?"), len(data))
        self.errors()
        return f, data.astype(float)

    def quick_peak(self):
        self.write(":INIT:IMM")
        self.wait(60)
        self.write(":CALC:MARK1:MODE POS")
        self.write(":CALC:MARK1:MAX")
        return self.qfloat(":CALC:MARK1:X?"), self.qfloat(":CALC:MARK1:Y?")

    def restore_live(self) -> None:
        self.write(":INIT:CONT ON")


class DSOX1204G(ScpiInstrument):
    """Keysight InfiniiVision 1000X oscilloscope."""
    MODEL = "DSOX1204G"

    def capture(self, channel: int, points: int):
        ch = f"CHAN{int(channel)}"
        self.write("*CLS")
        self.write(f":WAV:SOUR {ch}")
        self.write(":WAV:FORM BYTE")
        self.write(":WAV:UNS ON")
        self.write(":WAV:POIN:MODE MAX")    # take whatever record length the scope has
        self.write(f":DIG {ch}")            # one fresh acquisition, then stop
        self.wait(30)
        pre = [float(x) for x in self.query(":WAV:PRE?").split(",")]
        _fmt, _typ, _n, _cnt, xinc, xorg, xref, yinc, yorg, yref = pre[:10]
        raw = self.inst.query_binary_values(":WAV:DATA?", datatype="B", container=np.array)
        raw = raw[:int(points)]             # keep files small; enough for the RMS drift check
        t = (np.arange(len(raw)) - xref) * xinc + xorg
        v = (raw.astype(float) - yref) * yinc + yorg
        info = {
            "points": int(len(raw)),
            "sample_rate_Sa_s": (1.0 / xinc) if xinc else None,
            "vertical_scale_V_div": self.safe_float(f":{ch}:SCAL?"),
            "timebase_s_div": self.safe_float(":TIM:SCAL?"),
            "coupling": self.safe_query(f":{ch}:COUP?"),
            "bw_limit": self.safe_query(f":{ch}:BWL?"),
            "clipped": bool(raw.min() <= 1 or raw.max() >= 254),
        }
        self.write(":RUN")
        self.errors()
        return t, v, info


class N5173B(ScpiInstrument):
    MODEL = "N5173B"


def siggen_state(sg: ScpiInstrument) -> dict:
    return {"idn": sg.idn(), "freq_hz": sg.qfloat(":FREQ?"),
            "power_dbm": sg.qfloat(":POW?"), "rf_on": sg.query(":OUTP?")}


# ============================== SIMULATED DEVICES ==============================
class FakeESA:
    name = "ESA(sim)"

    def __init__(self, delay_m: float, cfg: dict, dnu: float = 4e3):
        self.tau = tau_from_length(delay_m, cfg)
        self.dnu = dnu
        self.s: dict = {}
        self.rng = np.random.default_rng()

    def idn(self): return "SIMULATED,N9010B,0,0"
    def setup_common(self): pass
    def errors(self): return []
    def restore_live(self): pass
    def close(self): pass

    def configure(self, center, span, rbw, points, averages):
        self.s = dict(center_hz=center, span_hz=span, rbw_hz=rbw, vbw_hz=rbw, points=int(points),
                      averages=int(averages), sweep_time_s=0.0, atten_db=10.0, ref_level_dbm=0.0)
        return dict(self.s)

    def acquire(self, timeout_s=None):
        s = self.s
        f = np.linspace(s["center_hz"] - s["span_hz"] / 2, s["center_hz"] + s["span_hz"] / 2, s["points"])
        d = f - s["center_hz"] - 37.0
        a = 1e-5 / np.pi                                     # total beat power -20 dBm
        e = np.exp(-2 * np.pi * self.dnu * self.tau)
        sig = s["rbw_hz"] / 2.355
        lin = sdsh_linear(d, self.dnu, self.tau, a) * s["rbw_hz"] + np.pi * a * e * np.exp(-0.5 * (d / sig) ** 2)
        floor = 1e-12 * s["rbw_hz"] / 1e3
        noise = 0.3 / math.sqrt(max(s["averages"], 1))
        lin = np.clip(lin, 0, None) * (1 + noise * self.rng.standard_normal(f.size)) \
            + floor * (1 + noise * self.rng.standard_normal(f.size))
        return f, 10 * np.log10(np.clip(lin, 1e-20, None))

    def quick_peak(self):
        time.sleep(0.3)
        return self.s["center_hz"], -25 + self.rng.normal(0, 0.5)


class FakeScope:
    name = "Scope(sim)"

    def idn(self): return "SIMULATED,DSOX1204G,0,0"
    def errors(self): return []
    def close(self): pass

    def capture(self, channel, points):
        t = np.arange(points) * 0.5e-9
        v = 0.1 * np.sin(2 * np.pi * 80e6 * t) + 0.003 * np.random.standard_normal(points)
        return t, v, {"points": points, "sample_rate_Sa_s": 2e9, "clipped": False}


# =================================== ANALYSIS ===================================
def tau_from_length(length_m: float, cfg: dict) -> float:
    return length_m * cfg["fiber_group_index"] / C_LIGHT


def sdsh_linear(d, dnu, tau, a):
    """Delayed self-heterodyne spectrum (continuum part), Lorentzian laser, delay tau.
    d = f - f_AOM, dnu = laser FWHM. tau -> inf gives a Lorentzian with FWHM = 2*dnu.
    The delta-function term at d=0 (coherent part) is not included."""
    d = np.where(np.abs(d) < 1e-3, 1e-3, d)
    e = np.exp(-2 * np.pi * dnu * tau)
    return a * dnu / (dnu ** 2 + d ** 2) * (
        1 - e * (np.cos(2 * np.pi * d * tau) + dnu / d * np.sin(2 * np.pi * d * tau)))


def _crossing(f, y, i_pk, level, step):
    i = i_pk
    while 0 < i < len(y) - 1 and y[i] > level:
        i += step
    if y[i] > level:
        return np.nan
    j = i - step
    return float(np.interp(level, [y[i], y[j]], [f[i], f[j]]))


def width_at(f, y, i_pk, drop_db):
    level = y[i_pk] - drop_db
    return _crossing(f, y, i_pk, level, +1) - _crossing(f, y, i_pk, level, -1)


def _fit_lorentz(x, y, rbw, g0, pk_db, floor_db):
    from scipy.optimize import curve_fit

    def model(x, x0, lg, la, lb):
        g = 10 ** lg
        return 10 * np.log10(10 ** la * (g / 2) ** 2 / ((x - x0) ** 2 + (g / 2) ** 2) + 10 ** lb)

    span = x.max() - x.min()
    lo = [-5 * max(rbw, g0), np.log10(rbw / 5), pk_db / 10 - 3, floor_db / 10 - 3]
    hi = [5 * max(rbw, g0), np.log10(span), pk_db / 10 + 3, floor_db / 10 + 3]
    p0 = np.clip([0.0, np.log10(g0), pk_db / 10, floor_db / 10], lo, hi)
    popt, _ = curve_fit(model, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
    return 10 ** popt[1], (lambda xx: model(xx, *popt))


def _fit_coherent(x, y, rbw, tau0, dnu0, env_pk_db, floor_db, mask_rbw=4):
    from scipy.optimize import curve_fit

    def model(x, x0, ldnu, tau_us, la, lb):
        s = sdsh_linear(x - x0, 10 ** ldnu, tau_us * 1e-6, 10 ** la)
        return 10 * np.log10(np.clip(s, 0, None) + 10 ** lb)

    m = np.abs(x) > mask_rbw * rbw              # exclude the coherent delta-peak at the center
    x0_lim = max(3 * rbw, 0.2 * dnu0)
    best = None
    for k in (1.0, 0.3, 3.0, 0.1, 10.0):        # several starts: fringes create local minima
        dn = float(np.clip(dnu0 * k, 10, 5e6))
        la0 = env_pk_db / 10 + np.log10(dn)
        lo = [-x0_lim, 1.0, tau0 * 1e6 * 0.9, la0 - 4, floor_db / 10 - 3]
        hi = [x0_lim, 7.0, tau0 * 1e6 * 1.1, la0 + 4, floor_db / 10 + 3]
        p0 = np.clip([0.0, np.log10(dn), tau0 * 1e6, la0, floor_db / 10], lo, hi)
        try:
            popt, pcov = curve_fit(model, x[m], y[m], p0=p0, bounds=(lo, hi), maxfev=20000)
        except Exception:
            continue
        sse = float(np.sum((model(x[m], *popt) - y[m]) ** 2))
        if best is None or sse < best[0]:
            best = (sse, popt, pcov)
    if best is None:
        raise RuntimeError("coherent fit did not converge")
    popt, pcov = best[1], best[2]
    at_bound = abs(popt[2] - tau0 * 1e6 * 0.9) < 1e-9 or abs(popt[2] - tau0 * 1e6 * 1.1) < 1e-9
    # relative 1-sigma uncertainty of dnu (from the log10 parameter)
    sig = float(np.sqrt(pcov[1, 1])) if np.all(np.isfinite(pcov)) else float("inf")
    rel_err = (10 ** sig - 1) if sig < 5 else float("inf")
    return 10 ** popt[1], popt[2] * 1e-6, at_bound, rel_err, (lambda xx: model(xx, *popt))


def analyze_trace(f, p_db, rbw, delay_m, cfg):
    """Returns (results dict, fits dict {label: callable(offset_Hz)->dB})."""
    res: dict = {"rbw_Hz": rbw}
    flags: list[str] = []
    fits: dict = {}

    sel = np.abs(f - cfg["aom_freq_hz"]) < cfg["peak_search_window_hz"]
    idx = np.flatnonzero(sel) if sel.any() else np.arange(len(f))
    i_pk = int(idx[np.argmax(p_db[idx])])
    f_pk, pk = float(f[i_pk]), float(p_db[i_pk])
    floor = float(np.percentile(p_db, 5))
    res.update(f_peak_Hz=f_pk, peak_dBm=pk, floor_dBm=floor, snr_dB=pk - floor)

    w3, w20 = width_at(f, p_db, i_pk, 3), width_at(f, p_db, i_pk, 20)
    res.update(w3dB_Hz=w3, dnu_3dB_Hz=w3 / 2, w20dB_Hz=w20, dnu_20dB_Hz=w20 / (2 * math.sqrt(99)))
    if np.isfinite(w3) and w3 < 5 * rbw:
        flags.append("3dB_width_RBW_limited")
    if pk - floor < 25:
        flags.append("SNR<25dB_20dB_width_unreliable")

    tau = tau_from_length(delay_m, cfg)
    res["tau_us"] = tau * 1e6
    if delay_m <= 0:
        res.update(regime="reference_no_delay", best_method="none", flags=";".join(flags))
        return res, fits
    if (f.max() - f.min()) < 0.5 / tau:
        # delay so short that not even half a fringe fits in the span: the trace is
        # practically identical to the zero-delay reference, no linewidth can be extracted
        flags.append("delay_too_short_for_this_span")
        res.update(regime="too_short_acts_as_reference", best_method="none", flags=";".join(flags))
        return res, fits
    if rbw > 0.1 / tau:
        flags.append("RBW_too_coarse_for_fringes")
    if (f.max() - f.min()) < 2 / tau:
        flags.append("span_<2_fringes_prefer_wider_span")

    x = f - f_pk
    # --- Lorentzian fit (valid when tau_d >> tau_c) ---
    g0 = w3 if (np.isfinite(w3) and w3 > rbw) else 10 * rbw
    try:
        g, fn = _fit_lorentz(x, p_db, rbw, g0, pk, floor)
        res["dnu_lorentz_Hz"] = g / 2
        fits["Lorentzian fit"] = fn
    except Exception as exc:
        log.warning("Lorentzian fit failed: %s", exc)

    # --- full DSH fit with partial coherence ---
    guesses = [res.get("dnu_lorentz_Hz"), res["dnu_20dB_Hz"], res["dnu_3dB_Hz"], 5e3]
    dnu0 = next(v for v in guesses if v is not None and np.isfinite(v) and v > 0)
    mask_rbw = cfg.get("center_mask_rbw", 4)
    env_pk = float(np.max(p_db[np.abs(x) > mask_rbw * rbw]))
    try:
        dnu_c, tau_fit, at_bound, rel_err, fn = _fit_coherent(x, p_db, rbw, tau, dnu0, env_pk, floor, mask_rbw)
        res.update(dnu_coherent_Hz=dnu_c, dnu_coherent_rel_err=rel_err, tau_fit_us=tau_fit * 1e6,
                   fiber_len_fit_m=tau_fit * C_LIGHT / cfg["fiber_group_index"])
        fits["DSH coherent fit"] = fn
        if at_bound:
            flags.append("tau_fit_at_bound_check_fiber_length")
        ratio = math.pi * dnu_c * tau_fit             # tau_d / tau_c
        res["tau_over_tauc"] = ratio
        res["regime"] = "incoherent" if math.exp(-2 * ratio) < 0.01 else "partially_coherent"
        if res["regime"] == "partially_coherent":
            flags.append("fringes_present_use_dnu_coherent")
    except Exception as exc:
        log.warning("Coherent DSH fit failed: %s", exc)

    rel_err = res.get("dnu_coherent_rel_err", float("inf"))
    if "dnu_coherent_Hz" in res and rel_err <= 0.3:
        res.update(dnu_best_Hz=res["dnu_coherent_Hz"], best_method="coherent_fit")
        if rel_err > 0.1:
            flags.append("dnu_uncertain_>10%")
    elif "dnu_coherent_Hz" in res:
        # fit converged but the data do not constrain dnu (typically: delay too short,
        # the incoherent part of the beat is buried near the noise floor)
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


# ================================ FILES & PLOTS ================================
def fmt_hz(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    for unit, k in (("MHz", 1e6), ("kHz", 1e3)):
        if abs(v) >= k:
            return f"{v / k:.3f} {unit}"
    return f"{v:.1f} Hz"


def save_xy(path: Path, x, y, header: str, fmt=("%.3f", "%.4f")) -> None:
    np.savetxt(path, np.column_stack([x, y]), delimiter=",", header=header, comments="", fmt=list(fmt))


def _csv_value(v):
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.6g}"
    return v


def write_rows(path: Path, rows: list[dict], append: bool) -> None:
    new = not path.exists() or not append
    with open(path, "a" if append else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: _csv_value(v) for k, v in r.items()})


def plot_trace(png: Path, f, p, res, fits, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = f - res["f_peak_Hz"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x / 1e3, p, lw=0.7, color="0.35", label="measured")
    for (label, fn), c in zip(fits.items(), ("tab:blue", "tab:red")):
        ax.plot(x / 1e3, fn(x), lw=1.3, color=c, label=label)
    ax.axhline(res["peak_dBm"] - 3, ls=":", lw=0.8, color="tab:green", label="-3 dB")
    ax.set_xlabel("Offset from beat peak [kHz]")
    ax.set_ylabel("Power [dBm]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    txt = "\n".join([
        f"RBW {fmt_hz(res['rbw_Hz'])}   SNR {res['snr_dB']:.1f} dB",
        f"laser dnu (-3 dB)    : {fmt_hz(res.get('dnu_3dB_Hz'))}",
        f"laser dnu (-20 dB)   : {fmt_hz(res.get('dnu_20dB_Hz'))}",
        f"laser dnu (Lorentz)  : {fmt_hz(res.get('dnu_lorentz_Hz'))}",
        f"laser dnu (coherent) : {fmt_hz(res.get('dnu_coherent_Hz'))}",
        f"beat 3dB FWHM equiv  : {fmt_hz(res.get('beat_3dB_FWHM_equiv_Hz'))}",
        f"regime: {res.get('regime', '')}",
    ])
    ax.text(0.01, 0.98, txt, transform=ax.transAxes, fontsize=8, family="monospace",
            va="top", bbox=dict(fc="white", alpha=0.85, ec="0.7"))
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n" + "=" * 78)
    print(f"Delay {rows[0]['delay_m']:.0f} m  (tau_d = {rows[0]['tau_us']:.2f} us)   "
          "- laser linewidth (FWHM), mean +/- std over repeats")
    print("  ('beat 3dB equiv' = beat FWHM = 2 x laser linewidth; all other lines = laser linewidth)")
    print("-" * 78)
    keys = [("dnu_best_Hz", "best"), ("beat_3dB_FWHM_equiv_Hz", "beat 3dB equiv"), ("dnu_coherent_Hz", "coherent fit"), ("dnu_lorentz_Hz", "Lorentz fit"),
            ("dnu_3dB_Hz", "-3 dB / 2"), ("dnu_20dB_Hz", "-20 dB")]
    for span in dict.fromkeys(r["span_name"] for r in rows):
        rr = [r for r in rows if r["span_name"] == span]
        print(f"[{span}]  RBW {fmt_hz(rr[0]['rbw_Hz'])}, regime: {rr[0].get('regime', '')}")
        for k, lbl in keys:
            v = np.array([r.get(k, np.nan) for r in rr], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                sd = f" +/- {fmt_hz(v.std(ddof=1))}" if v.size > 1 else ""
                print(f"    {lbl:<14}: {fmt_hz(v.mean())}{sd}")
        fl = sorted({f for r in rr for f in str(r.get("flags", "")).split(";") if f})
        if fl:
            print(f"    flags: {', '.join(fl)}")
    print("=" * 78 + "\n")


def add_file_logger(path: Path) -> None:
    h = logging.FileHandler(path, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(h)


# ==================================== MODES ====================================
def capture_scope(scope, path: Path, cfg: dict, meta: dict, key: str) -> float:
    """Saves a scope trace; returns AC RMS of the beat (nan on failure)."""
    try:
        t, v, info = scope.capture(cfg["scope_channel"], cfg["scope_points"])
        save_xy(path, t, v, "time_s,voltage_V", fmt=("%.6e", "%.6e"))
        rms = float(np.std(v))
        info.update(file=path.name, beat_rms_V=rms, mean_V=float(np.mean(v)))
        meta[key] = info
        if info.get("clipped"):
            log.warning("Scope trace is clipped - increase V/div on channel %d", cfg["scope_channel"])
        log.info("Scope %s: beat RMS = %.4g V", key, rms)
        return rms
    except Exception as exc:
        log.warning("Scope capture failed (%s) - continuing without it", exc)
        return float("nan")


def confirm_checklist(delay_m: float) -> None:
    print(f"\n--- Pre-measurement checklist (delay = {delay_m:.0f} m) ---")
    print("  1. Fiber spool connected / spliced, connectors cleaned")
    print("  2. Waited 5-10 min after handling the fiber (thermal + mechanical settling)")
    print("  3. Polarization optimized for max beat (run with --monitor)")
    print("  4. Laser in its low-noise (dither-off) mode, output power stable")
    print("  5. Nobody touching the table during the acquisition")
    input("Press Enter to start, or Ctrl+C to abort... ")


def run_check(cfg: dict) -> None:
    rm = open_rm()
    print("Instruments found on VISA:")
    for addr in rm.list_resources():
        try:
            r = rm.open_resource(addr)
            r.timeout = 3000
            print(f"  {addr:<48} {r.query('*IDN?').strip()}")
            r.close()
        except Exception as exc:
            print(f"  {addr:<48} (no answer: {exc})")
    print("\nAddresses in CONFIG:")
    items = [("ESA", cfg["esa_address"], N9010B), ("Scope", cfg["scope_address"], DSOX1204G),
             ("SigGen", cfg["siggen_address"], N5173B)]
    for name, addr, cls in items:
        try:
            dev = cls(rm, addr, name)
            print(f"  OK   {name:<7}{dev.idn()}")
            dev.close()
        except Exception as exc:
            print(f"  FAIL {name:<7}{exc}")


def run_diag(cfg: dict) -> None:
    """Send every ESA command one by one and report which ones the instrument rejects."""
    rm = open_rm()
    esa = N9010B(rm, cfg["esa_address"], "ESA")
    print("IDN        :", esa.idn())
    print("Mode       :", esa.safe_query(":INST:SEL?"))
    print("Modes inst.:", esa.safe_query(":INST:CAT?"))
    print("Measurement:", esa.safe_query(":CONF?"))
    esa.write("*CLS")
    print("\nSetup commands:")
    for cmd, _ in N9010B.SETUP_CMDS:
        errs = esa.write_checked(cmd, timeout_s=60)
        print(f"  {'OK  ' if not errs else 'FAIL'} {cmd:<28} {errs[0] if errs else ''}")
        if cmd == ":INST:SEL SA":
            time.sleep(2)
            print("       mode now:", esa.safe_query(":INST:SEL?"))
    print("\nConfiguration commands:")
    for cmd in [":FREQ:CENT 80e6", ":FREQ:SPAN 5e6", ":BAND:RES 1e3", ":SWE:POIN 20001",
                ":TRAC1:TYPE AVER", ":AVER:COUN 20"]:
        errs = esa.write_checked(cmd)
        print(f"  {'OK  ' if not errs else 'FAIL'} {cmd:<28} {errs[0] if errs else ''}")
    print("\nRead-back queries:")
    for q in [":FREQ:CENT?", ":FREQ:SPAN?", ":BAND:RES?", ":BAND:VID?", ":SWE:POIN?", ":SWE:TIME?"]:
        print(f"  {q:<28} {esa.safe_query(q)}")
    esa.write(":INIT:CONT ON")
    esa.close()


def run_monitor(args, cfg: dict) -> None:
    esa = FakeESA(0, cfg) if args.simulate else N9010B(open_rm(), cfg["esa_address"], "ESA")
    try:
        esa.setup_common()
        esa.configure(cfg["aom_freq_hz"], cfg["monitor_span_hz"], cfg["monitor_rbw_hz"], 1001, 1)
        best = -np.inf
        print("Live beat peak - adjust polarization for maximum. Ctrl+C to stop.")
        while True:
            fx, py = esa.quick_peak()
            best = max(best, py)
            bar = "#" * int(np.clip((py - (best - 20)) * 3, 0, 60))
            print(f"\r peak {py:7.2f} dBm @ {(fx - cfg['aom_freq_hz']) / 1e3:+9.2f} kHz | "
                  f"best {best:7.2f} dBm | {bar:<60}", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        esa.restore_live()
        esa.close()


def run_measurement(args, cfg: dict) -> None:
    delay = float(args.delay)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg["data_root"]) / f"delay_{int(round(delay)):05d}m" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    add_file_logger(run_dir / "run.log")
    tau = tau_from_length(delay, cfg)
    log.info("Run dir: %s", run_dir)
    log.info("Delay %g m -> tau_d = %.4g us, fringe spacing 1/tau_d = %s",
             delay, tau * 1e6, fmt_hz(1 / tau) if tau > 0 else "n/a")
    if not args.yes:
        confirm_checklist(delay)

    meta = {"delay_m": delay, "tau_us": tau * 1e6, "note": args.note, "simulated": args.simulate,
            "started": datetime.now().isoformat(timespec="seconds"), "repeats": args.repeats,
            "config": cfg, "traces": []}
    rows: list[dict] = []
    esa = scope = sg = None
    try:
        if args.simulate:
            esa, scope = FakeESA(delay, cfg), FakeScope()
        else:
            rm = open_rm()
            esa = N9010B(rm, cfg["esa_address"], "ESA")
            scope = DSOX1204G(rm, cfg["scope_address"], "Scope")
            if cfg["log_siggen"]:
                sg = N5173B(rm, cfg["siggen_address"], "SigGen")
                meta["siggen"] = siggen_state(sg)
        meta["instruments"] = {"esa": esa.idn(), "scope": scope.idn()}
        esa.setup_common()
        rms0 = capture_scope(scope, run_dir / "scope_before.csv", cfg, meta, "scope_before")

        for r in range(1, args.repeats + 1):
            for sp in cfg["esa_spans"]:
                tag = f"esa_{sp['name']}_r{r:02d}"
                log.info("Acquiring %s  (span %s, RBW %s, %d avg) ...", tag, fmt_hz(sp["span_hz"]),
                         fmt_hz(sp["rbw_hz"]), sp["averages"])
                actual = esa.configure(cfg["aom_freq_hz"], sp["span_hz"], sp["rbw_hz"],
                                       sp["points"], sp["averages"])
                t0 = time.time()
                f, p = esa.acquire(cfg["esa_sweep_timeout_s"])
                actual["acquisition_time_s"] = round(time.time() - t0, 2)
                save_xy(run_dir / f"{tag}.csv", f, p, "freq_Hz,power_dBm")
                meta["traces"].append({"file": f"{tag}.csv", "span_name": sp["name"], "repeat": r, **actual})

                res, fits = analyze_trace(f, p, actual["rbw_hz"], delay, cfg)
                row = {"timestamp": stamp, "run_dir": str(run_dir), "delay_m": delay, "span_name": sp["name"],
                       "repeat": r, "file": f"{tag}.csv", "note": args.note, **res}
                rows.append(row)
                plot_trace(run_dir / f"{tag}.png", f, p, res, fits, f"{delay:.0f} m - {tag}")
                log.info("  -> laser linewidth %s, beat 3dB FWHM equiv %s (%s)  %s",
                         fmt_hz(res.get("dnu_best_Hz")), fmt_hz(res.get("beat_3dB_FWHM_equiv_Hz")),
                         res.get("best_method"), res.get("flags", ""))

        rms1 = capture_scope(scope, run_dir / "scope_after.csv", cfg, meta, "scope_after")
        drift = abs(rms1 - rms0) / rms0 if rms0 > 0 else float("nan")
        meta["beat_rms_drift"] = drift
        if np.isfinite(drift) and drift > cfg["beat_drift_warn"]:
            log.warning("Beat amplitude changed by %.0f %% during the run - polarization/power drift, "
                        "consider repeating.", 100 * drift)
        for row in rows:
            row["beat_rms_drift"] = drift
    except KeyboardInterrupt:
        log.warning("Interrupted - partial data kept in %s", run_dir)
    except RuntimeError as exc:
        log.error("%s", exc)
    finally:
        for dev, fn in ((esa, "restore_live"), (esa, "close"), (scope, "close"), (sg, "close")):
            if dev is not None:
                try:
                    getattr(dev, fn)()
                except Exception:
                    pass
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        if rows:
            write_rows(run_dir / "results.csv", rows, append=False)
            write_rows(Path(cfg["data_root"]) / "summary.csv", rows, append=True)
    print_summary(rows)
    log.info("Done. Data in %s", run_dir)


def run_analysis(path: str, cfg: dict) -> None:
    run_dir = Path(path)
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = {**cfg, **{k: meta["config"][k] for k in ("aom_freq_hz", "fiber_group_index") if k in meta["config"]}}
    delay = float(meta["delay_m"])
    rows = []
    for tr in meta["traces"]:
        data = np.loadtxt(run_dir / tr["file"], delimiter=",", skiprows=1)
        f, p = data[:, 0], data[:, 1]
        res, fits = analyze_trace(f, p, tr["rbw_hz"], delay, cfg)
        rows.append({"timestamp": run_dir.name, "run_dir": str(run_dir), "delay_m": delay,
                     "span_name": tr["span_name"], "repeat": tr["repeat"], "file": tr["file"],
                     "note": meta.get("note", ""), "beat_rms_drift": meta.get("beat_rms_drift", ""), **res})
        plot_trace(run_dir / tr["file"].replace(".csv", ".png"), f, p, res, fits,
                   f"{delay:.0f} m - {tr['file']}")
    write_rows(run_dir / "results.csv", rows, append=False)
    print_summary(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Delayed self-heterodyne linewidth measurement")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--delay", type=float, help="fiber delay length in meters (0 = reference)")
    g.add_argument("--check", action="store_true", help="list VISA resources and query *IDN?")
    g.add_argument("--diag", action="store_true", help="test every ESA command and report rejects")
    g.add_argument("--monitor", action="store_true", help="live beat power for polarization alignment")
    g.add_argument("--analyze", metavar="RUN_DIR", help="re-analyze an existing run folder")
    ap.add_argument("--repeats", type=int, default=3, help="independent acquisitions per span (default 3)")
    ap.add_argument("--note", default="", help="free-text note stored with the data")
    ap.add_argument("--yes", action="store_true", help="skip the pre-measurement checklist prompt")
    ap.add_argument("--simulate", action="store_true", help="use simulated instruments")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = dict(CONFIG)
    if args.check:
        run_check(cfg)
    elif args.diag:
        run_diag(cfg)
    elif args.monitor:
        run_monitor(args, cfg)
    elif args.analyze:
        run_analysis(args.analyze, cfg)
    else:
        run_measurement(args, cfg)


if __name__ == "__main__":
    main()