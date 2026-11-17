"""Shared runner for the delayed self-heterodyne (DSH) fiber simulations.

Each sim_fiber_<L>m.py calls run_fiber(L). One run works like the lab script (sh_linewidth.py --delay L):
  1. Create a run folder  self_heterodyne/data_measured/delay_XXXXXm/<YYYY-MM-DD_HH-MM-SS>_sim/
     (next to the lab runs of the same fiber length; the _sim suffix tells them apart)
  2. For every repeat and every ESA acquisition (wide / narrow / ultra_wide): simulate the averaged
     ESA trace of the 80 MHz beat (DSH model + RBW filter + noise floor + averaging noise) and
     analyze it exactly like the lab data (-3/-20 dB widths, Lorentzian fit, coherent fit, flags).
  3. Save into the run folder:
        esa_<span>_rNN.csv    freq_Hz, power_dBm, expected_dBm (expected = without noise)
        esa_<span>_rNN.png    trace + Lorentzian fit + coherent fit + -3 dB line
        overview.png          repeat 1 of every acquisition
        results.csv           one row per trace (lab columns + true linewidth / coherent fraction)
        metadata.json         all simulation parameters + list of traces
        run.log               the printed summary
     and append the rows to self_heterodyne/data_measured/summary_sim.csv (the lab summary.csv
     stays measurement-only).
  4. Print mean +/- std over the repeats.

Browse the runs with:  python self_heterodyne/sh_data.py list --kind sim
All simulation parameters are defined below.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

SH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SH_ROOT))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

from dsh import analysis, model, runs  # noqa: E402
from dsh.model import Acquisition  # noqa: E402

DATA_ROOT = runs.DATA_ROOT         # self_heterodyne/data_measured (shared with the lab runs)
SUMMARY_FILE = "summary_sim.csv"

# ----------------------------- parameters ---------------------------------
LASER_LINEWIDTH_HZ = 52e3          # true laser FWHM dnu (Pure Photonics PPCL550, measured ~51.8 kHz)
AOM_FREQ_HZ = 80e6                 # AOM shift = beat center frequency
FIBER_GROUP_INDEX = model.N_GROUP_SMF28
BEAT_POWER_DBM = -23.0             # total beat power on the ESA (reproduces the measured peak levels)
NOISE_FLOOR_DBM_HZ = -145.0        # ESA + photodiode noise density
SWEEP_NOISE = 0.3                  # relative std of one sweep per point (divided by sqrt(averages))
BEAT_OFFSET_HZ = 0.0               # AOM and ESA share the 10 MHz reference -> beat at exactly 80 MHz
REPEATS = 3
SEED = None                        # None = new noise every run (like a real measurement); int = reproducible

# ESA acquisitions (same as the lab CONFIG; the ESA rounds the 5 kHz RBW to 5.1 kHz)
ACQUISITIONS = [
    Acquisition("wide",       span_hz=5e6,   rbw_hz=5.1e3, points=1001, averages=20),
    Acquisition("narrow",     span_hz=200e3, rbw_hz=200,   points=1001, averages=20),
    Acquisition("ultra_wide", span_hz=20e6,  rbw_hz=20e3,  points=1001, averages=20),
]
PEAK_SEARCH_WINDOW_HZ = 1e6        # search the beat within +/- this around AOM_FREQ_HZ
CENTER_MASK_RBW = 4                # coherent fit ignores +/- this many RBWs around the center
# -------------------------------------------------------------------------

# same columns as the lab results.csv / summary.csv, plus the simulation truth
RESULT_FIELDS = [
    "timestamp", "run_dir", "delay_m", "tau_us", "span_name", "repeat", "file", "rbw_Hz",
    "f_peak_Hz", "peak_dBm", "floor_dBm", "snr_dB",
    "w3dB_Hz", "dnu_3dB_Hz", "w20dB_Hz", "dnu_20dB_Hz",
    "dnu_lorentz_Hz", "dnu_coherent_Hz", "dnu_coherent_rel_err", "tau_fit_us", "fiber_len_fit_m",
    "tau_over_tauc", "regime", "dnu_best_Hz", "beat_3dB_FWHM_equiv_Hz", "best_method", "flags",
    "beat_rms_drift", "note", "dnu_true_Hz", "coherent_fraction_true",
]

SUMMARY_KEYS = [("dnu_best_Hz", "best"), ("beat_3dB_FWHM_equiv_Hz", "beat 3dB equiv"),
                ("dnu_coherent_Hz", "coherent fit"), ("dnu_lorentz_Hz", "Lorentz fit"),
                ("dnu_3dB_Hz", "-3 dB / 2"), ("dnu_20dB_Hz", "-20 dB")]


def fmt_hz(v) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    for unit, k in (("MHz", 1e6), ("kHz", 1e3)):
        if abs(v) >= k:
            return f"{v / k:.3f} {unit}"
    return f"{v:.1f} Hz"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_fiber(length_m: float, rng: np.random.Generator):
    """Simulate and analyze every repeat x acquisition. Returns (rows, traces)."""
    tau = model.delay_time(length_m, FIBER_GROUP_INDEX)
    rows, traces = [], []
    for rep in range(1, REPEATS + 1):
        for acq in ACQUISITIONS:
            f, p, p_expected = model.esa_trace(
                acq, AOM_FREQ_HZ, LASER_LINEWIDTH_HZ, tau,
                beat_power_dbm=BEAT_POWER_DBM, noise_floor_dbm_hz=NOISE_FLOOR_DBM_HZ,
                sweep_noise=SWEEP_NOISE, rng=rng, beat_offset_hz=BEAT_OFFSET_HZ)
            res, fits = analysis.analyze_trace(
                f, p, acq.rbw_hz, tau, center_hz=AOM_FREQ_HZ, peak_window_hz=PEAK_SEARCH_WINDOW_HZ,
                mask_rbw=CENTER_MASK_RBW, group_index=FIBER_GROUP_INDEX)
            row = {"delay_m": length_m, "span_name": acq.name, "repeat": rep,
                   "file": f"esa_{acq.name}_r{rep:02d}.csv", **res,
                   "dnu_true_Hz": LASER_LINEWIDTH_HZ,
                   "coherent_fraction_true": model.coherent_fraction(LASER_LINEWIDTH_HZ, tau)}
            rows.append(row)
            traces.append({"acq": acq, "repeat": rep, "f": f, "p": p, "p_expected": p_expected,
                           "res": res, "fits": fits})
    return rows, traces


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _csv_value(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.6g}"
    return v


def write_rows(path: Path, rows: list[dict], append: bool) -> None:
    new = not path.exists() or not append
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: _csv_value(v) for k, v in r.items()})


def save_trace_csv(path: Path, t: dict) -> None:
    np.savetxt(path, np.column_stack([t["f"], t["p"], t["p_expected"]]), delimiter=",",
               header="freq_Hz,power_dBm,expected_dBm", comments="", fmt=["%.3f", "%.4f", "%.4f"])


def draw_trace(ax, t: dict) -> None:
    res, acq = t["res"], t["acq"]
    x = t["f"] - res["f_peak_Hz"]
    ax.plot(x / 1e3, t["p"], lw=0.7, color="0.35", label="simulated ESA trace")
    ax.plot(x / 1e3, t["p_expected"], lw=0.8, color="0.7", ls="--", label="expected (no noise)")
    for (label, fn), c in zip(t["fits"].items(), ("tab:blue", "tab:red")):
        ax.plot(x / 1e3, fn(x), lw=1.3, color=c, label=label)
    ax.axhline(res["peak_dBm"] - 3, ls=":", lw=0.8, color="tab:green", label="-3 dB")
    ax.set_xlabel("Offset from beat peak [kHz]")
    ax.set_ylabel("Power [dBm]")
    ax.set_title(f"{acq.name} r{t['repeat']:02d}: span {fmt_hz(acq.span_hz)}, RBW {fmt_hz(acq.rbw_hz)}, "
                 f"{acq.averages} averages")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    txt = "\n".join([
        f"SNR {res['snr_dB']:.1f} dB   true dnu {fmt_hz(LASER_LINEWIDTH_HZ)}",
        f"laser dnu (-3 dB)    : {fmt_hz(res.get('dnu_3dB_Hz'))}",
        f"laser dnu (-20 dB)   : {fmt_hz(res.get('dnu_20dB_Hz'))}",
        f"laser dnu (Lorentz)  : {fmt_hz(res.get('dnu_lorentz_Hz'))}",
        f"laser dnu (coherent) : {fmt_hz(res.get('dnu_coherent_Hz'))}",
        f"beat 3dB FWHM equiv  : {fmt_hz(res.get('beat_3dB_FWHM_equiv_Hz'))}",
        f"regime: {res.get('regime', '')}",
    ])
    ax.text(0.01, 0.98, txt, transform=ax.transAxes, fontsize=8, family="monospace",
            va="top", bbox=dict(fc="white", alpha=0.85, ec="0.7"))


def save_trace_png(path: Path, length_m: float, t: dict) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    draw_trace(ax, t)
    ax.set_title(f"{length_m:g} m (simulation) - {ax.get_title()}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_overview(path: Path, length_m: float, traces: list[dict]):
    import matplotlib.pyplot as plt

    first = [t for t in traces if t["repeat"] == 1]
    tau = model.delay_time(length_m, FIBER_GROUP_INDEX)
    fig, axes = plt.subplots(len(first), 1, figsize=(11, 4.6 * len(first)))
    for ax, t in zip(np.atleast_1d(axes), first):
        draw_trace(ax, t)
    fig.suptitle(f"DSH simulation: {length_m:g} m fiber, tau_d = {tau * 1e6:.2f} us, "
                 f"tau_d/tau_c = {tau / model.coherence_time(LASER_LINEWIDTH_HZ):.2f}, coherent "
                 f"fraction {model.coherent_fraction(LASER_LINEWIDTH_HZ, tau):.2%}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def print_summary(length_m: float, rows: list[dict]) -> None:
    tau = model.delay_time(length_m, FIBER_GROUP_INDEX)
    print("=" * 78)
    print(f"Delay {length_m:g} m  (tau_d = {tau * 1e6:.2f} us, tau_d/tau_c = "
          f"{tau / model.coherence_time(LASER_LINEWIDTH_HZ):.2f}, coherent fraction "
          f"{model.coherent_fraction(LASER_LINEWIDTH_HZ, tau):.2%}, max RBW for fringes "
          f"{fmt_hz(model.max_rbw_for_fringes(tau))})")
    print(f"True laser linewidth {fmt_hz(LASER_LINEWIDTH_HZ)} - mean +/- std over {REPEATS} repeats")
    print("  ('beat 3dB equiv' = beat FWHM = 2 x laser linewidth; all other lines = laser linewidth)")
    print("-" * 78)
    for acq in ACQUISITIONS:
        rr = [r for r in rows if r["span_name"] == acq.name]
        print(f"[{acq.name}]  RBW {fmt_hz(acq.rbw_hz)}, regime: {rr[0].get('regime', '')}")
        for key, label in SUMMARY_KEYS:
            v = np.array([r.get(key, np.nan) for r in rr], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                sd = f" +/- {fmt_hz(v.std(ddof=1))}" if v.size > 1 else ""
                print(f"    {label:<14}: {fmt_hz(v.mean())}{sd}")
        fl = sorted({f for r in rr for f in str(r.get("flags", "")).split(";") if f})
        if fl:
            print(f"    flags: {', '.join(fl)}")
    print("=" * 78)


def run_fiber(length_m: float, show: bool = True, note: str = "") -> Path:
    """Simulate one fiber length into a new timestamped run folder. Returns the run folder."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = DATA_ROOT / runs.delay_dir_name(length_m) / f"{stamp}{runs.SIM_SUFFIX}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now().isoformat(timespec="seconds")
    tau = model.delay_time(length_m, FIBER_GROUP_INDEX)
    seed = SEED if SEED is not None else int(np.random.SeedSequence().entropy % 2**32)
    rows, traces = simulate_fiber(length_m, np.random.default_rng(seed))

    for t, row in zip(traces, rows):
        row.update(timestamp=stamp, run_dir=str(run_dir), note=note)
        save_trace_csv(run_dir / row["file"], t)
        save_trace_png(run_dir / row["file"].replace(".csv", ".png"), length_m, t)
    fig = plot_overview(run_dir / "overview.png", length_m, traces)

    meta = {
        "delay_m": length_m, "tau_us": tau * 1e6, "note": note, "simulated": True,
        "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
        "repeats": REPEATS, "seed": seed,
        "config": {"laser_linewidth_hz": LASER_LINEWIDTH_HZ, "aom_freq_hz": AOM_FREQ_HZ,
                   "fiber_group_index": FIBER_GROUP_INDEX, "beat_power_dbm": BEAT_POWER_DBM,
                   "noise_floor_dbm_hz": NOISE_FLOOR_DBM_HZ, "sweep_noise": SWEEP_NOISE,
                   "beat_offset_hz": BEAT_OFFSET_HZ, "peak_search_window_hz": PEAK_SEARCH_WINDOW_HZ,
                   "center_mask_rbw": CENTER_MASK_RBW, "esa_spans": [asdict(a) for a in ACQUISITIONS]},
        "traces": [{"file": r["file"], "span_name": r["span_name"], "repeat": r["repeat"],
                    "rbw_hz": t["acq"].rbw_hz, "span_hz": t["acq"].span_hz, "points": t["acq"].points,
                    "averages": t["acq"].averages} for r, t in zip(rows, traces)],
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_rows(run_dir / "results.csv", rows, append=False)
    write_rows(DATA_ROOT / SUMMARY_FILE, rows, append=True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_summary(length_m, rows)
    (run_dir / "run.log").write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue(), end="")
    print(f"Data in {run_dir}")
    print(f"Browse: python self_heterodyne/sh_data.py show {length_m:g} --run {run_dir.name}\n")
    if show:
        plt.show()
    plt.close(fig)
    return run_dir
