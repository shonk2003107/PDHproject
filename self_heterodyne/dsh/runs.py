"""Access to DSH run folders (lab measurements and simulations share the same layout).

Everything lives in one place: self_heterodyne/data_measured/ holds the lab runs (copied from the
lab script's data folder) and the simulation runs (folder name ends with _sim).

Layout (written by the lab script sh_linewidth.py --delay L, and by the simulations here):

    <root>/
        summary.csv                       lab: one row per ESA trace, appended across all runs
        summary_sim.csv                   the same for the simulation runs
        delay_XXXXXm/<YYYY-MM-DD_HH-MM-SS>/       lab run
        delay_XXXXXm/<YYYY-MM-DD_HH-MM-SS>_sim/   simulation run
            esa_<span>_rNN.csv            freq_Hz, power_dBm (+ expected_dBm for simulations)
            esa_<span>_rNN.png            trace + fits
            scope_before.csv, scope_after.csv   time_s, voltage_V (lab only)
            results.csv                   one row per ESA trace, all linewidth estimates
            metadata.json                 settings, notes, list of traces
            run.log

Usage from Python (run from the self_heterodyne folder, or add it to sys.path):

    from dsh import runs
    run = runs.find_run(runs.DATA_ROOT, 5000)              # latest 5 km run (lab or simulation)
    run = runs.find_run(runs.DATA_ROOT, 5000, "2026-11-12_16-51-43")
    rows = run.results                                    # list of dicts (numbers already floats)
    f, p = run.trace("wide", repeat=1)                    # ESA trace [Hz], [dBm]
    t, v = run.scope("before")                            # scope trace [s], [V]
    stats = runs.summarize(rows)                          # {span: {column: (mean, std, n)}}
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

SH_ROOT = Path(__file__).resolve().parents[1]
LAB_DATA_ROOT = Path(r"C:\Users\Owner\PycharmProjects\WelcomeScreen\data")   # data folder of sh_linewidth.py
DATA_ROOT = SH_ROOT / "data_measured"     # the project's single data folder: copied lab runs + simulations
SIM_SUFFIX = "_sim"                       # simulation run folders: <timestamp>_sim

_DELAY_RE = re.compile(r"^delay_(\d+)m$")
_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_sim)?$")
LINEWIDTH_COLUMNS = ["dnu_best_Hz", "beat_3dB_FWHM_equiv_Hz", "dnu_coherent_Hz", "dnu_lorentz_Hz",
                     "dnu_3dB_Hz", "dnu_20dB_Hz"]


def delay_dir_name(delay_m: float) -> str:
    """Folder name used for a fiber length: 500 -> 'delay_00500m'."""
    return f"delay_{int(round(delay_m)):05d}m"


def _number(v: str):
    """CSV cell -> float when numeric, None when empty, else the string."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return v


def read_rows(path: Path) -> list[dict]:
    """Read a one-header-row CSV into a list of dicts with numeric cells converted to float."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: _number(v) for k, v in row.items()} for row in csv.DictReader(fh)]


# ---------------------------------------------------------------------------
# One run folder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    path: Path
    delay_m: float
    timestamp: str

    @property
    def name(self) -> str:
        return f"{delay_dir_name(self.delay_m)}/{self.timestamp}"

    @cached_property
    def metadata(self) -> dict:
        p = self.path / "metadata.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    @cached_property
    def results(self) -> list[dict]:
        p = self.path / "results.csv"
        return read_rows(p) if p.exists() else []

    @property
    def note(self) -> str:
        return str(self.metadata.get("note", "") or "").strip()

    @property
    def simulated(self) -> bool:
        return self.timestamp.endswith(SIM_SUFFIX) or bool(self.metadata.get("simulated", False))

    @property
    def complete(self) -> bool:
        """metadata.json and results.csv are written at the end of a run; missing = interrupted."""
        return (self.path / "metadata.json").exists() and (self.path / "results.csv").exists()

    def files(self, pattern: str = "*") -> list[Path]:
        return sorted(p for p in self.path.glob(pattern) if p.is_file())

    def trace_files(self) -> list[Path]:
        return self.files("esa_*.csv")

    def trace_path(self, span_name: str, repeat: int = 1) -> Path:
        for tr in self.metadata.get("traces", []):
            if tr.get("span_name") == span_name and int(tr.get("repeat", 0)) == repeat:
                return self.path / tr["file"]
        # no metadata (interrupted run): try both spellings of the span name
        for name in (span_name, span_name.replace(" ", "_"), span_name.replace("_", " ")):
            p = self.path / f"esa_{name}_r{repeat:02d}.csv"
            if p.exists():
                return p
        raise FileNotFoundError(f"no trace '{span_name}' repeat {repeat} in {self.path}")

    def trace(self, span_name: str, repeat: int = 1):
        """ESA trace -> (freq_Hz, power_dBm)."""
        data = np.loadtxt(self.trace_path(span_name, repeat), delimiter=",", skiprows=1, ndmin=2)
        return data[:, 0], data[:, 1]

    def scope(self, which: str = "before"):
        """Scope trace -> (time_s, voltage_V); which = 'before' or 'after'."""
        data = np.loadtxt(self.path / f"scope_{which}.csv", delimiter=",", skiprows=1, ndmin=2)
        return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Finding runs
# ---------------------------------------------------------------------------


def list_runs(root: Path, delay_m: float | None = None, kind: str = "all") -> list[Run]:
    """Run folders under root, sorted by delay, then time.

    delay_m: only one fiber length.  kind: 'all', 'lab' (measurements) or 'sim' (simulations).
    """
    root = Path(root)
    out = []
    if not root.is_dir():
        return out
    for d in root.iterdir():
        m = _DELAY_RE.match(d.name)
        if not (d.is_dir() and m):
            continue
        delay = float(m.group(1))
        if delay_m is not None and int(round(delay)) != int(round(delay_m)):
            continue
        out += [Run(r, delay, r.name) for r in d.iterdir() if r.is_dir() and _STAMP_RE.match(r.name)]
    if kind != "all":
        out = [r for r in out if r.simulated == (kind == "sim")]
    return sorted(out, key=lambda r: (r.delay_m, r.timestamp))


def list_delays(root: Path) -> list[float]:
    return sorted({r.delay_m for r in list_runs(root)})


def find_run(root: Path, delay_m: float, run: str = "latest", kind: str = "all") -> Run:
    """A run of one fiber length: 'latest', a timestamp (or its unique prefix), or an index like '-2'."""
    runs = list_runs(root, delay_m, kind)
    if not runs:
        raise FileNotFoundError(f"no runs for {delay_m:g} m in {root}")
    if run in ("latest", "last"):
        return runs[-1]
    if re.fullmatch(r"-?\d+", run) and not _STAMP_RE.match(run):
        return runs[int(run)]
    hits = [r for r in runs if r.timestamp.startswith(run)]
    if len(hits) != 1:
        raise FileNotFoundError(f"{'no' if not hits else 'ambiguous'} run '{run}' for {delay_m:g} m "
                                f"(available: {', '.join(r.timestamp for r in runs)})")
    return hits[0]


# ---------------------------------------------------------------------------
# Summaries and copying
# ---------------------------------------------------------------------------


def summarize(rows: list[dict], columns=LINEWIDTH_COLUMNS) -> dict:
    """{span_name: {column: (mean, std, n)}} over the repeats; empty cells are skipped."""
    out: dict = {}
    for span in dict.fromkeys(r.get("span_name") for r in rows):
        rr = [r for r in rows if r.get("span_name") == span]
        stats = {}
        for c in columns:
            v = np.array([r[c] for r in rr if isinstance(r.get(c), float) and math.isfinite(r[c])])
            if v.size:
                stats[c] = (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan"), int(v.size))
        out[span] = stats
    return out


def span_flags(rows: list[dict], span: str) -> list[str]:
    return sorted({f for r in rows if r.get("span_name") == span
                   for f in str(r.get("flags") or "").split(";") if f})


def copy_run(run: Run, dest_root: Path = DATA_ROOT, overwrite: bool = False) -> Path:
    """Copy a run folder to dest_root/delay_XXXXXm/<timestamp>/ (keeps the same layout)."""
    dest = Path(dest_root) / delay_dir_name(run.delay_m) / run.timestamp
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"{dest} already exists")
        shutil.rmtree(dest)
    shutil.copytree(run.path, dest)
    return dest
