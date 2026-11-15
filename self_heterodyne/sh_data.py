"""Browse DSH run folders and take data out of them.

All data lives in one place: self_heterodyne/data_measured/
    delay_XXXXXm/<YYYY-MM-DD_HH-MM-SS>/       lab measurement (sh_linewidth.py writes here directly)
    delay_XXXXXm/<YYYY-MM-DD_HH-MM-SS>_sim/   simulation (sim_fiber_<L>m.py writes here directly)
Each run folder has the ESA CSV + PNG of every trace, results.csv, metadata.json and run.log.

Run from the project root:

    python self_heterodyne/sh_data.py                       # interactive menu
    python self_heterodyne/sh_data.py list                  # all runs (lab + simulation)
    python self_heterodyne/sh_data.py list --kind lab       # only measurements (or --kind sim)
    python self_heterodyne/sh_data.py list --delay 5000     # runs of one fiber length
    python self_heterodyne/sh_data.py show 5000             # latest 5 km run: results + file paths
    python self_heterodyne/sh_data.py show 5000 --kind lab  # latest 5 km measurement
    python self_heterodyne/sh_data.py show 5000 --run 2026-11-12_16-51-43
    python self_heterodyne/sh_data.py open 5000             # open the folder in Explorer
    python self_heterodyne/sh_data.py copy --all            # bring runs from the old PyCharm lab folder
    python self_heterodyne/sh_data.py copy 5000             # the latest 5 km run from the old PyCharm lab folder

'copy' brings in older runs from the old lab data folder in the PyCharm project:  C:\\Users\\Owner\\PycharmProjects\\WelcomeScreen\\data
Use --source lab to browse that folder directly, or --root PATH for any other folder.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsh import runs  # noqa: E402

SOURCES = {"project": runs.DATA_ROOT, "lab": runs.LAB_DATA_ROOT}
SPAN_ORDER = ["wide", "narrow", "ultra_wide", "ultra wide"]
LABELS = {"dnu_best_Hz": "best", "beat_3dB_FWHM_equiv_Hz": "beat 3dB equiv",
          "dnu_coherent_Hz": "coherent fit", "dnu_lorentz_Hz": "Lorentz fit",
          "dnu_3dB_Hz": "-3 dB / 2", "dnu_20dB_Hz": "-20 dB"}


def khz(v) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{v / 1e3:.2f} kHz"


def headline(run: runs.Run) -> str:
    """Best linewidth of the run's first span with a result (wide if present)."""
    stats = runs.summarize(run.results)
    for span in sorted(stats, key=lambda s: SPAN_ORDER.index(s) if s in SPAN_ORDER else 99):
        if "dnu_best_Hz" in stats[span]:
            mean, std, _ = stats[span]["dnu_best_Hz"]
            sd = f" +/- {std / 1e3:.2f}" if math.isfinite(std) else ""
            return f"{mean / 1e3:.2f}{sd} kHz ({span})"
    return "-"


def status(run: runs.Run) -> str:
    if not run.complete:
        return "INTERRUPTED"
    return "sim" if run.simulated else "measured"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def print_list(all_runs: list[runs.Run], root: Path) -> None:
    print(f"Data root: {root}")
    if not all_runs:
        print("  (no runs)")
        return
    print(f"  {'#':>3}  {'delay':>7}  {'run':<24} {'status':<12} {'traces':>6}  {'best laser linewidth':<28} note")
    for i, r in enumerate(all_runs, 1):
        print(f"  {i:>3}  {r.delay_m:>5.0f} m  {r.timestamp:<24} {status(r):<12} {len(r.trace_files()):>6}  "
              f"{headline(r):<28} {r.note}")


def print_run(run: runs.Run) -> None:
    md = run.metadata
    print("=" * 78)
    print(f"Run     : {run.name}   ({status(run)})")
    print(f"Folder  : {run.path}")
    if md:
        print(f"Delay   : {run.delay_m:g} m   tau_d = {md.get('tau_us', float('nan')):.2f} us   "
              f"repeats {md.get('repeats', '?')}   started {md.get('started', '?')}")
    if run.note:
        print(f"Note    : {run.note}")
    rows = run.results
    if rows:
        print("-" * 78)
        print("Laser linewidth, mean +/- std over repeats (beat 3dB equiv = 2 x linewidth)")
        for span, stats in runs.summarize(rows).items():
            rr = [r for r in rows if r.get("span_name") == span]
            print(f"[{span}]  RBW {khz(rr[0].get('rbw_Hz'))}, regime: {rr[0].get('regime') or ''}")
            for col, (mean, std, n) in stats.items():
                sd = f" +/- {std / 1e3:.2f}" if math.isfinite(std) else ""
                print(f"    {LABELS[col]:<14}: {mean / 1e3:.2f}{sd} kHz  (n={n})")
            fl = runs.span_flags(rows, span)
            if fl:
                print(f"    flags: {', '.join(fl)}")
    else:
        print("  no results.csv - the run was interrupted before it finished")
    print("-" * 78)
    print("Files:")
    for p in run.files():
        print(f"    {p.name:<28} {p.stat().st_size / 1024:>8.1f} kB")
    print("-" * 78)
    print("In Python (from the self_heterodyne folder):")
    print("    from dsh import runs")
    print(f"    run = runs.find_run(r\"{run.path.parents[1]}\", {run.delay_m:g}, \"{run.timestamp}\")")
    print("    rows = run.results;  f, p = run.trace(\"wide\", 1);  t, v = run.scope(\"before\")")
    print("=" * 78)


def open_folder(path: Path) -> None:
    if hasattr(os, "startfile"):
        os.startfile(path)  # Windows Explorer
        print(f"Opened {path}")
    else:
        print(f"Folder: {path}")


def do_copy(run: runs.Run, overwrite: bool) -> bool:
    """Copy one lab run into data_measured. Returns True if something was copied."""
    try:
        dest = runs.copy_run(run, overwrite=overwrite)
        print(f"Copied {run.name} -> {dest}")
        return True
    except FileExistsError:
        return False


def merge_summary(src: Path, dest: Path) -> int:
    """Append the rows of src that dest does not have yet. Never overwrites: sh_linewidth.py in
    this project appends its own rows to dest directly. Returns the number of rows added."""
    if not src.exists():
        return 0
    if not dest.exists():
        shutil.copy2(src, dest)
        return max(len(src.read_text(encoding="utf-8").splitlines()) - 1, 0)
    have = set(dest.read_text(encoding="utf-8").splitlines())
    new = [ln for ln in src.read_text(encoding="utf-8").splitlines()[1:] if ln and ln not in have]
    if new:
        text = dest.read_text(encoding="utf-8")
        with open(dest, "a", encoding="utf-8", newline="") as fh:
            fh.write(("" if text.endswith("\n") else "\n") + "\n".join(new) + "\n")
    return len(new)


def copy_all(overwrite: bool) -> None:
    """Copy every run from the old lab folder that is not yet in data_measured, and merge the lab
    summary.csv rows into ours (measurements only - the simulations write to summary_sim.csv)."""
    lab = runs.list_runs(runs.LAB_DATA_ROOT)
    copied = sum(do_copy(r, overwrite) for r in lab)
    print(f"{copied} new run(s) copied, {len(lab) - copied} already in {runs.DATA_ROOT}")
    added = merge_summary(runs.LAB_DATA_ROOT / "summary.csv", runs.DATA_ROOT / "summary.csv")
    print(f"summary.csv: {added} new row(s) merged from the lab folder")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------


def interactive(root: Path) -> None:
    while True:
        all_runs = runs.list_runs(root)
        print()
        print_list(all_runs, root)
        choice = input("\nRun number | u = update from lab (copy new runs) | q = quit: ").strip().lower()
        if choice in ("q", ""):
            return
        if choice == "u":
            copy_all(overwrite=False)
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(all_runs):
            print("Not a run number.")
            continue
        run = all_runs[int(choice) - 1]
        print_run(run)
        while True:
            act = input("o = open folder | b = back | q = quit: ").strip().lower()
            if act == "o":
                open_folder(run.path)
            elif act in ("b", ""):
                break
            elif act == "q":
                return


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=list(SOURCES), default="project",
                    help="project = self_heterodyne/data_measured (default), lab = the lab data folder")
    ap.add_argument("--root", type=Path, help="any other data root (overrides --source)")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("list", help="list runs")
    p.add_argument("--delay", type=float, help="only this fiber length [m]")
    p.add_argument("--kind", choices=["all", "lab", "sim"], default="all", help="measurements, simulations or both")
    for name, hlp in (("show", "results and files of a run"), ("open", "open the run folder"),
                      ("copy", "copy lab runs into self_heterodyne/data_measured/")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("delay", type=float, nargs="?", help="fiber length [m]")
        p.add_argument("--run", default="latest", help="'latest' (default), a timestamp or its prefix, or -2 ...")
        if name == "copy":
            p.add_argument("--all", action="store_true", help="copy every lab run not yet in data_measured")
            p.add_argument("--overwrite", action="store_true", help="replace runs that were already copied")
        else:
            p.add_argument("--kind", choices=["all", "lab", "sim"], default="all",
                           help="pick among measurements, simulations or both")
    args = ap.parse_args()
    root = args.root or SOURCES[args.source]

    try:
        if args.cmd is None:
            interactive(root)
        elif args.cmd == "list":
            print_list(runs.list_runs(root, args.delay, args.kind), root)
        elif args.cmd == "copy" and args.all:
            copy_all(args.overwrite)
        else:
            if args.delay is None:
                ap.error(f"{args.cmd}: give a fiber length, e.g. '{args.cmd} 5000'")
            if args.cmd == "copy":
                run = runs.find_run(runs.LAB_DATA_ROOT, args.delay, args.run)
                if not do_copy(run, args.overwrite):
                    print(f"{run.name} is already in {runs.DATA_ROOT} (use --overwrite to replace)")
                return
            run = runs.find_run(root, args.delay, args.run, args.kind)
            if args.cmd == "show":
                print_run(run)
            else:
                open_folder(run.path)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
