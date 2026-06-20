#!/usr/bin/env python3
"""
run_queue.py — Manage and run Quantum ESPRESSO calculations locally.

Scans a run root for directories containing pw.in, determines their status,
and optionally runs them one at a time.

Default behaviour is DRY RUN.  Pass --run to actually launch calculations.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── Status labels ─────────────────────────────────────────────────────────────

ST_NOT_STARTED          = "not_started"
ST_COMPLETED            = "completed"
ST_FAILED_OR_INCOMPLETE = "failed_or_incomplete"
ST_MISSING_PSEUDO       = "missing_pseudo"

# ── CSV columns ───────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "run_path",
    "status_before",
    "action",
    "status_after",
    "return_code",
    "elapsed_seconds",
    "output_path",
    "copied_to_results",
    "notes",
]


# ── Job discovery ─────────────────────────────────────────────────────────────

def find_job_dirs(run_root: Path) -> list:
    """Return all directories under run_root that contain pw.in, sorted."""
    jobs = []
    for dirpath, _dirs, files in os.walk(run_root):
        if "pw.in" in files:
            jobs.append(Path(dirpath))
    return sorted(jobs)


# ── Status detection ──────────────────────────────────────────────────────────

def job_status(job_dir: Path) -> str:
    pw_out = job_dir / "pw.out"
    if not pw_out.exists():
        return ST_NOT_STARTED
    try:
        text = pw_out.read_text(errors="replace")
    except OSError:
        return ST_FAILED_OR_INCOMPLETE
    return ST_COMPLETED if "JOB DONE" in text else ST_FAILED_OR_INCOMPLETE


# ── Pseudopotential validation ────────────────────────────────────────────────

def _resolve_pseudo_dir(pw_in: Path) -> Path:
    """Read pseudo_dir from pw.in and resolve it relative to job directory."""
    try:
        text = pw_in.read_text()
    except OSError:
        return pw_in.parent
    m = re.search(r"pseudo_dir\s*=\s*'([^']+)'", text, re.IGNORECASE)
    raw = m.group(1) if m else "./"
    p = Path(raw)
    if not p.is_absolute():
        p = (pw_in.parent / p).resolve()
    return p


def _upf_names_from_pw_in(pw_in: Path) -> list:
    """
    Extract UPF filenames from ATOMIC_SPECIES lines in pw.in.
    Matches lines of the form:  element  mass  file.UPF
    """
    try:
        text = pw_in.read_text()
    except OSError:
        return []
    return re.findall(
        r"^\s*\S+\s+[\d.]+\s+(\S+\.UPF)\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )


def check_pseudos(job_dir: Path) -> tuple:
    """
    Verify that all UPF files listed in pw.in exist on disk.
    Returns (ok: bool, missing_paths: list[str]).
    """
    pw_in = job_dir / "pw.in"
    pdir  = _resolve_pseudo_dir(pw_in)
    names = _upf_names_from_pw_in(pw_in)
    missing = [str(pdir / n) for n in names if not (pdir / n).exists()]
    return (not missing), missing


def _find_copyable_pseudos(job_dir: Path, pseudo_source: Path) -> tuple:
    """
    For each UPF missing from job_dir's pseudo_dir, look for it in pseudo_source.
    Returns (to_copy: list[(src, dst)], still_missing: list[str]).
    """
    pw_in = job_dir / "pw.in"
    pdir  = _resolve_pseudo_dir(pw_in)
    names = _upf_names_from_pw_in(pw_in)
    to_copy = []
    still_missing = []
    for name in names:
        dst = pdir / name
        if dst.exists():
            continue
        src = pseudo_source / name
        if src.exists():
            to_copy.append((src, dst))
        else:
            still_missing.append(str(dst))
    return to_copy, still_missing


# ── Result copying ────────────────────────────────────────────────────────────

def _is_reference_diamond(job_dir: Path, run_root: Path) -> bool:
    try:
        return "reference_diamond" in job_dir.relative_to(run_root).parts
    except ValueError:
        return False


def _results_dest(job_dir: Path, run_root: Path) -> Path:
    """Map runs/reference_diamond/X → results/reference_diamond/X."""
    rel = job_dir.relative_to(run_root)
    return run_root.parent / "results" / rel


def _slab_results_dest(job_dir: Path, run_root: Path) -> Path:
    """Map runs/<slab_name>/ → results/slabs/<slab_name>/."""
    return run_root.parent / "results" / "slabs" / job_dir.name


def _copy_results(job_dir: Path, dest: Path, force: bool = False) -> tuple:
    """
    Copy pw.in, pw.out, meta.json from job_dir to dest.
    Skips existing files unless force=True.
    Returns (copied: bool, dest_str: str, note: str).
    """
    candidates = ["pw.in", "pw.out", "meta.json"]
    present = [f for f in candidates if (job_dir / f).exists()]
    if not present:
        return False, "", "nothing to copy"
    dest.mkdir(parents=True, exist_ok=True)
    copied_files = []
    skipped_files = []
    for fname in present:
        dst_file = dest / fname
        if dst_file.exists() and not force:
            skipped_files.append(fname)
        else:
            shutil.copy2(job_dir / fname, dst_file)
            copied_files.append(fname)
    parts = []
    if copied_files:
        parts.append(f"copied {copied_files}")
    if skipped_files:
        parts.append(f"skipped existing {skipped_files} (use --force to overwrite)")
    note = "; ".join(parts) + f" → {dest}"
    return bool(copied_files or skipped_files), str(dest), note


def _copy_reference_results(job_dir: Path, run_root: Path, force: bool = False) -> tuple:
    dest = _results_dest(job_dir, run_root)
    return _copy_results(job_dir, dest, force)


def _copy_slab_results(job_dir: Path, run_root: Path, force: bool = False) -> tuple:
    dest = _slab_results_dest(job_dir, run_root)
    return _copy_results(job_dir, dest, force)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_job(job_dir: Path, pw_exe: str) -> tuple:
    """
    Execute:  pw.x < pw.in > pw.out   inside job_dir.
    Stderr is merged into pw.out so all QE output is in one file.
    Returns (returncode: int, elapsed_seconds: float).
    """
    pw_in  = job_dir / "pw.in"
    pw_out = job_dir / "pw.out"
    t0 = time.monotonic()
    with open(pw_in, "rb") as fin, open(pw_out, "wb") as fout:
        result = subprocess.run(
            [pw_exe],
            stdin=fin,
            stdout=fout,
            stderr=subprocess.STDOUT,
            cwd=str(job_dir),
        )
    return result.returncode, time.monotonic() - t0


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Manage and run Quantum ESPRESSO calculations locally.\n"
            "Default is DRY RUN — pass --run to actually execute jobs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_queue.py --dry-run
  python3 run_queue.py --dry-run --only reference_diamond
  python3 run_queue.py --run --only eps_-0.005 --max-jobs 1
  python3 run_queue.py --run --only reference_diamond/hydrostatic
  python3 run_queue.py --run --pw-exe ~/Downloads/qe-7.5/bin/pw.x --max-jobs 1
""",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would run; do not execute anything.",
    )
    mode.add_argument(
        "--run", action="store_true",
        help="Actually launch calculations (required for execution).",
    )

    p.add_argument(
        "--root", default="runs", metavar="DIR",
        help="Root directory to scan for pw.in jobs (default: runs/).",
    )
    p.add_argument(
        "--only", metavar="SUBSTRING",
        help="Only process jobs whose path contains SUBSTRING.",
    )
    p.add_argument(
        "--max-jobs", type=int, default=None, metavar="N",
        help="Stop after running (or dry-run printing) N jobs.",
    )
    p.add_argument(
        "--pw-exe", default="pw.x", metavar="EXE",
        help="Path to the pw.x executable (default: pw.x).",
    )
    p.add_argument(
        "--pseudo-source", metavar="DIR",
        help="Directory containing UPF files; missing pseudos are copied from "
             "here into each job folder before launching (or dry-run reporting).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run already-completed jobs, overwriting pw.out.",
    )
    p.add_argument(
        "--csv-out", default="queue_status.csv", metavar="FILE",
        help="Path for the queue-status CSV (default: queue_status.csv).",
    )
    return p


def main():
    args   = build_parser().parse_args()
    dry_run = args.dry_run

    run_root = Path(args.root).resolve()
    if not run_root.exists():
        print(f"Error: run root '{run_root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # ── Discover and filter jobs ──────────────────────────────────────────
    all_jobs = find_job_dirs(run_root)
    if args.only:
        all_jobs = [j for j in all_jobs if args.only in str(j)]

    # ── Header ───────────────────────────────────────────────────────────
    print(f"Run root  : {run_root}")
    print(f"Jobs found: {len(all_jobs)}" + (f"  (filter: '{args.only}')" if args.only else ""))
    if dry_run:
        print("[DRY RUN — no calculations will be launched]")
    if args.force:
        print("[--force — completed jobs will be re-run]")
    print()

    if not all_jobs:
        print("Nothing to do.")
        sys.exit(0)

    # ── Process each job ──────────────────────────────────────────────────
    csv_rows      = []
    jobs_launched = 0
    n_completed   = 0
    n_failed      = 0
    n_skip_done   = 0
    n_skip_pseudo = 0
    n_skip_limit  = 0

    for job_dir in all_jobs:
        rel           = str(job_dir.relative_to(run_root))
        status_before = job_status(job_dir)

        row = {
            "run_path":          rel,
            "status_before":     status_before,
            "action":            "",
            "status_after":      status_before,
            "return_code":       "",
            "elapsed_seconds":   "",
            "output_path":       str(job_dir / "pw.out") if (job_dir / "pw.out").exists() else "",
            "copied_to_results": "",
            "notes":             "",
        }

        # ── Guard: already done ───────────────────────────────────────────
        if status_before == ST_COMPLETED and not args.force:
            row["action"] = "skip_completed"
            row["notes"]  = "already completed; use --force to re-run"
            n_skip_done += 1
            print(f"  SKIP completed    {rel}")
            csv_rows.append(row)
            continue

        # ── Pseudopotential check (with optional auto-copy) ───────────────
        pseudo_ok, missing_upf = check_pseudos(job_dir)
        pseudo_copy_notes = []

        if not pseudo_ok and args.pseudo_source:
            pseudo_source_path = Path(args.pseudo_source).resolve()
            to_copy, still_missing = _find_copyable_pseudos(job_dir, pseudo_source_path)
            if to_copy:
                if dry_run:
                    for src, dst in to_copy:
                        pseudo_copy_notes.append(f"would copy {src.name}")
                    if not still_missing:
                        pseudo_ok, missing_upf = True, []
                else:
                    pdir = _resolve_pseudo_dir(job_dir / "pw.in")
                    pdir.mkdir(parents=True, exist_ok=True)
                    for src, dst in to_copy:
                        shutil.copy2(src, dst)
                        pseudo_copy_notes.append(f"copied {src.name}")
                    pseudo_ok, missing_upf = check_pseudos(job_dir)

        if not pseudo_ok:
            row["action"]       = "skip_missing_pseudo"
            row["status_after"] = ST_MISSING_PSEUDO
            row["notes"]        = "missing UPF: " + "; ".join(missing_upf)
            n_skip_pseudo += 1
            print(f"  SKIP no-pseudo    {rel}")
            for path in missing_upf:
                print(f"                    missing: {path}")
            csv_rows.append(row)
            continue

        # ── Guard: max-jobs quota ─────────────────────────────────────────
        if args.max_jobs is not None and jobs_launched >= args.max_jobs:
            row["action"] = "skip_max_jobs"
            row["notes"]  = f"--max-jobs {args.max_jobs} reached"
            n_skip_limit += 1
            csv_rows.append(row)
            continue

        # ── Dry run ───────────────────────────────────────────────────────
        if dry_run:
            print(f"  WOULD RUN         {rel}")
            note_parts = list(pseudo_copy_notes)
            note_parts.append(f"would run: {args.pw_exe} < pw.in > pw.out")
            if _is_reference_diamond(job_dir, run_root):
                dest = _results_dest(job_dir, run_root)
                note_parts.append(f"would copy results → {dest}")
            else:
                dest = _slab_results_dest(job_dir, run_root)
                note_parts.append(f"would copy slab results → {dest}")
            for n in pseudo_copy_notes:
                print(f"                    {n}")
            row["action"]       = "would_run"
            row["status_after"] = "?"
            row["notes"]        = "; ".join(note_parts)
            jobs_launched += 1
            csv_rows.append(row)
            continue

        # ── Execute ───────────────────────────────────────────────────────
        print(f"  RUNNING           {rel} ...", end="", flush=True)
        row["action"]      = "run"
        row["output_path"] = str(job_dir / "pw.out")
        if pseudo_copy_notes:
            row["notes"] = "; ".join(pseudo_copy_notes) + "; "

        rc, elapsed = run_job(job_dir, args.pw_exe)
        row["return_code"]    = rc
        row["elapsed_seconds"] = f"{elapsed:.1f}"
        jobs_launched += 1

        status_after      = job_status(job_dir)
        row["status_after"] = status_after

        if status_after == ST_COMPLETED:
            n_completed += 1
            print(f"  done  ({elapsed:.0f}s, rc={rc})")

            # Copy results and re-parse if applicable
            if _is_reference_diamond(job_dir, run_root):
                copied, dest_str, copy_note = _copy_reference_results(
                    job_dir, run_root, force=args.force)
                row["copied_to_results"] = dest_str if copied else ""
                notes = [copy_note]
                if copied:
                    print(f"    → results: {dest_str}")
                    print(f"    → rerunning parse_reference.py ...", end="", flush=True)
                    parse_rc = subprocess.run(
                        [sys.executable, "parse_reference.py"],
                        cwd=str(run_root.parent),
                        capture_output=True,
                    ).returncode
                    if parse_rc == 0:
                        print("  ok")
                    else:
                        print(f"  FAILED (rc={parse_rc})")
                        notes.append(f"parse_reference.py exited {parse_rc}")
                row["notes"] = "; ".join(notes)
            else:
                copied, dest_str, copy_note = _copy_slab_results(
                    job_dir, run_root, force=args.force)
                row["copied_to_results"] = dest_str if copied else ""
                if dest_str:
                    print(f"    → slab results: {dest_str}")
                row["notes"] = (row.get("notes") or "") + copy_note
        else:
            n_failed += 1
            row["notes"] = f"job did not complete (rc={rc})"
            print(f"  FAILED  (rc={rc}, {elapsed:.0f}s)")
            print(f"    check: {job_dir / 'pw.out'}")

        csv_rows.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────────
    csv_path = Path(args.csv_out)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("─" * 55)
    if dry_run:
        print(f"  Would run     : {jobs_launched}")
    else:
        print(f"  Ran           : {jobs_launched}")
        print(f"  Completed     : {n_completed}")
        print(f"  Failed        : {n_failed}")
    print(f"  Skip completed : {n_skip_done}")
    print(f"  Skip no-pseudo : {n_skip_pseudo}")
    if n_skip_limit:
        print(f"  Skip (limit)  : {n_skip_limit}")
    print(f"  CSV written    : {csv_path}")
    print("─" * 55)


if __name__ == "__main__":
    main()
