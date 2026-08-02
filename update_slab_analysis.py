"""
Wrapper that runs the slab-analysis pipeline in order:
  1. parse_slab.py
  2. analyze_slab_stress.py
  3. fit_slab_strain.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

EXPECTED_OUTPUTS = [
    "results/slabs/slab_summary.csv",
    "results/slabs/slab_summary.json",
    "results/slabs/slab_summary.md",
    "results/slabs/slab_stress_summary.csv",
    "results/slabs/slab_stress_summary.json",
    "results/slabs/slab_stress_summary.md",
    "results/slabs/slab_strain_fit_summary.csv",
    "results/slabs/slab_strain_fit_summary.json",
    "results/slabs/slab_strain_fit_summary.md",
]


def run_step(label: str, cmd: list[str], continue_on_error: bool) -> bool:
    """Run one pipeline step. Returns True on success, False on failure."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  cmd: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[FAIL] {label} exited with code {result.returncode}.")
        if not continue_on_error:
            sys.exit(result.returncode)
        return False

    print(f"\n[OK] {label} completed successfully.")
    return True


def print_summary(statuses: dict[str, bool]) -> None:
    print(f"\n{'='*60}")
    print("  Pipeline summary")
    print(f"{'='*60}")

    for step, ok in statuses.items():
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {step}")

    print(f"\n  Expected output files:")
    here = Path(".")
    missing = []
    for path_str in EXPECTED_OUTPUTS:
        p = here / path_str
        exists = p.exists()
        mark = "found  " if exists else "MISSING"
        print(f"  [{mark}]  {path_str}")
        if not exists:
            missing.append(path_str)

    if missing:
        print(f"\n  WARNING: {len(missing)} expected file(s) not found.")
    else:
        print("\n  All expected output files are present.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the slab-analysis pipeline (parse → stress → strain fit)."
    )
    parser.add_argument("--skip-parse", action="store_true", help="Skip parse_slab.py")
    parser.add_argument(
        "--skip-stress", action="store_true", help="Skip analyze_slab_stress.py"
    )
    parser.add_argument(
        "--skip-strain-fit", action="store_true", help="Skip fit_slab_strain.py"
    )
    parser.add_argument(
        "--run-nv", action="store_true",
        help="Also run nv_spin_strain.py (exact NV Hamiltonian; Python-only, "
             "not run by default)",
    )
    parser.add_argument(
        "--reference-config", default="config/reference_pbe_sssp.json",
        help="Bulk/elastic reference config passed to nv_spin_strain.py "
             "when --run-nv is given",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to subsequent steps even if one fails",
    )
    parser.add_argument(
        "--python",
        default="python3",
        metavar="EXE",
        help="Python executable to use (default: python3)",
    )
    args = parser.parse_args()

    steps = [
        ("parse_slab.py",         not args.skip_parse,       "parse_slab.py"),
        ("analyze_slab_stress.py", not args.skip_stress,     "analyze_slab_stress.py"),
        ("fit_slab_strain.py",     not args.skip_strain_fit, "fit_slab_strain.py"),
    ]
    if args.run_nv:
        steps.append((
            "nv_spin_strain.py", True,
            ["nv_spin_strain.py", "--reference-config", args.reference_config],
        ))

    statuses: dict[str, bool] = {}

    for label, enabled, script in steps:
        if not enabled:
            print(f"\n[SKIP] {label}")
            statuses[label] = True
            continue
        cmd = [args.python] + (script if isinstance(script, list) else [script])
        ok = run_step(label, cmd, args.continue_on_error)
        statuses[label] = ok

    print_summary(statuses)

    if not all(statuses.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
