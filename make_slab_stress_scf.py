#!/usr/bin/env python3
"""
make_slab_stress_scf.py - Generate stress-only SCF jobs from relaxed slabs.

Reads relaxed slab results from <results-root>/<name>/pw.in and pw.out, extracts
the final coordinates from pw.out, and writes new QE single-point inputs with
calculation='scf', tstress=.true., and tprnfor=.true.

The cell comes from pw.out when pw.out contains one
--------------------------------------------------
QE writes a "Begin final coordinates" block at the end of a relaxation. For a
fixed-cell `relax` that block holds ATOMIC_POSITIONS only; for a `vc-relax` it
also holds CELL_PARAMETERS, because the cell itself moved.

This script used to take CELL_PARAMETERS from the *source pw.in* in every case.
For a fixed-cell relax that is correct, since the cell never changed. For a
vc-relax it is silently wrong: it pairs the relaxed positions with the
*unrelaxed* cell, producing a structure that was never a stationary point of
anything, and a stress tensor that is meaningless. Nothing crashes and the
numbers look ordinary — the failure mode CLAUDE.md section 0 is about.

Now the final cell is taken from pw.out whenever pw.out provides one (invariant
7: pw.out is ground truth for what the code actually did), the pw.in cell is
used only when it does not, and a vc-relax source whose pw.out has no final cell
is a hard error rather than a silent fallback.
"""

import argparse
import re
import shutil
from pathlib import Path

_NUM = r"[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?"


def read(path):
    return Path(path).read_text()


def extract_block(text, start_pat, stop_pats):
    m = re.search(start_pat, text, re.IGNORECASE)
    if not m:
        return None
    start = m.start()
    rest = text[start:]
    stops = []
    for pat in stop_pats:
        sm = re.search(pat, rest[m.end() - start:], re.IGNORECASE)
        if sm:
            stops.append((m.end() - start) + sm.start())
    end = min(stops) if stops else len(rest)
    return rest[:end].rstrip() + "\n"


def extract_final_positions(out_text):
    m = re.search(
        r"Begin final coordinates.*?(ATOMIC_POSITIONS\s*\([^)]+\)\n.*?)(?:End final coordinates)",
        out_text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        lines = []
        for line in m.group(1).strip().splitlines():
            if line.strip() and not line.lstrip().startswith(("End", "CELL_PARAMETERS")):
                lines.append(line.rstrip())
        return "\n".join(lines) + "\n"

    matches = list(re.finditer(
        r"ATOMIC_POSITIONS\s*\([^)]+\)\n"
        r"(?:\s*[A-Za-z][A-Za-z0-9_+-]*\s+" + _NUM + r"\s+" + _NUM + r"\s+" + _NUM + r"(?:\s+[01]\s+[01]\s+[01])?\s*\n)+",
        out_text,
        re.IGNORECASE,
    ))
    if not matches:
        raise ValueError("Could not find final ATOMIC_POSITIONS block in pw.out")
    return matches[-1].group(0).rstrip() + "\n"


def extract_calculation(in_text):
    """The `calculation` string declared in the source &CONTROL, lowercased."""
    m = re.search(r"calculation\s*=\s*'([^']+)'", in_text, re.IGNORECASE)
    return m.group(1).strip().lower() if m else None


# Calculations in which the cell is a degree of freedom, so the input cell is
# not the cell the run finished at.
VARIABLE_CELL = {"vc-relax", "vc-md"}


def extract_final_cell(out_text):
    """
    CELL_PARAMETERS from pw.out's "Begin final coordinates" block, or None.

    Only variable-cell runs write one. Returning None for a fixed-cell relax is
    the correct, expected result, not a parse failure.
    """
    m = re.search(
        r"Begin final coordinates(.*?)End final coordinates",
        out_text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    block = m.group(1)
    cm = re.search(
        r"(CELL_PARAMETERS\s*\([^)]+\)\n"
        r"(?:\s*" + _NUM + r"\s+" + _NUM + r"\s+" + _NUM + r"\s*\n){3})",
        block,
        re.IGNORECASE,
    )
    return cm.group(1).rstrip() + "\n" if cm else None


def resolve_cell(in_text, out_text, src_dir):
    """
    Return (cell_block, provenance_string).

    pw.out wins whenever it supplies a cell. A variable-cell source whose pw.out
    supplies none is an error: falling back to the input cell there would pair
    relaxed coordinates with an unrelaxed cell.
    """
    calculation = extract_calculation(in_text)
    final_cell = extract_final_cell(out_text)

    if final_cell is not None:
        return final_cell, f"pw.out final coordinates (source calculation={calculation})"

    if calculation in VARIABLE_CELL:
        raise ValueError(
            f"{src_dir}: source calculation is {calculation!r}, so the cell "
            "relaxed, but pw.out has no CELL_PARAMETERS in its final "
            "coordinates block. Refusing to fall back to the pw.in cell, which "
            "would pair relaxed positions with an unrelaxed cell. Check whether "
            "the run finished."
        )

    cell = extract_block(in_text, r"CELL_PARAMETERS", [r"\nATOMIC_POSITIONS", r"\nK_POINTS"])
    return cell, f"pw.in (fixed-cell source calculation={calculation})"


def patch_control(control):
    control = re.sub(r"calculation\s*=\s*'[^']+'", "calculation   = 'scf'", control, flags=re.IGNORECASE)
    if re.search(r"\btstress\s*=", control, re.IGNORECASE):
        control = re.sub(r"\btstress\s*=\s*\.?\w+\.?", "tstress       = .true.", control, flags=re.IGNORECASE)
    else:
        control = control.replace("&CONTROL\n", "&CONTROL\n  tstress       = .true.\n", 1)
    if re.search(r"\btprnfor\s*=", control, re.IGNORECASE):
        control = re.sub(r"\btprnfor\s*=\s*\.?\w+\.?", "tprnfor       = .true.", control, flags=re.IGNORECASE)
    else:
        control = control.replace("&CONTROL\n", "&CONTROL\n  tprnfor       = .true.\n", 1)
    return control


def make_one(src_dir, dest_dir):
    in_text = read(src_dir / "pw.in")
    out_text = read(src_dir / "pw.out")

    control = extract_block(in_text, r"&CONTROL", [r"\n&SYSTEM"])
    system = extract_block(in_text, r"&SYSTEM", [r"\n&ELECTRONS"])
    electrons = extract_block(in_text, r"&ELECTRONS", [r"\n&IONS", r"\n&CELL", r"\nATOMIC_SPECIES"])
    species = extract_block(in_text, r"ATOMIC_SPECIES", [r"\nCELL_PARAMETERS", r"\nATOMIC_POSITIONS", r"\nK_POINTS"])
    cell, cell_source = resolve_cell(in_text, out_text, src_dir)
    kpoints = extract_block(in_text, r"K_POINTS", [r"\n[A-Z_]+"])

    if not all([control, system, electrons, species, cell, kpoints]):
        raise ValueError(f"Missing required QE block in {src_dir}")

    final_positions = extract_final_positions(out_text)
    control = patch_control(control)

    dest_dir.mkdir(parents=True, exist_ok=True)
    pw_in = dest_dir / "pw.in"
    pw_in.write_text(
        f"! Stress-only SCF generated from {src_dir}\n"
        f"! Uses final relaxed coordinates; no ionic relaxation.\n"
        f"! Cell taken from: {cell_source}\n"
        f"{control}\n"
        f"{system}\n"
        f"{electrons}\n"
        f"{species}\n"
        f"{cell}\n"
        f"{final_positions}\n"
        f"{kpoints}"
    )

    for pseudo in re.findall(r"^\s*\S+\s+\S+\s+(\S+)", species, re.MULTILINE):
        src_pseudo = src_dir / pseudo
        if src_pseudo.exists():
            shutil.copy2(src_pseudo, dest_dir / pseudo)

    return pw_in


RELAXATION_CALCULATIONS = {"relax", "vc-relax", "md", "vc-md"}


def discover_relaxations(results_root):
    """
    Every directory under results_root that holds a completed relaxation.

    Selection is by what the input declares, not by folder name: a directory
    qualifies if it has both pw.in and pw.out and its pw.in declares a
    calculation in which the ions move. That skips the `*_stress_scf` outputs
    this script itself produces, without pattern-matching their names.

    Replaces a hardcoded list of three run names from the original 6L campaign,
    which made the script unusable without --only on any later campaign.
    """
    if not results_root.is_dir():
        raise FileNotFoundError(f"results root not found: {results_root}")

    found = []
    for d in sorted(p for p in results_root.iterdir() if p.is_dir()):
        pw_in, pw_out = d / "pw.in", d / "pw.out"
        if not (pw_in.exists() and pw_out.exists()):
            continue
        if extract_calculation(pw_in.read_text()) in RELAXATION_CALCULATIONS:
            found.append(d.name)
    return found


def resolve_names(results_root, patterns):
    """Resolve --only patterns to actual result-folder names.

    A pattern that names an existing directory is used as-is. Otherwise it is
    treated as a substring filter against directories in results_root (e.g.
    "_sym" selects every folder whose name contains "_sym").
    """
    resolved = []
    for pattern in patterns:
        if (results_root / pattern).is_dir():
            resolved.append(pattern)
            continue
        matches = sorted(
            p.name for p in results_root.iterdir() if p.is_dir() and pattern in p.name
        )
        if not matches:
            raise FileNotFoundError(
                f"--only {pattern!r} did not match any directory in {results_root}"
            )
        resolved.extend(matches)
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for name in resolved:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/slabs")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--only", action="append", default=None)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    runs_root = Path(args.runs_root)

    if args.only is not None:
        names = resolve_names(results_root, args.only)
    else:
        names = discover_relaxations(results_root)
        if not names:
            raise SystemExit(
                f"No completed relaxations found under {results_root} "
                "(need pw.in + pw.out with a relaxing calculation)."
            )
        print(f"discovered {len(names)} relaxation(s) under {results_root}")

    for name in names:
        src = results_root / name
        dest = runs_root / f"{name}_stress_scf"
        pw_in = make_one(src, dest)
        print(f"wrote {pw_in}")


if __name__ == "__main__":
    main()
