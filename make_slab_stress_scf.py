#!/usr/bin/env python3
"""
make_slab_stress_scf.py - Generate stress-only SCF jobs from relaxed slabs.

Reads relaxed slab results from results/slabs/<name>/pw.in and pw.out, extracts
the final coordinates from pw.out, and writes new QE single-point inputs with:

  calculation = 'scf'
  tstress = .true.
  tprnfor = .true.

These jobs are for vacuum-corrected slab stress / lattice-pressure analysis.
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
        block = m.group(1).strip()
        lines = []
        for line in block.splitlines():
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
    cell = extract_block(in_text, r"CELL_PARAMETERS", [r"\nATOMIC_POSITIONS", r"\nK_POINTS"])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/slabs")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--only", action="append", default=[
        "C100_2x1_H_6L",
        "C110_1x1_H_6L_SSSP",
        "C111_1x1_H_6L",
    ])
    args = ap.parse_args()

    results_root = Path(args.results_root)
    runs_root = Path(args.runs_root)

    for name in args.only:
        src = results_root / name
        dest = runs_root / f"{name}_stress_scf"
        pw_in = make_one(src, dest)
        print(f"wrote {pw_in}")


if __name__ == "__main__":
    main()
