#!/usr/bin/env python3
"""
make_slab_strain_series.py - Generate in-plane strained slab SCF jobs.

Starts from stress-SCF slab inputs, applies homogeneous in-plane strain to the
first two lattice vectors, transforms atomic x/y coordinates consistently, and
writes stress-enabled SCF QE inputs for surface-stress fitting.

Default strain mode is biaxial:
  A1 -> (1 + eps) A1
  A2 -> (1 + eps) A2
  z/vacuum unchanged
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path

_NUM = r"[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?"


DEFAULT_SLABS = [
    "C100_2x1_H_6L_stress_scf",
    "C110_1x1_H_6L_SSSP_stress_scf",
    "C111_1x1_H_6L_stress_scf",
]

DEFAULT_STRAINS = [-0.010, -0.005, 0.000, 0.005, 0.010]


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


def parse_cell(block):
    lines = [ln for ln in block.splitlines() if ln.strip()]
    header = lines[0]
    rows = []
    for ln in lines[1:4]:
        vals = [float(x) for x in ln.split()[:3]]
        rows.append(vals)
    return header, rows


def format_cell(header, rows):
    out = [header]
    for r in rows:
        out.append(f"  {r[0]:16.10f} {r[1]:16.10f} {r[2]:16.10f}")
    return "\n".join(out) + "\n"


def parse_positions(block):
    lines = [ln for ln in block.splitlines() if ln.strip()]
    header = lines[0]
    atoms = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 4:
            continue
        atoms.append({
            "species": parts[0],
            "x": float(parts[1]),
            "y": float(parts[2]),
            "z": float(parts[3]),
            "flags": parts[4:],
        })
    return header, atoms


def format_positions(header, atoms):
    out = [header]
    for a in atoms:
        flags = (" " + " ".join(a["flags"])) if a["flags"] else ""
        out.append(f"  {a['species']:<2s} {a['x']:16.10f} {a['y']:16.10f} {a['z']:16.10f}{flags}")
    return "\n".join(out) + "\n"


def inv2(m):
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if abs(det) < 1e-12:
        raise ValueError("Cannot invert in-plane cell matrix")
    return [
        [ m[1][1] / det, -m[0][1] / det],
        [-m[1][0] / det,  m[0][0] / det],
    ]


def matvec2(m, v):
    return [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]


def scale_for_mode(eps, mode):
    if mode == "biaxial":
        return 1.0 + eps, 1.0 + eps
    if mode == "x":
        return 1.0 + eps, 1.0
    if mode == "y":
        return 1.0, 1.0 + eps
    raise ValueError(f"Unknown strain mode: {mode}")


def strain_cell_and_atoms(cell, atoms, eps, mode):
    sx, sy = scale_for_mode(eps, mode)

    new_cell = [
        [cell[0][0] * sx, cell[0][1] * sx, cell[0][2] * sx],
        [cell[1][0] * sy, cell[1][1] * sy, cell[1][2] * sy],
        list(cell[2]),
    ]

    # In-plane basis matrix maps fractional [f1, f2] to Cartesian [x, y].
    old_basis = [
        [cell[0][0], cell[1][0]],
        [cell[0][1], cell[1][1]],
    ]
    new_basis = [
        [new_cell[0][0], new_cell[1][0]],
        [new_cell[0][1], new_cell[1][1]],
    ]
    old_inv = inv2(old_basis)

    new_atoms = []
    for atom in atoms:
        frac = matvec2(old_inv, [atom["x"], atom["y"]])
        xy = matvec2(new_basis, frac)
        b = dict(atom)
        b["x"], b["y"] = xy[0], xy[1]
        new_atoms.append(b)

    return new_cell, new_atoms


def strain_tag(eps):
    sign = "p" if eps >= 0 else "m"
    val = abs(eps)
    return f"{sign}{val:.3f}".replace(".", "p")


def patch_control(control):
    control = re.sub(r"calculation\s*=\s*'[^']+'", "calculation   = 'scf'", control, flags=re.IGNORECASE)
    if "tstress" not in control.lower():
        control = control.replace("&CONTROL\n", "&CONTROL\n  tstress       = .true.\n", 1)
    if "tprnfor" not in control.lower():
        control = control.replace("&CONTROL\n", "&CONTROL\n  tprnfor       = .true.\n", 1)
    return control


def base_name_from_stress_name(name):
    return name.removesuffix("_stress_scf")


def make_one(src_dir, dest_dir, eps, mode):
    text = read(src_dir / "pw.in")

    control = extract_block(text, r"&CONTROL", [r"\n&SYSTEM"])
    system = extract_block(text, r"&SYSTEM", [r"\n&ELECTRONS"])
    electrons = extract_block(text, r"&ELECTRONS", [r"\n&IONS", r"\n&CELL", r"\nATOMIC_SPECIES"])
    species = extract_block(text, r"ATOMIC_SPECIES", [r"\nCELL_PARAMETERS", r"\nATOMIC_POSITIONS", r"\nK_POINTS"])
    cell_block = extract_block(text, r"CELL_PARAMETERS", [r"\nATOMIC_POSITIONS", r"\nK_POINTS"])
    pos_block = extract_block(text, r"ATOMIC_POSITIONS", [r"\nK_POINTS"])
    kpoints = extract_block(text, r"K_POINTS", [r"\n[A-Z_]+"])

    if not all([control, system, electrons, species, cell_block, pos_block, kpoints]):
        raise ValueError(f"Missing required QE block in {src_dir}")

    cell_header, cell = parse_cell(cell_block)
    pos_header, atoms = parse_positions(pos_block)

    new_cell, new_atoms = strain_cell_and_atoms(cell, atoms, eps, mode)
    control = patch_control(control)

    dest_dir.mkdir(parents=True, exist_ok=True)
    pw_in = dest_dir / "pw.in"
    pw_in.write_text(
        f"! Strained slab SCF generated from {src_dir}\n"
        f"! strain_mode = {mode}; epsilon = {eps:+.6f}\n"
        f"! In-plane lattice/coordinates strained; vacuum direction unchanged.\n"
        f"{control}\n"
        f"{system}\n"
        f"{electrons}\n"
        f"{species}\n"
        f"{format_cell(cell_header, new_cell)}\n"
        f"{format_positions(pos_header, new_atoms)}\n"
        f"{kpoints}"
    )

    meta = {
        "source": str(src_dir),
        "strain_mode": mode,
        "epsilon": eps,
        "calculation": "scf",
        "purpose": "in-plane slab strain series for surface-stress fitting",
    }
    (dest_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    for pseudo in re.findall(r"^\s*\S+\s+\S+\s+(\S+)", species, re.MULTILINE):
        src_pseudo = src_dir / pseudo
        if src_pseudo.exists():
            shutil.copy2(src_pseudo, dest_dir / pseudo)

    return pw_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--mode", choices=["biaxial", "x", "y"], default="biaxial")
    ap.add_argument("--strain", type=float, action="append", default=None,
                    help="Add one strain value, e.g. --strain -0.005. May be repeated.")
    ap.add_argument("--only", action="append", default=None,
                    help="Stress-SCF source run name. May be repeated.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    slabs = args.only or DEFAULT_SLABS
    strains = args.strain or DEFAULT_STRAINS

    for slab in slabs:
        src = runs_root / slab
        if not src.exists():
            raise FileNotFoundError(src)
        base = base_name_from_stress_name(slab)
        for eps in strains:
            tag = strain_tag(eps)
            dest = runs_root / f"{base}_strain_{args.mode}_{tag}_scf"
            pw_in = make_one(src, dest, eps, args.mode)
            print(f"wrote {pw_in}")


if __name__ == "__main__":
    main()
