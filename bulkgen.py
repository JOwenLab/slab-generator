"""
bulkgen.py — generate Quantum ESPRESSO pw.x input files for perfect periodic
diamond (bulk reference for surface-energy, stress, strain, Raman, and
NV-center analyses).

Does NOT run DFT calculations; only writes input files and metadata.

Usage
-----
  python3 bulkgen.py --help
  python3 bulkgen.py
  python3 bulkgen.py --pseudo-dir /path/to/SSSP --pseudo-C C.pbe-n-kjpaw_psl.1.0.0.UPF
  python3 bulkgen.py --a0 3.567 --ecutwfc 100 --ecutrho 800 --kpoints 10 10 10
  python3 bulkgen.py --cell primitive --kpoints 12 12 12
  python3 bulkgen.py --strain-values -0.02 -0.01 0.0 0.01 0.02

Output tree
-----------
  runs/reference_diamond/
  ├── relax/
  │   ├── pw.in       (vc-relax, full structural optimisation)
  │   └── meta.json
  └── hydrostatic/
      ├── eps_-0.010/
      │   ├── pw.in   (scf, cell scaled by a0*(1+eps))
      │   └── meta.json
      ├── eps_-0.005/
      ├── eps_+0.000/
      ├── eps_+0.005/
      └── eps_+0.010/
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Diamond crystal geometry
# ---------------------------------------------------------------------------

# Conventional cubic cell (Fd-3m, 8 atoms) — fractional coordinates
_CONV_FRAC = np.array([
    [0.00, 0.00, 0.00],
    [0.25, 0.25, 0.25],
    [0.50, 0.50, 0.00],
    [0.75, 0.75, 0.25],
    [0.50, 0.00, 0.50],
    [0.75, 0.25, 0.75],
    [0.00, 0.50, 0.50],
    [0.25, 0.75, 0.75],
])

# FCC primitive cell (2 atoms) — fractional in the FCC primitive basis
_PRIM_FRAC = np.array([
    [0.00, 0.00, 0.00],
    [0.25, 0.25, 0.25],
])


def _cell_vectors(a0, cell_type):
    """Return (A: 3×3 lattice matrix in Å, frac: N×3 fractional coords)."""
    if cell_type == "conventional":
        A = np.eye(3) * a0
        frac = _CONV_FRAC.copy()
    else:  # primitive FCC: a/2 * {[0,1,1],[1,0,1],[1,1,0]}
        h = a0 / 2.0
        A = np.array([
            [0.0,  h,   h  ],
            [h,    0.0, h  ],
            [h,    h,   0.0],
        ])
        frac = _PRIM_FRAC.copy()
    return A, frac


# ---------------------------------------------------------------------------
# QE input generators
# ---------------------------------------------------------------------------

def _fmt_cell(A):
    lines = []
    for v in A:
        v = v.copy()
        v[np.abs(v) < 1e-12] = 0.0
        lines.append(f"  {v[0]:14.8f} {v[1]:14.8f} {v[2]:14.8f}")
    return lines


def _atomic_positions(frac):
    return [f"  C  {f[0]:.8f}  {f[1]:.8f}  {f[2]:.8f}" for f in frac]


def write_relax_input(a0, cell_type, functional, pseudo_dir, pseudo_C,
                      ecutwfc, ecutrho, kpoints):
    """Return QE pw.x vc-relax input string for unperturbed diamond."""
    A, frac = _cell_vectors(a0, cell_type)
    nk1, nk2, nk3 = kpoints
    lines = [
        f"! bulk diamond vc-relax — bulkgen.py",
        f"! cell={cell_type}  a0={a0:.6f} Ang  functional={functional}",
        f"! verify pseudo, cutoffs, and k-mesh against your convergence tests",
        "&CONTROL",
        "  calculation   = 'vc-relax'",
        "  prefix        = 'diamond'",
        f"  pseudo_dir    = '{pseudo_dir}'",
        "  outdir        = './tmp'",
        "  forc_conv_thr = 1.0d-5",
        "  etot_conv_thr = 1.0d-6",
        "  tstress       = .true.",
        "  tprnfor       = .true.",
        "/",
        "&SYSTEM",
        "  ibrav    = 0",
        f"  nat      = {len(frac)}",
        "  ntyp     = 1",
        f"  ecutwfc  = {ecutwfc:.1f}",
        f"  ecutrho  = {ecutrho:.1f}",
        "  occupations = 'fixed'",
        "/",
        "&ELECTRONS",
        "  conv_thr    = 1.0d-10",
        "  mixing_beta = 0.4",
        "/",
        "&IONS",
        "  ion_dynamics = 'bfgs'",
        "/",
        "&CELL",
        "  cell_dynamics  = 'bfgs'",
        "  press_conv_thr = 0.1",
        "/",
        "ATOMIC_SPECIES",
        f"  C  12.011  {pseudo_C}",
        "CELL_PARAMETERS angstrom",
    ]
    lines += _fmt_cell(A)
    lines += ["ATOMIC_POSITIONS crystal"] + _atomic_positions(frac)
    lines += [f"K_POINTS automatic", f"  {nk1} {nk2} {nk3}  0 0 0"]
    return "\n".join(lines) + "\n"


def write_scf_input(a0, eps, cell_type, functional, pseudo_dir, pseudo_C,
                    ecutwfc, ecutrho, kpoints):
    """Return QE pw.x scf input string for a hydrostatically strained cell.

    Lattice is scaled by a = a0*(1+eps); fractional coordinates are unchanged.
    """
    a = a0 * (1.0 + eps)
    A, frac = _cell_vectors(a, cell_type)
    nk1, nk2, nk3 = kpoints
    eps_str = f"{'+' if eps >= 0 else ''}{eps:.4f}"
    lines = [
        f"! bulk diamond SCF  eps={eps_str} — bulkgen.py",
        f"! a = a0*(1+eps) = {a:.8f} Ang  functional={functional}",
        f"! verify pseudo, cutoffs, and k-mesh against your convergence tests",
        "&CONTROL",
        "  calculation   = 'scf'",
        "  prefix        = 'diamond'",
        f"  pseudo_dir    = '{pseudo_dir}'",
        "  outdir        = './tmp'",
        "  tstress       = .true.",
        "  tprnfor       = .true.",
        "/",
        "&SYSTEM",
        "  ibrav    = 0",
        f"  nat      = {len(frac)}",
        "  ntyp     = 1",
        f"  ecutwfc  = {ecutwfc:.1f}",
        f"  ecutrho  = {ecutrho:.1f}",
        "  occupations = 'fixed'",
        "/",
        "&ELECTRONS",
        "  conv_thr    = 1.0d-10",
        "  mixing_beta = 0.4",
        "/",
        "ATOMIC_SPECIES",
        f"  C  12.011  {pseudo_C}",
        "CELL_PARAMETERS angstrom",
    ]
    lines += _fmt_cell(A)
    lines += ["ATOMIC_POSITIONS crystal"] + _atomic_positions(frac)
    lines += [f"K_POINTS automatic", f"  {nk1} {nk2} {nk3}  0 0 0"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def make_metadata(calc_type, strain_type, eps, a_used, cell_type,
                  functional, pseudo_C, pseudo_dir, ecutwfc, ecutrho,
                  kpoints):
    return {
        "calculation": calc_type,
        "strain_type": strain_type,
        "epsilon": round(float(eps), 6),
        "lattice_constant_angstrom": round(float(a_used), 8),
        "functional": functional,
        "pseudopotential": pseudo_C,
        "pseudo_dir": pseudo_dir,
        "ecutwfc_Ry": ecutwfc,
        "ecutrho_Ry": ecutrho,
        "kpoints": list(kpoints),
        "cell_type": cell_type,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)


def _eps_label(eps):
    """Format a strain value as the canonical folder name: eps_+0.010."""
    sign = "+" if eps >= 0.0 else ""
    return f"eps_{sign}{eps:.3f}"


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _sanity(args):
    errors = []
    for eps in args.strain_values:
        if abs(eps) > 0.15:
            errors.append(
                f"strain value {eps:.4f} is unreasonably large (|eps| > 0.15); "
                "diamond would be non-physical at this strain")
    if args.ecutrho <= args.ecutwfc:
        errors.append(
            f"ecutrho ({args.ecutrho}) must be strictly greater than "
            f"ecutwfc ({args.ecutwfc})")
    if not args.pseudo_C.upper().endswith(".UPF"):
        errors.append(
            f"--pseudo-C must end in .UPF, got: '{args.pseudo_C}'")
    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        prog="bulkgen.py",
        description=(
            "Generate Quantum ESPRESSO pw.x inputs for perfect periodic diamond.\n"
            "Writes one vc-relax input and a hydrostatic strain series (SCF).\n"
            "No DFT calculations are run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  # Defaults (PBE, SSSP pseudo, 8x8x8 k-mesh, conventional cell)
  python3 bulkgen.py

  # Custom pseudo directory
  python3 bulkgen.py --pseudo-dir /path/to/SSSP \\
                     --pseudo-C C.pbe-n-kjpaw_psl.1.0.0.UPF

  # Tighter convergence settings
  python3 bulkgen.py --ecutwfc 100 --ecutrho 800 --kpoints 10 10 10

  # Primitive cell with denser k-mesh
  python3 bulkgen.py --cell primitive --kpoints 12 12 12

  # Custom strain range
  python3 bulkgen.py --strain-values -0.02 -0.01 0.0 0.01 0.02

  # Everything to a custom output tree
  python3 bulkgen.py --outdir /scratch/diamond_ref --a0 3.5670

Output tree
-----------
  runs/reference_diamond/
  ├── relax/
  │   ├── pw.in       (vc-relax — full structural optimisation)
  │   └── meta.json
  └── hydrostatic/
      ├── eps_-0.010/
      │   ├── pw.in   (scf, a = a0*(1+eps))
      │   └── meta.json
      ├── eps_-0.005/
      ├── eps_+0.000/
      ├── eps_+0.005/
      └── eps_+0.010/
""",
    )

    ap.add_argument(
        "--a0", type=float, default=3.567,
        help="cubic lattice constant in Å (default: 3.567)")
    ap.add_argument(
        "--cell", choices=["conventional", "primitive"], default="conventional",
        help="unit cell type: conventional (8 atoms) or primitive (2 atoms) "
             "(default: conventional)")
    ap.add_argument(
        "--functional", default="PBE",
        help="XC functional label for metadata only (default: PBE)")
    ap.add_argument(
        "--pseudo-dir", default="./",
        help="pseudo_dir path written into the QE input (default: ./)")
    ap.add_argument(
        "--pseudo-C", default="C.pbe-n-kjpaw_psl.1.0.0.UPF",
        help="carbon pseudopotential filename, must end in .UPF "
             "(default: C.pbe-n-kjpaw_psl.1.0.0.UPF)")
    ap.add_argument(
        "--ecutwfc", type=float, default=80.0,
        help="kinetic energy cutoff in Ry (default: 80)")
    ap.add_argument(
        "--ecutrho", type=float, default=640.0,
        help="charge density cutoff in Ry (default: 640)")
    ap.add_argument(
        "--kpoints", type=int, nargs=3, default=[8, 8, 8],
        metavar=("NK1", "NK2", "NK3"),
        help="Monkhorst-Pack mesh (default: 8 8 8)")
    ap.add_argument(
        "--outdir", default="runs/reference_diamond",
        help="root output directory (default: runs/reference_diamond)")
    ap.add_argument(
        "--strain-values", type=float, nargs="+",
        default=[-0.010, -0.005, 0.000, 0.005, 0.010],
        metavar="EPS",
        help="hydrostatic strain values eps where a=a0*(1+eps) "
             "(default: -0.010 -0.005 0.000 0.005 0.010)")

    args = ap.parse_args()
    _sanity(args)

    generated = []

    # ------------------------------------------------------------------
    # 1.  vc-relax input
    # ------------------------------------------------------------------
    relax_dir = os.path.join(args.outdir, "relax")
    pw_path   = os.path.join(relax_dir, "pw.in")
    meta_path = os.path.join(relax_dir, "meta.json")

    _write(pw_path, write_relax_input(
        args.a0, args.cell, args.functional, args.pseudo_dir, args.pseudo_C,
        args.ecutwfc, args.ecutrho, args.kpoints))
    generated.append(pw_path)

    _write(meta_path, json.dumps(make_metadata(
        calc_type="vc-relax", strain_type="none", eps=0.0,
        a_used=args.a0, cell_type=args.cell,
        functional=args.functional, pseudo_C=args.pseudo_C,
        pseudo_dir=args.pseudo_dir, ecutwfc=args.ecutwfc,
        ecutrho=args.ecutrho, kpoints=args.kpoints,
    ), indent=2) + "\n")
    generated.append(meta_path)

    # ------------------------------------------------------------------
    # 2.  Hydrostatic strain series (SCF, fixed cell)
    # ------------------------------------------------------------------
    for eps in args.strain_values:
        label      = _eps_label(eps)
        strain_dir = os.path.join(args.outdir, "hydrostatic", label)
        pw_path    = os.path.join(strain_dir, "pw.in")
        meta_path  = os.path.join(strain_dir, "meta.json")

        _write(pw_path, write_scf_input(
            args.a0, eps, args.cell, args.functional, args.pseudo_dir,
            args.pseudo_C, args.ecutwfc, args.ecutrho, args.kpoints))
        generated.append(pw_path)

        _write(meta_path, json.dumps(make_metadata(
            calc_type="scf", strain_type="hydrostatic", eps=eps,
            a_used=args.a0 * (1.0 + eps), cell_type=args.cell,
            functional=args.functional, pseudo_C=args.pseudo_C,
            pseudo_dir=args.pseudo_dir, ecutwfc=args.ecutwfc,
            ecutrho=args.ecutrho, kpoints=args.kpoints,
        ), indent=2) + "\n")
        generated.append(meta_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    nat = 8 if args.cell == "conventional" else 2
    print(f"\nbulkgen: wrote {len(generated)} files under {args.outdir}/\n")
    print(f"  cell       : {args.cell} ({nat} atoms)")
    print(f"  a0         : {args.a0} Å")
    print(f"  functional : {args.functional}")
    print(f"  pseudo-C   : {args.pseudo_C}")
    print(f"  pseudo-dir : {args.pseudo_dir}")
    print(f"  ecutwfc    : {args.ecutwfc} Ry")
    print(f"  ecutrho    : {args.ecutrho} Ry")
    print(f"  kpoints    : {args.kpoints[0]} {args.kpoints[1]} {args.kpoints[2]}")
    print(f"  strains    : {args.strain_values}")
    print()
    for p in generated:
        print(f"  {p}")
    print()
    print("Next steps:")
    print("  cd runs/reference_diamond/relax && pw.x < pw.in > pw.out")
    print("  for d in runs/reference_diamond/hydrostatic/eps_*/; do")
    print("    (cd \"$d\" && pw.x < pw.in > pw.out)")
    print("  done")


if __name__ == "__main__":
    main()
