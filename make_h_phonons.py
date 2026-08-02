#!/usr/bin/env python3
"""make_h_phonons.py — generate the adsorbed-H frozen-phonon campaign (NOT submitted).

Why this campaign
-----------------
Every surface energy in this project was built from DFT total energies with no
vibrational term on either side. Restoring zero-point energy consistently adds
N_H * ZPE_ads to E_slab and ZPE(H2)/2 to mu_H, which is algebraically a rigid
shift of the delta_mu axis by

    DELTA_ZPE = ZPE_ads(per H) - ZPE(H2)/2

`surface_energy.py` currently estimates this at about +0.20 eV from literature
C-H frequencies. That estimate is the DOMINANT uncertainty on the
interior-pressure sign change: it moves the crossing temperature by 100-200 K
(pressure dependent), against a few kelvin from the ideal-gas fit. This
campaign replaces the estimate with a computed number.

The correction was deliberately never applied, because applying the H2 side
alone would be unbalanced. This campaign computes the adsorbed side so that
both can be applied together.

What is generated
-----------------
Central-difference displacements of the hydrogen sublattice, as single-point
SCFs with forces, starting from the RELAXED production geometries:

    <out>/hphon~C111_8L~ref/pw.in            undisplaced reference
    <out>/hphon~C111_8L~H00_xm/pw.in         atom 0, -x
    <out>/hphon~C111_8L~H00_xp/pw.in         atom 0, +x
    ...
    <out>/hphon~H2~ref/pw.in                 gas-phase reference, same cutoff
    <out>/hphon~H2~H00_xm/pw.in              ...

6 displacements per displaced H atom, plus one reference. With `--face top`
(the default) only the top-face hydrogens move, which is 6 per surface on (111)
and 12 on (100)/(110).

Why only the top face is enough
-------------------------------
The slabs are inversion symmetric, so the two faces carry identical local
environments and their H modes are degenerate pairs, split only by
through-slab coupling. At 8 layers the faces are about 8 Angstrom apart and
that coupling is negligible for modes as localised and as stiff as C-H. The
per-H zero-point energy from one face therefore stands for both. `--face both`
doubles the cost and checks the assumption instead of relying on it.

Why H2 is recomputed here
-------------------------
DELTA_ZPE is a DIFFERENCE. Taking ZPE_ads from a frozen-phonon calculation and
ZPE(H2) from a spectroscopic table would leave the pseudopotential, cutoff, and
harmonic-approximation errors uncancelled in the difference. Computing both the
same way at the same cutoff cancels them to leading order. It costs 12 extra
single-point SCFs on a two-atom molecule, which is nothing.

Approximations, stated
----------------------
* Only the hydrogen sublattice is displaced. Carbon is held fixed, i.e. treated
  as infinitely heavy. This captures the C-H stretch and bends, which carry
  essentially all of the ZPE difference, but it slightly overstiffens them and
  it omits the change in the carbon-only phonons on hydrogenation. That
  remaining term is not computed here and is not claimed to be zero.
* Harmonic. No anharmonicity, which for a C-H stretch is a real few-percent
  effect on the fundamental.
* Gamma point only. The H modes are dispersionless to a very good
  approximation because they are localised, but this is an approximation.

`analyze_h_phonons.py` turns the resulting forces into frequencies, ZPE, and
DELTA_ZPE, and re-reports the sign-change temperature with it.

Usage
-----
    python3 make_h_phonons.py --out-dir runs/h_phonons
    python3 make_h_phonons.py --out-dir runs/h_phonons --layers 8 12 --face both
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import parse_slab
import preflight

DEFAULT_RUNS_DIR = "results/production"
DEFAULT_REFERENCE_DIR = "results/reference_90_720"
DEFAULT_OUT_DIR = "runs/h_phonons"
DEFAULT_PREFIX = "thick_a0corr"

# 0.015 A is small enough to stay harmonic for a C-H stretch and large enough
# that the force difference is far above the SCF force noise at conv_thr 1e-8.
DEFAULT_DISPLACEMENT_ANGSTROM = 0.015
DEFAULT_LAYERS = (8,)
AXES = ("x", "y", "z")

SURFACES = ("C100", "C110", "C111")


class HPhononError(Exception):
    """A load-bearing invariant of the phonon campaign was violated."""


def git_provenance() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return {"git_commit": run("git", "rev-parse", "HEAD"),
            "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(run("git", "status", "--porcelain"))}


SCF_CONTROL = """&CONTROL
  calculation   = 'scf'
  prefix        = 'phon'
  pseudo_dir    = './'
  outdir        = './tmp'
  tprnfor       = .true.
  tstress       = .false.
/
"""


def _system_block(nat: int, ntyp: int, ecutwfc: float, ecutrho: float,
                  molecule: bool) -> str:
    lines = ["&SYSTEM", "  ibrav    = 0", f"  nat      = {nat}",
             f"  ntyp     = {ntyp}", f"  ecutwfc  = {ecutwfc}",
             f"  ecutrho  = {ecutrho}"]
    if molecule:
        lines.append("  occupations = 'fixed'")
    else:
        lines += ["  occupations = 'smearing'", "  smearing    = 'mv'",
                  "  degauss     = 0.01"]
    lines.append("/")
    return "\n".join(lines) + "\n"


def write_scf(path: Path, header: str, cell, species_pseudo: dict,
              symbols, positions, ecutwfc: float, ecutrho: float,
              kpoints: str, molecule: bool) -> None:
    """A forces-only SCF. conv_thr is tightened: the Hessian is a difference of
    forces, so force noise propagates straight into the frequencies."""
    L = [f"! {header}"]
    L.append(SCF_CONTROL.rstrip())
    L.append(_system_block(len(symbols), len(species_pseudo),
                           ecutwfc, ecutrho, molecule).rstrip())
    L += ["&ELECTRONS", "  conv_thr    = 1.0d-10", "  mixing_beta = 0.3", "/"]
    L.append("ATOMIC_SPECIES")
    masses = {"C": 12.011, "H": 1.008}
    for el in sorted(species_pseudo, key=lambda e: ("C", "H").index(e)
                     if e in ("C", "H") else 99):
        L.append(f"  {el:2s} {masses.get(el, 1.0):>8.3f}  {species_pseudo[el]}")
    L.append("CELL_PARAMETERS angstrom")
    for v in np.asarray(cell, dtype=float):
        L.append(f"  {v[0]:16.10f} {v[1]:16.10f} {v[2]:16.10f}")
    L.append("ATOMIC_POSITIONS angstrom")
    for sym, p in zip(symbols, positions):
        L.append(f"  {sym:2s} {p[0]:16.10f} {p[1]:16.10f} {p[2]:16.10f}")
    L.append(kpoints)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


def select_hydrogens(symbols, positions, face: str) -> list:
    """Indices of the hydrogens to displace."""
    h_idx = [i for i, s in enumerate(symbols) if s == "H"]
    if not h_idx:
        raise HPhononError("no hydrogen atoms in the source geometry")
    if face == "both":
        return h_idx
    z = np.asarray([positions[i][2] for i in h_idx])
    z_mid = 0.5 * (z.min() + z.max())
    top = [i for i in h_idx if positions[i][2] > z_mid]
    if not top:
        raise HPhononError("could not identify a top face by z")
    if face == "top":
        return sorted(top)
    raise HPhononError(f"unknown face selector {face!r}")


def make_slab_set(surface: str, layers: int, runs_dir: Path, out_root: Path,
                  prefix: str, delta: float, face: str,
                  pseudo_dirs=()) -> dict:
    src = runs_dir / f"{prefix}~{surface}_{layers}L"
    if not (src / "pw.out").exists():
        raise HPhononError(f"no relaxed source run at {src}")
    pin = parse_slab.parse_pw_in(src / "pw.in")
    pout = parse_slab.parse_pw_out(src / "pw.out")
    if not pout["complete"] or pout["status"] != "JOB DONE":
        raise HPhononError(f"{src.name}: status {pout['status']!r}")
    if pout["relax_converged"] is False:
        raise HPhononError(
            f"{src.name}: relaxation did not converge; displacing an "
            f"unrelaxed geometry gives a Hessian with linear forces in it")
    atoms = pout["final_positions_ang"]
    if not atoms:
        raise HPhononError(f"{src.name}: no final coordinates in pw.out")

    symbols = [a["species"] for a in atoms]
    positions = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    cell = np.asarray(pin["cell_params_ang"], dtype=float)
    pseudo = pout["pseudo_files"] or pin["pseudopotentials"]
    kpoints = f"K_POINTS automatic\n  {pin['kpoints']}  0 0 0"
    ecutwfc, ecutrho = float(pin["ecutwfc"]), float(pin["ecutrho"])

    displaced = select_hydrogens(symbols, positions, face)
    tag = f"hphon~{surface}_{layers}L"
    made = []

    write_scf(out_root / f"{tag}~ref" / "pw.in",
              f"{tag} reference (undisplaced relaxed geometry)",
              cell, pseudo, symbols, positions, ecutwfc, ecutrho, kpoints,
              molecule=False)
    made.append(f"{tag}~ref")

    for n, atom in enumerate(displaced):
        for ax, axis in enumerate(AXES):
            for sign, label in ((+1.0, "p"), (-1.0, "m")):
                pos = positions.copy()
                pos[atom, ax] += sign * delta
                name = f"{tag}~H{n:02d}_{axis}{label}"
                write_scf(out_root / name / "pw.in",
                          f"{tag} atom {atom} ({axis}{label}) "
                          f"displaced {sign * delta:+.4f} A",
                          cell, pseudo, symbols, pos, ecutwfc, ecutrho,
                          kpoints, molecule=False)
                made.append(name)

    provenance = {
        "campaign": "h_phonons",
        "purpose": ("adsorbed-H zero-point energy, to replace the estimated "
                    "DELTA_ZPE that dominates the sign-change temperature"),
        "generator": "make_h_phonons.py",
        "kind": "slab",
        "surface": surface, "layers": layers,
        "source_relaxed_geometry": str(src),
        "displacement_angstrom": delta,
        "face": face,
        "displaced_atom_indices": displaced,
        "n_displaced_atoms": len(displaced),
        "n_displacements": 6 * len(displaced),
        "axes": list(AXES),
        "method": "central differences on the hydrogen sublattice only",
        "carbon_treatment": "held fixed (infinite mass)",
        "ecutwfc": ecutwfc, "ecutrho": ecutrho,
        "kpoints": pin["kpoints"],
        "pseudopotentials": pseudo,
        "conv_thr": "1.0d-10 (tighter than production: the Hessian is a "
                    "difference of forces)",
        "not_submitted": True,
        **git_provenance(),
    }
    for name in made:
        (out_root / name / "provenance.json").write_text(
            json.dumps({**provenance, "run": name}, indent=2) + "\n")

    n_failed = 0
    for name in made:
        skip = ("carbon_coordination", "name_matches_geometry")
        if not pseudo_dirs:
            skip = skip + ("pseudopotentials_exist",)
        try:
            passed, _ = preflight.check_directory(
                out_root / name, min_vacuum=4.0, skip=skip,
                pseudo_dirs=pseudo_dirs)
        except preflight.PreflightError:
            passed = False
        n_failed += 0 if passed else 1

    return {"tag": tag, "n_runs": len(made), "n_displaced": len(displaced),
            "atoms": len(symbols), "n_failed": n_failed}


def make_h2_set(reference_dir: Path, out_root: Path, delta: float) -> dict:
    """The gas-phase half of DELTA_ZPE, at the same cutoff and pseudopotential.

    All six degrees of freedom are displaced, so the resulting 6x6 Hessian
    carries five near-zero translation/rotation modes as a built-in check on
    the whole procedure alongside the one real stretch.
    """
    src = reference_dir / "H2"
    if not (src / "pw.out").exists():
        raise HPhononError(f"no H2 reference at {src}")
    pin = parse_slab.parse_pw_in(src / "pw.in")
    pout = parse_slab.parse_pw_out(src / "pw.out")
    atoms = pout["final_positions_ang"] or pin["initial_positions_ang"]
    if not atoms:
        raise HPhononError("could not read H2 coordinates")
    symbols = [a["species"] for a in atoms]
    positions = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)

    # The H2 reference uses ibrav=1 with celldm(1); rebuild the cubic cell.
    m = re.search(r"celldm\(1\)\s*=\s*([0-9.eEdD+-]+)", (src / "pw.in").read_text())
    if not m:
        raise HPhononError("could not read celldm(1) from the H2 reference")
    a_bohr = float(m.group(1).lower().replace("d", "e"))
    a_ang = a_bohr * 0.529177210903
    cell = np.eye(3) * a_ang

    pseudo = pout["pseudo_files"] or pin["pseudopotentials"]
    ecutwfc, ecutrho = float(pin["ecutwfc"]), float(pin["ecutrho"])
    kpoints = "K_POINTS gamma"

    made = []
    write_scf(out_root / "hphon~H2~ref" / "pw.in", "H2 reference (relaxed)",
              cell, pseudo, symbols, positions, ecutwfc, ecutrho, kpoints,
              molecule=True)
    made.append("hphon~H2~ref")
    for n in range(len(symbols)):
        for ax, axis in enumerate(AXES):
            for sign, label in ((+1.0, "p"), (-1.0, "m")):
                pos = positions.copy()
                pos[n, ax] += sign * delta
                name = f"hphon~H2~H{n:02d}_{axis}{label}"
                write_scf(out_root / name / "pw.in",
                          f"H2 atom {n} ({axis}{label}) "
                          f"displaced {sign * delta:+.4f} A",
                          cell, pseudo, symbols, pos, ecutwfc, ecutrho,
                          kpoints, molecule=True)
                made.append(name)

    provenance = {
        "campaign": "h_phonons", "kind": "molecule",
        "purpose": ("ZPE(H2) at the same cutoff and pseudopotential as the "
                    "slab side, so the DELTA_ZPE difference cancels method "
                    "error"),
        "generator": "make_h_phonons.py",
        "source_relaxed_geometry": str(src),
        "displacement_angstrom": delta,
        "n_displacements": 6 * len(symbols),
        "cell_angstrom": a_ang,
        "ecutwfc": ecutwfc, "ecutrho": ecutrho,
        "kpoints": "gamma",
        "pseudopotentials": pseudo,
        "expected_modes": ("1 stretch plus 5 near-zero translation/rotation "
                           "modes; the near-zero five are the built-in check"),
        "not_submitted": True,
        **git_provenance(),
    }
    for name in made:
        (out_root / name / "provenance.json").write_text(
            json.dumps({**provenance, "run": name}, indent=2) + "\n")
    return {"tag": "hphon~H2", "n_runs": len(made), "n_displaced": len(symbols),
            "atoms": len(symbols), "n_failed": 0}


def run(runs_dir, reference_dir, out_dir, prefix, layers_list, surfaces,
        delta, face, include_h2=True, pseudo_dirs=()) -> dict:
    out_root = Path(out_dir)
    sets = []
    for surface in surfaces:
        for layers in layers_list:
            sets.append(make_slab_set(surface, layers, Path(runs_dir), out_root,
                                      prefix, delta, face, pseudo_dirs))
    if include_h2:
        sets.append(make_h2_set(Path(reference_dir), out_root, delta))
    return {"sets": sets, "out_dir": str(out_root),
            "total_runs": sum(s["n_runs"] for s in sets),
            "pseudo_checked": bool(pseudo_dirs),
            "n_failed": sum(s["n_failed"] for s in sets)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--surfaces", nargs="+", default=list(SURFACES))
    ap.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS),
                    help="source thickness(es) (default: %(default)s). The ZPE "
                         "is a local quantity, but a second thickness is the "
                         "only way to show that.")
    ap.add_argument("--displacement", type=float,
                    default=DEFAULT_DISPLACEMENT_ANGSTROM, metavar="ANGSTROM")
    ap.add_argument("--face", choices=["top", "both"], default="top",
                    help="'top' relies on the slab's inversion symmetry and "
                         "halves the cost; 'both' checks it (default: %(default)s)")
    ap.add_argument("--no-h2", action="store_true",
                    help="skip the gas-phase H2 displacements. Not advised: "
                         "DELTA_ZPE is a difference, and computing both halves "
                         "the same way is what cancels the method error.")
    ap.add_argument("--pseudo-dir", action="append", default=[])
    args = ap.parse_args(argv)

    try:
        result = run(args.runs_dir, args.reference_dir, args.out_dir,
                     args.prefix, args.layers, args.surfaces,
                     args.displacement, args.face, not args.no_h2,
                     tuple(args.pseudo_dir))
    except HPhononError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"# adsorbed-H frozen-phonon campaign in {result['out_dir']}  "
          f"(NOT submitted)")
    print(f"# central differences, displacement {args.displacement:.4f} A, "
          f"face={args.face}, hydrogen sublattice only (carbon held fixed)")
    print(f"{'set':22s} {'atoms':>7s} {'H moved':>9s} {'runs':>6s}")
    for s in result["sets"]:
        print(f"{s['tag']:22s} {s['atoms']:7d} {s['n_displaced']:9d} "
              f"{s['n_runs']:6d}")
    print(f"# {result['total_runs']} single-point SCFs total "
          f"({result['n_failed']} failed pre-flight)")
    if not result["pseudo_checked"]:
        print("# GATE INCOMPLETE: pseudopotential presence not checked "
              "(no --pseudo-dir). Re-run preflight.py on the cluster.")
    print("# NOT SUBMITTED. Then: python3 analyze_h_phonons.py "
          f"--runs-dir {result['out_dir']}")
    return 1 if result["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
