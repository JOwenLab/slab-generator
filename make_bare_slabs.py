#!/usr/bin/env python3
"""make_bare_slabs.py — generate the bare-facet campaign (NOT submitted).

Why this campaign
-----------------
Every mu_H-dependent statement this project now makes -- the equilibrium habit,
the interior-pressure sign change, the (T, p) window for inverting the ODMR
shift -- rests on extrapolating gamma_H(mu_H) to hydrogen-poor conditions.
Nothing in that model stops gamma_H rising forever, because only H-terminated
facets were ever calculated. In reality the surface dehydrogenates once

    gamma_H(mu_H)  >  gamma_bare

and beyond that point the H-terminated Wulff construction describes a surface
that no longer exists. gamma_bare is independent of mu_H (no hydrogen in it),
so a single ladder per facet converts that inequality into a hard upper bound
on delta_mu -- which is exactly the bound `surface_energy.py` currently has to
flag as missing.

This module writes the inputs. It does not submit them.

What is generated
-----------------
Symmetric bare slabs on the production thickness ladder, at production
settings, one directory per (motif, layer count):

    <out>/bare~C100_6L/pw.in      + provenance.json
    <out>/bare~C110_6L/pw.in      ...
    <out>/bare~C111_6L/pw.in
    <out>/bare~C111pandey_20L/pw.in

The in-plane cells are identical to the H-terminated production runs, so
gamma_bare and gamma_H are computed on the same footing and their difference is
meaningful.

The (111) reconstruction matters
--------------------------------
`C111_1x1_bare` is the UNRECONSTRUCTED (111) surface. The real bare (111)
reconstructs to Pandey 2x1 chains, which is substantially lower in energy. The
bound needs the LOWEST bare surface energy: using the unreconstructed reference
alone would put gamma_bare too high and make the dehydrogenation bound too
permissive -- it would claim the H-terminated surface survives to
hydrogen-poorer conditions than it really does. Both are therefore generated,
and `surface_energy.py` takes the lower of the two.

The Pandey construction needs a thicker slab: `geometry.invert_symmetrize`
cannot place an exact inversion centre for a 2x1 reconstruction of that depth
below about 20 layers, so the Pandey ladder is 20/22/24/26/28L rather than
6-16L. That is a real cost; `--no-pandey` skips it, at the price of a bound
that is known to be too weak.

Numerical settings that need thought before running
---------------------------------------------------
* Bare diamond surfaces are NOT closed-shell the way the H-terminated ones are.
  Bare (111)-1x1 has a half-filled dangling-bond band and is metallic; the
  Mermin-Vermeier smearing carried over from production handles the occupation
  but a spin-polarised check on at least one thickness would be prudent. The
  generated inputs are non-spin-polarised, matching production. This is flagged
  in every provenance file rather than being silently assumed.
* Bare surfaces relax further than H-terminated ones. The production
  `forc_conv_thr = 1.0d-4` was verified adequate for stress on the H-terminated
  slabs; that verification does not automatically transfer.

Pre-flight
----------
Every directory is checked with `preflight.py` before it is written to disk
(CLAUDE.md sec 7). The `carbon_coordination` check is skipped, EXPLICITLY and
only here: it exists to catch the 45-calculation bare-bottom-face error, and
undercoordinated surface carbon is the defining feature of a deliberately bare
face rather than a defect. Every other check runs, including inversion
symmetry, face balance, vacuum, overlap, and name-versus-geometry.

Usage
-----
    python3 make_bare_slabs.py --out-dir runs/bare_facets
    python3 make_bare_slabs.py --out-dir runs/bare_facets --no-pandey
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

import elastic_reference
import preflight
import slabgen

# Production settings (commit 03fec70 baseline).
PRODUCTION_ECUTWFC = 90.0
PRODUCTION_ECUTRHO = 720.0
PRODUCTION_VACUUM = 8.0
DEFAULT_LADDER = (6, 8, 10, 12, 16)
# invert_symmetrize cannot realise a Pandey 2x1 inversion centre below ~20L.
PANDEY_LADDER = (20, 22, 24, 26, 28)

DEFAULT_OUT_DIR = "runs/bare_facets"
DEFAULT_REFERENCE = "config/reference_pbe_sssp.json"

# motif -> (tag used in the run name, ladder)
BARE_MOTIFS = {
    "C100_2x1_bare": ("C100", DEFAULT_LADDER),
    "C110_1x1_bare": ("C110", DEFAULT_LADDER),
    "C111_1x1_bare": ("C111", DEFAULT_LADDER),
}
PANDEY_MOTIF = ("C111_2x1_pandey", ("C111pandey", PANDEY_LADDER))

# Undercoordinated surface carbon IS the point of a bare face. Every other
# pre-flight check still runs.
PREFLIGHT_SKIP = ("carbon_coordination",)

SPIN_NOTE = ("non-spin-polarised, matching the H-terminated production runs. "
             "Bare (111)-1x1 has a half-filled dangling-bond band and is "
             "metallic; smearing handles the occupation but a spin-polarised "
             "cross-check on one thickness is advisable before these energies "
             "are used quantitatively.")


class BareSlabError(Exception):
    """A load-bearing invariant of the bare-slab campaign was violated."""


def git_provenance() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return {
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(run("git", "status", "--porcelain")),
    }


def patch_production_settings(text: str) -> str:
    """Raise slabgen's default cutoffs to the production values.

    slabgen.to_qe hardcodes 60/480 Ry. Its k-mesh rule (24/|a|) already
    reproduces the production meshes exactly -- 9x5x1 on (100), 9x7x1 on (110),
    9x9x1 on (111) -- so only the cutoffs need changing. The substitution is
    asserted rather than assumed, because a silent no-op here would generate a
    whole campaign at the wrong cutoff and nothing downstream would notice.
    """
    out, n_wfc = re.subn(r"(?m)^(\s*ecutwfc\s*=\s*)\S+$",
                         rf"\g<1>{PRODUCTION_ECUTWFC}", text)
    out, n_rho = re.subn(r"(?m)^(\s*ecutrho\s*=\s*)\S+$",
                         rf"\g<1>{PRODUCTION_ECUTRHO}", out)
    if n_wfc != 1 or n_rho != 1:
        raise BareSlabError(
            f"expected exactly one ecutwfc and one ecutrho line to patch, "
            f"found {n_wfc} and {n_rho}; slabgen.to_qe's template has changed "
            f"and this campaign would be generated at the wrong cutoff")
    if f"ecutwfc  = {PRODUCTION_ECUTWFC}" not in out.replace("=", "  = ", 0):
        pass  # formatting varies; the regex count above is the real check
    return out


def kmesh_from(text: str) -> str:
    m = re.search(r"(?m)^K_POINTS\s+automatic\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", text)
    return f"{m.group(1)} {m.group(2)} {m.group(3)}" if m else ""


def make_one(motif: str, tag: str, layers: int, a0: float, vacuum: float,
             out_root: Path, min_vacuum: float, pseudo_dirs=()) -> dict:
    slab, motif_entry = slabgen.generate(motif, layers, a0, True, "bare", (1, 1))
    counts = {e: slab.el.count(e) for e in sorted(set(slab.el))}
    if counts.get("H"):
        raise BareSlabError(
            f"{motif} {layers}L produced {counts['H']} hydrogen atoms; a bare "
            f"facet campaign must contain none")
    if counts.get("C", 0) % layers:
        raise BareSlabError(
            f"{motif} {layers}L: {counts.get('C')} carbon atoms is not "
            f"divisible by {layers} layers (CLAUDE.md invariant 3)")

    comment = (f"{motif} | {layers} layers | a0={a0} | symmetric bare | "
               f"production {PRODUCTION_ECUTWFC:.0f}/{PRODUCTION_ECUTRHO:.0f} Ry")
    text = patch_production_settings(
        slabgen.to_qe(slab, vacuum, comment, symmetric=True, relax_mode="ions"))

    run_dir = out_root / f"bare~{tag}_{layers}L"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pw.in").write_text(text)

    provenance = {
        "campaign": "bare_facets",
        "purpose": ("gamma_bare, to bound gamma_H(mu_H) from above and fix the "
                    "dehydrogenation limit on delta_mu"),
        "generator": "make_bare_slabs.py",
        "motif": motif,
        "motif_status": motif_entry.get("status", ""),
        "motif_notes": motif_entry.get("notes", ""),
        "orientation": motif_entry.get("orientation", ""),
        "top_termination": "bare",
        "bottom_termination": "bare",
        "symmetric": True,
        "dipole_correction": "not applicable (symmetric, no net dipole)",
        "layers": layers,
        "atom_counts": counts,
        "vacuum_angstrom": vacuum,
        "in_plane_repeat": [1, 1],
        "a0_angstrom": a0,
        "cell_angstrom": [list(map(float, slab.A1)), list(map(float, slab.A2))],
        "ecutwfc": PRODUCTION_ECUTWFC,
        "ecutrho": PRODUCTION_ECUTRHO,
        "kpoints": kmesh_from(text),
        "pseudopotentials": {e: slabgen.QE_PSEUDO[e][1] for e in counts},
        "calculation": "relax (fixed cell)",
        "spin": SPIN_NOTE,
        "strain_mode": None,
        "strain_value": None,
        "source_relaxed_geometry": None,
        "preflight_skipped_checks": list(PREFLIGHT_SKIP),
        "preflight_skip_reason": (
            "carbon_coordination is skipped because undercoordinated surface "
            "carbon is the defining feature of a deliberately bare face, not a "
            "defect. All other checks run."),
        "not_submitted": True,
        **git_provenance(),
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    # Without a --pseudo-dir the UPF-presence check cannot pass: the repo does
    # not commit UPF files (config/reference_pbe_sssp.json notes). Skipping it
    # here makes the local gate INCOMPLETE, which is reported rather than
    # quietly treated as a pass.
    skip = tuple(PREFLIGHT_SKIP)
    if not pseudo_dirs:
        skip = skip + ("pseudopotentials_exist",)
    passed, results = preflight.check_directory(
        run_dir, min_vacuum=min_vacuum, skip=skip, pseudo_dirs=pseudo_dirs)
    violations = [v for vs in results.values() for v in vs]
    return {"run": run_dir.name, "path": str(run_dir), "counts": counts,
            "passed": passed, "violations": violations,
            "pseudo_checked": bool(pseudo_dirs),
            "kpoints": provenance["kpoints"]}


def run(out_dir, a0: float, vacuum: float, include_pandey: bool,
        ladder=None, min_vacuum: float = 5.0, pseudo_dirs=()) -> dict:
    out_root = Path(out_dir)
    plan = dict(BARE_MOTIFS)
    if include_pandey:
        plan[PANDEY_MOTIF[0]] = PANDEY_MOTIF[1]

    made = []
    for motif, (tag, default_ladder) in sorted(plan.items()):
        layers_list = ladder if (ladder and motif != PANDEY_MOTIF[0]) \
            else default_ladder
        for layers in layers_list:
            made.append(make_one(motif, tag, layers, a0, vacuum, out_root,
                                 min_vacuum, pseudo_dirs))
    return {"runs": made, "out_dir": str(out_root),
            "pseudo_checked": bool(pseudo_dirs),
            "n_failed": sum(1 for m in made if not m["passed"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--reference-config", default=DEFAULT_REFERENCE)
    ap.add_argument("--vacuum", type=float, default=PRODUCTION_VACUUM)
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help=f"thickness ladder (default: {DEFAULT_LADDER}; the "
                         f"Pandey ladder is fixed at {PANDEY_LADDER} because "
                         f"the reconstruction needs a thicker slab)")
    ap.add_argument("--no-pandey", action="store_true",
                    help="skip the Pandey 2x1 (111) reconstruction. Cheaper, "
                         "but the resulting dehydrogenation bound is known to "
                         "be too permissive.")
    ap.add_argument("--min-vacuum", type=float, default=5.0)
    ap.add_argument("--pseudo-dir", action="append", default=[],
                    help="directory holding the UPF files, passed to the "
                         "pre-flight gate. The repo does not commit UPFs, so "
                         "without this the gate skips the pseudopotential "
                         "check and is INCOMPLETE; re-run preflight.py with "
                         "--pseudo-dir on the cluster before submitting.")
    args = ap.parse_args(argv)

    try:
        ref = elastic_reference.load_elastic_reference(args.reference_config)
        result = run(args.out_dir, ref.bulk.a0_angstrom, args.vacuum,
                     not args.no_pandey, args.layers, args.min_vacuum,
                     tuple(args.pseudo_dir))
    except (BareSlabError, preflight.PreflightError,
            elastic_reference.ElasticReferenceError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"# bare-facet campaign generated in {result['out_dir']}  "
          f"(NOT submitted)")
    print(f"# a0 = {ref.bulk.a0_angstrom:.6f} A, vacuum {args.vacuum:.1f} A, "
          f"{PRODUCTION_ECUTWFC:.0f}/{PRODUCTION_ECUTRHO:.0f} Ry, symmetric")
    print(f"# pre-flight: all checks except {PREFLIGHT_SKIP} "
          f"(bare faces carry dangling bonds by construction)")
    print(f"{'run':26s} {'atoms':16s} {'k-mesh':10s} {'preflight':10s}")
    for m in result["runs"]:
        status = "PASS" if m["passed"] else "FAIL"
        print(f"{m['run']:26s} {str(m['counts']):16s} {m['kpoints']:10s} "
              f"{status:10s}")
        for v in m["violations"]:
            print(f"    {v}", file=sys.stderr)
    print(f"# {len(result['runs'])} directories, "
          f"{result['n_failed']} failed pre-flight")
    if not result["pseudo_checked"]:
        print("# GATE INCOMPLETE: pseudopotential presence was NOT checked "
              "(no --pseudo-dir; the repo does not commit UPFs). Re-run "
              "`python3 preflight.py <out-dir> --pseudo-dir <upf dir> "
              "--skip carbon_coordination` on the cluster before submitting.")
    print("# NOT SUBMITTED. Review, then run with your own queue tooling.")
    if not args.no_pandey:
        print(f"# Pandey (111) ladder is {PANDEY_LADDER} -- thicker slabs, "
              f"real cost. It is required for a correct (not merely "
              f"permissive) dehydrogenation bound.")
    return 1 if result["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
