#!/usr/bin/env python3
"""analyze_h_phonons.py — adsorbed-H zero-point energy from frozen phonons.

Turns the displacement campaign written by `make_h_phonons.py` into the one
number that currently dominates the uncertainty on the interior-pressure sign
change:

    DELTA_ZPE = ZPE_ads(per H) - ZPE(H2)/2

`surface_energy.py` estimates this at about +0.20 eV from literature C-H
frequencies. That estimate moves the predicted crossing temperature by 100-200 K
depending on pressure -- roughly thirty times the ideal-gas fit residual -- so
replacing it with a computed value is what makes the (T, p) prediction
quotable.

Method
------
Central differences on the hydrogen sublattice. For hydrogens i, j displaced by
+/- delta along Cartesian axes alpha, beta:

    Phi[i.alpha, j.beta] = -[ F_{i.alpha}(+delta_{j.beta})
                              - F_{i.alpha}(-delta_{j.beta}) ] / (2 delta)

The force-constant matrix is symmetrised, mass-weighted by the hydrogen mass,
and diagonalised. Its asymmetry before symmetrisation is reported: it is a
direct measure of the numerical quality of the whole campaign, since an exact
Hessian is symmetric by construction.

For H2 all six degrees of freedom are displaced, so five of the six modes must
come out near zero (three translations, two rotations) and one is the stretch.
Those five near-zero modes are a built-in end-to-end check on the parsing, the
sign convention, the units, and the displacement magnitude; the module refuses
to report a ZPE if they are not small.

What this does and does not capture
-----------------------------------
* Carbon is held fixed. This captures the C-H stretch and bends, which carry
  essentially all of the ZPE difference, but slightly overstiffens them and
  omits the change in the carbon-only phonon spectrum on hydrogenation. That
  remaining term is NOT computed and is not claimed to be zero.
* Harmonic, gamma point only.
* ZPE(H2) is computed from this campaign rather than taken from a table,
  precisely so the pseudopotential, cutoff, and harmonic errors cancel in the
  DIFFERENCE. A tabulated ZPE(H2) would leave them uncancelled.

Numerical floor
---------------
QE prints forces to eight decimal places in Ry/bohr. A finite-difference
Hessian is a difference of two such numbers, so the printed precision -- not
the algebra -- sets the accuracy, and the displacement has to be large enough
to lift the force difference well clear of the last printed digit. At the 0.015
Angstrom default and a C-H force constant the difference is around 1e-2
Ry/bohr, six orders above the print granularity, and the resulting frequency
error is well under 1 cm^-1. Shrinking the displacement to chase harmonicity
makes this worse, not better; see
tests/test_h_phonons.py::test_force_print_precision_limits_the_hessian.

Epistemic level
---------------
L1 until it has actually run. The frequencies themselves would be L2 once the
campaign completes (converged production settings, central differences with a
reported symmetry residual), but DELTA_ZPE inherits the frozen-carbon
approximation, so it stays L1 pending a full phonon calculation.

Usage
-----
    python3 analyze_h_phonons.py --runs-dir runs/h_phonons
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import surface_energy

DEFAULT_RUNS_DIR = "runs/h_phonons"
DEFAULT_OUT_DIR = "results/production"

RY_TO_EV = 13.605693
BOHR_TO_ANG = 0.529177210903
# Ry/bohr^2 -> eV/Angstrom^2
RY_PER_BOHR2_TO_EV_PER_A2 = RY_TO_EV / (BOHR_TO_ANG ** 2)
MASS_H_AMU = 1.008
CM1_TO_EV = 1.23984198e-4

# sqrt(eV / Angstrom^2 / amu) -> cm^-1
_EV = 1.602176634e-19
_AMU = 1.66053906660e-27
_C_CM_S = 2.99792458e10
FREQ_FACTOR_CM1 = math.sqrt(_EV / (1e-20 * _AMU)) / (2.0 * math.pi * _C_CM_S)

# The five H2 translation/rotation modes must come out below this, in cm^-1.
H2_ZERO_MODE_TOL_CM1 = 120.0
# Warn if the raw force-constant matrix is less symmetric than this (eV/A^2).
HESSIAN_ASYMMETRY_TOL = 0.20

RUN_RE = re.compile(
    r"^hphon~(?P<tag>[^~]+)~(?:ref|H(?P<atom>\d+)_(?P<axis>[xyz])(?P<sign>[pm]))$")
AXES = ("x", "y", "z")

EPISTEMIC_LEVEL = "L1"


class HPhononAnalysisError(Exception):
    """A load-bearing invariant of the phonon analysis was violated."""


# ------------------------------------------------------------------ parsing
_NUM = r"[-+]?\d+\.?\d*(?:[EedD][+-]?\d+)?"


def parse_forces(pw_out_path: Path) -> np.ndarray:
    """Per-atom forces in Ry/bohr from the last force block of a pw.out."""
    text = pw_out_path.read_text()
    if "JOB DONE" not in text:
        raise HPhononAnalysisError(f"{pw_out_path.parent.name}: no 'JOB DONE'")
    blocks = list(re.finditer(
        r"Forces acting on atoms.*?\n(.*?)(?:\n\s*\n|\nTotal force)",
        text, re.DOTALL))
    if not blocks:
        raise HPhononAnalysisError(
            f"{pw_out_path.parent.name}: no force block; was tprnfor set?")
    rows = re.findall(
        r"atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*"
        rf"({_NUM})\s+({_NUM})\s+({_NUM})", blocks[-1].group(1))
    if not rows:
        raise HPhononAnalysisError(
            f"{pw_out_path.parent.name}: force block present but unparseable")
    forces = np.zeros((len(rows), 3))
    for idx, fx, fy, fz in rows:
        forces[int(idx) - 1] = [float(fx), float(fy), float(fz)]
    return forces


def load_set(runs_dir: Path, tag: str) -> dict:
    """Collect the reference and displaced runs for one tag."""
    prov_path = None
    runs = {}
    for child in sorted(runs_dir.iterdir()):
        m = RUN_RE.match(child.name)
        if not m or m.group("tag") != tag:
            continue
        if not (child / "pw.out").exists():
            raise HPhononAnalysisError(
                f"{child.name}: no pw.out (campaign not run yet?)")
        if child.name.endswith("~ref"):
            runs["ref"] = child
            prov_path = child / "provenance.json"
        else:
            key = (int(m.group("atom")), m.group("axis"), m.group("sign"))
            runs[key] = child
    if "ref" not in runs:
        raise HPhononAnalysisError(f"{tag}: no reference run")
    provenance = {}
    if prov_path and prov_path.exists():
        provenance = json.loads(prov_path.read_text())
    return {"runs": runs, "provenance": provenance}


# ------------------------------------------------------------------ hessian
@dataclass
class ModeSet:
    tag: str
    frequencies_cm1: np.ndarray
    imaginary_cm1: np.ndarray
    zpe_ev: float
    n_displaced: int
    hessian_asymmetry: float
    max_residual_force: float
    displacement_angstrom: float
    provenance: dict = field(default_factory=dict)

    @property
    def zpe_per_h_ev(self) -> float:
        return self.zpe_ev / self.n_displaced


def build_hessian(runs: dict, displaced_atoms: list, delta_ang: float):
    """Force-constant matrix over the displaced sublattice, eV/Angstrom^2."""
    n = len(displaced_atoms)
    phi = np.zeros((3 * n, 3 * n))
    for j, atom_j in enumerate(displaced_atoms):
        for bx, axis in enumerate(AXES):
            try:
                fp = parse_forces(runs[(j, axis, "p")] / "pw.out")
                fm = parse_forces(runs[(j, axis, "m")] / "pw.out")
            except KeyError as exc:
                raise HPhononAnalysisError(
                    f"missing displacement {j} {axis}: {exc}") from exc
            dF = (fp - fm) / (2.0 * delta_ang)          # Ry/bohr per Angstrom
            for i, atom_i in enumerate(displaced_atoms):
                for ax in range(3):
                    phi[3 * i + ax, 3 * j + bx] = -dF[atom_i, ax]
    # Ry/bohr/Angstrom -> eV/Angstrom^2
    phi *= RY_TO_EV / BOHR_TO_ANG
    asymmetry = float(np.abs(phi - phi.T).max())
    return 0.5 * (phi + phi.T), asymmetry


def frequencies_from_hessian(phi: np.ndarray, mass_amu: float = MASS_H_AMU):
    """Mass-weight, diagonalise, and convert to cm^-1.

    Returns (real frequencies, imaginary-mode magnitudes), both cm^-1.
    """
    dyn = phi / mass_amu
    eigenvalues = np.linalg.eigvalsh(0.5 * (dyn + dyn.T))
    real, imag = [], []
    for lam in eigenvalues:
        w = FREQ_FACTOR_CM1 * math.sqrt(abs(lam))
        (real if lam >= 0 else imag).append(w)
    return np.array(sorted(real)), np.array(sorted(imag))


def zero_point_energy_ev(frequencies_cm1) -> float:
    return 0.5 * float(np.sum(np.asarray(frequencies_cm1))) * CM1_TO_EV


def analyse_set(runs_dir: Path, tag: str) -> ModeSet:
    data = load_set(runs_dir, tag)
    runs, prov = data["runs"], data["provenance"]
    delta = float(prov.get("displacement_angstrom", 0.0))
    if delta <= 0:
        raise HPhononAnalysisError(
            f"{tag}: displacement magnitude missing from provenance.json; "
            f"the Hessian cannot be scaled without it")
    displaced = prov.get("displaced_atom_indices")
    if displaced is None:
        n_atoms = max(k[0] for k in runs if k != "ref") + 1
        displaced = list(range(n_atoms))
    phi, asymmetry = build_hessian(runs, displaced, delta)
    real, imag = frequencies_from_hessian(phi)
    residual = float(np.abs(parse_forces(runs["ref"] / "pw.out")).max())
    return ModeSet(tag=tag, frequencies_cm1=real, imaginary_cm1=imag,
                   zpe_ev=zero_point_energy_ev(real),
                   n_displaced=len(displaced), hessian_asymmetry=asymmetry,
                   max_residual_force=residual, displacement_angstrom=delta,
                   provenance=prov)


def analyse_h2(runs_dir: Path) -> ModeSet:
    """H2 stretch. Five of the six modes must be near zero; that is the check."""
    ms = analyse_set(runs_dir, "H2")
    near_zero = ms.frequencies_cm1[:-1] if len(ms.frequencies_cm1) else np.array([])
    all_small = np.concatenate([near_zero, ms.imaginary_cm1]) \
        if len(ms.imaginary_cm1) else near_zero
    if len(ms.frequencies_cm1) + len(ms.imaginary_cm1) != 6:
        raise HPhononAnalysisError(
            f"H2: expected 6 modes, got "
            f"{len(ms.frequencies_cm1) + len(ms.imaginary_cm1)}")
    if len(all_small) and float(np.max(all_small)) > H2_ZERO_MODE_TOL_CM1:
        raise HPhononAnalysisError(
            f"H2: the five translation/rotation modes should be near zero but "
            f"reach {float(np.max(all_small)):.1f} cm^-1 (tolerance "
            f"{H2_ZERO_MODE_TOL_CM1:.0f}). Something is wrong with the forces, "
            f"the displacement, the units, or the sign convention -- and if it "
            f"is wrong here it is wrong for the slabs too. Refusing to report "
            f"a ZPE.")
    # Only the stretch carries zero-point energy worth counting.
    stretch = float(ms.frequencies_cm1[-1])
    ms.zpe_ev = 0.5 * stretch * CM1_TO_EV
    return ms


# ------------------------------------------------------------------ outputs
def delta_zpe(slab_sets: list, h2: ModeSet) -> dict:
    """DELTA_ZPE per surface, and the shift it implies on the delta_mu axis."""
    zpe_h2_per_h = h2.zpe_ev / 2.0
    out = {}
    for ms in slab_sets:
        per_h = ms.zpe_per_h_ev
        out[ms.tag] = {
            "zpe_ads_per_h_ev": per_h,
            "zpe_h2_per_h_ev": zpe_h2_per_h,
            "delta_zpe_ev": per_h - zpe_h2_per_h,
            "frequencies_cm1": [float(f) for f in ms.frequencies_cm1],
            "n_imaginary": int(len(ms.imaginary_cm1)),
        }
    return out


SUMMARY_FIELDS = ["set", "n_displaced", "n_modes", "n_imaginary",
                  "frequencies_cm1", "zpe_ev", "zpe_per_h_ev",
                  "delta_zpe_ev", "hessian_asymmetry_ev_per_a2",
                  "max_residual_force_ry_bohr", "displacement_angstrom",
                  "epistemic_level", "method_caveat"]

METHOD_CAVEAT = ("hydrogen sublattice only, carbon held fixed; harmonic; gamma "
                 "point. The carbon-only phonon change on hydrogenation is not "
                 "computed and is not claimed to be zero.")


def run(runs_dir, out_dir) -> dict:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise HPhononAnalysisError(f"no runs directory at {runs_dir}")
    tags = sorted({m.group("tag") for m in
                   (RUN_RE.match(c.name) for c in runs_dir.iterdir()) if m})
    if not tags:
        raise HPhononAnalysisError(
            f"no 'hphon~<tag>~...' directories under {runs_dir}; run "
            f"make_h_phonons.py first, then the campaign")
    if "H2" not in tags:
        raise HPhononAnalysisError(
            "no H2 set. DELTA_ZPE is a difference and the gas-phase half must "
            "be computed the same way for the method error to cancel; rerun "
            "make_h_phonons.py without --no-h2.")

    h2 = analyse_h2(runs_dir)
    slab_sets = [analyse_set(runs_dir, t) for t in tags if t != "H2"]
    deltas = delta_zpe(slab_sets, h2)

    rows = []
    for ms in [h2] + slab_sets:
        d = deltas.get(ms.tag, {})
        rows.append({
            "set": ms.tag, "n_displaced": ms.n_displaced,
            "n_modes": len(ms.frequencies_cm1) + len(ms.imaginary_cm1),
            "n_imaginary": len(ms.imaginary_cm1),
            "frequencies_cm1": " ".join(f"{f:.1f}" for f in ms.frequencies_cm1),
            "zpe_ev": f"{ms.zpe_ev:.6f}",
            "zpe_per_h_ev": f"{ms.zpe_per_h_ev:.6f}",
            "delta_zpe_ev": (f"{d['delta_zpe_ev']:.6f}" if d else ""),
            "hessian_asymmetry_ev_per_a2": f"{ms.hessian_asymmetry:.6f}",
            "max_residual_force_ry_bohr": f"{ms.max_residual_force:.6e}",
            "displacement_angstrom": f"{ms.displacement_angstrom:.4f}",
            "epistemic_level": EPISTEMIC_LEVEL,
            "method_caveat": METHOD_CAVEAT,
        })

    values = [v["delta_zpe_ev"] for v in deltas.values()]
    mean_delta = float(np.mean(values)) if values else float("nan")
    spread = (float(np.max(values) - np.min(values)) if len(values) > 1 else 0.0)
    estimate = surface_energy.delta_zpe_ev()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "h_phonon_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "epistemic_level": EPISTEMIC_LEVEL,
        "delta_zpe_mean_ev": mean_delta,
        "delta_zpe_spread_across_facets_ev": spread,
        "delta_zpe_literature_estimate_ev": estimate,
        "delta_zpe_estimate_error_ev": mean_delta - estimate,
        "per_surface": deltas,
        "h2_stretch_cm1": (float(h2.frequencies_cm1[-1])
                           if len(h2.frequencies_cm1) else None),
        "method_caveat": METHOD_CAVEAT,
        "runs_dir": str(runs_dir),
    }
    (out_dir / "h_phonon_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"rows": rows, "meta": meta, "h2": h2, "slab_sets": slab_sets}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)

    try:
        result = run(args.runs_dir, args.out_dir)
    except HPhononAnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    meta = result["meta"]
    print(f"# adsorbed-H zero-point energy from frozen phonons  "
          f"[{EPISTEMIC_LEVEL}]")
    print(f"# {METHOD_CAVEAT}")
    print(f"# H2 stretch {meta['h2_stretch_cm1']:.1f} cm^-1")
    print(f"{'set':16s} {'nH':>4s} {'ZPE/H (eV)':>11s} {'dZPE (eV)':>10s} "
          f"{'asym':>8s} {'imag':>5s}")
    for row in result["rows"]:
        print(f"{row['set']:16s} {row['n_displaced']:>4} "
              f"{float(row['zpe_per_h_ev']):11.4f} "
              f"{(row['delta_zpe_ev'] or 'n/a'):>10s} "
              f"{float(row['hessian_asymmetry_ev_per_a2']):8.4f} "
              f"{row['n_imaginary']:>5}")
    print(f"# DELTA_ZPE = {meta['delta_zpe_mean_ev']:+.4f} eV "
          f"(spread across facets {meta['delta_zpe_spread_across_facets_ev']:.4f})")
    print(f"# literature estimate was {meta['delta_zpe_literature_estimate_ev']:+.4f} eV; "
          f"error {meta['delta_zpe_estimate_error_ev']:+.4f} eV")
    print(f"# next: feed this into surface_energy.py / particle_strain.py in "
          f"place of the estimate")
    for name in ("h_phonon_summary.csv", "h_phonon_meta.json"):
        print(f"wrote {os.path.join(args.out_dir, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
