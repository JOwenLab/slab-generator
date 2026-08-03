#!/usr/bin/env python3
"""surface_energy.py — H-terminated diamond surface energies as a function of mu_H.

What it computes
----------------
For a symmetric slab with N_C carbon and N_H hydrogen atoms and in-plane cell
area A, the surface energy per face is

    gamma(mu_H) = [ E_slab - N_C * mu_C - N_H * mu_H ] / (2 A)

The factor 2 is the slab's two equivalent faces (CLAUDE.md invariant 1: these
are inversion-symmetric H/H slabs, so both faces are the same surface and each
carries half the excess energy).

mu_C comes from the thickness ladder itself
-------------------------------------------
Rather than subtracting a separately computed bulk energy per carbon, mu_C is
obtained as the SLOPE of a linear fit of E_slab against N_C across the ladder:

    E_slab(N_C) = mu_C * N_C + [ N_H * mu_H + 2 A gamma ]

This is the standard fit (Boettger) construction. It matters: subtracting a
bulk energy computed in a different cell, with a different k-mesh, accumulates
an error proportional to N_C, which grows with slab thickness and masquerades
as a thickness-dependent surface energy. Taking mu_C from the slab's own
interior cancels that to leading order.

The fitted mu_C is then compared against the project bulk reference
(the Birch-Murnaghan E0 per carbon in results/reference_90_720/). Agreement is
a real check, not a formality: a fitted slope that drifts from the bulk value
means the slab interior is not bulk-like, or the ladder is not converged.
`--mu-c-tol-mry` controls when that becomes a warning.

A drifting slope does not merely add noise -- it BIASES gamma
--------------------------------------------------------------
The least-squares intercept is b = <E> - mu_C_fit * <N_C>, so a slope error
`drift` moves the intercept by exactly -drift * <N_C> and moves gamma by

    bias = -drift * <N_C>_used / (2A) * RY_PER_A2_TO_J_PER_M2      [J/m^2]

This is an identity, not an estimate: it is exactly the gap between the
Boettger gamma and the mean of the per-slab gammas taken with the independent
bulk mu_C. It is the quantity that matters, because it is proportional to
<N_C>: the same drift hurts a thick ladder more than a thin one, and hurts a
small-area cell more than a large one. `--mu-c-tol-mry` alone cannot see that.

`--mu-c-reject-j-m2` therefore REJECTS the fit -- raises, writes nothing --
when the bias exceeds a budget in the units gamma is actually reported in.
This is not a warning, because the consumers of gamma do not read warnings.
The specific failure that motivated it: a bare (111) ladder whose 6L and 8L
points sit outside the asymptotic regime fitted a slope tens of mRy from bulk
and produced a large NEGATIVE gamma_bare, which `dehydrogenation_bound` then
took as the binding ceiling on delta_mu. A ceiling built on a bad fit is worse
than a missing one: the missing one is reported as missing, and the bad one
silently binds every hydrogen-poor statement downstream of it.

The bare facet's SPIN STATE is part of the ceiling
--------------------------------------------------
A bare face carries one unpaired electron per dangling bond; a spin-polarised
bare (111) settles at 2.00 Bohr magnetons per symmetric cell, one per face.
nspin = 2 is variational over nspin = 1, so a non-spin-polarised bare energy is
an upper bound, gamma_bare is an upper bound, and the ceiling built from it is
too PERMISSIVE. No fit diagnostic can see this: the ladder can be perfectly
linear with zero mu_C drift and still be the energy of the wrong electronic
state. `check_bare_spin_states` therefore reads the spin state from pw.out
(not pw.in), refuses a facet that mixes polarised and non-polarised rungs, and
warns loudly on a facet where no polarised run exists anywhere. See the block
above `dehydrogenation_bound`.

E_s is measured, not guessed. Given same-thickness nspin=1/nspin=2 pairs
(`--bare-spin-runs-dir`), `measure_spin_stabilisation` returns the exchange
stabilisation per dangling bond, and the correction to the ceiling is exactly
E_s on every unreconstructed facet -- area, coverage and thickness all cancel.
Both the corrected and uncorrected ceilings are reported. The correction is
withheld where the dangling bonds are quenched or partially quenched, which the
total magnetization of the polarised run reveals.

Exclusions are per surface, not global
--------------------------------------
Which thicknesses are outside the asymptotic regime is a property of the
SURFACE, not of the campaign. The H-terminated ladders need 6L dropped; a bare
ladder, with two interacting dangling-bond faces instead of two passivated
ones, converges more slowly and can need 6L and 8L dropped. `--exclude-layers`
sets the default and `--exclude-layers-for C111=6,8` overrides it per surface
(`--bare-exclude-layers` / `--bare-exclude-layers-for` for the bare ladder,
which is keyed by the same surface tags). Every fit reports the thicknesses it
used, the thicknesses it dropped, and why.

Why mu_H is a variable and not a number
---------------------------------------
The H-rich limit is mu_H = E(H2)/2, the most hydrogen-rich condition under
which H2 gas does not condense. At any lower hydrogen chemical potential each
gamma RISES, at a rate set by the facet's hydrogen coverage:

    dgamma / d(-mu_H) = N_H / (2 A)

Those coefficients differ per facet, so the ORDERING of the surface energies is
not a fixed fact -- it is a function of mu_H, and so is the equilibrium particle
shape that follows from it. Reporting a single H-rich number and calling it
"the" surface energy hides that. This module therefore reports gamma over a
range of mu_H and locates:

  * where each gamma crosses zero (below which a Wulff construction exists at
    all -- see `particle_strain.wulff_area_fractions`);
  * where each PAIR of gammas crosses, i.e. where the stability ordering and
    hence the equilibrium habit changes.

Limits of the mu_H range
------------------------
The upper limit (H-rich, mu_H = E(H2)/2) is computed here. The lower limit is
NOT: bounding how hydrogen-poor the system can get requires a reference for the
competing carbon-hydrogen phase (CH4, or graphite plus H2), and no such
calculation exists in this repository. The scanned range is therefore a
convention set by `--mu-h-range`, not a computed physical bound, and the
outputs say so.

Epistemic level
---------------
L2 for gamma at fixed mu_H on (110) and (111): converged production
relaxations at 90/720 Ry, a0 = 3.572997 Angstrom, with the fit residual and the
per-slab scatter reported.

L1 for (100). Its E(N_C) fit is far worse than the other two and its fitted
mu_C drifts measurably from the bulk value; the per-slab gammas scatter by a
few hundredths of a J/m^2. This is the same even/odd parity oscillation that
makes (100) the worst surface in the tau extrapolation
(`fit_tau_infinity.py`), and it persists past 8 layers. Do not quote (100)
surface energies to better than about 0.03 J/m^2.

Usage
-----
    python3 surface_energy.py
    python3 surface_energy.py --mu-h-range 0 3 --write-config
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import elastic_reference
import parse_slab

RY_TO_EV = 13.605693
# 1 eV / Angstrom^2 = 1.602176634e-19 J / 1e-20 m^2
EV_PER_A2_TO_J_PER_M2 = 16.02176634
RY_PER_A2_TO_J_PER_M2 = RY_TO_EV * EV_PER_A2_TO_J_PER_M2

DEFAULT_RUNS_DIR = "results/production"
DEFAULT_REFERENCE_DIR = "results/reference_90_720"
DEFAULT_OUT_DIR = "results/production"
DEFAULT_CONFIG_OUT = "config/surface_energies_h.json"
DEFAULT_PREFIX = "thick_a0corr"

# Warn if the ladder's fitted mu_C drifts from the bulk reference by more than
# this. (110) and (111) sit at ~0.003 mRy; (100) at ~0.05 mRy.
DEFAULT_MU_C_TOL_MRY = 0.02

# REJECT the fit if the drift biases gamma by more than this many J/m^2.
#
# Why 0.10 J/m^2, and why in J/m^2 rather than mRy:
#
#  * gamma is reported in J/m^2, so the budget belongs in J/m^2. A mRy
#    threshold is not comparable across surfaces -- the same drift biases a
#    16L, 5.5 Angstrom^2 (111) ladder about four times harder than an 8L,
#    12.8 Angstrom^2 (100) one, because the bias scales as <N_C>/(2A).
#  * 0.10 J/m^2 is roughly 4x the worst honest per-slab scatter in the
#    production set ((100), 0.024 J/m^2) and 3x the 0.03 J/m^2 that the module
#    docstring already says is the floor on quoting (100). A bias below the
#    scatter is indistinguishable from the noise already declared; a bias
#    several times above it is a different number, not a noisier one.
#  * In the units that actually consume gamma: the dehydrogenation ceiling is
#    (gamma_bare - gamma_H) / (N_H/2A), with slopes of 2.5-3.5 (J/m^2)/eV, so
#    0.10 J/m^2 moves a ceiling by at most 0.04 eV. The margin that ceiling has
#    to resolve -- against the delta_mu = 1.295 eV interior-pressure sign
#    change -- is a few tenths of an eV. 0.04 eV is an order of magnitude
#    below it; a budget ten times looser would not be.
#
# The production H ladders land at 0.010 J/m^2 ((100)) and below 0.001 J/m^2
# ((110), (111)). The bare (111) failure this guard exists for lands near
# 8 J/m^2. The separation is ~3 orders of magnitude; the threshold is not
# balanced on a knife edge.
DEFAULT_MU_C_REJECT_J_M2 = 0.10

RUN_RE = re.compile(r"^(?P<prefix>.+)~(?P<surface>C\d{3}[a-z]*)_(?P<layers>\d+)L$")

ORIENTATION = {"C100": "(100)", "C110": "(110)", "C111": "(111)"}


class SurfaceEnergyError(Exception):
    """A load-bearing invariant of the surface-energy derivation was violated."""


# ------------------------------------------------------------------ loading
@dataclass(frozen=True)
class Reference:
    mu_h_rich_ry: float          # E(H2) / 2
    e_h2_ry: float
    mu_c_bulk_ry: float          # Birch-Murnaghan E0 per carbon
    h2_run: str
    bulk_source: str
    pseudo_C: str
    pseudo_H: str
    ecutwfc: float
    ecutrho: float


def load_references(reference_dir, bulk_fit_json=None) -> Reference:
    """mu_H at the H-rich limit and mu_C from the project bulk fit.

    mu_C is taken from the Birch-Murnaghan E0, i.e. the energy at the FITTED
    equilibrium lattice constant, not from the nearest sampled point. The
    sampled eps=0 run sits at a0 = 3.567 Angstrom, which is not the production
    lattice constant, and using it would put a spurious offset into every
    surface energy.
    """
    reference_dir = Path(reference_dir)
    h2_dir = reference_dir / "H2"
    if not (h2_dir / "pw.out").exists():
        raise SurfaceEnergyError(
            f"no H2 reference at {h2_dir}. gamma cannot be referenced to a "
            f"hydrogen chemical potential without it.")
    h2_in = parse_slab.parse_pw_in(h2_dir / "pw.in")
    h2_out = parse_slab.parse_pw_out(h2_dir / "pw.out")
    if not h2_out["complete"] or h2_out["status"] != "JOB DONE":
        raise SurfaceEnergyError(f"H2 reference status is {h2_out['status']!r}")
    n_h = sum(1 for a in h2_in["initial_positions_ang"] if a["species"] == "H")
    if n_h != 2:
        raise SurfaceEnergyError(
            f"H2 reference holds {n_h} hydrogen atoms, not 2; mu_H = E/2 would "
            f"be wrong")
    e_h2 = float(h2_out["energy_ry"])

    fit_path = Path(bulk_fit_json or (reference_dir / "bulk_fit_summary.json"))
    if not fit_path.exists():
        raise SurfaceEnergyError(f"no bulk fit summary at {fit_path}")
    fit = json.loads(fit_path.read_text())
    if "bm3_E0_ry" not in fit:
        raise SurfaceEnergyError(
            f"{fit_path} has no 'bm3_E0_ry'; mu_C must come from the fitted "
            f"equilibrium energy, not a sampled point")
    e0 = float(fit["bm3_E0_ry"])

    # The bulk cell's carbon count, read rather than assumed.
    bulk_dirs = sorted(reference_dir.glob("bulk*eps_+0.000"))
    if not bulk_dirs:
        raise SurfaceEnergyError(f"no bulk reference run under {reference_dir}")
    bulk_in = parse_slab.parse_pw_in(bulk_dirs[0] / "pw.in")
    bulk_out = parse_slab.parse_pw_out(bulk_dirs[0] / "pw.out")
    n_c_bulk = bulk_in["nat"]
    if not n_c_bulk:
        raise SurfaceEnergyError(f"could not read nat from {bulk_dirs[0]}/pw.in")

    pseudo_c = (bulk_out["pseudo_files"] or bulk_in["pseudopotentials"]).get("C", "")
    pseudo_h = (h2_out["pseudo_files"] or h2_in["pseudopotentials"]).get("H", "")
    if h2_in["ecutwfc"] != bulk_in["ecutwfc"] or h2_in["ecutrho"] != bulk_in["ecutrho"]:
        raise SurfaceEnergyError(
            f"H2 reference ({h2_in['ecutwfc']}/{h2_in['ecutrho']} Ry) and bulk "
            f"reference ({bulk_in['ecutwfc']}/{bulk_in['ecutrho']} Ry) use "
            f"different cutoffs; their energies cannot be combined")

    return Reference(
        mu_h_rich_ry=e_h2 / 2.0, e_h2_ry=e_h2, mu_c_bulk_ry=e0 / n_c_bulk,
        h2_run=str(h2_dir), bulk_source=str(fit_path),
        pseudo_C=pseudo_c, pseudo_H=pseudo_h,
        ecutwfc=float(bulk_in["ecutwfc"]), ecutrho=float(bulk_in["ecutrho"]))


@dataclass(frozen=True)
class SlabEnergy:
    surface: str
    layers: int
    run: str
    n_C: int
    n_H: int
    area_angstrom2: float
    energy_ry: float
    pseudo_C: str
    pseudo_H: str
    ecutwfc: float
    ecutrho: float
    nspin_declared: int = 1          # from pw.in (QE defaults to 1 if absent)
    nspin_effective: int = None      # from pw.out; ground truth for what ran
    total_magnetization_bohr: float = None

    @property
    def spin_polarised(self) -> bool:
        """True only when pw.out shows the run really was spin-polarised."""
        if self.nspin_effective is not None:
            return self.nspin_effective == 2
        return self.nspin_declared == 2


def load_slab_energies(runs_dir, prefix: str) -> dict:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise SurfaceEnergyError(f"runs directory not found: {runs_dir}")
    by_surface = {}
    for child in sorted(runs_dir.iterdir()):
        m = RUN_RE.match(child.name)
        if not m or m.group("prefix") != prefix or not (child / "pw.out").exists():
            continue
        pin = parse_slab.parse_pw_in(child / "pw.in")
        pout = parse_slab.parse_pw_out(child / "pw.out")
        if not pout["complete"] or pout["status"] != "JOB DONE":
            raise SurfaceEnergyError(
                f"{child.name}: pw.out status is {pout['status']!r}")
        if pout["relax_converged"] is False:
            raise SurfaceEnergyError(
                f"{child.name}: relaxation did not converge; its energy is not "
                f"a relaxed-geometry energy")
        if pout["energy_ry"] is None:
            raise SurfaceEnergyError(f"{child.name}: no total energy in pw.out")
        cell = np.asarray(pin["cell_params_ang"], dtype=float)
        area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        atoms = pout["final_positions_ang"] or pin["initial_positions_ang"]
        n_C = sum(1 for a in atoms if a["species"] == "C")
        n_H = sum(1 for a in atoms if a["species"] == "H")
        layers = int(m.group("layers"))
        if n_C % layers:
            raise SurfaceEnergyError(
                f"{child.name}: {n_C} carbon atoms is not divisible by the "
                f"{layers} layers the folder name claims (CLAUDE.md invariant 3)")
        pseudo = pout["pseudo_files"] or pin["pseudopotentials"]
        by_surface.setdefault(m.group("surface"), []).append(SlabEnergy(
            surface=m.group("surface"), layers=layers, run=child.name,
            n_C=n_C, n_H=n_H, area_angstrom2=area,
            energy_ry=float(pout["energy_ry"]),
            pseudo_C=pseudo.get("C", ""), pseudo_H=pseudo.get("H", ""),
            ecutwfc=float(pin["ecutwfc"]), ecutrho=float(pin["ecutrho"]),
            nspin_declared=int(pin.get("nspin") or 1),
            nspin_effective=pout.get("nspin_effective"),
            total_magnetization_bohr=pout.get("total_magnetization_bohr")))
    if not by_surface:
        raise SurfaceEnergyError(
            f"no '{prefix}~<surface>_<N>L' relaxations under {runs_dir}")
    for pts in by_surface.values():
        pts.sort(key=lambda p: p.layers)
    return by_surface


def check_reference_consistency(points: list, ref: Reference) -> None:
    """A chemical-potential subtraction mixes energies from different runs.

    Doing that across different pseudopotentials or cutoffs produces a number
    that looks perfectly reasonable and is meaningless (CLAUDE.md invariant 7).
    """
    for p in points:
        if p.pseudo_C and ref.pseudo_C and p.pseudo_C != ref.pseudo_C:
            raise SurfaceEnergyError(
                f"{p.run}: carbon pseudopotential {p.pseudo_C!r} differs from "
                f"the bulk reference's {ref.pseudo_C!r}; E_slab - N_C*mu_C is "
                f"not a meaningful subtraction")
        if p.n_H and p.pseudo_H and ref.pseudo_H and p.pseudo_H != ref.pseudo_H:
            raise SurfaceEnergyError(
                f"{p.run}: hydrogen pseudopotential {p.pseudo_H!r} differs from "
                f"the H2 reference's {ref.pseudo_H!r}")
        if (p.ecutwfc, p.ecutrho) != (ref.ecutwfc, ref.ecutrho):
            raise SurfaceEnergyError(
                f"{p.run}: cutoffs {p.ecutwfc}/{p.ecutrho} Ry differ from the "
                f"references' {ref.ecutwfc}/{ref.ecutrho} Ry; the energies "
                f"cannot be combined")


# ------------------------------------------------------- per-surface exclusion
def parse_layer_exclusions(specs) -> dict:
    """['C111=6,8', 'C100=6'] -> {'C111': [6, 8], 'C100': [6]}.

    Which thicknesses lie outside the asymptotic regime is a property of the
    surface. A bare facet has two interacting dangling-bond faces where an
    H-terminated one has two passivated faces, so it converges more slowly and
    needs more of the thin end dropped. A single global rule cannot express
    that without either keeping points it should drop on one surface or
    throwing away good points on another.
    """
    out = {}
    for spec in specs or []:
        if spec.count("=") != 1:
            raise SurfaceEnergyError(
                f"layer exclusion {spec!r} is not of the form SURFACE=N[,N...] "
                f"(for example C111=6,8)")
        surface, rhs = (s.strip() for s in spec.split("="))
        if not surface:
            raise SurfaceEnergyError(f"layer exclusion {spec!r} names no surface")
        if surface in out:
            raise SurfaceEnergyError(
                f"surface {surface!r} is given a layer exclusion twice; the "
                f"second would silently win")
        layers = []
        for tok in rhs.split(","):
            tok = tok.strip().rstrip("Ll")
            if not tok:
                continue
            try:
                layers.append(int(tok))
            except ValueError:
                raise SurfaceEnergyError(
                    f"layer exclusion {spec!r}: {tok!r} is not a layer count")
        out[surface] = sorted(set(layers))
    return out


def resolve_exclusions(surface: str, default_layers, per_surface: dict):
    """(layers_to_exclude, human-readable reason) for one surface."""
    per_surface = per_surface or {}
    if surface in per_surface:
        layers = list(per_surface[surface])
        why = (f"explicit per-surface exclusion for {surface} "
               f"({', '.join(f'{n}L' for n in layers) or 'none'})")
        return layers, why
    layers = list(default_layers or [])
    why = (f"campaign default ({', '.join(f'{n}L' for n in layers)})"
           if layers else "no exclusions")
    return layers, why


# ---------------------------------------------------------------- the fit
@dataclass
class GammaFit:
    """gamma(mu_H) = gamma_h_rich + slope * delta_mu, delta_mu >= 0 in eV."""
    surface: str
    orientation: str
    area_angstrom2: float
    n_H: int
    layers_used: list
    layers_excluded: list
    mu_c_fit_ry: float
    mu_c_bulk_ry: float
    fit_rms_ry: float
    gamma_h_rich_j_m2: float
    slope_j_m2_per_ev: float
    coverage_per_a2: float          # N_H / (2A), the slope in Angstrom^-2
    per_slab_gamma_j_m2: dict       # layers -> gamma using the bulk mu_C
    gamma_scatter_j_m2: float
    mean_n_c_used: float = 0.0
    exclusion_reason: str = ""
    epistemic_level: str = "L2"
    rejected: bool = False
    reject_reason: str = ""
    notes: list = field(default_factory=list)

    @property
    def mu_c_drift_mry(self) -> float:
        return 1000.0 * (self.mu_c_fit_ry - self.mu_c_bulk_ry)

    @property
    def gamma_bias_from_mu_c_drift_j_m2(self) -> float:
        """How far the slope drift moves gamma, in the units gamma is reported.

        Exact, not an estimate. The least-squares intercept is
        b = <E> - mu_C_fit*<N_C>, so re-fitting with the slope pinned at the
        bulk mu_C would move it by +drift*<N_C>, and gamma by that over 2A.
        Equivalently: this is gamma_boettger minus the mean of the per-slab
        gammas over the fitted ladder. Signed, so it says which way.
        """
        return (-(self.mu_c_fit_ry - self.mu_c_bulk_ry) * self.mean_n_c_used
                / (2.0 * self.area_angstrom2)) * RY_PER_A2_TO_J_PER_M2

    def gamma_at(self, delta_mu_ev: float) -> float:
        """gamma in J/m^2 at mu_H = mu_H(H-rich) - delta_mu_ev."""
        return self.gamma_h_rich_j_m2 + self.slope_j_m2_per_ev * delta_mu_ev

    def zero_crossing_ev(self):
        """delta_mu at which gamma turns positive, or None if it never does."""
        if self.slope_j_m2_per_ev <= 0:
            return None
        return -self.gamma_h_rich_j_m2 / self.slope_j_m2_per_ev


def fit_surface_energy(surface: str, points: list, ref: Reference,
                       exclude_layers, exclusion_reason: str = "") -> GammaFit:
    exclude = set(exclude_layers)
    used = [p for p in points if p.layers not in exclude]
    dropped = [p for p in points if p.layers in exclude]
    absent = sorted(exclude - {p.layers for p in points})
    if absent:
        raise SurfaceEnergyError(
            f"{surface}: asked to exclude "
            f"{', '.join(f'{n}L' for n in absent)}, which the ladder does not "
            f"contain (it has {', '.join(f'{p.layers}L' for p in points)}). A "
            f"misdirected exclusion silently leaves the bad point in the fit")
    if len(used) < 3:
        raise SurfaceEnergyError(
            f"{surface}: {len(used)} points left after excluding "
            f"{', '.join(f'{p.layers}L' for p in dropped) or 'nothing'}; too "
            f"few for a two-parameter fit with a meaningful residual")
    areas = {round(p.area_angstrom2, 6) for p in used}
    if len(areas) != 1:
        raise SurfaceEnergyError(
            f"{surface}: in-plane cell area differs across the ladder "
            f"({sorted(areas)}); gamma per unit area is then not comparable")
    n_h = {p.n_H for p in used}
    if len(n_h) != 1:
        raise SurfaceEnergyError(
            f"{surface}: hydrogen count differs across the ladder "
            f"({sorted(n_h)}); the termination is not constant")

    area = used[0].area_angstrom2
    n_H = used[0].n_H

    N = np.array([p.n_C for p in used], dtype=float)
    E = np.array([p.energy_ry for p in used], dtype=float)
    design = np.vstack([N, np.ones_like(N)]).T
    coeffs, *_ = np.linalg.lstsq(design, E, rcond=None)
    mu_c_fit, intercept = float(coeffs[0]), float(coeffs[1])
    resid = E - design @ coeffs

    gamma_ry_per_a2 = (intercept - n_H * ref.mu_h_rich_ry) / (2.0 * area)
    gamma = gamma_ry_per_a2 * RY_PER_A2_TO_J_PER_M2

    coverage = n_H / (2.0 * area)
    slope = coverage * EV_PER_A2_TO_J_PER_M2

    # Cross-check: gamma per slab using the independent bulk mu_C. Its spread
    # across the ladder is an honest uncertainty on gamma.
    per_slab = {}
    for p in points:
        g = ((p.energy_ry - p.n_C * ref.mu_c_bulk_ry - p.n_H * ref.mu_h_rich_ry)
             / (2.0 * p.area_angstrom2)) * RY_PER_A2_TO_J_PER_M2
        per_slab[p.layers] = g
    used_gammas = [per_slab[p.layers] for p in used]
    scatter = max(used_gammas) - min(used_gammas)

    return GammaFit(
        surface=surface, orientation=ORIENTATION.get(surface, f"({surface[1:]})"),
        area_angstrom2=area, n_H=n_H,
        layers_used=[p.layers for p in used],
        layers_excluded=[p.layers for p in dropped],
        mu_c_fit_ry=mu_c_fit, mu_c_bulk_ry=ref.mu_c_bulk_ry,
        fit_rms_ry=float(np.sqrt(np.mean(resid ** 2))),
        gamma_h_rich_j_m2=gamma, slope_j_m2_per_ev=slope,
        coverage_per_a2=coverage, per_slab_gamma_j_m2=per_slab,
        gamma_scatter_j_m2=scatter, mean_n_c_used=float(np.mean(N)),
        exclusion_reason=exclusion_reason)


def _exclusion_hint(f: GammaFit, flag: str) -> str:
    """The concrete next command, given the per-slab gammas already computed.

    Points whose per-slab gamma sits far from the ladder's own median are the
    ones dragging the slope; naming them is more useful than telling the user
    to go and look.
    """
    used = {n: g for n, g in f.per_slab_gamma_j_m2.items() if n in f.layers_used}
    if len(used) < 4:
        return (f"the ladder has {len(used)} usable points; excluding any leaves "
                f"too few to fit. Extend it before trying to salvage it.")
    med = float(np.median(list(used.values())))
    outliers = sorted(n for n, g in used.items()
                      if abs(g - med) > max(0.05, 3 * f.gamma_scatter_j_m2 / 4))
    keep = [n for n in sorted(used) if n not in outliers]
    if not outliers or len(keep) < 3:
        return ("no subset of the ladder is obviously to blame; the whole "
                "ladder is likely outside the asymptotic regime.")
    drop = sorted(set(f.layers_excluded) | set(outliers))
    return (f"per-slab gamma singles out "
            f"{', '.join(f'{n}L' for n in outliers)} "
            f"({', '.join(f'{n}L {used[n]:+.3f}' for n in sorted(used))} J/m^2, "
            f"median {med:+.3f}). If those are the unconverged ones, refit with "
            f"`{flag} {f.surface}={','.join(str(n) for n in drop)}` and say so "
            f"in the report.")


def grade_fits(fits: list, mu_c_tol_mry: float,
               mu_c_reject_j_m2: float = DEFAULT_MU_C_REJECT_J_M2,
               exclusion_flag: str = "--exclude-layers-for") -> list:
    """Attach an epistemic level and warnings based on the fit's own diagnostics.

    Raises rather than warning once the slope drift biases gamma by more than
    `mu_c_reject_j_m2`. Nothing downstream reads warnings, and a gamma biased
    by several J/m^2 is not a noisy number -- it is a different number. See
    DEFAULT_MU_C_REJECT_J_M2 for why the budget is stated in J/m^2.
    """
    rejected = []
    for f in fits:
        bias = f.gamma_bias_from_mu_c_drift_j_m2
        if abs(bias) > mu_c_reject_j_m2:
            f.rejected = True
            f.epistemic_level = "REJECTED"
            f.reject_reason = (
                f"{f.surface}: fitted mu_C = {f.mu_c_fit_ry:.9f} Ry drifts "
                f"{f.mu_c_drift_mry:+.4f} mRy from the bulk reference "
                f"{f.mu_c_bulk_ry:.9f} Ry. Over the fitted ladder "
                f"({', '.join(f'{n}L' for n in f.layers_used)}, <N_C> = "
                f"{f.mean_n_c_used:.1f}, A = {f.area_angstrom2:.4f} "
                f"Angstrom^2) that biases gamma by {bias:+.3f} J/m^2, past the "
                f"{mu_c_reject_j_m2:.3f} J/m^2 budget. Excluded so far: "
                f"{', '.join(f'{n}L' for n in f.layers_excluded) or 'nothing'} "
                f"({f.exclusion_reason or 'unspecified'}). "
                f"gamma = {f.gamma_h_rich_j_m2:+.4f} J/m^2 from this fit is "
                f"NOT reported. {_exclusion_hint(f, exclusion_flag)}")
            rejected.append(f)
    if rejected:
        raise SurfaceEnergyError(
            "the fitted carbon chemical potential is too far from bulk for "
            + ("this ladder" if len(rejected) == 1 else
               f"{len(rejected)} ladders")
            + " to give a usable surface energy:\n\n"
            + "\n\n".join(f"  * {f.reject_reason}" for f in rejected))

    warnings = []
    for f in fits:
        if abs(f.mu_c_drift_mry) > mu_c_tol_mry:
            f.epistemic_level = "L1"
            f.notes.append(
                f"fitted mu_C drifts {f.mu_c_drift_mry:+.4f} mRy from the bulk "
                f"reference (tolerance {mu_c_tol_mry:.3f} mRy)")
            warnings.append(
                f"{f.surface}: fitted mu_C = {f.mu_c_fit_ry:.9f} Ry drifts "
                f"{f.mu_c_drift_mry:+.4f} mRy from the bulk reference "
                f"{f.mu_c_bulk_ry:.9f} Ry. The slab interior is not perfectly "
                f"bulk-like, or the ladder is not in the asymptotic regime. "
                f"gamma for this surface is downgraded to L1; per-slab scatter "
                f"is {f.gamma_scatter_j_m2:.4f} J/m^2 and the drift biases "
                f"gamma by {f.gamma_bias_from_mu_c_drift_j_m2:+.4f} J/m^2 "
                f"(budget {mu_c_reject_j_m2:.3f}).")
        if f.gamma_scatter_j_m2 > 0.01:
            f.notes.append(
                f"per-slab gamma scatters by {f.gamma_scatter_j_m2:.4f} J/m^2 "
                f"across the fitted ladder")
    return warnings


# ============================================================================
# Mapping delta_mu -> (T, p_H2) via the ideal-gas chemical potential of H2
# ============================================================================
# WHAT IS COMPUTED HERE AND WHAT IS NOT
#
# Computed by DFT in this project:
#   * E(H2), the total energy of the relaxed molecule;
#   * the slab total energies, hence gamma at the H-rich limit and its slope
#     dgamma/d(-mu_H) = N_H/2A.
#
# NOT computed here -- textbook ideal-gas statistical thermodynamics, using
# spectroscopic constants of H2 taken from the literature:
#   * the translational, rotational and vibrational contributions to
#     mu_H2(T, p) below. No molecular dynamics, no phonons, no anharmonicity.
#
# The model is rigid-rotor / harmonic-oscillator (RRHO) with an explicit sum
# over rotational levels rather than the high-temperature limit, because H2's
# rotational temperature (87.6 K) is large enough that the high-T limit is a
# visible approximation at room temperature. Validated against the NIST-JANAF
# tabulation in tests/test_surface_energy.py: agreement is better than 15 meV
# from 298 K to 2000 K, which is roughly +/-15 K on any temperature this
# mapping reports.
#
# ZERO-POINT ENERGY IS EXCLUDED BY DEFAULT, and that is a deliberate
# consistency choice, not an oversight. gamma was derived from DFT total
# energies with no vibrational contribution on either side. Adding the H2 ZPE
# (0.273 eV, so 0.136 eV per H) without the corresponding zero-point energy of
# the ADSORBED hydrogen would be an unbalanced correction, and the adsorbed-H
# ZPE cannot be computed here -- this repository has no phonon calculation
# (CLAUDE.md sec 3). The two terms are of similar size and partially cancel.
# `--include-zpe` applies the H2 side alone for sensitivity testing and is
# labelled as unbalanced wherever it is used.
# ============================================================================

KB_SI = 1.380649e-23             # J/K
H_PLANCK_SI = 6.62607015e-34     # J s
AMU_KG = 1.66053906660e-27
KB_EV = 8.617333262e-5           # eV/K
PA_PER_BAR = 1.0e5
PA_PER_TORR = 133.322368421

# H2 spectroscopic constants (Huber & Herzberg, Constants of Diatomic
# Molecules). Literature values, not computed in this project.
H2_MASS_AMU = 2.01588
H2_THETA_ROT_K = 87.55           # from B0 = 60.853 cm^-1
H2_THETA_VIB_K = 6332.5          # from nu = 4401.21 cm^-1
H2_SYMMETRY_NUMBER = 2           # homonuclear

# NIST-JANAF H2 reference values, G(T) - H(0) at 1 bar, eV. Used ONLY to
# validate and to quote the uncertainty of the RRHO model; never in the
# computation path.
JANAF_H2_G_MINUS_H0_EV = {
    298.15: -0.3160, 500.0: -0.6065, 1000.0: -1.4207,
    1500.0: -2.3053, 2000.0: -3.2690,
}
RRHO_VS_JANAF_MAX_DEV_EV = 0.015


def h2_rotational_partition_function(T: float, j_max: int = 80) -> float:
    """Explicit sum over rigid-rotor levels, divided by the symmetry number.

    Below roughly 150 K this is not right for hydrogen: ortho/para nuclear-spin
    statistics dominate and the equilibrium composition is temperature
    dependent. The mapping refuses to run below `MIN_VALID_TEMPERATURE_K`.
    """
    return sum((2 * J + 1) * math.exp(-J * (J + 1) * H2_THETA_ROT_K / T)
               for J in range(j_max)) / H2_SYMMETRY_NUMBER


MIN_VALID_TEMPERATURE_K = 150.0


def h2_mu_shift_ev(T: float, p_pa: float, include_zpe: bool = False) -> float:
    """mu_H2(T, p) - E(H2), in eV: everything beyond the DFT total energy.

    Translational + rotational + vibrational, ideal gas. Negative and growing
    in magnitude with T (the entropy term dominates) and with falling pressure.
    """
    if T < MIN_VALID_TEMPERATURE_K:
        raise SurfaceEnergyError(
            f"T = {T} K is below {MIN_VALID_TEMPERATURE_K} K, where the "
            f"rigid-rotor treatment of H2 breaks down (ortho/para nuclear-spin "
            f"statistics). Refusing to extrapolate.")
    if p_pa <= 0:
        raise SurfaceEnergyError(f"pressure must be positive, got {p_pa} Pa")
    lam = H_PLANCK_SI / math.sqrt(
        2.0 * math.pi * H2_MASS_AMU * AMU_KG * KB_SI * T)
    kT = KB_EV * T
    mu_trans = -kT * math.log(KB_SI * T / (p_pa * lam ** 3))
    mu_rot = -kT * math.log(h2_rotational_partition_function(T))
    mu_vib = kT * math.log(1.0 - math.exp(-H2_THETA_VIB_K / T))
    if include_zpe:
        mu_vib += 0.5 * KB_EV * H2_THETA_VIB_K
    return mu_trans + mu_rot + mu_vib


def delta_mu_from_tp(T: float, p_pa: float, include_zpe: bool = False) -> float:
    """delta_mu = mu_H(H-rich) - mu_H(T, p), in eV.

    mu_H = mu_H2 / 2, so delta_mu = -[mu_H2(T,p) - E(H2)] / 2. Positive means
    hydrogen-poorer than the H-rich limit, which is the direction that raises
    every surface energy.
    """
    return -h2_mu_shift_ev(T, p_pa, include_zpe) / 2.0


def temperature_for_delta_mu(delta_mu_ev: float, p_pa: float,
                             include_zpe: bool = False,
                             t_lo: float = MIN_VALID_TEMPERATURE_K,
                             t_hi: float = 4000.0):
    """Temperature at which delta_mu(T, p) reaches the target, or None.

    None means the condition is out of reach at this pressure within the
    bracketed temperature range -- which is itself a reportable result.
    """
    def f(T):
        return delta_mu_from_tp(T, p_pa, include_zpe) - delta_mu_ev
    if f(t_lo) > 0:
        return t_lo if abs(f(t_lo)) < 1e-9 else None
    if f(t_hi) < 0:
        return None
    lo, hi = t_lo, t_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


DEFAULT_TP_PRESSURES_BAR = (1.0, 1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12)


def tp_curve(delta_mu_ev: float, pressures_bar=DEFAULT_TP_PRESSURES_BAR,
             include_zpe: bool = False, t_hi: float = 4000.0) -> list:
    """The (T, p) locus along which delta_mu equals the given value."""
    rows = []
    for p_bar in pressures_bar:
        p_pa = p_bar * PA_PER_BAR
        T = temperature_for_delta_mu(delta_mu_ev, p_pa, include_zpe, t_hi=t_hi)
        rows.append({
            "delta_mu_ev": delta_mu_ev,
            "p_h2_bar": p_bar,
            "p_h2_pa": p_pa,
            "p_h2_torr": p_pa / PA_PER_TORR,
            "temperature_k": T,
            "temperature_c": None if T is None else T - 273.15,
            "reachable": T is not None,
        })
    return rows


# ---------------------------------------------------------------------------
# The zero-point-energy systematic -- the DOMINANT uncertainty on any
# temperature this mapping reports, roughly thirty times the ideal-gas fit
# residual. It is quoted, not applied.
#
# gamma was built from DFT total energies with no vibrational term anywhere.
# Restoring zero-point energy consistently adds N_H * ZPE_ads to E_slab and
# ZPE(H2)/2 to mu_H, so
#
#     gamma_with_zpe(delta_mu) = gamma(delta_mu + DELTA_ZPE)
#
# with DELTA_ZPE = ZPE_ads(per H) - ZPE(H2)/2. Because the C-H zero-point
# energy is nearly the same on all three facets, the correction enters through
# N_H/2A exactly as the mu_H term does -- it is a rigid SHIFT OF THE delta_mu
# AXIS, not a per-facet reshuffle. Every facet ordering and every crossing
# therefore moves together, and the shape at a given delta_mu is unchanged; it
# is the (T, p) required to reach that delta_mu that moves.
#
# DELTA_ZPE > 0, so the shift makes every condition MORE accessible: the same
# surface chemistry is reached at a lower temperature than the no-ZPE numbers
# suggest.
#
# The frequencies below are literature values for monohydride C-H on diamond,
# used only to size the systematic. `make_h_phonons.py` / `analyze_h_phonons.py`
# compute the real number from frozen-phonon displacements; until that campaign
# has run, this is an estimate and is labelled as one everywhere it appears.
# ---------------------------------------------------------------------------
CM1_TO_EV = 1.23984198e-4

# Monohydride C-H on diamond: one stretch, two bends.
ZPE_CH_STRETCH_CM1 = 2900.0
ZPE_CH_BEND_CM1 = 1250.0
ZPE_ESTIMATE_SOURCE = ("literature monohydride C-H frequencies on diamond "
                       "(stretch ~2900 cm^-1, two bends ~1250 cm^-1); an "
                       "ESTIMATE pending make_h_phonons.py / "
                       "analyze_h_phonons.py")


def zpe_adsorbed_h_ev(stretch_cm1: float = ZPE_CH_STRETCH_CM1,
                      bend_cm1: float = ZPE_CH_BEND_CM1) -> float:
    """Zero-point energy of one adsorbed H: half the sum of its three modes."""
    return 0.5 * (stretch_cm1 + 2.0 * bend_cm1) * CM1_TO_EV


def zpe_h2_per_h_ev() -> float:
    return 0.5 * (0.5 * KB_EV * H2_THETA_VIB_K) * 2.0 / 2.0


def delta_zpe_ev(stretch_cm1: float = ZPE_CH_STRETCH_CM1,
                 bend_cm1: float = ZPE_CH_BEND_CM1) -> float:
    """DELTA_ZPE = ZPE_ads(per H) - ZPE(H2)/2, in eV.

    Positive: an adsorbed H is stiffer than half an H2 molecule. Adding it
    shifts the delta_mu axis by this amount, in the direction that makes every
    condition easier to reach.
    """
    return zpe_adsorbed_h_ev(stretch_cm1, bend_cm1) - zpe_h2_per_h_ev()


def zpe_systematic_on_temperature(delta_mu_ev: float, p_pa: float,
                                  delta_zpe_ev_value: float = None) -> dict:
    """How far the missing ZPE moves a crossing temperature at one pressure.

    Returns the no-ZPE temperature, the ZPE-corrected temperature, and the
    shift. The shift is pressure dependent because d(delta_mu)/dT carries the
    (k/2)*ln(p0/p) term, so it grows as the pressure falls.
    """
    dz = delta_zpe_ev() if delta_zpe_ev_value is None else delta_zpe_ev_value
    t_plain = temperature_for_delta_mu(delta_mu_ev, p_pa)
    t_zpe = temperature_for_delta_mu(max(delta_mu_ev - dz, 0.0), p_pa)
    return {
        "delta_zpe_ev": dz,
        "delta_mu_no_zpe_ev": delta_mu_ev,
        "delta_mu_with_zpe_ev": delta_mu_ev - dz,
        "temperature_no_zpe_k": t_plain,
        "temperature_with_zpe_k": t_zpe,
        "shift_k": (None if (t_plain is None or t_zpe is None)
                    else t_zpe - t_plain),
    }


def rrho_validation() -> dict:
    """Deviation of this RRHO model from the NIST-JANAF tabulation."""
    devs = {}
    for T, ref in sorted(JANAF_H2_G_MINUS_H0_EV.items()):
        model = h2_mu_shift_ev(T, PA_PER_BAR, include_zpe=False)
        devs[T] = {"model_ev": model, "janaf_ev": ref, "deviation_ev": model - ref}
    worst = max(abs(v["deviation_ev"]) for v in devs.values())
    return {"points": devs, "max_abs_deviation_ev": worst,
            "passes": worst <= RRHO_VS_JANAF_MAX_DEV_EV}


def temperature_uncertainty_k(delta_mu_ev: float, p_pa: float,
                              dev_ev: float = RRHO_VS_JANAF_MAX_DEV_EV) -> float:
    """Convert the RRHO-vs-JANAF deviation into a temperature uncertainty."""
    T = temperature_for_delta_mu(delta_mu_ev, p_pa)
    if T is None:
        return float("nan")
    h = 5.0
    slope = ((delta_mu_from_tp(T + h, p_pa) - delta_mu_from_tp(T - h, p_pa))
             / (2 * h))
    return abs(dev_ev / 2.0 / slope) if slope else float("nan")


# ===================================================== dehydrogenation bound
# gamma_H(mu_H) rises without limit in this model because only H-terminated
# facets were ever calculated. The physical stop is dehydrogenation: once
#
#     gamma_H(delta_mu)  >  gamma_bare
#
# the bare surface is the stable one and every H-terminated statement beyond
# that delta_mu describes a surface that is not there. gamma_bare carries no
# hydrogen, so it does not move with mu_H, and the inequality becomes a hard
# ceiling:
#
#     delta_mu_max = (gamma_bare - gamma_H_rich) / (N_H / 2A)
#
# The bound needs the LOWEST bare surface energy for each facet. On (111) the
# unreconstructed 1x1 is well above the Pandey 2x1 reconstruction, so using it
# alone gives a ceiling that is too high -- i.e. too permissive, claiming the
# H-terminated surface survives further than it does. `make_bare_slabs.py`
# generates both and this function takes the minimum.
# ===========================================================================
#
# THE SPIN STATE OF THE BARE FACET IS PART OF THE CEILING
#
# A bare (111) face carries one unpaired electron per dangling bond. A
# spin-polarised run settles at a total magnetization of 2.00 Bohr magnetons
# per symmetric cell -- exactly one per face -- and holds it. A
# non-spin-polarised run cannot represent that state at all: it forces the two
# spin channels to be identical and pays an energy penalty for it, so it
# returns a total energy ABOVE the true ground state.
#
# The DIRECTION of the error is fixed, and it is the unsafe one. nspin = 2 is
# variational over nspin = 1 -- the closed-shell solution is available to the
# spin-polarised calculation and is simply not the minimum -- so
#
#     E_bare(non-polarised)  >=  E_bare(polarised)
#
# always, at the same geometry and smearing. E_slab enters gamma with a plus
# sign, so a non-spin-polarised bare ladder gives a gamma_bare that is too
# HIGH, hence a delta_mu ceiling that is too HIGH: too PERMISSIVE. It claims
# the H-terminated surface survives to hydrogen-poorer conditions than it
# does. That is the same direction of error `make_bare_slabs.py` already warns
# about for the (111) reconstruction, and the one direction this bound must
# not fail in.
#
# The error also does not average out and is invisible to every fit
# diagnostic: the ladder can be perfectly linear, the mu_C drift zero, the
# per-slab scatter 0.0001 J/m^2, and the number is still the energy of the
# wrong electronic state. Same class of failure as the mu_C artifact -- a
# ceiling that silently binds with every internal check passing.
#
# So the spin state is checked against pw.out, not assumed, and it is checked
# ACROSS the campaign: once a spin-polarised result exists for any thickness of
# a facet, a non-spin-polarised energy for that facet is known-wrong and is
# refused rather than warned about. Where no spin-polarised run exists anywhere
# for that facet, the ceiling is still computed -- refusing outright would
# leave no bound at all, which is worse -- but it is labelled and warned about
# loudly, and the warning names the facet so it can be queued.
#
# (100) is expected to quench its dangling bonds by dimerisation and so to be
# genuinely closed-shell, but "expected" is not "tested"; it gets the same
# treatment as the others until a spin-polarised run says otherwise.
#
BARE_SURFACE_KEY = re.compile(r"^(C\d{3})([a-z]*)$")

# A dangling bond holds one unpaired electron; a symmetric slab has two faces.
# Used only to say whether an observed magnetization looks like the expected
# state or like something else that needs a human.
EXPECTED_MAG_PER_SYMMETRIC_CELL = 2.0


def spin_audit(bare_by_surface: dict) -> dict:
    """Per facet: which thicknesses ran spin-polarised, and what that implies.

    Keyed by BASE facet (C111, not C111pandey): a reconstruction variant is the
    same dangling-bond chemistry, so a spin-polarised result on any variant is
    evidence about the facet.
    """
    audit = {}
    for surface, points in bare_by_surface.items():
        base = bare_base_surface(surface)
        a = audit.setdefault(base, {
            "facet": base, "variants": [], "polarised": [], "unpolarised": [],
            "magnetizations": {}, "declared_but_not_run": [],
        })
        if surface not in a["variants"]:
            a["variants"].append(surface)
        for p in points:
            tag = f"{surface}_{p.layers}L"
            if p.spin_polarised:
                a["polarised"].append(tag)
                if p.total_magnetization_bohr is not None:
                    a["magnetizations"][tag] = p.total_magnetization_bohr
            else:
                a["unpolarised"].append(tag)
                # Declared nspin = 2 but pw.out shows no spin channels: the run
                # is not the calculation the input claims (invariant 7).
                if p.nspin_declared == 2 and p.nspin_effective == 1:
                    a["declared_but_not_run"].append(tag)
    for a in audit.values():
        a["has_polarised_reference"] = bool(a["polarised"])
        a["mixed"] = bool(a["polarised"]) and bool(a["unpolarised"])
    return audit


# ---------------------------------------------------------------------------
# MEASURING E_s, AND WHY THE CEILING CORRECTION IS EXACTLY E_s
#
# Pair a spin-polarised bare relaxation with its non-polarised counterpart at
# the same thickness and the difference is the exchange stabilisation of the
# slab's unpaired electrons. Divided by the number of dangling bonds in the
# cell it is a per-dangling-bond energy:
#
#     E_s = [ E_bare(nspin=1) - E_bare(nspin=2) ] / n_DB      [eV per DB]
#
# n_DB is one per dangling bond per face, two faces. On (111)-1x1 that is 2 per
# cell, so E_s is the difference over 2.
#
# Applying it to the ceiling needs no refit, because the geometry cancels
# exactly. With one H per dangling bond on the H-terminated face:
#
#     slope        = N_H / 2A * EV_PER_A2 = n_DB / 2A * EV_PER_A2
#     d(gamma_bare) = n_DB * E_s / 2A * EV_PER_A2       (both faces, per face)
#     d(ceiling)    = d(gamma_bare) / slope = E_s
#
# So every facet's ceiling drops by the SAME E_s, in eV, independent of area,
# coverage and thickness. The ranking of facets is untouched; the whole
# delta_mu axis of ceilings slides down by E_s.
#
# WHERE THE IDENTITY BREAKS: it needs n_DB(bare) = N_H(H-terminated), i.e. an
# unreconstructed 1x1 bare face whose dangling bonds are all still there. A
# Pandey-reconstructed (111) and a dimerised (100) both quench dangling bonds,
# so for them n_DB(bare) < N_H and the cancellation fails.
#
# The measurement validates its own premise: one unpaired electron per intact
# dangling bond means the total magnetization of the polarised run should equal
# n_DB. So
#
#   * |M| ~ n_DB      -> dangling bonds intact, identity holds, apply E_s
#   * |M| ~ 0         -> fully quenched, E_s ~ 0, no correction (this is the
#                        expected answer for dimerised (100))
#   * anything else   -> partially quenched; E_s is reported but NOT applied,
#                        because the cancellation is no longer exact and the
#                        correct n_DB is a structural question for a human
#
# This is what makes (110) decisive. It is the candidate binding facet; if its
# bare face comes back at 0.00 Bohr magnetons it takes no correction and the
# margin to the interior-pressure sign change survives intact, and if it comes
# back magnetic like (111) the margin shrinks by E_s.
# ---------------------------------------------------------------------------
MAG_TOLERANCE_BOHR = 0.1


@dataclass(frozen=True)
class SpinStabilisation:
    """E_s per dangling bond for one facet, from paired nspin=1/2 runs."""
    surface: str
    n_db_per_cell: int
    per_thickness_ev: dict          # layers -> E_s, eV per dangling bond
    magnetization_bohr: dict        # layers -> total magnetization of nspin=2
    e_s_ev: float
    spread_ev: float
    quenched: bool
    identity_applies: bool
    reason: str

    @property
    def thicknesses(self) -> list:
        return sorted(self.per_thickness_ev)


def measure_spin_stabilisation(unpolarised: list, polarised: list,
                               n_db_per_cell: int) -> SpinStabilisation:
    """E_s from same-thickness nspin=1 / nspin=2 pairs of one bare facet."""
    surface = (polarised or unpolarised)[0].surface
    if n_db_per_cell <= 0:
        raise SurfaceEnergyError(
            f"{surface}: dangling-bond count per cell must be positive, got "
            f"{n_db_per_cell}; E_s per dangling bond is undefined without it")
    nsp = {p.layers: p for p in unpolarised if not p.spin_polarised}
    sp = {p.layers: p for p in polarised if p.spin_polarised}
    shared = sorted(set(nsp) & set(sp))
    if not shared:
        raise SurfaceEnergyError(
            f"{surface}: no thickness has both a spin-polarised and a "
            f"non-spin-polarised run (nspin=1 at "
            f"{', '.join(f'{n}L' for n in sorted(nsp)) or 'none'}; nspin=2 at "
            f"{', '.join(f'{n}L' for n in sorted(sp)) or 'none'}). E_s is the "
            f"difference of a PAIR; it cannot be taken across thicknesses, "
            f"which would fold in the bulk term the pairing exists to cancel")

    per, mags = {}, {}
    for n in shared:
        if nsp[n].n_C != sp[n].n_C or abs(
                nsp[n].area_angstrom2 - sp[n].area_angstrom2) > 1e-6:
            raise SurfaceEnergyError(
                f"{surface} {n}L: the nspin=1 and nspin=2 runs differ in cell "
                f"({nsp[n].n_C} vs {sp[n].n_C} carbon, "
                f"{nsp[n].area_angstrom2:.4f} vs {sp[n].area_angstrom2:.4f} "
                f"Angstrom^2). Their energy difference is not a spin "
                f"stabilisation")
        d_ry = nsp[n].energy_ry - sp[n].energy_ry
        if d_ry < -1e-6:
            raise SurfaceEnergyError(
                f"{surface} {n}L: the spin-polarised run is HIGHER in energy "
                f"than the non-polarised one by {-d_ry * RY_TO_EV:.4f} eV. "
                f"nspin=2 is variational over nspin=1, so this cannot happen "
                f"for the same geometry and settings -- the pair is not a "
                f"pair, or one did not reach its ground state")
        per[n] = d_ry * RY_TO_EV / n_db_per_cell
        if sp[n].total_magnetization_bohr is not None:
            mags[n] = sp[n].total_magnetization_bohr

    e_s = float(np.mean(list(per.values())))
    spread = max(per.values()) - min(per.values())

    observed = [abs(m) for m in mags.values()]
    quenched = bool(observed) and all(m < MAG_TOLERANCE_BOHR for m in observed)
    intact = bool(observed) and all(
        abs(m - n_db_per_cell) < MAG_TOLERANCE_BOHR for m in observed)
    if not observed:
        applies, why = False, (
            "the spin-polarised runs recorded no total magnetization, so "
            "whether the dangling bonds are intact cannot be confirmed")
    elif quenched:
        applies, why = False, (
            f"total magnetization is 0 within {MAG_TOLERANCE_BOHR} Bohr "
            f"mag/cell: the dangling bonds are quenched, E_s = "
            f"{e_s:+.4f} eV/DB is consistent with zero, and no correction "
            f"applies")
    elif intact:
        applies, why = True, (
            f"total magnetization {', '.join(f'{v:.2f}' for v in mags.values())}"
            f" matches n_DB = {n_db_per_cell}: the dangling bonds are intact, "
            f"one unpaired electron each, so n_DB(bare) = N_H and the "
            f"cancellation is exact")
    else:
        applies, why = False, (
            f"total magnetization {', '.join(f'{v:.2f}' for v in mags.values())}"
            f" is neither 0 nor n_DB = {n_db_per_cell}: the face is PARTIALLY "
            f"quenched, so n_DB(bare) != N_H and d(ceiling) = E_s no longer "
            f"holds. E_s is reported but not applied; the right dangling-bond "
            f"count is a structural question")
    return SpinStabilisation(
        surface=surface, n_db_per_cell=n_db_per_cell, per_thickness_ev=per,
        magnetization_bohr=mags, e_s_ev=e_s, spread_ev=spread,
        quenched=quenched, identity_applies=applies, reason=why)


def apply_spin_correction(bound: dict, stabilisations: dict) -> dict:
    """Slide each ceiling down by its facet's E_s, keeping the raw one visible.

    The corrected and uncorrected ceilings are both reported, and the binding
    facet is recomputed on the corrected values -- correcting some facets and
    not others can in principle change which one binds, even though a UNIFORM
    E_s cannot.
    """
    corrected = {}
    for s, e in bound.get("per_surface", {}).items():
        if not e.get("available"):
            continue
        st = stabilisations.get(s)
        e["delta_mu_max_uncorrected_ev"] = e["delta_mu_max_ev"]
        if st is None:
            e["spin_correction_applied"] = False
            e["spin_correction_reason"] = (
                "no paired nspin=1/nspin=2 runs for this facet; E_s not "
                "measured")
        else:
            e["e_s_ev"] = st.e_s_ev
            e["e_s_spread_ev"] = st.spread_ev
            e["e_s_thicknesses"] = st.thicknesses
            e["e_s_n_db_per_cell"] = st.n_db_per_cell
            e["spin_correction_applied"] = st.identity_applies
            e["spin_correction_reason"] = st.reason
            if st.identity_applies:
                e["delta_mu_max_ev"] = e["delta_mu_max_ev"] - st.e_s_ev
        corrected[s] = e["delta_mu_max_ev"]
    if corrected:
        binding = min(corrected, key=corrected.get)
        bound["binding_surface_uncorrected"] = bound.get("binding_surface")
        bound["delta_mu_max_uncorrected_ev"] = bound.get("delta_mu_max_ev")
        bound["binding_surface"] = binding
        bound["delta_mu_max_ev"] = corrected[binding]
    bound["spin_stabilisation"] = {
        s: {"e_s_ev": st.e_s_ev, "spread_ev": st.spread_ev,
            "n_db_per_cell": st.n_db_per_cell,
            "per_thickness_ev": st.per_thickness_ev,
            "magnetization_bohr": st.magnetization_bohr,
            "quenched": st.quenched, "identity_applies": st.identity_applies,
            "reason": st.reason}
        for s, st in stabilisations.items()}
    return bound


def check_bare_spin_states(bare_by_surface: dict) -> list:
    """Refuse known-wrong spin states; warn loudly about untested ones.

    Returns warnings. Raises when a facet has a spin-polarised result AND a
    non-spin-polarised one, because then the non-polarised energies are not
    merely unvalidated -- they are known to be the wrong electronic state, and
    mixing the two inside one ladder also breaks the Boettger fit, whose whole
    premise is that every rung differs only by bulk carbon.
    """
    audit = spin_audit(bare_by_surface)
    warnings, fatal = [], []
    for base, a in sorted(audit.items()):
        if a["declared_but_not_run"]:
            fatal.append(
                f"{base}: {', '.join(a['declared_but_not_run'])} declare "
                f"nspin = 2 in pw.in but pw.out shows no spin channels. The "
                f"run is not the calculation the input describes; do not use "
                f"its energy (CLAUDE.md invariant 7).")
            continue
        if a["mixed"]:
            mags = ", ".join(f"{k} {v:+.2f}" for k, v in
                             sorted(a["magnetizations"].items())) or "none recorded"
            fatal.append(
                f"{base}: a spin-polarised result exists "
                f"({', '.join(a['polarised'])}; total magnetization {mags} "
                f"Bohr mag/cell) but {', '.join(a['unpolarised'])} ran "
                f"non-spin-polarised. Those energies are the wrong electronic "
                f"state, and a ladder mixing the two is not a ladder differing "
                f"only by bulk carbon. Rerun them spin-polarised, or exclude "
                f"those thicknesses explicitly with --bare-exclude-layers-for.")
            continue
        if not a["has_polarised_reference"]:
            warnings.append(
                f"bare {base} ({', '.join(a['variants'])}): NO spin-polarised "
                f"run exists at any thickness, so gamma_bare and the "
                f"dehydrogenation ceiling derived from it assume a closed-shell "
                f"bare surface that has not been tested. A bare face carries "
                f"one unpaired electron per dangling bond; if this facet does "
                f"not quench them, the non-polarised energy is too high, so "
                f"gamma_bare is too high and this ceiling is too PERMISSIVE. "
                f"Queue a spin-polarised run at one thickness.")
            continue
        odd = {k: v for k, v in a["magnetizations"].items()
               if abs(abs(v) - EXPECTED_MAG_PER_SYMMETRIC_CELL) > 0.1
               and abs(v) > 0.1}
        if odd:
            warnings.append(
                f"bare {base}: total magnetization "
                + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(odd.items()))
                + f" Bohr mag/cell, not the "
                  f"{EXPECTED_MAG_PER_SYMMETRIC_CELL:.1f} expected for one "
                  f"unpaired electron per dangling bond on each of two faces. "
                  f"Either the face is partly quenched or the slab is not the "
                  f"structure assumed; check before using the ceiling.")
    if fatal:
        raise SurfaceEnergyError(
            "bare-facet spin states are not usable for a dehydrogenation "
            "ceiling:\n\n" + "\n\n".join(f"  * {m}" for m in fatal))
    return warnings


def bare_base_surface(tag: str) -> str:
    """'C111pandey' -> 'C111'. The reconstruction is a variant of the facet."""
    m = BARE_SURFACE_KEY.match(tag)
    if not m:
        raise SurfaceEnergyError(f"unrecognised bare surface tag {tag!r}")
    return m.group(1)


def lowest_bare_gamma(bare_fits: list) -> dict:
    """Per facet, the lowest gamma_bare across all reconstruction variants."""
    best = {}
    for f in bare_fits:
        if f.n_H:
            raise SurfaceEnergyError(
                f"{f.surface}: a bare-facet fit must contain no hydrogen, "
                f"found N_H = {f.n_H}")
        if f.rejected:
            # Belt and braces: grade_fits already raises. Dropping a rejected
            # variant here instead would raise the minimum over the remaining
            # ones, which makes the ceiling too PERMISSIVE -- the one direction
            # of error this bound must never fail in.
            raise SurfaceEnergyError(
                f"{f.surface}: rejected fit reached lowest_bare_gamma. "
                f"{f.reject_reason}")
        base = bare_base_surface(f.surface)
        if base not in best or f.gamma_h_rich_j_m2 < best[base].gamma_h_rich_j_m2:
            best[base] = f
    return best


def dehydrogenation_bound(h_fits: list, bare_fits: list) -> dict:
    """delta_mu ceiling per facet, and the binding one overall.

    Returns per-facet entries plus 'binding_surface' and
    'delta_mu_max_ev' -- the smallest ceiling, since the first facet to
    dehydrogenate invalidates the H-terminated Wulff construction.
    """
    if not bare_fits:
        return {
            "available": False,
            "reason": ("no bare-facet ladder found. gamma_H(mu_H) is therefore "
                       "unbounded above in this model and every hydrogen-poor "
                       "statement is an extrapolation with no stop. Generate "
                       "the campaign with make_bare_slabs.py and rerun with "
                       "--bare-runs-dir."),
            "per_surface": {}, "delta_mu_max_ev": None,
            "binding_surface": None,
        }
    best_bare = lowest_bare_gamma(bare_fits)
    per_surface, ceilings = {}, {}
    for f in h_fits:
        bare = best_bare.get(f.surface)
        if bare is None:
            per_surface[f.surface] = {
                "available": False,
                "reason": f"no bare ladder for {f.surface}",
            }
            continue
        if f.slope_j_m2_per_ev <= 0:
            continue
        ceiling = ((bare.gamma_h_rich_j_m2 - f.gamma_h_rich_j_m2)
                   / f.slope_j_m2_per_ev)
        per_surface[f.surface] = {
            "available": True,
            "gamma_h_rich_j_m2": f.gamma_h_rich_j_m2,
            "gamma_bare_j_m2": bare.gamma_h_rich_j_m2,
            "bare_variant": bare.surface,
            "slope_j_m2_per_ev": f.slope_j_m2_per_ev,
            "delta_mu_max_ev": ceiling,
            "bare_epistemic_level": bare.epistemic_level,
            # The ceiling is only as good as the bare ladder behind it, so the
            # bare ladder's own bookkeeping travels with the number.
            "bare_layers_used": list(bare.layers_used),
            "bare_layers_excluded": list(bare.layers_excluded),
            "bare_exclusion_reason": bare.exclusion_reason,
            "bare_mu_c_drift_mry": bare.mu_c_drift_mry,
            "bare_gamma_bias_j_m2": bare.gamma_bias_from_mu_c_drift_j_m2,
            "bare_gamma_scatter_j_m2": bare.gamma_scatter_j_m2,
            "h_layers_used": list(f.layers_used),
            "h_layers_excluded": list(f.layers_excluded),
        }
        ceilings[f.surface] = ceiling
    binding = min(ceilings, key=ceilings.get) if ceilings else None
    return {
        "available": bool(ceilings),
        "reason": "",
        "per_surface": per_surface,
        "delta_mu_max_ev": ceilings.get(binding) if binding else None,
        "binding_surface": binding,
    }


# ------------------------------------------------------- mu_H dependence
@dataclass
class Crossing:
    kind: str            # "zero" or "ordering"
    surfaces: tuple
    delta_mu_ev: float
    in_scanned_range: bool
    physical: bool       # delta_mu >= 0, i.e. at or below the H-rich limit
    note: str = ""


def find_crossings(fits: list, mu_range) -> list:
    lo, hi = mu_range
    out = []
    for f in sorted(fits, key=lambda x: x.surface):
        z = f.zero_crossing_ev()
        if z is None:
            continue
        out.append(Crossing(
            kind="zero", surfaces=(f.surface,), delta_mu_ev=z,
            in_scanned_range=lo <= z <= hi, physical=z >= 0.0,
            note=(f"gamma({f.surface}) turns positive below mu_H(H-rich) "
                  f"- {z:.4f} eV")))
    for a, b in itertools.combinations(sorted(fits, key=lambda x: x.surface), 2):
        dslope = a.slope_j_m2_per_ev - b.slope_j_m2_per_ev
        dgamma = a.gamma_h_rich_j_m2 - b.gamma_h_rich_j_m2
        if abs(dslope) < 1e-12:
            out.append(Crossing(
                kind="ordering", surfaces=(a.surface, b.surface),
                delta_mu_ev=float("inf"), in_scanned_range=False,
                physical=False,
                note=(f"{a.surface} and {b.surface} have equal hydrogen "
                      f"coverage per area; their ordering never changes")))
            continue
        x = -dgamma / dslope
        out.append(Crossing(
            kind="ordering", surfaces=(a.surface, b.surface), delta_mu_ev=x,
            in_scanned_range=lo <= x <= hi, physical=x >= 0.0,
            note=(f"{a.surface} and {b.surface} swap order at "
                  f"delta_mu = {x:.4f} eV")))
    return out


def wulff_available_from_ev(fits: list):
    """Smallest delta_mu at which EVERY gamma is positive, so a Wulff
    construction exists. None if no such delta_mu exists."""
    needed = []
    for f in fits:
        if f.gamma_h_rich_j_m2 > 0:
            needed.append(0.0)
            continue
        z = f.zero_crossing_ev()
        if z is None:
            return None
        needed.append(z)
    return max(needed)


def ordering_at(fits: list, delta_mu_ev: float) -> list:
    """Surfaces sorted from most stable (lowest gamma) to least."""
    return [f.surface for f in sorted(fits, key=lambda f: f.gamma_at(delta_mu_ev))]


def scan(fits: list, mu_range, n_points: int) -> list:
    lo, hi = mu_range
    grid = np.linspace(lo, hi, n_points)
    rows = []
    for d in grid:
        gammas = {f.surface: f.gamma_at(float(d)) for f in fits}
        order = ordering_at(fits, float(d))
        rows.append({
            "delta_mu_h_ev": float(d),
            **{f"gamma_{s}_j_m2": gammas[s] for s in sorted(gammas)},
            "most_stable": order[0],
            "ordering": " < ".join(order),
            "all_positive": all(v > 0 for v in gammas.values()),
            "wulff_defined": all(v > 0 for v in gammas.values()),
        })
    return rows


# ---------------------------------------------------------------- outputs
SUMMARY_FIELDS = [
    "surface", "orientation", "epistemic_level",
    "gamma_h_rich_j_m2", "gamma_scatter_j_m2",
    "dgamma_dmu_j_m2_per_ev", "coverage_n_h_over_2a_per_a2",
    "zero_crossing_delta_mu_ev",
    "n_H", "cell_area_angstrom2",
    "mu_c_fit_ry", "mu_c_bulk_ry", "mu_c_drift_mry",
    "gamma_bias_from_mu_c_drift_j_m2", "fit_rms_ry",
    "layers_used", "layers_excluded", "exclusion_reason",
    "per_slab_gamma_j_m2",
    "mu_h_rich_ry", "mu_c_reference", "notes",
]


def _f(v, nd=6):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def summary_rows(fits: list, ref: Reference) -> list:
    rows = []
    for f in sorted(fits, key=lambda x: x.surface):
        z = f.zero_crossing_ev()
        rows.append({
            "surface": f.surface, "orientation": f.orientation,
            "epistemic_level": f.epistemic_level,
            "gamma_h_rich_j_m2": _f(f.gamma_h_rich_j_m2, 4),
            "gamma_scatter_j_m2": _f(f.gamma_scatter_j_m2, 4),
            "dgamma_dmu_j_m2_per_ev": _f(f.slope_j_m2_per_ev, 4),
            "coverage_n_h_over_2a_per_a2": _f(f.coverage_per_a2, 4),
            "zero_crossing_delta_mu_ev": _f(z, 4),
            "n_H": f.n_H,
            "cell_area_angstrom2": _f(f.area_angstrom2, 4),
            "mu_c_fit_ry": _f(f.mu_c_fit_ry, 9),
            "mu_c_bulk_ry": _f(f.mu_c_bulk_ry, 9),
            "mu_c_drift_mry": _f(f.mu_c_drift_mry, 4),
            "gamma_bias_from_mu_c_drift_j_m2": _f(
                f.gamma_bias_from_mu_c_drift_j_m2, 4),
            "fit_rms_ry": _f(f.fit_rms_ry, 9),
            "layers_used": " ".join(f"{n}L" for n in f.layers_used),
            "layers_excluded": " ".join(f"{n}L" for n in f.layers_excluded) or "none",
            "exclusion_reason": f.exclusion_reason,
            "per_slab_gamma_j_m2": "; ".join(
                f"{n}L:{f.per_slab_gamma_j_m2[n]:+.4f}"
                for n in sorted(f.per_slab_gamma_j_m2)),
            "mu_h_rich_ry": _f(ref.mu_h_rich_ry, 9),
            "mu_c_reference": ref.bulk_source,
            "notes": "; ".join(f.notes),
        })
    return rows


def write_csv(rows, fields, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_config(fits: list, ref: Reference, meta: dict) -> dict:
    """The generated replacement for the hand-entered surface-energy config."""
    return {
        "name": "surface_energies_h",
        "description": ("H-terminated diamond surface energies as a function of "
                        "the hydrogen chemical potential, derived from the "
                        "production slab relaxations."),
        "units": "J/m^2",
        "termination": "H",
        "generated_by": "surface_energy.py",
        "source_type": "project_dft_fit",
        "sign_convention": (
            "gamma > 0 costs energy to create the surface. gamma < 0 means the "
            "H-terminated surface is more stable than the reference state, so "
            "the surface-area term lowers the total energy; a Wulff "
            "construction does not exist in that regime."),
        "chemical_potentials": {
            "mu_H_rich_ry": ref.mu_h_rich_ry,
            "mu_H_rich_definition": "E(H2)/2 from a relaxed H2 molecule",
            "E_H2_ry": ref.e_h2_ry,
            "H2_run": ref.h2_run,
            "mu_C_ry": ref.mu_c_bulk_ry,
            "mu_C_definition": ("Birch-Murnaghan E0 per carbon at the fitted "
                                "equilibrium lattice constant"),
            "mu_C_source": ref.bulk_source,
            "pseudo_C": ref.pseudo_C,
            "pseudo_H": ref.pseudo_H,
            "ecutwfc": ref.ecutwfc,
            "ecutrho": ref.ecutrho,
        },
        "mu_h_dependence": {
            "formula": ("gamma(mu_H) = gamma_h_rich + (N_H / 2A) * delta_mu, "
                        "with delta_mu = mu_H(H-rich) - mu_H >= 0 in eV"),
            "note": ("The ordering of the surface energies, and therefore the "
                     "equilibrium particle shape, depends on mu_H. A single "
                     "H-rich number is not the whole result."),
            "h_poor_limit": ("NOT COMPUTED. Bounding how hydrogen-poor the "
                             "system can get needs a CH4 or graphite+H2 "
                             "reference, which does not exist in this "
                             "repository. Any scanned range is a convention."),
            "wulff_available_from_delta_mu_ev": meta["wulff_available_from_ev"],
        },
        "surface_energies": {
            f.surface[1:]: {
                "value": round(f.gamma_h_rich_j_m2, 6),
                "value_at": "mu_H = E(H2)/2 (H-rich limit)",
                "scatter_j_m2": round(f.gamma_scatter_j_m2, 6),
                "dgamma_dmu_j_m2_per_ev": round(f.slope_j_m2_per_ev, 6),
                "coverage_n_h_over_2a_per_a2": round(f.coverage_per_a2, 6),
                "zero_crossing_delta_mu_ev": (
                    None if f.zero_crossing_ev() is None
                    else round(f.zero_crossing_ev(), 6)),
                "epistemic_level": f.epistemic_level,
                "n_H": f.n_H,
                "cell_area_angstrom2": round(f.area_angstrom2, 6),
                "layers_used": f.layers_used,
                "layers_excluded": f.layers_excluded,
                "mu_c_fit_ry": f.mu_c_fit_ry,
                "mu_c_drift_mry": f.mu_c_drift_mry,
                "fit_rms_ry": f.fit_rms_ry,
                "facet": {
                    "111": "(111)-1x1-H, single-dangling-bond face",
                    "110": "(110)-1x1-H",
                    "100": "(100)-2x1-H, monohydride dimer",
                }.get(f.surface[1:], ""),
                "notes": f.notes,
            }
            for f in sorted(fits, key=lambda x: x.surface)
        },
        "method": {
            "formula": "gamma = [E_slab - N_C*mu_C - N_H*mu_H] / (2A)",
            "mu_c_from": ("slope of a linear fit of E_slab against N_C across "
                          "the thickness ladder (Boettger construction), "
                          "cross-checked against the bulk reference"),
            "excluded_layers": sorted({n for f in fits for n in f.layers_excluded}),
            "excluded_layers_per_surface": {
                f.surface: {"used": f.layers_used,
                            "excluded": f.layers_excluded,
                            "reason": f.exclusion_reason}
                for f in sorted(fits, key=lambda x: x.surface)},
            "excluded_reason": ("per surface; see excluded_layers_per_surface. "
                                "The campaign default drops 6L, which is "
                                "outside the asymptotic regime on all three "
                                "H-terminated surfaces, consistently with "
                                "fit_tau_infinity.py. Bare ladders carry their "
                                "own policy: they converge more slowly and can "
                                "need more of the thin end dropped."),
            "mu_c_reject_j_m2": meta["mu_c_reject_j_m2"],
            "mu_c_reject_meaning": (
                "a fit whose mu_C drift biases gamma by more than this is "
                "rejected outright, not warned about"),
            "runs_dir": meta["runs_dir"],
            "run_prefix": meta["run_prefix"],
        },
        "warnings": meta["warnings"],
    }


def render_report(fits: list, ref: Reference, crossings: list, meta: dict) -> str:
    lo, hi = meta["mu_h_range"]
    L = ["# H-terminated diamond surface energies vs hydrogen chemical potential",
         "",
         "gamma = [E_slab - N_C*mu_C - N_H*mu_H] / (2A), per face, from the "
         "production thickness ladder.", "",
         "Epistemic level is per surface (see the table); it is not uniform.",
         "",
         "## Chemical potentials", "",
         f"* mu_H(H-rich) = E(H2)/2 = {ref.mu_h_rich_ry:.9f} Ry "
         f"(E(H2) = {ref.e_h2_ry:.9f} Ry, `{ref.h2_run}`)",
         f"* mu_C = {ref.mu_c_bulk_ry:.9f} Ry, the Birch-Murnaghan E0 per carbon "
         f"at the fitted equilibrium lattice constant (`{ref.bulk_source}`). "
         f"The sampled eps=0 run sits at a0 = 3.567 Angstrom and is NOT used.",
         f"* Pseudopotentials: C `{ref.pseudo_C}`, H `{ref.pseudo_H}`; "
         f"{ref.ecutwfc:.0f}/{ref.ecutrho:.0f} Ry throughout, checked against "
         f"every slab.", "",
         "## Surface energies at the H-rich limit", "",
         "| surface | level | gamma (J/m^2) | scatter | dgamma/d(-mu_H) | "
         "N_H/2A | mu_C drift | fit rms |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for f in sorted(fits, key=lambda x: x.surface):
        L.append(
            f"| {f.orientation} | {f.epistemic_level} | "
            f"{f.gamma_h_rich_j_m2:+.4f} | {f.gamma_scatter_j_m2:.4f} | "
            f"{f.slope_j_m2_per_ev:.4f} | {f.coverage_per_a2:.4f} | "
            f"{f.mu_c_drift_mry:+.4f} mRy | {f.fit_rms_ry * 1e6:.2f} uRy |")
    L += ["", "gamma in J/m^2; dgamma/d(-mu_H) in (J/m^2)/eV; N_H/2A in "
          "Angstrom^-2. `scatter` is the spread of the per-slab gamma across "
          "the fitted ladder using the independent bulk mu_C, and is the honest "
          "uncertainty on each number.", ""]

    L += ["## Which thicknesses entered each fit", "",
          "Exclusions are per surface, not global. A fit is REJECTED outright "
          "-- nothing written, not a warning -- when the mu_C drift biases "
          f"gamma by more than {meta['mu_c_reject_j_m2']:.3f} J/m^2, since a "
          "biased gamma is a different number rather than a noisier one. The "
          "bias column below is exact: it is gamma from the Boettger fit minus "
          "the mean per-slab gamma taken with the independent bulk mu_C.", "",
          "| surface | ladder | used | dropped | why dropped | gamma bias from "
          "mu_C drift |",
          "| --- | --- | --- | --- | --- | ---: |"]
    for f in sorted(fits, key=lambda x: x.surface):
        ladder = " ".join(f"{n}L" for n in sorted(f.per_slab_gamma_j_m2))
        L.append(
            f"| {f.orientation} | {ladder} | "
            f"{' '.join(f'{n}L' for n in f.layers_used)} | "
            f"{' '.join(f'{n}L' for n in f.layers_excluded) or 'none'} | "
            f"{f.exclusion_reason or '-'} | "
            f"{f.gamma_bias_from_mu_c_drift_j_m2:+.4f} J/m^2 |")
    L.append("")

    bound = meta.get("dehydrogenation_bound") or {}
    L += ["## Dehydrogenation ceiling on delta_mu", ""]
    if not bound.get("available"):
        L += [f"**MISSING.** {bound.get('reason', 'no bare-facet ladder.')}", ""]
    else:
        L += ["gamma_bare carries no hydrogen, so it does not move with mu_H "
              "and the condition gamma_H(delta_mu) > gamma_bare becomes a hard "
              "ceiling: delta_mu_max = (gamma_bare - gamma_H_rich) / (N_H/2A). "
              "The BINDING facet is the one with the smallest ceiling -- the "
              "first to dehydrogenate invalidates the H-terminated Wulff "
              "construction, whatever the others do.", "",
              "| facet | gamma_H rich | gamma_bare | bare variant | bare ladder "
              "used | bare dropped | spin | delta_mu_max (eV, after any E_s correction) |",
              "| --- | ---: | ---: | --- | --- | --- | --- | ---: |"]
        for s, e in sorted(bound.get("per_surface", {}).items()):
            if not e.get("available"):
                L.append(f"| {s} | | | {e.get('reason', 'unavailable')} | | | | |")
                continue
            if e.get("bare_spin_polarised"):
                spin = "nspin=2"
            elif e.get("spin_correction_applied"):
                spin = f"nspin=1 + E_s {e['e_s_ev']:+.3f} eV/DB"
            elif e.get("e_s_ev") is not None:
                spin = f"nspin=1, E_s {e['e_s_ev']:+.3f} eV/DB not applied"
            else:
                spin = "**nspin=1, UNTESTED**"
            L.append(
                f"| {s} | {e['gamma_h_rich_j_m2']:+.4f} | "
                f"{e['gamma_bare_j_m2']:+.4f} | {e['bare_variant']} "
                f"({e['bare_epistemic_level']}) | "
                f"{' '.join(f'{n}L' for n in e['bare_layers_used'])} | "
                f"{' '.join(f'{n}L' for n in e['bare_layers_excluded']) or 'none'} "
                f"| {spin} | {e['delta_mu_max_ev']:.4f} |")
        L += ["",
              f"**Binding facet {bound['binding_surface']}: delta_mu <= "
              f"{bound['delta_mu_max_ev']:.4f} eV.** Beyond it the H-terminated "
              f"surface described everywhere else in this report is not the "
              f"one that is there.", ""]
        stab = bound.get("spin_stabilisation") or {}
        if stab:
            L += ["### Spin stabilisation E_s, and the corrected ceiling", "",
                  "Pairing each bare relaxation with its counterpart in the "
                  "other spin state gives the exchange stabilisation directly:",
                  "", "    E_s = [ E_bare(nspin=1) - E_bare(nspin=2) ] / n_DB "
                  "   [eV per dangling bond]", "",
                  "with n_DB the dangling bonds per cell, read from the "
                  "matching H facet's hydrogen count (one H caps one dangling "
                  "bond). E_s is a SAME-THICKNESS difference; taking it across "
                  "thicknesses would fold back in the bulk term the pairing "
                  "exists to cancel.", "",
                  "Applying it needs no refit, because the geometry cancels "
                  "exactly. With one H per dangling bond, slope = n_DB/2A and "
                  "d(gamma_bare) = n_DB*E_s/2A, so", "",
                  "    d(delta_mu_max) = E_s", "",
                  "identically on every facet, independent of area, coverage "
                  "and thickness. A uniform E_s slides the whole ceiling axis "
                  "down without touching the ranking.", "",
                  "**The identity holds only for unreconstructed 1x1 bare "
                  "faces with their dangling bonds intact.** It needs "
                  "n_DB(bare) = N_H, and a reconstruction breaks that: Pandey "
                  "(111) and dimerised (100) both quench dangling bonds, so "
                  "neither takes this correction. The measurement checks its "
                  "own premise -- one unpaired electron per intact dangling "
                  "bond means the total magnetization should equal n_DB, so "
                  "|M| ~ n_DB confirms the identity, |M| ~ 0 means quenched "
                  "and no correction, and anything between means partially "
                  "quenched and the correction is reported but withheld.", "",
                  "| facet | n_DB | pairs | E_s (eV/DB) | spread | M (Bohr "
                  "mag/cell) | applied? |",
                  "| --- | ---: | --- | ---: | ---: | ---: | --- |"]
            for s, st in sorted(stab.items()):
                mags = ", ".join(f"{v:.2f}"
                                 for _, v in sorted(st["magnetization_bohr"].items()))
                L.append(
                    f"| {s} | {st['n_db_per_cell']} | "
                    f"{' '.join(f'{n}L' for n in sorted(st['per_thickness_ev']))} | "
                    f"{st['e_s_ev']:+.4f} | {st['spread_ev']:.4f} | "
                    f"{mags or '-'} | "
                    f"{'yes' if st['identity_applies'] else 'NO'} |")
            L += ["", "Why, per facet:", ""]
            L += [f"* {s}: {st['reason']}" for s, st in sorted(stab.items())]
            L.append("")
            applied = {s: st for s, st in stab.items() if st["identity_applies"]}
            if applied:
                L += ["| facet | ceiling before (eV) | E_s | ceiling after "
                      "(eV) |", "| --- | ---: | ---: | ---: |"]
                for s, e in sorted(bound.get("per_surface", {}).items()):
                    if not e.get("available"):
                        continue
                    es = e.get("e_s_ev")
                    L.append(
                        f"| {s} | {e['delta_mu_max_uncorrected_ev']:.4f} | "
                        + (f"{es:+.4f} |" if e.get("spin_correction_applied")
                           else ("not applied |" if es is not None
                                 else "not measured |"))
                        + f" {e['delta_mu_max_ev']:.4f} |")
                L += ["",
                      f"Uncorrected binding facet "
                      f"{bound.get('binding_surface_uncorrected')} at "
                      f"{bound.get('delta_mu_max_uncorrected_ev'):.4f} eV; "
                      f"corrected binding facet {bound['binding_surface']} at "
                      f"{bound['delta_mu_max_ev']:.4f} eV.", ""]

        unpol = sorted(s for s, e in bound.get("per_surface", {}).items()
                       if e.get("available") and not e.get("bare_spin_polarised")
                       and not (bound.get("spin_stabilisation") or {}).get(s))
        if unpol:
            L += [f"**The bare energies for {', '.join(unpol)} are "
                  f"non-spin-polarised.** A bare face carries one unpaired "
                  f"electron per dangling bond, and nspin = 2 is variational "
                  f"over nspin = 1, so those energies are upper bounds, "
                  f"gamma_bare is an upper bound, and the ceilings above are "
                  f"too PERMISSIVE by an amount no diagnostic in this report "
                  f"can size. Each 1 J/m^2 of overestimate in gamma_bare "
                  f"inflates that facet's ceiling by "
                  + ", ".join(
                      f"{1.0 / next(f.slope_j_m2_per_ev for f in fits if f.surface == s):.2f} eV ({s})"
                      for s in unpol if any(f.surface == s for f in fits))
                  + ".", ""]
        mags = {k: v for a in (bound.get("bare_spin_audit") or {}).values()
                for k, v in a["magnetizations"].items()}
        if mags:
            L += ["Total magnetization of the spin-polarised bare runs, "
                  "Bohr magnetons per cell (2.00 = one unpaired electron per "
                  "dangling bond on each of the two faces):", ""]
            L += [f"* {k}: {v:+.2f}" for k, v in sorted(mags.items())]
            L.append("")

    L += ["## Dependence on mu_H", "",
          "Lowering mu_H below the H-rich limit raises every gamma, at a rate "
          "set by that facet's hydrogen coverage N_H/2A. Because the rates "
          "differ, the stability ORDERING is a function of mu_H, not a fixed "
          "fact.", "",
          f"Scanned range: delta_mu = mu_H(H-rich) - mu_H from {lo:.2f} to "
          f"{hi:.2f} eV. The lower end of that range is a CONVENTION, not a "
          f"computed bound: fixing the H-poor limit needs a CH4 or graphite+H2 "
          f"reference, which this repository does not have.", "",
          "### Where each gamma turns positive", ""]
    for c in crossings:
        if c.kind != "zero":
            continue
        where = "inside" if c.in_scanned_range else "outside"
        L.append(f"* {c.surfaces[0]}: delta_mu = {c.delta_mu_ev:+.4f} eV "
                 f"({where} the scanned range)")
    avail = meta["wulff_available_from_ev"]
    L += ["",
          (f"**A Wulff construction exists only for delta_mu >= "
           f"{avail:.4f} eV**, where all three gammas are positive."
           if avail is not None else
           "**No delta_mu makes every gamma positive; a Wulff construction "
           "never exists in this reference state.**"),
          "", "### Where the ordering changes", ""]
    for c in crossings:
        if c.kind != "ordering":
            continue
        if not math.isfinite(c.delta_mu_ev):
            L.append(f"* {c.surfaces[0]} vs {c.surfaces[1]}: never (equal coverage)")
            continue
        tag = ("occurs at" if c.physical else
               "would occur at (UNPHYSICAL: needs mu_H ABOVE the H-rich limit, "
               "where H2 condenses)")
        L.append(f"* {c.surfaces[0]} vs {c.surfaces[1]}: {tag} "
                 f"delta_mu = {c.delta_mu_ev:+.4f} eV")
    L += ["", "### Stability ordering across the range", "",
          "| delta_mu (eV) | ordering (most stable first) | all gamma > 0 |",
          "| ---: | --- | --- |"]
    marks = sorted({0.0}
                   | {round(c.delta_mu_ev, 4) for c in crossings
                      if math.isfinite(c.delta_mu_ev) and lo <= c.delta_mu_ev <= hi}
                   | {hi})
    probes = []
    for i, m in enumerate(marks):
        probes.append(m)
        if i + 1 < len(marks):
            probes.append(0.5 * (m + marks[i + 1]))
    for d in probes:
        order = ordering_at(fits, d)
        allpos = all(f.gamma_at(d) > 0 for f in fits)
        L.append(f"| {d:.4f} | {' < '.join(order)} | {'yes' if allpos else 'no'} |")
    L += ["", "## Per-slab cross-check", "",
          "gamma computed for each individual slab using the independent bulk "
          "mu_C rather than the fitted slope. Constancy across the ladder is "
          "the evidence that the interior is bulk-like.", ""]
    for f in sorted(fits, key=lambda x: x.surface):
        vals = "  ".join(f"{n}L {f.per_slab_gamma_j_m2[n]:+.4f}"
                         for n in sorted(f.per_slab_gamma_j_m2))
        L.append(f"* {f.orientation}: {vals}")
    L += ["", "## Translating delta_mu into (T, p_H2)", "",
          "delta_mu is an abstract axis until it is anchored to conditions a "
          "furnace can reach. The mapping below uses the ideal-gas chemical "
          "potential of H2:", "",
          "    delta_mu(T, p) = -[ mu_H2(T,p) - E(H2) ] / 2", "",
          "with mu_H2 built from the standard translational, rotational and "
          "vibrational terms.", "",
          "**What is computed here versus what is textbook thermodynamics:**", "",
          "* COMPUTED (this project's DFT): E(H2); the slab energies; hence "
          "gamma at the H-rich limit and its slope N_H/2A.",
          "* NOT COMPUTED (ideal-gas statistical thermodynamics with "
          "literature spectroscopic constants of H2): every T- and p-dependent "
          "term below. Rigid rotor with an explicit level sum, harmonic "
          "oscillator, ideal gas. No anharmonicity, no real-gas correction.", ""]
    val = meta.get("rrho_validation")
    if val:
        L += [f"The model reproduces the NIST-JANAF tabulation of "
              f"G(T) - H(0) for H2 to within "
              f"{val['max_abs_deviation_ev'] * 1000:.0f} meV between 298 K and "
              f"2000 K, which is the dominant systematic on any temperature "
              f"quoted below (of order +/-15 K).", "",
              "| T (K) | this model (eV) | NIST-JANAF (eV) | deviation (meV) |",
              "| ---: | ---: | ---: | ---: |"]
        for T, v in sorted(val["points"].items()):
            L.append(f"| {T:.2f} | {v['model_ev']:+.4f} | {v['janaf_ev']:+.4f} "
                     f"| {v['deviation_ev'] * 1000:+.1f} |")
        L.append("")
    L += ["Zero-point energy is EXCLUDED. gamma was derived from DFT total "
          "energies with no vibrational term on either side; adding the H2 ZPE "
          "(0.136 eV per H) without the adsorbed-H ZPE would be an unbalanced "
          "correction, and the adsorbed-H ZPE needs a phonon calculation this "
          "repository does not have. The two are of similar size and partially "
          "cancel.", ""]
    tp = meta.get("tp_landmarks") or []
    if tp:
        L += ["### Conditions for each landmark", "",
              "| landmark | delta_mu (eV) | p_H2 (bar) | p_H2 (Torr) | T (K) | "
              "T (C) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in tp:
            tk = ("unreachable below 4000 K" if r["temperature_k"] is None
                  else f"{r['temperature_k']:.0f}")
            tc = ("" if r["temperature_c"] is None
                  else f"{r['temperature_c']:.0f}")
            name = (f"{r['landmark']} {r['surfaces']}"
                    if r["surfaces"] != "all" else r["landmark"])
            L.append(f"| {name} | {r['delta_mu_ev']:.3f} | "
                     f"{r['p_h2_bar']:.0e} | {r['p_h2_torr']:.1e} | {tk} | {tc} |")
        L.append("")

    L += ["", "## Caveats", ""]
    if bound.get("available"):
        L += [f"* **The H-terminated surface remains the stable termination "
              f"only up to delta_mu = {bound['delta_mu_max_ev']:.4f} eV**, "
              f"where {bound['binding_surface']} dehydrogenates. Every number "
              f"above at a larger delta_mu describes a surface that is not "
              f"there. The ceiling is only as good as the bare ladder behind "
              f"it: it inherits that ladder's thickness convergence, and it "
              f"assumes the lowest bare reconstruction generated is the lowest "
              f"one that exists.",]
    else:
        L += ["* **The H-terminated surface is ASSUMED to remain the stable "
              "termination at every mu_H.** Only H-terminated facets were "
              "calculated, so gamma_H can be extrapolated to arbitrarily "
              "hydrogen-poor conditions without anything stopping it. In "
              "reality the surface dehydrogenates or reconstructs once gamma_H "
              "rises above the bare or reconstructed surface energy, and that "
              "bound cannot be computed from this data set. It is the single "
              "largest limitation on everything above, and it bites hardest "
              "exactly where delta_mu is large.",]
    L += [
          "* Competing kinetics are not modelled at all. Nanodiamond surfaces "
          "graphitize on annealing in vacuum, and that process is not in this "
          "thermodynamic picture. A (T, p) point being thermodynamically "
          "reachable does not mean the H-terminated diamond surface survives "
          "the trip.",
          "* The H-poor end of the mu_H range is not bounded by any calculation "
          "here. Statements about very hydrogen-poor conditions are "
          "extrapolations of a straight line, nothing more.",
          "* gamma is a fixed-cell, relaxed-ion quantity at the bulk in-plane "
          "lattice constant.",
          "* Zero-point and finite-temperature contributions to mu_H are "
          "omitted entirely; at room temperature and realistic H2 partial "
          "pressures these shift mu_H by a few tenths of an eV, which is the "
          "same scale as the crossings above.",
          "* k-meshes differ between the slabs (9x5x1, 9x7x1, 9x9x1) and the "
          "bulk reference (8x8x8). The fitted-slope construction cancels the "
          "leading effect of that mismatch; it does not cancel it exactly.", ""]
    if meta["warnings"]:
        L += ["## Warnings", ""] + [f"* {w}" for w in meta["warnings"]] + [""]
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------- main
def run(runs_dir, reference_dir, out_dir, prefix, exclude_layers, mu_h_range,
        mu_h_points, mu_c_tol_mry, config_out=None,
        bare_runs_dir=None, bare_prefix="bare", *,
        exclude_layers_for=None,
        bare_exclude_layers=None, bare_exclude_layers_for=None,
        mu_c_reject_j_m2=DEFAULT_MU_C_REJECT_J_M2,
        bare_spin_runs_dir=None, bare_spin_prefix=None) -> dict:
    ref = load_references(reference_dir)
    by_surface = load_slab_energies(runs_dir, prefix)
    for points in by_surface.values():
        check_reference_consistency(points, ref)

    exclude_layers_for = exclude_layers_for or {}
    fits = []
    for s, pts in sorted(by_surface.items()):
        excl, why = resolve_exclusions(s, exclude_layers, exclude_layers_for)
        fits.append(fit_surface_energy(s, pts, ref, excl, why))
    warnings = grade_fits(fits, mu_c_tol_mry, mu_c_reject_j_m2,
                          "--exclude-layers-for")

    # --- bare facets: the ceiling on delta_mu, if the campaign has been run
    #
    # The bare ladder gets its own exclusion policy. It is not the H ladder's:
    # two interacting dangling-bond faces converge more slowly than two
    # passivated ones, so the thin end of a bare ladder can be unusable at
    # thicknesses where the H-terminated one is already asymptotic. Defaulting
    # the bare policy to the H one (as this used to, and then only when the
    # ladder had more than three points) is how a 6L/8L-contaminated bare fit
    # got as far as the ceiling.
    bare_fits = []
    bare_spin_audit = {}
    if bare_runs_dir and Path(bare_runs_dir).is_dir():
        bare_exclude = (list(exclude_layers) if bare_exclude_layers is None
                        else list(bare_exclude_layers))
        bare_exclude_layers_for = bare_exclude_layers_for or {}
        try:
            bare_by_surface = load_slab_energies(bare_runs_dir, bare_prefix)
        except SurfaceEnergyError:
            bare_by_surface = {}
        # Before any fitting: a ladder in the wrong electronic state cannot be
        # rescued by a better fit, so this gate runs first.
        spin_warnings = check_bare_spin_states(bare_by_surface)
        warnings.extend(spin_warnings)
        bare_spin_audit = spin_audit(bare_by_surface)
        for surface, pts in sorted(bare_by_surface.items()):
            check_reference_consistency(pts, ref)
            excl, why = resolve_exclusions(
                surface, bare_exclude, bare_exclude_layers_for)
            # A bare ladder that simply does not have the default thickness
            # (the Pandey 20-28L ladder does not have a 6L rung) is not an
            # error; a MISDIRECTED exclusion is, and fit_surface_energy still
            # catches that for anything named explicitly.
            if surface not in bare_exclude_layers_for:
                present = {p.layers for p in pts}
                excl = [n for n in excl if n in present]
                why = (f"campaign default "
                       f"({', '.join(f'{n}L' for n in excl)})" if excl
                       else "campaign default, none present in this ladder")
            bare_fits.append(
                fit_surface_energy(surface, pts, ref, excl, why))
        grade_fits(bare_fits, mu_c_tol_mry, mu_c_reject_j_m2,
                   "--bare-exclude-layers-for")
    # --- E_s from paired nspin=1 / nspin=2 bare runs, if the pairs exist.
    #
    # These live in their own directory and are deliberately NOT rungs of the
    # fitted ladder: the ladder must be spin-consistent, and E_s is a
    # same-thickness DIFFERENCE, which is what cancels the bulk term. n_DB per
    # cell is read from the matching H facet's hydrogen count -- one H caps one
    # dangling bond -- rather than assumed from the orientation.
    stabilisations = {}
    if bare_spin_runs_dir and Path(bare_spin_runs_dir).is_dir():
        n_db = {f.surface: f.n_H for f in fits}
        try:
            spin_by_surface = load_slab_energies(
                bare_spin_runs_dir, bare_spin_prefix or bare_prefix)
        except SurfaceEnergyError:
            spin_by_surface = {}
        for surface, sp_pts in sorted(spin_by_surface.items()):
            check_reference_consistency(sp_pts, ref)
            base = bare_base_surface(surface)
            nsp_pts = bare_by_surface.get(surface, []) if bare_runs_dir else []
            if not nsp_pts or base not in n_db:
                warnings.append(
                    f"spin-polarised runs found for {surface} but no "
                    f"{'non-polarised counterpart' if not nsp_pts else 'H-terminated facet'}"
                    f" to pair them with; E_s not measured")
                continue
            if surface != base:
                warnings.append(
                    f"{surface} is a reconstruction variant; a reconstruction "
                    f"quenches dangling bonds, so n_DB(bare) != N_H and "
                    f"d(ceiling) = E_s does not apply to it. E_s is measured "
                    f"for the record only.")
            st = measure_spin_stabilisation(nsp_pts, sp_pts, n_db[base])
            if surface == base:
                stabilisations[base] = st
            if not st.identity_applies:
                warnings.append(
                    f"bare {surface}: E_s = {st.e_s_ev:+.4f} eV/DB measured "
                    f"from {', '.join(f'{n}L' for n in st.thicknesses)}, but "
                    f"NOT applied to the ceiling -- {st.reason}")

    bound = dehydrogenation_bound(fits, bare_fits)
    bound["bare_spin_audit"] = bare_spin_audit
    for s, e in bound.get("per_surface", {}).items():
        a = bare_spin_audit.get(s)
        if e.get("available") and a is not None:
            e["bare_spin_polarised"] = a["has_polarised_reference"]
            e["bare_total_magnetization_bohr"] = a["magnetizations"]
            if not a["has_polarised_reference"] and s not in stabilisations:
                e["ceiling_caveat"] = (
                    "non-spin-polarised bare energy; nspin=2 is variational "
                    "over nspin=1, so gamma_bare is an upper bound and this "
                    "ceiling is too permissive")
    apply_spin_correction(bound, stabilisations)
    if not bound["available"]:
        warnings.append(
            "dehydrogenation bound MISSING: " + bound["reason"])
    crossings = find_crossings(fits, mu_h_range)
    avail = wulff_available_from_ev(fits)

    validation = rrho_validation()
    meta = {
        "runs_dir": str(runs_dir), "run_prefix": prefix,
        "reference_dir": str(reference_dir),
        "mu_h_range": list(mu_h_range), "mu_h_points": mu_h_points,
        "mu_c_tol_mry": mu_c_tol_mry,
        "mu_c_reject_j_m2": mu_c_reject_j_m2,
        "exclusions": {f.surface: {"used": list(f.layers_used),
                                   "excluded": list(f.layers_excluded),
                                   "reason": f.exclusion_reason}
                       for f in fits},
        "bare_exclusions": {f.surface: {"used": list(f.layers_used),
                                        "excluded": list(f.layers_excluded),
                                        "reason": f.exclusion_reason}
                            for f in bare_fits},
        "wulff_available_from_ev": avail,
        "rrho_validation": validation,
        "dehydrogenation_bound": bound,
        "delta_zpe_estimate_ev": delta_zpe_ev(),
        "delta_zpe_source": ZPE_ESTIMATE_SOURCE,
        "warnings": warnings,
    }

    # Every landmark on the delta_mu axis, expressed as a (T, p_H2) locus.
    tp_rows = []
    landmarks = [("wulff_available", ("all",), avail)] + [
        (c.kind, c.surfaces, c.delta_mu_ev) for c in crossings
        if math.isfinite(c.delta_mu_ev) and c.physical]
    for kind, surfaces, d in landmarks:
        if d is None:
            continue
        for row in tp_curve(d):
            tp_rows.append({
                "landmark": kind, "surfaces": "+".join(surfaces), **row,
                "temperature_uncertainty_k": temperature_uncertainty_k(
                    d, row["p_h2_pa"]),
            })

    out_dir = Path(out_dir)
    write_csv(summary_rows(fits, ref), SUMMARY_FIELDS,
              out_dir / "surface_energy.csv")
    scan_rows = scan(fits, mu_h_range, mu_h_points)
    write_csv(scan_rows, list(scan_rows[0].keys()),
              out_dir / "surface_energy_vs_mu_h.csv")
    if tp_rows:
        write_csv(tp_rows, list(tp_rows[0].keys()),
                  out_dir / "surface_energy_tp_map.csv")
    meta["tp_landmarks"] = tp_rows
    (out_dir / "surface_energy_report.md").write_text(
        render_report(fits, ref, crossings, meta))

    config = build_config(fits, ref, meta)
    if config_out:
        p = Path(config_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(config, indent=2) + "\n")

    return {"fits": fits, "bare_fits": bare_fits, "reference": ref,
            "crossings": crossings, "meta": meta, "config": config,
            "dehydrogenation_bound": bound}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--exclude-layers", type=int, nargs="*", default=[6],
                    metavar="N",
                    help="default thicknesses to drop from every H-terminated "
                         "ladder (default: %(default)s)")
    ap.add_argument("--exclude-layers-for", action="append", default=[],
                    metavar="SURFACE=N[,N...]",
                    help="per-surface override of --exclude-layers, e.g. "
                         "C111=6,8. Repeatable. Which thicknesses are outside "
                         "the asymptotic regime is a property of the surface, "
                         "not of the campaign.")
    ap.add_argument("--bare-exclude-layers", type=int, nargs="*", default=None,
                    metavar="N",
                    help="default thicknesses to drop from every BARE ladder. "
                         "Defaults to --exclude-layers, but bare facets have "
                         "two interacting dangling-bond faces and converge "
                         "more slowly, so they often need more.")
    ap.add_argument("--bare-exclude-layers-for", action="append", default=[],
                    metavar="SURFACE=N[,N...]",
                    help="per-surface override for the bare ladder, e.g. "
                         "C111=6,8. Repeatable.")
    ap.add_argument("--mu-h-range", type=float, nargs=2, default=[0.0, 3.0],
                    metavar=("LO", "HI"),
                    help="delta_mu = mu_H(H-rich) - mu_H, in eV. The upper "
                         "bound is a scan convention, not a computed physical "
                         "limit (default: %(default)s)")
    ap.add_argument("--mu-h-points", type=int, default=121)
    ap.add_argument("--mu-c-tol-mry", type=float, default=DEFAULT_MU_C_TOL_MRY,
                    help="warn and downgrade to L1 if the fitted mu_C drifts "
                         "from the bulk reference by more than this "
                         "(default: %(default)s)")
    ap.add_argument("--mu-c-reject-j-m2", type=float,
                    default=DEFAULT_MU_C_REJECT_J_M2, metavar="J_M2",
                    help="REJECT the fit -- raise, write nothing -- if the "
                         "mu_C drift biases gamma by more than this many "
                         "J/m^2. Stated in J/m^2 and not mRy because the bias "
                         "scales as <N_C>/2A and is therefore not comparable "
                         "between surfaces in mRy (default: %(default)s)")
    ap.add_argument("--bare-runs-dir", default=None,
                    help="directory holding the bare-facet ladder from "
                         "make_bare_slabs.py. Without it the dehydrogenation "
                         "ceiling on delta_mu cannot be computed and is "
                         "reported as MISSING.")
    ap.add_argument("--bare-prefix", default="bare")
    ap.add_argument("--bare-spin-runs-dir", default=None,
                    help="directory holding SPIN-POLARISED bare relaxations, "
                         "named the same as their non-polarised counterparts "
                         "in --bare-runs-dir. Same-thickness pairs give the "
                         "exchange stabilisation E_s per dangling bond, and "
                         "the ceiling drops by exactly E_s on every "
                         "unreconstructed facet. Kept separate from the fitted "
                         "ladder on purpose: the ladder must be "
                         "spin-consistent, and E_s must be a same-thickness "
                         "difference.")
    ap.add_argument("--bare-spin-prefix", default=None,
                    help="run prefix inside --bare-spin-runs-dir "
                         "(default: --bare-prefix)")
    ap.add_argument("--write-config", nargs="?", const=DEFAULT_CONFIG_OUT,
                    default=None, metavar="PATH",
                    help=f"regenerate the surface-energy config "
                         f"(default path: {DEFAULT_CONFIG_OUT})")
    args = ap.parse_args(argv)

    try:
        result = run(args.runs_dir, args.reference_dir, args.out_dir,
                     args.prefix, args.exclude_layers, args.mu_h_range,
                     args.mu_h_points, args.mu_c_tol_mry, args.write_config,
                     args.bare_runs_dir, args.bare_prefix,
                     exclude_layers_for=parse_layer_exclusions(
                         args.exclude_layers_for),
                     bare_exclude_layers=args.bare_exclude_layers,
                     bare_exclude_layers_for=parse_layer_exclusions(
                         args.bare_exclude_layers_for),
                     mu_c_reject_j_m2=args.mu_c_reject_j_m2,
                     bare_spin_runs_dir=args.bare_spin_runs_dir,
                     bare_spin_prefix=args.bare_spin_prefix)
    except SurfaceEnergyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ref, fits, meta = result["reference"], result["fits"], result["meta"]
    print("# H-terminated surface energies from the production ladder")
    print(f"# mu_H(H-rich) = {ref.mu_h_rich_ry:.9f} Ry = E(H2)/2; "
          f"mu_C = {ref.mu_c_bulk_ry:.9f} Ry (BM3 E0/atom)")
    print(f"# gamma = [E_slab - N_C*mu_C - N_H*mu_H] / 2A, per face")
    print(f"{'surface':9s} {'lvl':4s} {'gamma(H-rich)':>14s} {'scatter':>9s} "
          f"{'dg/d(-muH)':>11s} {'g=0 at':>9s} {'muC drift':>11s} "
          f"{'g bias':>9s}")
    for f in fits:
        z = f.zero_crossing_ev()
        print(f"{f.orientation:9s} {f.epistemic_level:4s} "
              f"{f.gamma_h_rich_j_m2:+14.4f} {f.gamma_scatter_j_m2:9.4f} "
              f"{f.slope_j_m2_per_ev:11.4f} "
              f"{(f'{z:.4f}' if z is not None else 'never'):>9s} "
              f"{f.mu_c_drift_mry:+10.4f}m "
              f"{f.gamma_bias_from_mu_c_drift_j_m2:+9.4f}")
    print("# gamma in J/m^2, dg/d(-muH) in (J/m^2)/eV, g=0 at delta_mu in eV")
    print(f"# 'g bias' = gamma shift from the mu_C drift, J/m^2; rejection "
          f"budget {meta['mu_c_reject_j_m2']:.3f}")
    for label, key in (("H", "exclusions"), ("bare", "bare_exclusions")):
        for s, e in sorted(meta[key].items()):
            print(f"# {label} ladder {s}: used "
                  f"{' '.join(f'{n}L' for n in e['used'])}, dropped "
                  f"{' '.join(f'{n}L' for n in e['excluded']) or 'nothing'} "
                  f"({e['reason']})")
    avail = meta["wulff_available_from_ev"]
    print(f"# Wulff construction available from delta_mu >= "
          + (f"{avail:.4f} eV" if avail is not None else "never"))
    for c in result["crossings"]:
        if c.kind == "ordering" and math.isfinite(c.delta_mu_ev):
            tag = "" if c.physical else "  [UNPHYSICAL: mu_H above H-rich]"
            print(f"# ordering swap {c.surfaces[0]}/{c.surfaces[1]} at "
                  f"delta_mu = {c.delta_mu_ev:+.4f} eV{tag}")
    for w in meta["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    bound = result["dehydrogenation_bound"]
    if bound["available"]:
        print(f"# dehydrogenation ceiling: delta_mu <= "
              f"{bound['delta_mu_max_ev']:.4f} eV "
              f"(binding facet {bound['binding_surface']})")
        for s, e in sorted(bound["per_surface"].items()):
            if e.get("available"):
                print(f"#   {s}: gamma_bare {e['gamma_bare_j_m2']:+.4f} "
                      f"({e['bare_variant']}) -> delta_mu_max "
                      f"{e['delta_mu_max_ev']:.4f} eV")
    else:
        print("# dehydrogenation ceiling: MISSING (no bare-facet ladder). "
              "gamma_H is unbounded above in this model; run "
              "make_bare_slabs.py.")
    dz = result["meta"]["delta_zpe_estimate_ev"]
    print(f"# DELTA_ZPE (estimated, NOT applied) = {dz:+.4f} eV -- shifts the "
          f"delta_mu axis and is the DOMINANT systematic on any temperature "
          f"below; see make_h_phonons.py / analyze_h_phonons.py")
    val = result["meta"]["rrho_validation"]
    print(f"# H2 ideal-gas mapping (RRHO, literature constants, no ZPE): "
          f"max deviation from NIST-JANAF "
          f"{val['max_abs_deviation_ev'] * 1000:.0f} meV over 298-2000 K")
    for r in result["meta"].get("tp_landmarks", []):
        if r["p_h2_bar"] not in (1.0, 1e-6, 1e-12):
            continue
        name = (f"{r['landmark']} {r['surfaces']}" if r["surfaces"] != "all"
                else r["landmark"])
        tk = ("unreachable <4000K" if r["temperature_k"] is None
              else f"T = {r['temperature_k']:7.0f} K")
        print(f"#   {name:28s} d_mu={r['delta_mu_ev']:5.3f} eV  "
              f"p={r['p_h2_bar']:7.0e} bar  {tk}")
    for name in ("surface_energy.csv", "surface_energy_vs_mu_h.csv",
                 "surface_energy_tp_map.csv", "surface_energy_report.md"):
        print(f"wrote {os.path.join(args.out_dir, name)}")
    if args.write_config:
        print(f"wrote {args.write_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
