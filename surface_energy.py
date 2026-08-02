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

RUN_RE = re.compile(r"^(?P<prefix>.+)~(?P<surface>C\d{3})_(?P<layers>\d+)L$")

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
            ecutwfc=float(pin["ecutwfc"]), ecutrho=float(pin["ecutrho"])))
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
    epistemic_level: str = "L2"
    notes: list = field(default_factory=list)

    @property
    def mu_c_drift_mry(self) -> float:
        return 1000.0 * (self.mu_c_fit_ry - self.mu_c_bulk_ry)

    def gamma_at(self, delta_mu_ev: float) -> float:
        """gamma in J/m^2 at mu_H = mu_H(H-rich) - delta_mu_ev."""
        return self.gamma_h_rich_j_m2 + self.slope_j_m2_per_ev * delta_mu_ev

    def zero_crossing_ev(self):
        """delta_mu at which gamma turns positive, or None if it never does."""
        if self.slope_j_m2_per_ev <= 0:
            return None
        return -self.gamma_h_rich_j_m2 / self.slope_j_m2_per_ev


def fit_surface_energy(surface: str, points: list, ref: Reference,
                       exclude_layers) -> GammaFit:
    exclude = set(exclude_layers)
    used = [p for p in points if p.layers not in exclude]
    dropped = [p for p in points if p.layers in exclude]
    if len(used) < 3:
        raise SurfaceEnergyError(
            f"{surface}: {len(used)} points left after exclusion; too few for a "
            f"two-parameter fit with a meaningful residual")
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
        gamma_scatter_j_m2=scatter)


def grade_fits(fits: list, mu_c_tol_mry: float) -> list:
    """Attach an epistemic level and warnings based on the fit's own diagnostics."""
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
                f"is {f.gamma_scatter_j_m2:.4f} J/m^2.")
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
    "mu_c_fit_ry", "mu_c_bulk_ry", "mu_c_drift_mry", "fit_rms_ry",
    "layers_used", "layers_excluded", "per_slab_gamma_j_m2",
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
            "fit_rms_ry": _f(f.fit_rms_ry, 9),
            "layers_used": " ".join(f"{n}L" for n in f.layers_used),
            "layers_excluded": " ".join(f"{n}L" for n in f.layers_excluded) or "none",
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
            "excluded_reason": ("6L is outside the asymptotic regime on all "
                                "three surfaces; excluded consistently with "
                                "fit_tau_infinity.py"),
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

    L += ["", "## Caveats", "",
          "* **The H-terminated surface is ASSUMED to remain the stable "
          "termination at every mu_H.** Only H-terminated facets were "
          "calculated, so gamma_H can be extrapolated to arbitrarily "
          "hydrogen-poor conditions without anything stopping it. In reality "
          "the surface dehydrogenates or reconstructs once gamma_H rises above "
          "the bare or reconstructed surface energy, and that bound cannot be "
          "computed from this data set. It is the single largest limitation on "
          "everything above, and it bites hardest exactly where delta_mu is "
          "large.",
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
        mu_h_points, mu_c_tol_mry, config_out=None) -> dict:
    ref = load_references(reference_dir)
    by_surface = load_slab_energies(runs_dir, prefix)
    for points in by_surface.values():
        check_reference_consistency(points, ref)

    fits = [fit_surface_energy(s, pts, ref, exclude_layers)
            for s, pts in sorted(by_surface.items())]
    warnings = grade_fits(fits, mu_c_tol_mry)
    crossings = find_crossings(fits, mu_h_range)
    avail = wulff_available_from_ev(fits)

    validation = rrho_validation()
    meta = {
        "runs_dir": str(runs_dir), "run_prefix": prefix,
        "reference_dir": str(reference_dir),
        "mu_h_range": list(mu_h_range), "mu_h_points": mu_h_points,
        "mu_c_tol_mry": mu_c_tol_mry,
        "wulff_available_from_ev": avail,
        "rrho_validation": validation,
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

    return {"fits": fits, "reference": ref, "crossings": crossings,
            "meta": meta, "config": config}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--exclude-layers", type=int, nargs="*", default=[6],
                    metavar="N")
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
    ap.add_argument("--write-config", nargs="?", const=DEFAULT_CONFIG_OUT,
                    default=None, metavar="PATH",
                    help=f"regenerate the surface-energy config "
                         f"(default path: {DEFAULT_CONFIG_OUT})")
    args = ap.parse_args(argv)

    try:
        result = run(args.runs_dir, args.reference_dir, args.out_dir,
                     args.prefix, args.exclude_layers, args.mu_h_range,
                     args.mu_h_points, args.mu_c_tol_mry, args.write_config)
    except SurfaceEnergyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ref, fits, meta = result["reference"], result["fits"], result["meta"]
    print("# H-terminated surface energies from the production ladder")
    print(f"# mu_H(H-rich) = {ref.mu_h_rich_ry:.9f} Ry = E(H2)/2; "
          f"mu_C = {ref.mu_c_bulk_ry:.9f} Ry (BM3 E0/atom)")
    print(f"# gamma = [E_slab - N_C*mu_C - N_H*mu_H] / 2A, per face")
    print(f"{'surface':9s} {'lvl':4s} {'gamma(H-rich)':>14s} {'scatter':>9s} "
          f"{'dg/d(-muH)':>11s} {'g=0 at':>9s} {'muC drift':>11s}")
    for f in fits:
        z = f.zero_crossing_ev()
        print(f"{f.orientation:9s} {f.epistemic_level:4s} "
              f"{f.gamma_h_rich_j_m2:+14.4f} {f.gamma_scatter_j_m2:9.4f} "
              f"{f.slope_j_m2_per_ev:11.4f} "
              f"{(f'{z:.4f}' if z is not None else 'never'):>9s} "
              f"{f.mu_c_drift_mry:+10.4f}m")
    print("# gamma in J/m^2, dg/d(-muH) in (J/m^2)/eV, g=0 at delta_mu in eV")
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
