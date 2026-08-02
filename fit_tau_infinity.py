#!/usr/bin/env python3
"""fit_tau_infinity.py — thickness extrapolation of the surface stress tau.

Purpose
-------
The tau_inf values the rest of the project depends on (the facet inputs to
`particle_strain.py`, the numbers quoted in the production baseline commit)
were originally computed in throwaway one-liners and existed nowhere in the
repository.  This module is their permanent, re-runnable home.

Model
-----
A symmetric slab of interior thickness t carries two identical surfaces plus a
bulk-like interior.  Its cell-averaged in-plane stress therefore separates into
a surface term that does not scale with thickness and an interior term that
does:

    sigma_ij * Lz = 2 * tau_ij^inf  +  sigma_res * t

with

    sigma_ij   QE in-plane stress of the slab cell            [kbar]
    Lz         height of the periodic cell (slab + vacuum)    [Angstrom]
    t          bulk-equivalent thickness of the carbon slab   [Angstrom]
    tau_ij^inf thickness-extrapolated surface stress          [kbar*Angstrom]
    sigma_res  residual interior stress                       [kbar]

The intercept is 2*tau^inf because the slab has two surfaces (CLAUDE.md sec 2).
Conversion to the reported unit uses 1 kbar*Angstrom = 0.01 N/m:

    tau[N/m] = 0.01 * intercept[kbar*Angstrom] / 2

Sign conventions (CLAUDE.md sec 2, and stated again in every output file):
  * sigma > 0  -> the cell is COMPRESSED (pressure-like).
  * tau   > 0  -> TENSILE surface stress: the surface wants to contract, and
                  so compresses the material beneath it.
  * sigma_res < 0 -> the interior is in slight tension at the reference
                  lattice constant, i.e. a0 is marginally too small.

Why t is measured by atom count, not by geometry
------------------------------------------------
t is *not* a geometric slab thickness (z_max - z_min over the carbon atoms, or
any variant that does or does not count the hydrogen).  It is

    t = N_C * Omega_bulk / A

with Omega_bulk = a0^3 / 8 the bulk volume per carbon atom (diamond has 8
atoms in the conventional cubic cell) and A the in-plane cell area.  Counting
bulk-equivalent volume by atom number removes the "where exactly does the slab
end" convention, which is worth roughly 1 Angstrom of arbitrary offset and
shifts tau_inf by several percent depending on whether the terminating
hydrogen is counted.

NOTE: this choice moves only the INTERCEPT, never the slope.  The two
definitions of t differ by an additive constant per surface (t_atomic =
t_geometric + c), so sigma_res is identical either way while 2*tau^inf absorbs
sigma_res * c.  Verified numerically in tests/test_fit_tau_infinity.py.

The 6L point is excluded from every fit
---------------------------------------
6L lies outside the asymptotic regime on all three surfaces: it shows the
even/odd parity oscillation on C100 and it is the outlier in energy slope,
surface energy, and tau_aniso.  Its exclusion is recorded in every output file
rather than being a silent filter, and the 6L residual against the fit is
reported so the size of the exclusion is visible.

Epistemic level
---------------
L2 (CLAUDE.md sec 4) for tau^inf: the underlying stress SCFs are the converged
production set (90/720 Ry, 8 Angstrom vacuum, a0 = 3.572997 Angstrom) with
cutoff, vacuum, k-point, and thickness convergence demonstrated and retained
under results/convergence/.  The extrapolation itself is a four-point linear
fit; its residuals are reported so that quality is inspectable rather than
asserted.

Usage
-----
    python3 fit_tau_infinity.py
    python3 fit_tau_infinity.py --runs-dir results/production --out-dir results/production
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

import elastic_reference
import parse_slab

# 1 kbar * Angstrom = 1e8 Pa * 1e-10 m = 0.01 N/m
KBAR_ANGSTROM_TO_N_PER_M = 0.01

# Carbon atoms in the conventional cubic diamond cell.
C_PER_CONVENTIONAL_CELL = 8

DEFAULT_RUNS_DIR = "results/production"
DEFAULT_OUT_DIR = "results/production"
DEFAULT_PREFIX = "thick_a0corr_stress"
DEFAULT_REFERENCE = "config/reference_pbe_sssp.json"

# sigma_res is a property of the bulk interior at the reference lattice
# constant. It must therefore come out the same on every surface. A spread
# larger than this across the three surfaces means something surface-specific
# is leaking into the slope -- most often a wrong reference lattice constant.
DEFAULT_SIGMA_RES_TOL_KBAR = 0.3

EPISTEMIC_LEVEL = "L2"

RUN_RE = re.compile(r"^(?P<prefix>.+)~(?P<surface>C\d{3})_(?P<layers>\d+)L_stress_scf$")


class TauFitError(Exception):
    """A load-bearing invariant of the tau extrapolation was violated."""


# --------------------------------------------------------------- data model
@dataclass(frozen=True)
class StressPoint:
    """One converged stress SCF on the thickness ladder."""
    surface: str
    layers: int
    run: str
    n_C: int
    n_H: int
    area_angstrom2: float
    Lz_angstrom: float
    thickness_angstrom: float          # t = N_C * Omega_bulk / A
    thickness_geometric_angstrom: float  # carbon z-extent, diagnostic only
    sigma_xx_kbar: float
    sigma_yy_kbar: float
    pseudopotentials: str

    def sigma(self, component: str) -> float:
        return self.sigma_xx_kbar if component == "xx" else self.sigma_yy_kbar


@dataclass
class ComponentFit:
    """Least-squares fit of sigma_ij * Lz = 2*tau_inf + sigma_res * t."""
    component: str
    tau_inf_n_per_m: float
    sigma_res_kbar: float
    rms_kbar_angstrom: float
    max_abs_residual_kbar_angstrom: float
    n_points: int
    residuals: dict = field(default_factory=dict)   # layers -> kbar*Angstrom

    @property
    def rms_as_tau_n_per_m(self) -> float:
        """The fit rms expressed as an equivalent uncertainty on tau."""
        return KBAR_ANGSTROM_TO_N_PER_M * self.rms_kbar_angstrom / 2.0


@dataclass
class SurfaceFit:
    surface: str
    orientation: str
    fit_xx: ComponentFit
    fit_yy: ComponentFit
    layers_used: list
    layers_excluded: list
    excluded_residuals: dict          # layers -> {"xx": kbar*A, "yy": kbar*A}
    area_angstrom2: float
    atoms_per_layer: int
    n_H: int

    @property
    def tau_mean_n_per_m(self) -> float:
        return 0.5 * (self.fit_xx.tau_inf_n_per_m + self.fit_yy.tau_inf_n_per_m)

    @property
    def tau_aniso_n_per_m(self) -> float:
        """tau_xx - tau_yy. Sign is meaningful and axis-convention dependent;
        the axis assignment is reported alongside it (CLAUDE.md sec 2)."""
        return self.fit_xx.tau_inf_n_per_m - self.fit_yy.tau_inf_n_per_m

    @property
    def sigma_res_mean_kbar(self) -> float:
        return 0.5 * (self.fit_xx.sigma_res_kbar + self.fit_yy.sigma_res_kbar)


# The in-plane Cartesian axes of each surface cell, taken from
# geometry._frame() (and mirrored in nv_spin_strain.slab_frame). Reported with
# every anisotropy number because the sign of tau_xx - tau_yy is meaningless
# without it (CLAUDE.md sec 2).
SURFACE_AXES = {
    "C100": {"orientation": "(100)", "x": "[110]", "y": "[1-10]",
             "note": "2x1 dimer bond lies along y = [1-10]; dimer rows run "
                     "along x = [110]"},
    "C110": {"orientation": "(110)", "x": "[1-10]", "y": "[001]",
             "note": "1x1; x is the zig-zag chain direction"},
    "C111": {"orientation": "(111)", "x": "[1-10]", "y": "[11-2]",
             "note": "1x1 H-terminated; in-plane isotropic by 3-fold symmetry"},
}


# ------------------------------------------------------------------ loading
def bulk_volume_per_carbon(a0_angstrom: float) -> float:
    """Omega_bulk, the bulk diamond volume per carbon atom, in Angstrom^3."""
    return a0_angstrom ** 3 / C_PER_CONVENTIONAL_CELL


def _cell_invariants(cell, run: str):
    """Return (area, Lz) after checking the cell is slab-shaped.

    The decomposition sigma*Lz = 2*tau + sigma_res*t assumes the third cell
    vector is normal to the surface plane, so that sigma_xx/sigma_yy really
    are the in-plane components and Lz really is the repeat distance.
    """
    a, b, c = (np.asarray(v, dtype=float) for v in cell)
    if not np.all(np.isfinite(np.array([a, b, c]))):
        raise TauFitError(f"{run}: non-finite cell vectors")
    Lz = float(np.linalg.norm(c))
    for label, v in (("a", a), ("b", b)):
        cosang = float(np.dot(v, c)) / (np.linalg.norm(v) * Lz)
        if abs(cosang) > 1e-6:
            raise TauFitError(
                f"{run}: cell vector {label} is not perpendicular to c "
                f"(cos = {cosang:.3e}); sigma_xx/sigma_yy are then not the "
                f"in-plane components and Lz is not the slab repeat")
    area = float(np.linalg.norm(np.cross(a, b)))
    return area, Lz


def load_stress_point(run_dir: Path, omega_c: float) -> StressPoint:
    """Read one *_stress_scf directory. pw.in and pw.out are read-only."""
    name = run_dir.name
    m = RUN_RE.match(name)
    if not m:
        raise TauFitError(f"{name}: not a recognised stress-SCF directory name")
    surface = m.group("surface")
    layers = int(m.group("layers"))

    pin = parse_slab.parse_pw_in(run_dir / "pw.in")
    pout = parse_slab.parse_pw_out(run_dir / "pw.out")

    if not pout["complete"] or pout["status"] != "JOB DONE":
        raise TauFitError(f"{name}: pw.out status is {pout['status']!r}, not 'JOB DONE'")
    if pout["errors"]:
        raise TauFitError(f"{name}: pw.out reports errors: {pout['errors'][:2]}")
    if not pout["stress"]:
        raise TauFitError(f"{name}: no stress tensor in pw.out (tstress not set?)")
    if pin["calculation_type"] != "scf":
        raise TauFitError(
            f"{name}: calculation = {pin['calculation_type']!r}; the tau ladder "
            f"needs single-point stress SCFs on already-relaxed geometries")

    cell = pin["cell_params_ang"]
    if cell is None:
        raise TauFitError(f"{name}: could not read CELL_PARAMETERS from pw.in")
    area, Lz = _cell_invariants(cell, name)

    atoms = pin["initial_positions_ang"]
    if not atoms:
        raise TauFitError(f"{name}: could not read ATOMIC_POSITIONS from pw.in")
    n_C = sum(1 for a in atoms if a["species"] == "C")
    n_H = sum(1 for a in atoms if a["species"] == "H")
    if n_C + n_H != len(atoms):
        other = sorted({a["species"] for a in atoms} - {"C", "H"})
        raise TauFitError(f"{name}: unexpected species {other}; this ladder is C/H only")

    # CLAUDE.md invariant 3: the name must describe the geometry. A folder
    # called _16L that does not hold a whole number of equal carbon layers is
    # not the structure the analysis claims it is.
    if n_C % layers != 0:
        raise TauFitError(
            f"{name}: {n_C} carbon atoms is not divisible by the {layers} layers "
            f"the folder name claims")

    # pw.out is ground truth for the pseudopotential actually opened
    # (CLAUDE.md invariant 7), not the ATOMIC_SPECIES card in pw.in.
    pseudo = pout["pseudo_files"] or pin["pseudopotentials"]
    pseudo_str = "; ".join(f"{k}:{v}" for k, v in sorted(pseudo.items()))

    thickness = n_C * omega_c / area
    zc = [a["z"] for a in atoms if a["species"] == "C"]

    return StressPoint(
        surface=surface, layers=layers, run=name, n_C=n_C, n_H=n_H,
        area_angstrom2=area, Lz_angstrom=Lz,
        thickness_angstrom=thickness,
        thickness_geometric_angstrom=max(zc) - min(zc),
        sigma_xx_kbar=float(pout["stress"]["xx"]),
        sigma_yy_kbar=float(pout["stress"]["yy"]),
        pseudopotentials=pseudo_str,
    )


def discover_points(runs_dir, prefix: str, omega_c: float) -> dict:
    """Load every stress SCF under runs_dir matching prefix, keyed by surface."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise TauFitError(f"runs directory not found: {runs_dir}")
    by_surface = {}
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir() or not (child / "pw.out").exists():
            continue
        m = RUN_RE.match(child.name)
        if not m or m.group("prefix") != prefix:
            continue
        pt = load_stress_point(child, omega_c)
        by_surface.setdefault(pt.surface, []).append(pt)
    if not by_surface:
        raise TauFitError(
            f"no '{prefix}~<surface>_<N>L_stress_scf' directories under {runs_dir}")
    for surface, pts in by_surface.items():
        pts.sort(key=lambda p: p.layers)
        _check_ladder_consistency(surface, pts)
    return by_surface


def _check_ladder_consistency(surface: str, pts: list) -> None:
    """A thickness ladder must vary thickness and nothing else."""
    areas = {round(p.area_angstrom2, 6) for p in pts}
    if len(areas) != 1:
        raise TauFitError(
            f"{surface}: in-plane cell area differs across the ladder ({sorted(areas)}); "
            f"the fit assumes a fixed surface cell")
    per_layer = {p.n_C // p.layers for p in pts}
    if len(per_layer) != 1:
        raise TauFitError(
            f"{surface}: carbon atoms per layer differs across the ladder "
            f"({sorted(per_layer)}); the folder layer counts and the geometries disagree")
    n_h = {p.n_H for p in pts}
    if len(n_h) != 1:
        raise TauFitError(
            f"{surface}: hydrogen count differs across the ladder ({sorted(n_h)}); "
            f"the termination is not constant")
    pseudos = {p.pseudopotentials for p in pts}
    if len(pseudos) != 1:
        raise TauFitError(
            f"{surface}: pseudopotentials differ across the ladder: {sorted(pseudos)}")


# ---------------------------------------------------------------- the fit
def fit_component(points: list, component: str) -> ComponentFit:
    """sigma_ij * Lz = 2*tau_inf + sigma_res * t, ordinary least squares."""
    if len(points) < 3:
        raise TauFitError(
            f"component {component}: {len(points)} points is too few for a "
            f"two-parameter fit with a meaningful residual")
    t = np.array([p.thickness_angstrom for p in points])
    y = np.array([p.sigma(component) * p.Lz_angstrom for p in points])
    design = np.vstack([np.ones_like(t), t]).T
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept, slope = float(coeffs[0]), float(coeffs[1])
    resid = y - design @ coeffs
    return ComponentFit(
        component=component,
        tau_inf_n_per_m=KBAR_ANGSTROM_TO_N_PER_M * intercept / 2.0,
        sigma_res_kbar=slope,
        rms_kbar_angstrom=float(np.sqrt(np.mean(resid ** 2))),
        max_abs_residual_kbar_angstrom=float(np.max(np.abs(resid))),
        n_points=len(points),
        residuals={p.layers: float(r) for p, r in zip(points, resid)},
    )


def _predict(fit: ComponentFit, t: float) -> float:
    """sigma*Lz predicted by the fit at bulk-equivalent thickness t."""
    intercept = 2.0 * fit.tau_inf_n_per_m / KBAR_ANGSTROM_TO_N_PER_M
    return intercept + fit.sigma_res_kbar * t


def fit_surface(surface: str, points: list, exclude_layers) -> SurfaceFit:
    exclude = set(exclude_layers)
    used = [p for p in points if p.layers not in exclude]
    dropped = [p for p in points if p.layers in exclude]
    if not used:
        raise TauFitError(f"{surface}: every point was excluded by --exclude-layers")

    fit_xx = fit_component(used, "xx")
    fit_yy = fit_component(used, "yy")

    excluded_residuals = {
        p.layers: {
            "xx": p.sigma_xx_kbar * p.Lz_angstrom - _predict(fit_xx, p.thickness_angstrom),
            "yy": p.sigma_yy_kbar * p.Lz_angstrom - _predict(fit_yy, p.thickness_angstrom),
        }
        for p in dropped
    }

    axes = SURFACE_AXES.get(surface, {})
    return SurfaceFit(
        surface=surface,
        orientation=axes.get("orientation", f"({surface[1:]})"),
        fit_xx=fit_xx, fit_yy=fit_yy,
        layers_used=[p.layers for p in used],
        layers_excluded=[p.layers for p in dropped],
        excluded_residuals=excluded_residuals,
        area_angstrom2=used[0].area_angstrom2,
        atoms_per_layer=used[0].n_C // used[0].layers,
        n_H=used[0].n_H,
    )


def check_sigma_res_consistency(fits: list, tol_kbar: float) -> list:
    """sigma_res is a bulk property: it must agree across the three surfaces.

    Returns a list of human-readable warning strings (empty if consistent).
    A large *magnitude* means the reference lattice constant is wrong; a large
    *spread* means something surface-specific is leaking into the slope.
    """
    warnings = []
    if len(fits) < 2:
        return warnings
    means = {f.surface: f.sigma_res_mean_kbar for f in fits}
    spread = max(means.values()) - min(means.values())
    if spread > tol_kbar:
        detail = ", ".join(f"{k} {v:+.3f}" for k, v in sorted(means.items()))
        warnings.append(
            f"sigma_res differs by {spread:.3f} kbar across surfaces "
            f"(tolerance {tol_kbar:.3f} kbar): {detail}. sigma_res is a bulk "
            f"property and should be surface-independent; a large spread "
            f"usually means the reference lattice constant is wrong or one "
            f"ladder is not in the asymptotic regime.")
    for f in fits:
        gap = abs(f.fit_xx.sigma_res_kbar - f.fit_yy.sigma_res_kbar)
        if gap > tol_kbar:
            warnings.append(
                f"{f.surface}: sigma_res differs between the xx and yy fits by "
                f"{gap:.3f} kbar ({f.fit_xx.sigma_res_kbar:+.3f} vs "
                f"{f.fit_yy.sigma_res_kbar:+.3f}); the interior residual stress "
                f"should be in-plane isotropic at a fixed bulk lattice constant.")
    return warnings


# ---------------------------------------------------------------- outputs
SUMMARY_FIELDS = [
    "surface", "orientation", "epistemic_level",
    "axis_x", "axis_y", "axis_note",
    "tau_xx_inf_n_per_m", "tau_yy_inf_n_per_m",
    "tau_mean_n_per_m", "tau_aniso_n_per_m",
    "sigma_res_xx_kbar", "sigma_res_yy_kbar", "sigma_res_mean_kbar",
    "rms_xx_kbar_angstrom", "rms_yy_kbar_angstrom",
    "rms_xx_as_tau_n_per_m", "rms_yy_as_tau_n_per_m",
    "max_abs_residual_xx_kbar_angstrom", "max_abs_residual_yy_kbar_angstrom",
    "n_points", "layers_used", "layers_excluded",
    "excluded_residual_xx_kbar_angstrom", "excluded_residual_yy_kbar_angstrom",
    "thickness_definition", "cell_area_angstrom2", "carbon_atoms_per_layer",
    "hydrogen_atoms", "sign_convention",
]

SIGN_CONVENTION = (
    "sigma>0 compressed (CLAUDE.md sec 2); tau = sigma*Lz/2 inherits that, so "
    "tau>0 is COMPRESSIVE surface stress: the surface pushes outward and a "
    "released cell expands. This equals MINUS the continuum surface stress f "
    "(for which positive is tensile); consumers wanting f must negate. "
    "Verified against free 2D vc-relax (cell_dofree='2Dxy') at 16L on all "
    "three surfaces: positive tau expanded the cell in 6 of 6 axes "
    "((100)[110] +0.139%, (110)[1-10] +0.145%, (110)[001] +0.429%, "
    "(111) +0.052% both axes), negative tau contracted it "
    "((100)[1-10] -0.664%).")
THICKNESS_DEFINITION = "t = N_C * a0^3/8 / A (bulk-equivalent, atom-counted)"


def _fmt(v, nd=6):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def summary_rows(fits: list) -> list:
    rows = []
    for f in sorted(fits, key=lambda x: x.surface):
        axes = SURFACE_AXES.get(f.surface, {})
        excl_xx = "; ".join(
            f"{n}L:{f.excluded_residuals[n]['xx']:+.3f}" for n in sorted(f.excluded_residuals))
        excl_yy = "; ".join(
            f"{n}L:{f.excluded_residuals[n]['yy']:+.3f}" for n in sorted(f.excluded_residuals))
        rows.append({
            "surface": f.surface,
            "orientation": f.orientation,
            "epistemic_level": EPISTEMIC_LEVEL,
            "axis_x": axes.get("x", ""),
            "axis_y": axes.get("y", ""),
            "axis_note": axes.get("note", ""),
            "tau_xx_inf_n_per_m": _fmt(f.fit_xx.tau_inf_n_per_m),
            "tau_yy_inf_n_per_m": _fmt(f.fit_yy.tau_inf_n_per_m),
            "tau_mean_n_per_m": _fmt(f.tau_mean_n_per_m),
            "tau_aniso_n_per_m": _fmt(f.tau_aniso_n_per_m),
            "sigma_res_xx_kbar": _fmt(f.fit_xx.sigma_res_kbar),
            "sigma_res_yy_kbar": _fmt(f.fit_yy.sigma_res_kbar),
            "sigma_res_mean_kbar": _fmt(f.sigma_res_mean_kbar),
            "rms_xx_kbar_angstrom": _fmt(f.fit_xx.rms_kbar_angstrom),
            "rms_yy_kbar_angstrom": _fmt(f.fit_yy.rms_kbar_angstrom),
            "rms_xx_as_tau_n_per_m": _fmt(f.fit_xx.rms_as_tau_n_per_m),
            "rms_yy_as_tau_n_per_m": _fmt(f.fit_yy.rms_as_tau_n_per_m),
            "max_abs_residual_xx_kbar_angstrom": _fmt(f.fit_xx.max_abs_residual_kbar_angstrom),
            "max_abs_residual_yy_kbar_angstrom": _fmt(f.fit_yy.max_abs_residual_kbar_angstrom),
            "n_points": f.fit_xx.n_points,
            "layers_used": " ".join(f"{n}L" for n in f.layers_used),
            "layers_excluded": " ".join(f"{n}L" for n in f.layers_excluded),
            "excluded_residual_xx_kbar_angstrom": excl_xx,
            "excluded_residual_yy_kbar_angstrom": excl_yy,
            "thickness_definition": THICKNESS_DEFINITION,
            "cell_area_angstrom2": _fmt(f.area_angstrom2),
            "carbon_atoms_per_layer": f.atoms_per_layer,
            "hydrogen_atoms": f.n_H,
            "sign_convention": SIGN_CONVENTION,
        })
    return rows


POINT_FIELDS = [
    "surface", "layers", "run", "in_fit", "n_C", "n_H",
    "cell_area_angstrom2", "Lz_angstrom",
    "thickness_atomic_angstrom", "thickness_geometric_angstrom",
    "sigma_xx_kbar", "sigma_yy_kbar",
    "sigma_xx_Lz_kbar_angstrom", "sigma_yy_Lz_kbar_angstrom",
    "residual_xx_kbar_angstrom", "residual_yy_kbar_angstrom",
]


def point_rows(by_surface: dict, fits: dict) -> list:
    rows = []
    for surface in sorted(by_surface):
        f = fits[surface]
        for p in by_surface[surface]:
            in_fit = p.layers in f.layers_used
            if in_fit:
                rxx = f.fit_xx.residuals[p.layers]
                ryy = f.fit_yy.residuals[p.layers]
            else:
                rxx = f.excluded_residuals[p.layers]["xx"]
                ryy = f.excluded_residuals[p.layers]["yy"]
            rows.append({
                "surface": surface, "layers": p.layers, "run": p.run,
                "in_fit": in_fit, "n_C": p.n_C, "n_H": p.n_H,
                "cell_area_angstrom2": _fmt(p.area_angstrom2),
                "Lz_angstrom": _fmt(p.Lz_angstrom),
                "thickness_atomic_angstrom": _fmt(p.thickness_angstrom),
                "thickness_geometric_angstrom": _fmt(p.thickness_geometric_angstrom),
                "sigma_xx_kbar": _fmt(p.sigma_xx_kbar, 3),
                "sigma_yy_kbar": _fmt(p.sigma_yy_kbar, 3),
                "sigma_xx_Lz_kbar_angstrom": _fmt(p.sigma_xx_kbar * p.Lz_angstrom, 4),
                "sigma_yy_Lz_kbar_angstrom": _fmt(p.sigma_yy_kbar * p.Lz_angstrom, 4),
                "residual_xx_kbar_angstrom": _fmt(rxx, 4),
                "residual_yy_kbar_angstrom": _fmt(ryy, 4),
            })
    return rows


def write_csv(rows: list, fields: list, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(fits: list, warnings: list, meta: dict) -> str:
    L = ["# Thickness-extrapolated surface stress (tau_inf)", "",
         f"Epistemic level: **{EPISTEMIC_LEVEL}** (CLAUDE.md sec 4). Converged "
         f"production stress SCFs at "
         f"{meta['ecutwfc']:.0f}/{meta['ecutrho']:.0f} Ry, "
         f"a0 = {meta['a0_angstrom']:.6f} Angstrom.", "",
         "## Conventions", "",
         f"* Stress sign: {SIGN_CONVENTION}.",
         "* `tau[N/m] = 0.01 * (sigma[kbar] * Lz[Angstrom]) / 2`; the factor 2 "
         "is the slab's two surfaces.",
         f"* Bulk-equivalent thickness: `{THICKNESS_DEFINITION}`, with "
         f"Omega_bulk = {meta['omega_c_angstrom3']:.6f} Angstrom^3 per carbon.",
         "* `tau_aniso = tau_xx - tau_yy`; its sign is meaningless without the "
         "axis assignment given per surface below.", "",
         "## Fit set", "",
         f"* Included: {meta['layers_used_note']}",
         f"* Excluded: {meta['layers_excluded_note']} — outside the asymptotic "
         "regime on all three surfaces (parity oscillation on C100; the outlier "
         "in energy slope, surface energy, and tau_aniso).", "",
         "## Results", "",
         "| surface | axes (x, y) | tau_xx | tau_yy | tau_mean | tau_aniso | "
         "sigma_res | rms |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for f in sorted(fits, key=lambda x: x.surface):
        axes = SURFACE_AXES.get(f.surface, {})
        rms = max(f.fit_xx.rms_kbar_angstrom, f.fit_yy.rms_kbar_angstrom)
        L.append(
            f"| {f.orientation} | {axes.get('x','?')}, {axes.get('y','?')} | "
            f"{f.fit_xx.tau_inf_n_per_m:+.4f} | {f.fit_yy.tau_inf_n_per_m:+.4f} | "
            f"{f.tau_mean_n_per_m:+.4f} | {f.tau_aniso_n_per_m:+.4f} | "
            f"{f.sigma_res_mean_kbar:+.3f} | {rms:.3f} |")
    L += ["", "tau in N/m, sigma_res in kbar, rms in kbar*Angstrom "
          "(worse of the two components).", ""]

    L += ["## Per-surface detail", ""]
    for f in sorted(fits, key=lambda x: x.surface):
        axes = SURFACE_AXES.get(f.surface, {})
        L += [f"### {f.orientation} ({f.surface})", "",
              f"* Surface-frame axes: x = {axes.get('x','?')}, "
              f"y = {axes.get('y','?')}. {axes.get('note','')}",
              f"* Layers in fit: {' '.join(str(n) + 'L' for n in f.layers_used)}; "
              f"excluded: {' '.join(str(n) + 'L' for n in f.layers_excluded) or 'none'}",
              f"* tau_xx = {f.fit_xx.tau_inf_n_per_m:+.4f} N/m, "
              f"tau_yy = {f.fit_yy.tau_inf_n_per_m:+.4f} N/m",
              f"* sigma_res = {f.fit_xx.sigma_res_kbar:+.3f} (xx) / "
              f"{f.fit_yy.sigma_res_kbar:+.3f} (yy) kbar",
              f"* fit rms = {f.fit_xx.rms_kbar_angstrom:.3f} / "
              f"{f.fit_yy.rms_kbar_angstrom:.3f} kbar*Angstrom, equivalent to "
              f"{f.fit_xx.rms_as_tau_n_per_m:.4f} / "
              f"{f.fit_yy.rms_as_tau_n_per_m:.4f} N/m on tau"]
        if f.excluded_residuals:
            for n in sorted(f.excluded_residuals):
                r = f.excluded_residuals[n]
                L.append(f"* {n}L residual against the fit: {r['xx']:+.3f} (xx) / "
                         f"{r['yy']:+.3f} (yy) kbar*Angstrom")
        L.append("")

    L += ["## sigma_res consistency", "",
          "sigma_res is a bulk property at the reference lattice constant and "
          "must agree across surfaces. Its magnitude is a direct diagnostic of "
          "a0: a large value means the reference lattice constant is wrong.", ""]
    if warnings:
        L += ["**WARNINGS**", ""] + [f"* {w}" for w in warnings] + [""]
    else:
        L += [f"* Consistent within the {meta['sigma_res_tol_kbar']:.3f} kbar "
              f"tolerance.", ""]

    L += ["## What this is and is not", "",
          "* This is the surface stress of the H-terminated slab extrapolated "
          "to infinite thickness. It is a slab-cell quantity: the interior "
          "elastic response has been separated out by the fit, not by an "
          "independent bulk calculation.",
          "* It is a clamped-cell, relaxed-ion quantity: ions were relaxed at "
          "the fixed bulk in-plane lattice constant. It is not a relaxed-cell "
          "surface stress and says nothing about the strain response "
          "(CLAUDE.md sec 3).",
          "* The fit is four points and two parameters per component. The "
          "residuals above, not the closeness of the numbers to expectation, "
          "are the evidence of fit quality.", ""]
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------- main
def run(runs_dir, out_dir, prefix, exclude_layers, reference_config,
        sigma_res_tol) -> dict:
    ref = elastic_reference.load_elastic_reference(reference_config)
    a0 = ref.bulk.a0_angstrom
    omega_c = bulk_volume_per_carbon(a0)

    by_surface = discover_points(runs_dir, prefix, omega_c)
    fits = {s: fit_surface(s, pts, exclude_layers) for s, pts in by_surface.items()}
    fit_list = list(fits.values())
    warnings = check_sigma_res_consistency(fit_list, sigma_res_tol)

    any_point = next(iter(by_surface.values()))[0]
    pin = parse_slab.parse_pw_in(Path(runs_dir) / any_point.run / "pw.in")
    used = sorted({n for f in fit_list for n in f.layers_used})
    excl = sorted({n for f in fit_list for n in f.layers_excluded})
    meta = {
        "a0_angstrom": a0,
        "a0_source": ref.bulk.source_type,
        "reference_config": str(reference_config),
        "omega_c_angstrom3": omega_c,
        "ecutwfc": pin["ecutwfc"], "ecutrho": pin["ecutrho"],
        "runs_dir": str(runs_dir), "run_prefix": prefix,
        "epistemic_level": EPISTEMIC_LEVEL,
        "thickness_definition": THICKNESS_DEFINITION,
        "sign_convention": SIGN_CONVENTION,
        "layers_used_note": " ".join(f"{n}L" for n in used),
        "layers_excluded_note": " ".join(f"{n}L" for n in excl) or "none",
        "sigma_res_tol_kbar": sigma_res_tol,
        "sigma_res_consistent": not warnings,
        "warnings": warnings,
    }

    out_dir = Path(out_dir)
    summary = summary_rows(fit_list)
    write_csv(summary, SUMMARY_FIELDS, out_dir / "tau_infinity.csv")
    write_csv(point_rows(by_surface, fits), POINT_FIELDS,
              out_dir / "tau_infinity_points.csv")
    (out_dir / "tau_infinity_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    (out_dir / "tau_infinity_report.md").write_text(
        render_report(fit_list, warnings, meta))

    return {"fits": fits, "meta": meta, "warnings": warnings,
            "points": by_surface, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX,
                    help="run-name prefix before '~' (default: %(default)s)")
    ap.add_argument("--exclude-layers", type=int, nargs="*", default=[6],
                    metavar="N",
                    help="layer counts to keep out of the fit (default: 6, "
                         "which is outside the asymptotic regime)")
    ap.add_argument("--reference-config", default=DEFAULT_REFERENCE)
    ap.add_argument("--sigma-res-tol", type=float,
                    default=DEFAULT_SIGMA_RES_TOL_KBAR, metavar="KBAR",
                    help="warn if sigma_res spreads by more than this across "
                         "surfaces (default: %(default)s)")
    args = ap.parse_args(argv)

    try:
        result = run(args.runs_dir, args.out_dir, args.prefix,
                     args.exclude_layers, args.reference_config,
                     args.sigma_res_tol)
    except (TauFitError, elastic_reference.ElasticReferenceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    meta = result["meta"]
    print(f"# tau_inf thickness extrapolation  [{EPISTEMIC_LEVEL}]")
    print(f"# a0 = {meta['a0_angstrom']:.6f} A ({meta['a0_source']}), "
          f"Omega_C = {meta['omega_c_angstrom3']:.6f} A^3, "
          f"{meta['ecutwfc']:.0f}/{meta['ecutrho']:.0f} Ry")
    print(f"# thickness: {THICKNESS_DEFINITION}")
    print(f"# sign: {SIGN_CONVENTION}")
    print(f"# fitted layers: {meta['layers_used_note']}   "
          f"EXCLUDED: {meta['layers_excluded_note']}")
    print(f"{'surface':9s} {'axes(x,y)':18s} {'tau_xx':>9s} {'tau_yy':>9s} "
          f"{'tau_mean':>9s} {'tau_aniso':>10s} {'sig_res':>9s} {'rms':>8s}")
    for f in sorted(result["fits"].values(), key=lambda x: x.surface):
        axes = SURFACE_AXES.get(f.surface, {})
        rms = max(f.fit_xx.rms_kbar_angstrom, f.fit_yy.rms_kbar_angstrom)
        print(f"{f.orientation:9s} "
              f"{axes.get('x','?') + ', ' + axes.get('y','?'):18s} "
              f"{f.fit_xx.tau_inf_n_per_m:+9.4f} {f.fit_yy.tau_inf_n_per_m:+9.4f} "
              f"{f.tau_mean_n_per_m:+9.4f} {f.tau_aniso_n_per_m:+10.4f} "
              f"{f.sigma_res_mean_kbar:+9.3f} {rms:8.3f}")
    print("# tau in N/m, sigma_res in kbar, rms in kbar*Angstrom")
    for w in result["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    for name in ("tau_infinity.csv", "tau_infinity_points.csv",
                 "tau_infinity_meta.json", "tau_infinity_report.md"):
        print(f"wrote {os.path.join(args.out_dir, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
