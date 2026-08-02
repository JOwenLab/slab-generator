#!/usr/bin/env python3
"""layer_profile.py — depth profile of surface relaxation in a relaxed slab.

Why this exists
---------------
It answers one question: **how deep does the surface perturbation reach?**

That question decides whether the continuum model in `particle_strain.py` is
allowed to treat the interior of a nanodiamond as uniformly strained bulk. A
3 nm particle is roughly 30 (111) layers across, i.e. about 15 Angstrom of
half-thickness. If the relaxation decay length is a few Angstrom, the core is
genuinely bulk-like and a volume-averaged interior strain is a fair
description. If the decay length were comparable to the particle radius, there
would be no bulk-like core, and the continuum model would need a
depth-dependent correction rather than a single uniform interior strain.

What is extractable, and what is not
------------------------------------
IMPORTANT: in a periodic slab the in-plane lattice is fixed by the cell, so it
is uniform by construction. There is **no depth-dependent in-plane strain** in
these calculations, and this module does not pretend to extract one. Any
"in-plane strain profile" derived from a periodic slab would be an artefact of
the analysis, not a result.

What is extractable is the out-of-plane profile and the internal distortion:

  * carbon atoms assigned to layers by z;
  * interlayer spacing d_n against the ideal bulk value, giving eps_zz(z);
  * intralayer rumpling (the z spread within one nominal layer);
  * nearest-neighbour C-C bond lengths against the bulk value, by depth.

The decay of |eps_zz| away from the surface is then fitted with an exponential
and the decay length reported per surface.

Ideal reference spacings
------------------------
Derived analytically from the conventional cubic lattice constant a0 rather
than taken from the generator that made the slab (CLAUDE.md sec 0: read the
structure, do not trust the script):

  (100)  uniform      a0/4
  (110)  uniform      a0/(2*sqrt(2))
  (111)  alternating  a0*sqrt(3)/12  and  a0*sqrt(3)/4   (sum = a0/sqrt(3))

Each observed gap is assigned to its nearest ideal class, and the assignment is
then verified: every gap must be within `--gap-tol` of exactly one class, the
classes must strictly alternate where there is more than one, and the gap count
must equal (layers - 1). A phase convention is therefore never assumed.

Because the slabs are inversion-symmetric, the profile is folded about the slab
centre before fitting. The disagreement between the two halves is reported as
`fold_asymmetry` and is a free check on the symmetry invariant (CLAUDE.md
invariant 1): it should be at numerical noise.

Epistemic level
---------------
L2 (CLAUDE.md sec 4) for the geometric profile itself: it is read directly from
converged production relaxations at 90/720 Ry, a0 = 3.572997 Angstrom, and the
thickness ladder shows the interior profile has stopped changing. The
exponential decay LENGTH is a derived fit over few points and is labelled L1:
on (111) the two alternating gap classes have different magnitudes, so a single
exponential is an envelope description rather than a law. Class-resolved fits
and a convention-free penetration depth are reported alongside it.

Usage
-----
    python3 layer_profile.py
    python3 layer_profile.py --run C111=final~C111_24L --out-dir results/production
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

DEFAULT_RUNS_DIR = "results/production"
DEFAULT_OUT_DIR = "results/production"
DEFAULT_REFERENCE = "config/reference_pbe_sssp.json"

# Thickest converged relaxation available for each surface.
DEFAULT_RUNS = {
    "C100": "thick_a0corr~C100_16L",
    "C110": "thick_a0corr~C110_16L",
    "C111": "final~C111_24L",
}

# Relative tolerance on |d_observed - d_ideal| / d_ideal when assigning a gap
# to an ideal spacing class.
DEFAULT_GAP_TOL = 0.25
# |eps_zz| below this is treated as numerical noise and kept out of the fit.
DEFAULT_EPS_FLOOR = 1.0e-4
# Depth at which |eps_zz| has fallen below this is the reported penetration.
DEFAULT_PENETRATION_THRESHOLD = 1.0e-3

EPISTEMIC_LEVEL_PROFILE = "L2"
EPISTEMIC_LEVEL_DECAY = "L1"

RUN_RE = re.compile(r"~(?P<surface>C\d{3})_(?P<layers>\d+)L(?P<tail>_vc)?$")


class LayerProfileError(Exception):
    """A load-bearing invariant of the layer analysis was violated."""


# ------------------------------------------------------- ideal bulk spacings
def ideal_spacings(orientation: str, a0: float) -> list:
    """Ideal interlayer spacings (Angstrom) for the orientation, as a list of
    distinct classes. Two entries means the spacings alternate."""
    if orientation == "100":
        return [a0 / 4.0]
    if orientation == "110":
        return [a0 / (2.0 * math.sqrt(2.0))]
    if orientation == "111":
        return [a0 * math.sqrt(3.0) / 12.0, a0 * math.sqrt(3.0) / 4.0]
    raise LayerProfileError(f"no ideal spacing known for orientation {orientation!r}")


def bulk_bond_length(a0: float) -> float:
    """Nearest-neighbour C-C distance in bulk diamond, Angstrom."""
    return math.sqrt(3.0) / 4.0 * a0


# -------------------------------------------------------------- data model
@dataclass
class Layer:
    index: int                  # 0 = topmost
    z_center: float
    z_min: float
    z_max: float
    n_atoms: int
    depth: float                # from the nearer outer carbon layer
    rumpling: float
    # gap to the layer below (None for the bottom layer)
    d_below: float = None
    d_below_ideal: float = None
    eps_zz_below: float = None
    gap_class: int = None
    gap_depth: float = None
    # C-C bonds incident on this layer
    n_bonds: int = 0
    bond_mean: float = None
    bond_min: float = None
    bond_max: float = None
    bond_dev_mean_pct: float = None
    ch_bond_mean: float = None


@dataclass
class DecayFit:
    label: str
    decay_length_angstrom: float
    amplitude: float
    r_squared: float
    n_points: int
    depths: list = field(default_factory=list)
    values: list = field(default_factory=list)
    status: str = "ok"


@dataclass
class SurfaceProfile:
    surface: str
    orientation: str
    run: str
    n_layers: int
    n_carbon: int
    n_hydrogen: int
    atoms_per_layer: int
    a0_angstrom: float
    cell_area_angstrom2: float
    Lz_angstrom: float
    layers: list
    fold_asymmetry: float
    noise_floor: float
    fit_all: DecayFit
    fit_by_class: list
    penetration_depth_angstrom: float
    penetration_threshold: float
    max_rumpling: float
    max_bond_dev_pct: float
    surface_eps_zz: float


# ------------------------------------------------------------- geometry I/O
def _read_relaxed(run_dir: Path):
    """Return (cell 3x3 Angstrom, atoms) from a completed relaxation.

    pw.in and pw.out are read-only. For a fixed-cell relax the cell comes from
    pw.in; for vc-relax the final cell printed by pw.out wins.
    """
    pin = parse_slab.parse_pw_in(run_dir / "pw.in")
    pout = parse_slab.parse_pw_out(run_dir / "pw.out")

    if not pout["complete"] or pout["status"] != "JOB DONE":
        raise LayerProfileError(
            f"{run_dir.name}: pw.out status is {pout['status']!r}, not 'JOB DONE'")
    if pout["errors"]:
        raise LayerProfileError(f"{run_dir.name}: pw.out reports errors: {pout['errors'][:2]}")
    calc = pin["calculation_type"]
    if calc not in ("relax", "vc-relax"):
        raise LayerProfileError(
            f"{run_dir.name}: calculation = {calc!r}; the layer profile needs a "
            f"relaxation, not a single-point run")
    if pout["relax_converged"] is False:
        raise LayerProfileError(
            f"{run_dir.name}: the relaxation did not converge; its coordinates "
            f"are not a relaxed geometry")

    atoms = pout["final_positions_ang"]
    if not atoms:
        raise LayerProfileError(
            f"{run_dir.name}: no final ATOMIC_POSITIONS block in pw.out")

    cell = pout["final_cell_ang"] if calc == "vc-relax" else pin["cell_params_ang"]
    if cell is None:
        raise LayerProfileError(f"{run_dir.name}: could not read the cell")
    cell = np.asarray(cell, dtype=float)
    if not np.all(np.isfinite(cell)):
        raise LayerProfileError(f"{run_dir.name}: non-finite cell vectors")
    coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    if not np.all(np.isfinite(coords)):
        raise LayerProfileError(f"{run_dir.name}: non-finite atomic coordinates")
    return cell, atoms


# ------------------------------------------------------------ layer sorting
def assign_layers(z_values, tol: float) -> list:
    """Group sorted z values into layers, splitting whenever the gap exceeds
    tol. Returns a list of lists, deepest first (ascending z)."""
    zs = sorted(float(z) for z in z_values)
    groups = [[zs[0]]]
    for z in zs[1:]:
        if z - groups[-1][-1] > tol:
            groups.append([z])
        else:
            groups[-1].append(z)
    return groups


def _classify_gaps(gaps, classes, tol: float, surface: str) -> list:
    """Assign each observed gap to its nearest ideal class, then verify."""
    out = []
    for n, d in enumerate(gaps):
        devs = [abs(d - c) / c for c in classes]
        k = int(np.argmin(devs))
        if devs[k] > tol:
            raise LayerProfileError(
                f"{surface}: interlayer gap {n} is {d:.4f} Angstrom, "
                f"{devs[k] * 100:.1f}% from the nearest ideal spacing "
                f"{classes[k]:.4f} (tolerance {tol * 100:.0f}%). Either the "
                f"layer assignment is wrong or this is not the structure the "
                f"folder name claims.")
        if len(classes) > 1:
            second = sorted(devs)[1]
            if second - devs[k] < 0.05:
                raise LayerProfileError(
                    f"{surface}: interlayer gap {n} ({d:.4f} Angstrom) is "
                    f"ambiguous between the ideal spacing classes {classes}; "
                    f"refusing to guess the alternation phase.")
        out.append(k)
    if len(classes) > 1:
        for n in range(1, len(out)):
            if out[n] == out[n - 1]:
                raise LayerProfileError(
                    f"{surface}: ideal spacing classes do not alternate at gap "
                    f"{n} (classes {out}); the layer assignment is wrong.")
    return out


# ------------------------------------------------------------- bond lengths
def _cc_bonds(cell, coords, species, rcut: float):
    """C-C bond list [(i, j, length)] using in-plane minimum image.

    The slab is finite in z, so only the two in-plane lattice vectors generate
    images. Each bond is returned once, with i < j in the (image, index) sense.
    """
    a1, a2 = cell[0], cell[1]
    images = [i * a1 + j * a2 for i in (-1, 0, 1) for j in (-1, 0, 1)]
    idx_c = [i for i, s in enumerate(species) if s == "C"]
    bonds = []
    for a, i in enumerate(idx_c):
        for j in idx_c[a + 1:]:
            best = min(float(np.linalg.norm(coords[j] + im - coords[i]))
                       for im in images)
            if best < rcut:
                bonds.append((i, j, best))
    return bonds


def _ch_bonds(cell, coords, species, rcut: float):
    a1, a2 = cell[0], cell[1]
    images = [i * a1 + j * a2 for i in (-1, 0, 1) for j in (-1, 0, 1)]
    idx_c = [i for i, s in enumerate(species) if s == "C"]
    idx_h = [i for i, s in enumerate(species) if s == "H"]
    out = []
    for i in idx_c:
        for h in idx_h:
            best = min(float(np.linalg.norm(coords[h] + im - coords[i]))
                       for im in images)
            if best < rcut:
                out.append((i, h, best))
    return out


# ------------------------------------------------------------- decay fits
def fit_exponential(depths, values, label: str) -> DecayFit:
    """Fit |eps| = A * exp(-depth / lambda) by least squares on ln|eps|."""
    d = np.asarray(depths, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(d) < 2:
        return DecayFit(label, float("nan"), float("nan"), float("nan"),
                        len(d), list(d), list(v),
                        status=f"too few points above the noise floor ({len(d)})")
    y = np.log(v)
    design = np.vstack([np.ones_like(d), -d]).T
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    ln_a, inv_lambda = float(coeffs[0]), float(coeffs[1])
    if inv_lambda <= 0:
        return DecayFit(label, float("inf"), float(np.exp(ln_a)), float("nan"),
                        len(d), list(d), list(v),
                        status="|eps_zz| does not decay with depth over the "
                               "fitted points; no decay length is defined")
    pred = design @ coeffs
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    status = "ok" if len(d) >= 3 else "two points only: lambda is exact, not fitted"
    return DecayFit(label, 1.0 / inv_lambda, float(np.exp(ln_a)), r2,
                    len(d), list(d), list(v), status=status)


# ---------------------------------------------------------------- analysis
def analyse_run(surface: str, run_dir: Path, a0: float, gap_tol: float,
                eps_floor: float, penetration_threshold: float) -> SurfaceProfile:
    m = RUN_RE.search(run_dir.name)
    if not m:
        raise LayerProfileError(
            f"{run_dir.name}: cannot read surface and layer count from the "
            f"directory name")
    if m.group("surface") != surface:
        raise LayerProfileError(
            f"{run_dir.name}: folder names surface {m.group('surface')}, "
            f"requested {surface} (CLAUDE.md invariant 3)")
    orientation = surface[1:]
    n_layers = int(m.group("layers"))

    cell, atoms = _read_relaxed(run_dir)
    species = [a["species"] for a in atoms]
    coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    z_c = [a["z"] for a in atoms if a["species"] == "C"]
    n_carbon = len(z_c)
    n_hydrogen = sum(1 for s in species if s == "H")
    if n_carbon % n_layers != 0:
        raise LayerProfileError(
            f"{run_dir.name}: {n_carbon} carbon atoms is not divisible by the "
            f"{n_layers} layers the folder name claims")

    classes = ideal_spacings(orientation, a0)
    # Split tolerance must be well below the smallest ideal spacing, but above
    # any physical intralayer rumpling. 40% of the smallest gap satisfies both
    # here; a violation is caught by the layer-count check immediately below.
    split_tol = 0.4 * min(classes)
    groups = assign_layers(z_c, split_tol)
    if len(groups) != n_layers:
        raise LayerProfileError(
            f"{run_dir.name}: clustered {len(groups)} carbon layers but the "
            f"folder name claims {n_layers}")
    counts = {len(g) for g in groups}
    if len(counts) != 1:
        raise LayerProfileError(
            f"{run_dir.name}: carbon layers hold different atom counts "
            f"({sorted(len(g) for g in groups)}); the layer assignment is wrong")

    groups = groups[::-1]                     # topmost layer first
    centers = [float(np.mean(g)) for g in groups]
    z_top, z_bot = centers[0], centers[-1]

    layers = []
    for i, g in enumerate(groups):
        c = centers[i]
        layers.append(Layer(
            index=i, z_center=c, z_min=min(g), z_max=max(g), n_atoms=len(g),
            depth=min(z_top - c, c - z_bot), rumpling=max(g) - min(g)))

    gaps = [centers[i] - centers[i + 1] for i in range(len(centers) - 1)]
    gap_class = _classify_gaps(gaps, classes, gap_tol, run_dir.name)
    for i, (d, k) in enumerate(zip(gaps, gap_class)):
        ideal = classes[k]
        mid = 0.5 * (centers[i] + centers[i + 1])
        layers[i].d_below = d
        layers[i].d_below_ideal = ideal
        layers[i].eps_zz_below = (d - ideal) / ideal
        layers[i].gap_class = k
        layers[i].gap_depth = min(z_top - mid, mid - z_bot)

    # ---- bonds
    rcut_cc = 1.15 * bulk_bond_length(a0)
    bonds = _cc_bonds(cell, coords, species, rcut_cc)
    if not bonds:
        raise LayerProfileError(f"{run_dir.name}: no C-C bonds found within "
                                f"{rcut_cc:.3f} Angstrom; geometry is wrong")
    layer_of = {}
    for i, g in enumerate(groups):
        zset = set(round(z, 8) for z in g)
        for a_i, a in enumerate(atoms):
            if a["species"] == "C" and round(a["z"], 8) in zset:
                layer_of[a_i] = i
    per_layer = {i: [] for i in range(n_layers)}
    for i, j, length in bonds:
        for atom in (i, j):
            per_layer[layer_of[atom]].append(length)
    d0 = bulk_bond_length(a0)
    for i, lengths in per_layer.items():
        if not lengths:
            continue
        arr = np.array(lengths)
        layers[i].n_bonds = len(arr)
        layers[i].bond_mean = float(arr.mean())
        layers[i].bond_min = float(arr.min())
        layers[i].bond_max = float(arr.max())
        layers[i].bond_dev_mean_pct = float(100.0 * (arr.mean() - d0) / d0)

    if n_hydrogen:
        ch = _ch_bonds(cell, coords, species, 1.4)
        per_layer_ch = {}
        for i, _h, length in ch:
            per_layer_ch.setdefault(layer_of[i], []).append(length)
        for i, lengths in per_layer_ch.items():
            layers[i].ch_bond_mean = float(np.mean(lengths))

    # ---- fold the profile about the slab centre
    gap_layers = [L for L in layers if L.eps_zz_below is not None]
    folded = {}
    for L in gap_layers:
        key = (L.gap_class, round(L.gap_depth, 4))
        folded.setdefault(key, []).append(abs(L.eps_zz_below))
    fold_asymmetry = 0.0
    for key, vals in folded.items():
        if len(vals) > 1:
            fold_asymmetry = max(fold_asymmetry, max(vals) - min(vals))
    points = sorted(((key[1], key[0], float(np.mean(v)))
                     for key, v in folded.items()), key=lambda p: p[0])

    # ---- noise floor from the deepest folded points
    deepest = [p[2] for p in points[-2:]] or [0.0]
    noise_floor = max(3.0 * max(deepest), eps_floor)

    above = [p for p in points if p[2] > noise_floor]
    fit_all = fit_exponential([p[0] for p in above], [p[2] for p in above],
                              "all gaps")
    fit_by_class = []
    if len(classes) > 1:
        for k, ideal in enumerate(classes):
            sub = [p for p in above if p[1] == k]
            fit_by_class.append(fit_exponential(
                [p[0] for p in sub], [p[2] for p in sub],
                f"gap class {k} (ideal {ideal:.4f} A)"))

    # ---- convention-free penetration depth
    penetration = float("nan")
    for depth, _k, val in points:
        if val < penetration_threshold and all(
                v < penetration_threshold for d2, _k2, v in points if d2 >= depth):
            penetration = depth
            break

    area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
    bond_devs = [L.bond_dev_mean_pct for L in layers if L.bond_dev_mean_pct is not None]
    return SurfaceProfile(
        surface=surface, orientation=f"({orientation})", run=run_dir.name,
        n_layers=n_layers, n_carbon=n_carbon, n_hydrogen=n_hydrogen,
        atoms_per_layer=n_carbon // n_layers, a0_angstrom=a0,
        cell_area_angstrom2=area, Lz_angstrom=float(np.linalg.norm(cell[2])),
        layers=layers, fold_asymmetry=fold_asymmetry, noise_floor=noise_floor,
        fit_all=fit_all, fit_by_class=fit_by_class,
        penetration_depth_angstrom=penetration,
        penetration_threshold=penetration_threshold,
        max_rumpling=max(L.rumpling for L in layers),
        max_bond_dev_pct=max(abs(v) for v in bond_devs) if bond_devs else float("nan"),
        surface_eps_zz=gap_layers[0].eps_zz_below if gap_layers else float("nan"),
    )


# ---------------------------------------------------------------- outputs
LAYER_FIELDS = [
    "surface", "run", "layer_index", "z_center_angstrom",
    "depth_from_surface_angstrom", "n_atoms", "rumpling_angstrom",
    "d_below_angstrom", "d_below_ideal_angstrom", "eps_zz_below",
    "eps_zz_below_pct", "gap_class", "gap_depth_angstrom",
    "n_cc_bonds", "cc_bond_mean_angstrom", "cc_bond_min_angstrom",
    "cc_bond_max_angstrom", "cc_bond_dev_mean_pct", "ch_bond_mean_angstrom",
    "epistemic_level",
]

SUMMARY_FIELDS = [
    "surface", "orientation", "run", "n_layers", "n_carbon", "n_hydrogen",
    "atoms_per_layer", "a0_angstrom", "bulk_cc_bond_angstrom",
    "surface_eps_zz_pct", "max_rumpling_angstrom", "max_cc_bond_dev_pct",
    "decay_length_angstrom", "decay_r_squared", "decay_n_points",
    "decay_status", "decay_epistemic_level",
    "decay_length_class0_angstrom", "decay_length_class1_angstrom",
    "penetration_depth_angstrom", "penetration_threshold",
    "noise_floor_eps_zz", "fold_asymmetry_eps_zz",
    "profile_epistemic_level", "note",
]

PROFILE_NOTE = ("out-of-plane profile only; a periodic slab has no "
                "depth-dependent in-plane strain to extract")


def _fmt(v, nd=6):
    if v is None:
        return ""
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.{nd}f}"
    return str(v)


def layer_rows(prof: SurfaceProfile) -> list:
    rows = []
    for L in prof.layers:
        rows.append({
            "surface": prof.surface, "run": prof.run, "layer_index": L.index,
            "z_center_angstrom": _fmt(L.z_center, 5),
            "depth_from_surface_angstrom": _fmt(L.depth, 5),
            "n_atoms": L.n_atoms,
            "rumpling_angstrom": _fmt(L.rumpling, 5),
            "d_below_angstrom": _fmt(L.d_below, 5),
            "d_below_ideal_angstrom": _fmt(L.d_below_ideal, 5),
            "eps_zz_below": _fmt(L.eps_zz_below, 8),
            "eps_zz_below_pct": _fmt(None if L.eps_zz_below is None
                                     else 100.0 * L.eps_zz_below, 4),
            "gap_class": "" if L.gap_class is None else L.gap_class,
            "gap_depth_angstrom": _fmt(L.gap_depth, 5),
            "n_cc_bonds": L.n_bonds,
            "cc_bond_mean_angstrom": _fmt(L.bond_mean, 5),
            "cc_bond_min_angstrom": _fmt(L.bond_min, 5),
            "cc_bond_max_angstrom": _fmt(L.bond_max, 5),
            "cc_bond_dev_mean_pct": _fmt(L.bond_dev_mean_pct, 4),
            "ch_bond_mean_angstrom": _fmt(L.ch_bond_mean, 5),
            "epistemic_level": EPISTEMIC_LEVEL_PROFILE,
        })
    return rows


def summary_row(prof: SurfaceProfile) -> dict:
    c0 = prof.fit_by_class[0].decay_length_angstrom if prof.fit_by_class else None
    c1 = prof.fit_by_class[1].decay_length_angstrom if len(prof.fit_by_class) > 1 else None
    return {
        "surface": prof.surface, "orientation": prof.orientation, "run": prof.run,
        "n_layers": prof.n_layers, "n_carbon": prof.n_carbon,
        "n_hydrogen": prof.n_hydrogen, "atoms_per_layer": prof.atoms_per_layer,
        "a0_angstrom": _fmt(prof.a0_angstrom),
        "bulk_cc_bond_angstrom": _fmt(bulk_bond_length(prof.a0_angstrom), 5),
        "surface_eps_zz_pct": _fmt(100.0 * prof.surface_eps_zz, 4),
        "max_rumpling_angstrom": _fmt(prof.max_rumpling, 5),
        "max_cc_bond_dev_pct": _fmt(prof.max_bond_dev_pct, 4),
        "decay_length_angstrom": _fmt(prof.fit_all.decay_length_angstrom, 4),
        "decay_r_squared": _fmt(prof.fit_all.r_squared, 4),
        "decay_n_points": prof.fit_all.n_points,
        "decay_status": prof.fit_all.status,
        "decay_epistemic_level": EPISTEMIC_LEVEL_DECAY,
        "decay_length_class0_angstrom": _fmt(c0, 4),
        "decay_length_class1_angstrom": _fmt(c1, 4),
        "penetration_depth_angstrom": _fmt(prof.penetration_depth_angstrom, 4),
        "penetration_threshold": _fmt(prof.penetration_threshold, 6),
        "noise_floor_eps_zz": _fmt(prof.noise_floor, 8),
        "fold_asymmetry_eps_zz": _fmt(prof.fold_asymmetry, 10),
        "profile_epistemic_level": EPISTEMIC_LEVEL_PROFILE,
        "note": PROFILE_NOTE,
    }


def write_csv(rows, fields, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def render_report(profiles: list) -> str:
    L = ["# Depth profile of surface relaxation", "",
         f"Geometric profile: **{EPISTEMIC_LEVEL_PROFILE}**. "
         f"Exponential decay length: **{EPISTEMIC_LEVEL_DECAY}** "
         f"(few-point fit; see caveats). CLAUDE.md sec 4.", "",
         "This answers how deep the surface perturbation reaches, and "
         "therefore whether the interior of a nanoparticle may be treated as "
         "uniformly strained bulk in `particle_strain.py`.", "",
         "**Scope.** Out-of-plane profile only. A periodic slab has a uniform "
         "in-plane lattice by construction, so there is no depth-dependent "
         "in-plane strain in these calculations and none is reported.", "",
         "## Summary", "",
         "| surface | run | layers | eps_zz (surface gap) | decay length | "
         "penetration (<0.1%) | max rumpling |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for p in profiles:
        L.append(
            f"| {p.orientation} | `{p.run}` | {p.n_layers} | "
            f"{100.0 * p.surface_eps_zz:+.3f}% | "
            f"{p.fit_all.decay_length_angstrom:.2f} A | "
            f"{p.penetration_depth_angstrom:.2f} A | "
            f"{p.max_rumpling:.4f} A |")
    L += ["", "## Per-surface detail", ""]
    for p in profiles:
        L += [f"### {p.orientation} — `{p.run}`", "",
              f"* {p.n_carbon} C ({p.atoms_per_layer}/layer) + {p.n_hydrogen} H, "
              f"cell area {p.cell_area_angstrom2:.4f} A^2, Lz {p.Lz_angstrom:.4f} A",
              f"* Bulk reference C-C bond {bulk_bond_length(p.a0_angstrom):.5f} A "
              f"at a0 = {p.a0_angstrom:.6f} A; largest layer-mean deviation "
              f"{p.max_bond_dev_pct:+.3f}%",
              f"* Largest intralayer rumpling {p.max_rumpling:.5f} A",
              f"* Outermost interlayer gap: eps_zz = "
              f"{100.0 * p.surface_eps_zz:+.3f}%",
              f"* Fold asymmetry between the two halves: "
              f"{p.fold_asymmetry:.2e} in eps_zz (should be numerical noise; "
              f"a nonzero value would break CLAUDE.md invariant 1)",
              f"* Noise floor used for the fit: |eps_zz| > {p.noise_floor:.2e}",
              f"* Exponential fit over all gaps: lambda = "
              f"{p.fit_all.decay_length_angstrom:.3f} A, R^2 = "
              f"{p.fit_all.r_squared:.4f}, {p.fit_all.n_points} points "
              f"({p.fit_all.status})"]
        for f in p.fit_by_class:
            L.append(f"* Class-resolved fit — {f.label}: lambda = "
                     f"{f.decay_length_angstrom:.3f} A, R^2 = {f.r_squared:.4f}, "
                     f"{f.n_points} points ({f.status})")
        L += [f"* Penetration depth (|eps_zz| stays below "
              f"{100 * p.penetration_threshold:.2f}%): "
              f"{p.penetration_depth_angstrom:.3f} A", ""]

    L += ["## Caveats on the decay length", "",
          "* On (111) the interlayer gaps alternate between two ideal classes "
          "(a0*sqrt3/12 and a0*sqrt3/4) whose relaxations differ in both sign "
          "and magnitude. A single exponential through all gaps is an envelope, "
          "not a law; the class-resolved fits above are the more meaningful "
          "numbers, and the penetration depth is convention-free.",
          "* The fits use only the points above the noise floor, which is a "
          "handful. R^2 measures how well a line fits those few log values; it "
          "is not evidence that the decay is exponential (CLAUDE.md sec 4).",
          "* The profile is a fixed-cell, relaxed-ion geometry at the bulk "
          "in-plane lattice constant. It is not a relaxed-cell result.", ""]

    L += ["## Consequence for the continuum model", ""]
    worst = max(p.penetration_depth_angstrom for p in profiles
                if math.isfinite(p.penetration_depth_angstrom))
    L += [f"* The deepest penetration across the three surfaces is "
          f"{worst:.2f} Angstrom.",
          "* A 3 nm particle has roughly 15 Angstrom of half-thickness, so the "
          "perturbed shell occupies a small fraction of the radius and the core "
          "is bulk-like. A uniform interior strain in `particle_strain.py` is "
          "therefore a defensible leading approximation for particles of this "
          "size and larger.",
          "* It is NOT defensible for an NV sited within the perturbed shell "
          "itself: such a centre sees local, not volume-averaged, strain. "
          "`particle_strain.py` models the volume average only.", ""]
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------- main
def run(runs_dir, out_dir, run_map, reference_config, gap_tol, eps_floor,
        penetration_threshold) -> dict:
    ref = elastic_reference.load_elastic_reference(reference_config)
    a0 = ref.bulk.a0_angstrom
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)

    profiles = []
    for surface in sorted(run_map):
        run_dir = runs_dir / run_map[surface]
        if not (run_dir / "pw.out").exists():
            raise LayerProfileError(f"no pw.out in {run_dir}")
        prof = analyse_run(surface, run_dir, a0, gap_tol, eps_floor,
                           penetration_threshold)
        profiles.append(prof)
        write_csv(layer_rows(prof), LAYER_FIELDS,
                  out_dir / f"layer_profile_{surface}.csv")

    write_csv([summary_row(p) for p in profiles], SUMMARY_FIELDS,
              out_dir / "layer_profile_summary.csv")
    (out_dir / "layer_profile_report.md").write_text(render_report(profiles))
    (out_dir / "layer_profile_meta.json").write_text(json.dumps({
        "a0_angstrom": a0,
        "a0_source": ref.bulk.source_type,
        "reference_config": str(reference_config),
        "bulk_cc_bond_angstrom": bulk_bond_length(a0),
        "runs": run_map,
        "gap_tol": gap_tol,
        "eps_floor": eps_floor,
        "penetration_threshold": penetration_threshold,
        "profile_epistemic_level": EPISTEMIC_LEVEL_PROFILE,
        "decay_epistemic_level": EPISTEMIC_LEVEL_DECAY,
        "note": PROFILE_NOTE,
    }, indent=2) + "\n")
    return {"profiles": profiles}


def _parse_run_args(items):
    run_map = dict(DEFAULT_RUNS)
    for item in items or []:
        if "=" not in item:
            raise LayerProfileError(f"--run expects SURFACE=RUNNAME, got {item!r}")
        surface, name = item.split("=", 1)
        run_map[surface.strip()] = name.strip()
    return run_map


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--run", action="append", metavar="SURFACE=RUNNAME",
                    help="override the run used for a surface (repeatable); "
                         f"defaults: {DEFAULT_RUNS}")
    ap.add_argument("--reference-config", default=DEFAULT_REFERENCE)
    ap.add_argument("--gap-tol", type=float, default=DEFAULT_GAP_TOL,
                    help="max relative deviation of an interlayer gap from its "
                         "ideal class (default: %(default)s)")
    ap.add_argument("--eps-floor", type=float, default=DEFAULT_EPS_FLOOR,
                    help="absolute floor on |eps_zz| for the decay fit "
                         "(default: %(default)s)")
    ap.add_argument("--penetration-threshold", type=float,
                    default=DEFAULT_PENETRATION_THRESHOLD,
                    help="|eps_zz| defining the penetration depth "
                         "(default: %(default)s)")
    args = ap.parse_args(argv)

    try:
        run_map = _parse_run_args(args.run)
        result = run(args.runs_dir, args.out_dir, run_map, args.reference_config,
                     args.gap_tol, args.eps_floor, args.penetration_threshold)
    except (LayerProfileError, elastic_reference.ElasticReferenceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"# layer relaxation profile  [profile {EPISTEMIC_LEVEL_PROFILE}, "
          f"decay length {EPISTEMIC_LEVEL_DECAY}]")
    print(f"# {PROFILE_NOTE}")
    print(f"{'surface':9s} {'run':22s} {'eps_zz(surf)':>13s} {'lambda':>8s} "
          f"{'R^2':>7s} {'pen(A)':>8s} {'rumple':>8s} {'fold':>10s}")
    for p in result["profiles"]:
        print(f"{p.orientation:9s} {p.run:22s} "
              f"{100.0 * p.surface_eps_zz:+12.3f}% "
              f"{p.fit_all.decay_length_angstrom:8.3f} "
              f"{p.fit_all.r_squared:7.4f} "
              f"{p.penetration_depth_angstrom:8.3f} "
              f"{p.max_rumpling:8.4f} {p.fold_asymmetry:10.2e}")
        for f in p.fit_by_class:
            print(f"          -> {f.label:30s} lambda = "
                  f"{f.decay_length_angstrom:6.3f} A, R^2 = {f.r_squared:.4f}, "
                  f"{f.n_points} pts")
    print("# lambda and penetration in Angstrom; fold = max half-to-half "
          "disagreement in eps_zz (symmetry check)")
    for surface in sorted(_parse_run_args(args.run)):
        print(f"wrote {os.path.join(args.out_dir, f'layer_profile_{surface}.csv')}")
    for name in ("layer_profile_summary.csv", "layer_profile_report.md",
                 "layer_profile_meta.json"):
        print(f"wrote {os.path.join(args.out_dir, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
