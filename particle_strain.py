#!/usr/bin/env python3
"""particle_strain.py — facet surface stress -> particle interior strain -> NV ZFS.

This is the continuum stage of the chain in CLAUDE.md sec 3. It takes the
thickness-extrapolated facet surface stresses from `fit_tau_infinity.py` and
turns them into a prediction for the NV zero-field-splitting shifts of a
faceted nanodiamond:

    tau^(facet)  ->  <sigma> in the particle interior  ->  <eps>  ->  Delta D, E

Each arrow is a separate inferential step (CLAUDE.md sec 3) and each is
labelled below.

Epistemic level
---------------
**L1 at best**, and the output files say so themselves. The reasons are
structural, not fixable by running longer:

  * the cubic elastic constants C11/C12/C44 are LITERATURE values, not fitted
    from this project's DFT (config/reference_pbe_sssp.json flags them
    `source_type: literature`);
  * the interior strain is a continuum volume average over an idealised
    polyhedron -- there is no explicit NV defect anywhere in the calculation,
    no depth model, and no relaxation of the particle;
  * the spin-strain couplings are a published DFT parameter set whose axial
    channel is about 1.7x smaller than the experimental value;
  * the facet tau values are for infinite planar surfaces and ignore edges,
    corners, and facet-size effects, which for a 3 nm particle are not
    obviously small.

The defensible outputs are the RANKING across shapes and the CHANNEL
SEPARATION (which shapes produce a D shift with no E, and which produce E at
all), not the absolute MHz values.

Step 1 — facet frames (the part that is easy to get silently wrong)
-------------------------------------------------------------------
tau_xx and tau_yy are components in the SURFACE frame of each slab. To combine
facets they must be rotated into the cubic crystal frame, which requires
knowing which crystallographic direction each surface axis is. That is taken
from `geometry._frame()` -- the same rotation that built the slabs -- via
`nv_spin_strain.slab_frame()`, and it is VERIFIED at run time in two ways
before any number is produced:

  * `_verify_slab_frames()` checks that `nv_spin_strain.slab_frame(face)`
    still equals `geometry._frame(face, a0)`'s rotation matrix, so the two
    copies of the convention cannot drift apart;
  * `verify_production_cells()` reads the CELL_PARAMETERS actually used by the
    production stress SCFs and checks each is a positive integer multiple of
    `geometry._frame`'s A1 and A2. This closes the loop from the tau numbers
    back to crystal directions: it proves the cell whose sigma_xx was measured
    has its x axis along `slab_frame(face)[0]`.

For (100) that verification also reports the supercell multiples (1 x 2), i.e.
the 2x1 doubling is along A2. The surface-frame axes are then x = [110] and
y = [1-10], with the dimer bond along y -- so tau_yy is the along-dimer-bond
component and tau_xx is along the dimer rows. Swapping them would flip the
sign of the off-diagonal tau_xy in the cubic frame, which is where the whole
(100) anisotropy lives (the surface axes sit at 45 degrees to the cubic axes,
so a purely diagonal surface-frame anisotropy is purely off-diagonal in the
crystal frame).

Step 1b — the facet orbit, and why the group is Td and not Oh
-------------------------------------------------------------
A particle carries every symmetry-equivalent copy of each facet, and their
in-plane frames are not independent: the copy at normal g*n carries the
reference facet's structure transported by g. The correct group is **Td**, the
point group of the diamond structure, not the full Oh of the cubic lattice.
This is not pedantry:

  * under Td the transported tau is well defined -- every operation that fixes
    a facet normal leaves that facet's tau unchanged. Under Oh it is not: Oh's
    stabiliser of a [001] normal contains the 4-fold rotation that exchanges
    [110] and [1-10], which would force tau_xx = tau_yy and erase the dimer
    anisotropy. `facet_family()` asserts the well-definedness it relies on.
  * Td automatically produces the physical fact that the (001) and (00-1)
    facets of a diamond particle carry dimer rows at 90 degrees to each other,
    because they terminate on different sublattices.

The eight {111} facets fall into two separate Td orbits (the A and B faces of
a diamond octahedron, which in a real crystal are not equivalent). This module
gives both the same H-terminated tau. That is safe here and not merely assumed:
the Td site symmetry of a [111] normal is C3v, whose 3-fold rotation already
forces the (111) tau to be in-plane isotropic for the well-definedness check to
pass at all -- and an isotropic in-plane tau is inversion-invariant. A
reconstructed, anisotropic (111) surface would be rejected outright rather than
silently averaged.

Step 2 — interior stress
------------------------
    <sigma_ij> = -(1/V) * sum_f A_f * tau_ij^(f)
               = -(3/R) * sum_f x_f * tau_ij^(f),     R := 3V / A_total

with x_f the facet area fractions. Writing it with the effective radius R
removes any need to construct the polyhedron's absolute size and makes the
Laplace limit exact: for isotropic tau on a facet set with sum_f x_f n_f n_f =
I/3, this gives <sigma> = -(2 tau / R) I, i.e. a compressive pressure
2*tau/R. `laplace_check()` runs that validation at import-time cost zero and
it is reported in every run's output.

SIGN CONVENTIONS -- read this before using any number below
-----------------------------------------------------------
Two opposite conventions meet in this module and both are reported explicitly:

  * **Project / QE convention** (CLAUDE.md sec 2): sigma > 0 means the cell is
    COMPRESSED. This is how tau was measured and how `sigma_*_project_gpa` is
    reported.
  * **Continuum mechanics convention**: sigma > 0 means TENSION. The formula
    above, the compliance tensor, and `nv_spin_strain` (whose strain sign is
    "negative strain = compression") all live in this convention.

The module works internally in the mechanics convention and reports both. The
scalar pressure `pressure_gpa` is COMPRESSIVE-POSITIVE, so that it can be
compared directly against the Laplace value 2*tau/R.

Usage
-----
    python3 particle_strain.py --shape octahedron --radius-nm 1.5
    python3 particle_strain.py --shape mixture --fraction 111=0.6 --fraction 100=0.4
    python3 particle_strain.py --shape wulff        # refuses at the H-rich limit
    python3 particle_strain.py --shape wulff --mu-h-offset 1.0

Surface energies and the hydrogen chemical potential
-----------------------------------------------------
The Wulff preset reads `config/surface_energies_h.json`, generated by
`surface_energy.py`. Those gammas are NOT constants: they depend on mu_H, and
at the H-rich limit (mu_H = E(H2)/2) two of the three are negative, so no
equilibrium shape exists and this module refuses to invent one. Lowering mu_H
raises every gamma at a facet-dependent rate N_H/2A; `--mu-h-offset` is that
control, in eV below the H-rich limit. The refusal names the threshold at
which the construction becomes available. Because the rates differ, the
equilibrium habit is a function of mu_H, and so is the sign of the interior
pressure it produces.

Spin-strain coupling sets
-------------------------
Both published parameter sets are emitted by default. They differ by ~1.65x on
the axial channel for identical strain, which is the largest single uncertainty
in the MHz numbers -- larger than the elastic constants, the shape, or the
facet tau values. Every NV output carries a Delta D band spanning them.
Requesting one set with `--params` is possible but warns, because it hides that
band behind a default.
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
import geometry
import nv_spin_strain
import parse_slab

DEFAULT_TAU_CSV = "results/production/tau_infinity.csv"
DEFAULT_RUNS_DIR = "results/production"
DEFAULT_OUT_DIR = "results/production"
DEFAULT_REFERENCE = "config/reference_pbe_sssp.json"
DEFAULT_SURFACE_ENERGIES = "config/surface_energies_h.json"
DEFAULT_RADIUS_NM = 1.5

GPA = 1.0e9
EPISTEMIC_LEVEL = "L1"

FAMILIES = ("100", "110", "111")

# A fully symmetric shape must give E identically zero. This is the numerical
# tolerance on "identically", not a physical threshold.
E_ZERO_TOL_MHZ = 1.0e-9

SIGN_NOTE = (
    "tau in tau_infinity.csv is in the PROJECT convention, where positive tau "
    "is COMPRESSIVE surface stress (the released cell expands; verified "
    "against free 2D vc-relax on all three surfaces, six axes). load_taus() "
    "negates it once to obtain the continuum surface stress f, positive = "
    "TENSILE. sigma_*_project: positive = COMPRESSED (CLAUDE.md sec 2, QE "
    "convention). sigma_*_mech: positive = TENSION (continuum convention, used "
    "internally and by nv_spin_strain). pressure_gpa: positive = compressive.")


class ParticleStrainError(Exception):
    """A load-bearing invariant of the continuum model was violated."""


# ------------------------------------------------------- crystal point group
_TETRAHEDRON = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                        dtype=float)


def oh_operations() -> list:
    """The 48 operations of Oh, as signed permutation matrices."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            ops.append(M)
    return ops


def td_operations() -> list:
    """The 24 operations of Td: those of Oh that map the diamond structure to
    itself, identified as the ones preserving the tetrahedron of bond
    directions around a carbon site."""
    out = []
    for M in oh_operations():
        image = (M @ _TETRAHEDRON.T).T
        if all(any(np.allclose(v, w) for w in _TETRAHEDRON) for v in image):
            out.append(M)
    if len(out) != 24:
        raise ParticleStrainError(
            f"generated {len(out)} tetrahedral operations, expected 24")
    return out


TD = td_operations()
INVERSION = -np.eye(3)


# ------------------------------------------------------- frame verification
def _verify_slab_frames(a0: float, tol: float = 1e-12) -> dict:
    """nv_spin_strain.slab_frame must still equal geometry._frame's rotation.

    Two copies of a convention are two chances for it to drift. This makes the
    drift a loud failure instead of a silent sign error in the crystal frame.
    """
    out = {}
    for face in FAMILIES:
        _A1, _A2, R_geom = geometry._frame(face, a0)
        R_nv = nv_spin_strain.slab_frame(face)
        if not np.allclose(R_geom, R_nv, atol=tol):
            raise ParticleStrainError(
                f"({face}): nv_spin_strain.slab_frame disagrees with "
                f"geometry._frame.\n geometry: {R_geom.tolist()}\n"
                f" nv_spin_strain: {R_nv.tolist()}\n"
                f"The surface-frame axes are then unknown and every crystal-frame "
                f"tau below would be wrong. Refusing to continue.")
        out[face] = R_geom
    return out


_CELL_RE = re.compile(r"~(C\d{3})_(\d+)L_stress_scf$")


def verify_production_cells(runs_dir, a0: float, tol: float = 1e-4) -> dict:
    """Check the production stress-SCF cells against geometry._frame.

    Each production in-plane cell vector must be a positive integer multiple of
    the corresponding `geometry._frame` surface vector. That is what proves the
    sigma_xx measured in those runs is the component along
    `slab_frame(face)[0]`, and hence what licenses rotating tau into the cubic
    frame with that matrix. Returns {face: (n1, n2)} supercell multiples.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise ParticleStrainError(
            f"cannot verify surface frames: no runs directory at {runs_dir}. "
            f"Pass --runs-dir, or --no-frame-check to proceed without the "
            f"check (the output will be marked unverified).")
    found = {}
    for child in sorted(runs_dir.iterdir()):
        m = _CELL_RE.search(child.name)
        if not m or not (child / "pw.in").exists():
            continue
        face = m.group(1)[1:]
        if face in found:
            continue
        cell = parse_slab.parse_pw_in(child / "pw.in")["cell_params_ang"]
        if cell is None:
            continue
        A1, A2, _R = geometry._frame(face, a0)
        mult = []
        for label, obs, ref in (("a", np.array(cell[0]), A1),
                                ("b", np.array(cell[1]), A2)):
            n = float(np.dot(obs, ref) / np.dot(ref, ref))
            if abs(n - round(n)) > tol or round(n) < 1:
                raise ParticleStrainError(
                    f"{child.name}: cell vector {label} = {obs.tolist()} is not "
                    f"a positive integer multiple of geometry._frame's "
                    f"{ref.tolist()} (got {n:.6f}). The surface-frame axes of "
                    f"the production runs are not the ones this module assumes; "
                    f"tau cannot be rotated into the crystal frame.")
            if not np.allclose(obs, round(n) * ref, atol=1e-3):
                raise ParticleStrainError(
                    f"{child.name}: cell vector {label} is parallel to but not "
                    f"equal to {round(n)} x geometry._frame's vector; the "
                    f"lattice constant or frame convention differs.")
            mult.append(int(round(n)))
        found[face] = (mult[0], mult[1], child.name)
    missing = [f for f in FAMILIES if f not in found]
    if missing:
        raise ParticleStrainError(
            f"no production stress SCF found to verify the surface frame of "
            f"{missing}; run fit_tau_infinity.py's ladder first or pass "
            f"--no-frame-check.")
    return found


# ------------------------------------------------------------------- facets
@dataclass
class Facet:
    family: str                 # "100" / "110" / "111"
    normal: np.ndarray          # unit normal in cubic coordinates
    frame: np.ndarray           # rows = surface x, y, z in cubic coordinates
    tau_surface: np.ndarray     # 3x3, surface frame, N/m
    tau_cubic: np.ndarray       # 3x3, cubic frame, N/m
    orbit: str                  # "Td" or "Td*inversion" (the {111} B faces)
    area_fraction: float = 0.0

    @property
    def miller(self) -> str:
        v = self.normal / np.max(np.abs(self.normal))
        ints = np.round(v).astype(int)
        return "".join(f"{i}" if i >= 0 else f"-{abs(i)}" for i in ints)


def facet_family(family: str, tau_xx: float, tau_yy: float,
                 tau_xy: float = 0.0, tol: float = 1e-9) -> list:
    """All symmetry-equivalent facets of one family, with tau in the cubic frame.

    Normals are the full Oh orbit (6 for {100}, 12 for {110}, 8 for {111}).
    In-plane frames are transported by Td, the point group of the diamond
    structure. Td already covers all 6 {100} and all 12 {110} normals; it
    covers only 4 of the 8 {111} normals, and the remaining four (the "B" faces
    of the octahedron) are reached by composing with inversion.

    Reaching the B faces that way assigns them the reference facet's tau, i.e.
    it assumes the two octahedral face types carry the same surface structure.
    That is legitimate here, and provably so rather than by assertion: the Td
    stabiliser of a [111] normal is C3v, which contains a 3-fold rotation, so
    the well-definedness check below already admits only an in-plane ISOTROPIC
    (111) tau -- and an isotropic in-plane tau is invariant under inversion.
    An anisotropic (111) tau (a reconstructed (111)-2x1, say) is rejected by
    that check before it ever reaches the inversion step.
    """
    R0 = nv_spin_strain.slab_frame(family)
    tau_surf = np.array([[tau_xx, tau_xy, 0.0],
                         [tau_xy, tau_yy, 0.0],
                         [0.0, 0.0, 0.0]])

    def orbit_of(ops):
        by_normal = {}
        for g in ops:
            R = R0 @ g.T
            by_normal.setdefault(tuple(np.round(R[2], 6)), []).append(R)
        return by_normal

    # The Td orbit is the primary one: Td is the point group of the diamond
    # structure, so transporting the reference facet by a Td operation carries
    # its actual surface decoration. Inversion is NOT in Td and is NOT a
    # symmetry of that decoration -- applying it to a (100) facet would assert
    # that opposite faces carry parallel dimer rows, when in fact Td makes them
    # perpendicular. It is therefore used only to reach normals Td cannot reach
    # at all, which for diamond is exactly the four "B" faces of the {111}
    # octahedron.
    primary = orbit_of(TD)
    secondary = {k: v for k, v in orbit_of([INVERSION @ g for g in TD]).items()
                 if k not in primary}

    facets = []
    for key, entries, orbit in (
            [(k, primary[k], "Td") for k in sorted(primary)]
            + [(k, secondary[k], "Td*inversion") for k in sorted(secondary)]):
        taus = [R.T @ tau_surf @ R for R in entries]
        # Well-definedness: every operation reaching this normal must give the
        # same cubic-frame tau. If it does not, the surface structure is not
        # invariant under its own site symmetry as transported, and the whole
        # construction is meaningless.
        spread = max(float(np.abs(t - taus[0]).max()) for t in taus)
        if spread > tol:
            raise ParticleStrainError(
                f"({family}) facet {key}: tau in the cubic frame depends on "
                f"which symmetry operation is used to reach it (spread "
                f"{spread:.3e} N/m). The facet's tau is not invariant under "
                f"its own site symmetry, so there is no single well-defined "
                f"tau for this facet and the orbit construction is invalid. "
                f"For ({family}) the site symmetry within Td is "
                f"{ {'100': 'C2v', '110': 'Cs', '111': 'C3v'}.get(family, '?') }; "
                f"note C3v contains a 3-fold rotation, which admits only an "
                f"in-plane ISOTROPIC tau -- a reconstructed, anisotropic (111) "
                f"surface cannot be used with this construction.")
        facets.append(Facet(family=family, normal=np.array(key),
                            frame=entries[0], tau_surface=tau_surf,
                            tau_cubic=taus[0], orbit=orbit))

    expected = {"100": 6, "110": 12, "111": 8}[family]
    if len(facets) != expected:
        raise ParticleStrainError(
            f"({family}): built {len(facets)} distinct facet normals, expected "
            f"{expected}")
    return facets


def build_facets(taus: dict, family_fractions: dict,
                 facet_overrides: dict = None) -> list:
    """Assemble the facet list with area fractions.

    family_fractions: {"111": 0.6, ...}, summing to 1 over the families used.
    Within a family the area is split equally over its symmetry-equivalent
    facets unless facet_overrides names individual Miller indices, which is how
    a deliberately non-equilibrium (symmetry-broken) shape is specified.
    """
    facet_overrides = facet_overrides or {}
    out = []
    for family, frac in sorted(family_fractions.items()):
        if frac < 0:
            raise ParticleStrainError(f"negative area fraction for {family}: {frac}")
        if frac == 0:
            continue
        if family not in taus:
            raise ParticleStrainError(
                f"no tau available for family {family}; have {sorted(taus)}")
        t = taus[family]
        fam = facet_family(family, t["tau_xx"], t["tau_yy"])
        weights = np.array([facet_overrides.get(f.miller, 1.0) for f in fam])
        if weights.sum() <= 0:
            raise ParticleStrainError(
                f"all facet weights for family {family} are zero")
        weights = weights / weights.sum()
        for f, w in zip(fam, weights):
            f.area_fraction = frac * float(w)
            out.append(f)
    total = sum(f.area_fraction for f in out)
    if abs(total - 1.0) > 1e-9:
        raise ParticleStrainError(
            f"facet area fractions sum to {total}, not 1")
    return out


# ---------------------------------------------------------- interior stress
def interior_stress_mech(facets: list, radius_m: float) -> np.ndarray:
    """<sigma_ij> = -(3/R) * sum_f x_f tau_ij^(f), in Pa, TENSION POSITIVE.

    Equivalent to -(1/V) sum_f A_f tau^(f) with R = 3V/A_total; writing it this
    way keeps the Laplace limit exact and needs no absolute particle size.
    """
    if radius_m <= 0:
        raise ParticleStrainError(f"particle radius must be positive, got {radius_m}")
    acc = np.zeros((3, 3))
    for f in facets:
        acc += f.area_fraction * f.tau_cubic
    return -(3.0 / radius_m) * acc


def pressure_gpa(sigma_mech: np.ndarray) -> float:
    """Compressive-positive hydrostatic pressure, GPa."""
    return -float(np.trace(sigma_mech)) / 3.0 / GPA


# ------------------------------------------------- coupling-free observable
def lattice_strain(eps_cubic: np.ndarray) -> float:
    """
    Linear lattice strain, eps_lin = tr(eps)/3. Dimensionless.

    THE COUPLING-INDEPENDENT OBSERVABLE.

    Every MHz number this module produces is (interior strain) x (spin-strain
    coupling), and the couplings are the weakest link in the chain: the two
    published parameter sets differ by ~1.65x on the axial channel alone (see
    COUPLING_NOTE), which is a larger uncertainty than everything else combined.

    The lattice strain is the same interior strain BEFORE that multiplication.
    It therefore carries none of the coupling uncertainty, and it is directly
    measurable: the cubic lattice parameter of a nanodiamond ensemble versus
    particle size is a standard powder-XRD experiment. A size-resolved PXRD
    series measures exactly this quantity, with a 1/R dependence that
    `size_sweep()` emits for direct comparison.

    For a symmetry-complete facet set the interior stress is hydrostatic, so
    eps is isotropic and eps_lin is simply its diagonal entry; tr/3 is used so
    the definition still means something for a deliberately asymmetric shape.
    """
    return float(np.trace(eps_cubic)) / 3.0


def lattice_parameter_angstrom(eps_cubic: np.ndarray, a0_angstrom: float) -> float:
    """Strained cubic lattice parameter, a = a0 (1 + eps_lin), in Angstrom."""
    return a0_angstrom * (1.0 + lattice_strain(eps_cubic))


def isotropic_facet_set(tau_n_per_m: float, n_directions: int = 2000) -> list:
    """A sphere discretised into equal-area facets carrying isotropic tau.

    Used by `laplace_check` and by the test suite. Each direction gets an
    in-plane isotropic tau, so tau_cubic = tau * (I - n n^T).
    """
    facets = []
    # Fibonacci sphere: near-equal solid angle per direction.
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_directions):
        z = 1.0 - 2.0 * (i + 0.5) / n_directions
        r = math.sqrt(max(0.0, 1.0 - z * z))
        th = ga * i
        n = np.array([r * math.cos(th), r * math.sin(th), z])
        n = n / np.linalg.norm(n)
        tau_c = tau_n_per_m * (np.eye(3) - np.outer(n, n))
        facets.append(Facet(family="sphere", normal=n, frame=np.eye(3),
                            tau_surface=np.diag([tau_n_per_m, tau_n_per_m, 0.0]),
                            tau_cubic=tau_c, orbit="none",
                            area_fraction=1.0 / n_directions))
    return facets


def laplace_check(tau_n_per_m: float = 1.0, radius_m: float = 1.5e-9,
                  n_directions: int = 2000) -> dict:
    """REQUIRED VALIDATION: a sphere with isotropic tau must give P = 2*tau/R.

    If this fails, the geometric factors in `interior_stress_mech` are wrong
    and nothing downstream is trustworthy.
    """
    facets = isotropic_facet_set(tau_n_per_m, n_directions)
    sigma = interior_stress_mech(facets, radius_m)
    model = pressure_gpa(sigma)
    laplace = 2.0 * tau_n_per_m / radius_m / GPA
    off_diag = float(np.abs(sigma - np.diag(np.diag(sigma))).max())
    aniso = float(np.abs(np.diag(sigma) - np.mean(np.diag(sigma))).max())
    return {
        "pressure_model_gpa": model,
        "pressure_laplace_gpa": laplace,
        "relative_error": abs(model - laplace) / abs(laplace),
        "max_off_diagonal_pa": off_diag,
        "max_diagonal_anisotropy_pa": aniso,
        "n_directions": n_directions,
        "tau_n_per_m": tau_n_per_m,
        "radius_m": radius_m,
    }


# --------------------------------------------------------------- elasticity
def compliance_strain(sigma: np.ndarray,
                      elastic: nv_spin_strain.ElasticConstants) -> np.ndarray:
    """Solve C_ijkl eps_kl = sigma_ij for the strain tensor.

    Done as a 6x6 solve on the independent components rather than via a
    hand-written Voigt compliance, so there is no factor-of-two convention to
    get wrong. sigma in Pa (tension positive) gives eps dimensionless
    (negative = compression), matching nv_spin_strain's convention.
    """
    C = elastic.tensor() * GPA          # ElasticConstants are in GPa
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    M = np.zeros((6, 6))
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            M[a, b] = C[i, j, k, l] * (2.0 if k != l else 1.0)
    rhs = np.array([sigma[i, j] for (i, j) in pairs])
    sol = np.linalg.solve(M, rhs)
    eps = np.zeros((3, 3))
    for b, (k, l) in enumerate(pairs):
        eps[k, l] = eps[l, k] = sol[b]
    return eps


# -------------------------------------------------------------------- shape
SINGLE_FAMILY_SHAPES = {
    "octahedron": {"111": 1.0},
    "cube": {"100": 1.0},
    "rhombic-dodecahedron": {"110": 1.0},
}


def parse_fraction_spec(items) -> dict:
    """Parse repeated --fraction KEY=VALUE into family fractions + overrides.

    KEY is a family (100 / 110 / 111) or an explicit signed Miller index
    (e.g. 1-11), the latter being a per-facet weight override used to build a
    deliberately symmetry-broken shape.
    """
    families, overrides = {}, {}
    for item in items or []:
        if "=" not in item:
            raise ParticleStrainError(
                f"--fraction expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key, value = key.strip(), float(value)
        if key in FAMILIES:
            families[key] = value
        else:
            overrides[key] = value
    if not families:
        raise ParticleStrainError(
            "--shape mixture needs at least one --fraction FAMILY=VALUE "
            f"with FAMILY in {FAMILIES}")
    total = sum(families.values())
    if total <= 0:
        raise ParticleStrainError("family area fractions must sum to > 0")
    families = {k: v / total for k, v in families.items()}
    return families, overrides


def load_surface_energies(path, delta_mu_h_ev: float = 0.0):
    """Surface energies at a chosen hydrogen chemical potential.

    Reads the config generated by `surface_energy.py`, which stores each gamma
    at the H-rich limit together with its slope dgamma/d(-mu_H) = N_H/2A. The
    surface energies of a hydrogenated surface are NOT single numbers: lowering
    mu_H below the H-rich limit raises every gamma, at a facet-dependent rate,
    so both the sign of each gamma and their ordering are functions of mu_H.

    delta_mu_h_ev is the reduction below the H-rich limit,
    delta_mu = mu_H(H-rich) - mu_H, in eV. It must be >= 0: a positive mu_H
    excursion above E(H2)/2 is the regime where H2 condenses.

    Returns (gammas at that mu_H, config, provenance dict).
    """
    cfg = json.loads(Path(path).read_text())
    se = cfg.get("surface_energies")
    if not se:
        raise ParticleStrainError(f"{path}: no 'surface_energies' block")
    if delta_mu_h_ev < 0:
        raise ParticleStrainError(
            f"delta_mu_H = {delta_mu_h_ev} eV is negative, i.e. hydrogen richer "
            f"than mu_H = E(H2)/2. That is the regime where H2 condenses and is "
            f"not a physical chemical potential for this system.")

    gammas, slopes = {}, {}
    for family, entry in se.items():
        gammas[family] = float(entry["value"])
        slope = entry.get("dgamma_dmu_j_m2_per_ev")
        if slope is None:
            raise ParticleStrainError(
                f"{path}: surface '{family}' has no 'dgamma_dmu_j_m2_per_ev'. "
                f"This config predates the mu_H-dependent treatment; "
                f"regenerate it with `python3 surface_energy.py --write-config`.")
        slopes[family] = float(slope)

    at_mu = {k: gammas[k] + slopes[k] * delta_mu_h_ev for k in gammas}
    available_from = (cfg.get("mu_h_dependence", {})
                      .get("wulff_available_from_delta_mu_ev"))
    provenance = {
        "surface_energy_config": str(path),
        "surface_energy_source_type": cfg.get("source_type", "unknown"),
        "delta_mu_h_ev": delta_mu_h_ev,
        "gamma_h_rich_j_m2": gammas,
        "dgamma_dmu_j_m2_per_ev": slopes,
        "gamma_at_mu_h_j_m2": at_mu,
        "wulff_available_from_delta_mu_ev": available_from,
        "epistemic_level": {k: se[k].get("epistemic_level", "") for k in se},
        "gamma_scatter_j_m2": {k: se[k].get("scatter_j_m2") for k in se},
    }
    return at_mu, cfg, provenance


def wulff_available_from(gammas: dict, slopes: dict):
    """Smallest delta_mu_H at which every gamma is positive, or None."""
    needed = []
    for k, g in gammas.items():
        if g > 0:
            needed.append(0.0)
            continue
        if slopes.get(k, 0.0) <= 0:
            return None
        needed.append(-g / slopes[k])
    return max(needed) if needed else None


def wulff_area_fractions(gammas: dict, offset: float = 0.0,
                         available_from=None) -> dict:
    """Equilibrium (Wulff) facet area fractions from surface energies.

    `gammas` must already be evaluated at the desired mu_H (see
    `load_surface_energies`). `offset` is a bare additive shift retained only
    for testing the geometry; the physical control is mu_H.

    The Wulff shape is the intersection of half-spaces n_f . x <= gamma_f. That
    construction is only defined for gamma > 0: a negative surface energy has
    no distance interpretation and the equilibrium shape is unbounded, because
    the crystal lowers its energy without limit by creating more surface. At
    the H-rich limit the H-terminated diamond gammas are negative, so this
    raises rather than returning a plausible-looking shape -- but unlike a bare
    offset, lowering mu_H is a physical knob that makes the construction valid.
    """
    shifted = {k: v + offset for k, v in gammas.items()}
    bad = {k: v for k, v in shifted.items() if v <= 0}
    if bad:
        hint = ("Lower the hydrogen chemical potential with --mu-h-offset: "
                + (f"every gamma is positive from delta_mu_H >= "
                   f"{available_from:.4f} eV."
                   if available_from is not None else
                   "no delta_mu_H makes them all positive for this data."))
        raise ParticleStrainError(
            "Wulff construction is undefined for non-positive surface energies "
            f"{ {k: round(v, 4) for k, v in bad.items()} } J/m^2. The Wulff "
            "shape places facet f at distance proportional to gamma_f from the "
            "centre; a negative gamma has no such interpretation and the "
            "equilibrium shape is unbounded (more surface always lowers the "
            "energy). Negative gamma is expected for hydrogenated diamond at "
            "the H-rich limit and is not a bug in the energies -- it means the "
            "equilibrium-shape question is not well posed at that hydrogen "
            f"chemical potential. {hint} "
            "Or use --shape octahedron/cube/rhombic-dodecahedron/mixture.")

    planes = []
    for family, gamma in sorted(shifted.items()):
        for f in facet_family(family, 0.0, 0.0):
            planes.append((family, f.normal, gamma))

    verts = []
    for (i, j, k) in itertools.combinations(range(len(planes)), 3):
        A = np.array([planes[i][1], planes[j][1], planes[k][1]])
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        x = np.linalg.solve(A, np.array([planes[i][2], planes[j][2], planes[k][2]]))
        if all(float(np.dot(n, x)) <= h + 1e-9 for _fam, n, h in planes):
            verts.append(x)
    if not verts:
        raise ParticleStrainError("Wulff construction produced no vertices")

    areas = {}
    for family, n, h in planes:
        on = [v for v in verts if abs(float(np.dot(n, v)) - h) < 1e-7]
        on = _dedupe(on)
        if len(on) < 3:
            continue
        areas[family] = areas.get(family, 0.0) + _polygon_area(on, n)
    total = sum(areas.values())
    if total <= 0:
        raise ParticleStrainError("Wulff construction produced zero surface area")
    return {k: v / total for k, v in areas.items() if v / total > 1e-12}


def _dedupe(points, tol: float = 1e-7) -> list:
    out = []
    for p in points:
        if not any(np.linalg.norm(p - q) < tol for q in out):
            out.append(p)
    return out


def _polygon_area(points: list, normal: np.ndarray) -> float:
    """Area of a planar convex polygon given its (unordered) vertices."""
    pts = np.array(points)
    c = pts.mean(axis=0)
    n = normal / np.linalg.norm(normal)
    u = pts[0] - c
    u = u - np.dot(u, n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    ang = np.arctan2((pts - c) @ v, (pts - c) @ u)
    pts = pts[np.argsort(ang)]
    total = 0.0
    for i in range(1, len(pts) - 1):
        total += 0.5 * abs(float(np.dot(n, np.cross(pts[i] - pts[0],
                                                    pts[i + 1] - pts[0]))))
    return total


# ------------------------------------------------------------------ tau I/O
def load_taus(csv_path) -> dict:
    """
    Read tau_xx/tau_yy per surface from fit_tau_infinity.py's output,
    converting from the project sign convention to the continuum one.

    THE SIGN FLIP HAPPENS HERE, ONCE.

    tau_infinity.csv stores tau = sigma * Lz / 2 with sigma in the QE/project
    convention, where positive sigma means the cell is COMPRESSED (CLAUDE.md
    section 2, anchored on bulk diamond at -1% strain giving sigma = +162.86
    kbar). That is MINUS the continuum surface stress f, for which positive
    means TENSILE. Everything downstream of this function — facet_family,
    interior_stress_mech, nv_spin_strain — is written in the continuum
    convention, so the negation belongs at this boundary and nowhere else.

    Measured, not argued. Free 2D cell relaxations (cell_dofree='2Dxy') at 16
    layers, predicted direction from tau versus what the cell actually did:

        surface  axis      tau_project   released cell
        (100)    [110]        +1.032        EXPANDED   (+0.139 %)
        (100)    [1-10]       -5.024        contracted (-0.664 %)
        (110)    [1-10]       +2.144        EXPANDED   (+0.145 %)
        (110)    [001]        +4.397        EXPANDED   (+0.429 %)
        (111)    [1-10]       +0.483        EXPANDED   (+0.052 %)
        (111)    [11-2]       +0.483        EXPANDED   (+0.052 %)

    A surface under tensile stress pulls the cell in, so releasing it must
    shrink. Positive tau_project expands the cell in all six cases, so positive
    tau_project is COMPRESSIVE and f = -tau_project. Six of six; the opposite
    convention is refuted in every case.

    Consequence: a free (111)-H particle expands, its interior goes into
    TENSION, and Delta D shifts DOWN. Magnitudes, anisotropy and rankings are
    unaffected — only the sign.
    """
    path = Path(csv_path)
    if not path.exists():
        raise ParticleStrainError(
            f"no tau summary at {path}. Run `python3 fit_tau_infinity.py` first.")
    taus = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            family = row["surface"][1:]
            taus[family] = {
                # negated: project convention -> continuum f, tension positive
                "tau_xx": -float(row["tau_xx_inf_n_per_m"]),
                "tau_yy": -float(row["tau_yy_inf_n_per_m"]),
                "axis_x": row.get("axis_x", ""),
                "axis_y": row.get("axis_y", ""),
                "epistemic_level": row.get("epistemic_level", ""),
                "layers_used": row.get("layers_used", ""),
                "layers_excluded": row.get("layers_excluded", ""),
            }
    if not taus:
        raise ParticleStrainError(f"{path} contained no rows")
    return taus


# ------------------------------------------------------------------ the run
@dataclass
class ParticleResult:
    shape: str
    radius_nm: float
    family_fractions: dict
    facets: list
    sigma_mech_pa: np.ndarray
    eps_cubic: np.ndarray
    pressure_gpa: float
    nv_rows: list
    e_max_mhz: float
    delta_d_spread_mhz: float
    laplace: dict
    param_sets: list = field(default_factory=list)
    facet_overrides: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    a0_angstrom: float = None

    # -- the coupling-independent observable -------------------------------
    @property
    def lattice_strain(self) -> float:
        """Linear lattice strain tr(eps)/3. No spin-strain coupling involved."""
        return lattice_strain(self.eps_cubic)

    @property
    def lattice_strain_percent(self) -> float:
        return 100.0 * self.lattice_strain

    @property
    def lattice_parameter_angstrom(self):
        if self.a0_angstrom is None:
            return None
        return lattice_parameter_angstrom(self.eps_cubic, self.a0_angstrom)

    @property
    def shape_is_symmetric(self) -> bool:
        """True when every facet of every family carries the same area, which
        is what makes the shape Td-symmetric and forces E = 0 exactly."""
        return not self.facet_overrides

    def rows_for(self, param_set: str) -> list:
        return [r for r in self.nv_rows if r["param_set"] == param_set]

    def delta_d_band_mhz(self, nv_axis: str = None):
        """(min, max) Delta D across ALL spin-strain parameter sets.

        This band, not any single number, is the honest output: the two
        published parameter sets differ by a factor of ~1.65 on the axial
        channel, which is the dominant uncertainty in the final MHz values.
        """
        rows = [r for r in self.nv_rows
                if nv_axis is None or r["nv_axis"] == nv_axis]
        vals = [r["delta_D_mhz"] for r in rows]
        return (min(vals), max(vals))

    def e_band_mhz(self, nv_axis: str = None):
        rows = [r for r in self.nv_rows
                if nv_axis is None or r["nv_axis"] == nv_axis]
        vals = [abs(r["E_mhz"]) for r in rows]
        return (min(vals), max(vals))

    @property
    def coupling_spread_factor(self) -> float:
        """How much the parameter-set choice alone moves Delta D."""
        lo, hi = self.delta_d_band_mhz()
        if min(abs(lo), abs(hi)) < 1e-12:
            return float("inf") if max(abs(lo), abs(hi)) > 1e-12 else 1.0
        return max(abs(lo), abs(hi)) / min(abs(lo), abs(hi))


def evaluate(taus: dict, family_fractions: dict, radius_nm: float,
             elastic, param_sets, facet_overrides: dict = None,
             shape_label: str = "mixture",
             a0_angstrom: float = None) -> ParticleResult:
    """Evaluate the particle for EVERY supplied spin-strain parameter set.

    param_sets is a list of SpinStrainParams (a bare one is accepted and
    wrapped). Emitting all of them is deliberate: the axial couplings of the
    two published sets differ by ~1.65x on identical input, and that factor
    dominates every other uncertainty in the MHz numbers. Reporting one set and
    relegating the other to a flag would hide the largest error bar in the
    result behind a default.
    """
    if isinstance(param_sets, nv_spin_strain.SpinStrainParams):
        param_sets = [param_sets]
    param_sets = list(param_sets)
    if not param_sets:
        raise ParticleStrainError("no spin-strain parameter set supplied")

    facets = build_facets(taus, family_fractions, facet_overrides)
    radius_m = radius_nm * 1.0e-9
    sigma = interior_stress_mech(facets, radius_m)
    eps = compliance_strain(sigma, elastic)

    rows = []
    for params in param_sets:
        for label, frame in nv_spin_strain.nv_frames():
            obs = nv_spin_strain.nv_observables(eps, frame, params)
            eps_nv = frame @ eps @ frame.T
            rows.append({
                "param_set": params.name,
                "param_set_reference": params.reference,
                "nv_axis": label,
                "eps_axial": float(eps_nv[2, 2]),
                "eps_transverse_xx": float(eps_nv[0, 0]),
                "eps_transverse_yy": float(eps_nv[1, 1]),
                "eps_transverse_xy": float(eps_nv[0, 1]),
                **obs,
            })
    e_max = max(abs(r["E_mhz"]) for r in rows)
    dspread = max(
        max(r["delta_D_mhz"] for r in rows if r["param_set"] == p.name)
        - min(r["delta_D_mhz"] for r in rows if r["param_set"] == p.name)
        for p in param_sets)
    return ParticleResult(
        shape=shape_label, radius_nm=radius_nm,
        family_fractions=family_fractions, facets=facets,
        sigma_mech_pa=sigma, eps_cubic=eps, pressure_gpa=pressure_gpa(sigma),
        nv_rows=rows, e_max_mhz=e_max, delta_d_spread_mhz=dspread,
        laplace=laplace_check(), param_sets=[p.name for p in param_sets],
        facet_overrides=dict(facet_overrides or {}),
        a0_angstrom=a0_angstrom,
    )


DEFAULT_SWEEP_RADII_NM = (0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
                          7.5, 10.0, 15.0, 20.0)


CHANNEL_NOTE = (
    "D and E are NOT equally well constrained. delta_D depends only on h41 and "
    "h43, and for the hydrostatic interior stress of a symmetry-complete "
    "particle only through the single combination (2*h41 + h43) -- which is "
    "exactly what a hydrostatic-pressure ODMR measurement determines. E depends "
    "on a DISJOINT pair, h15 and h16, which no hydrostatic measurement "
    "constrains at all; which of the two dominates depends on the strain state "
    "(h15 multiplies the in-plane difference eps_xx - eps_yy and eps_xy, h16 "
    "the shears eps_xz and eps_yz, both in the NV frame), so neither can be "
    "neglected in general. E additionally vanishes identically for any "
    "symmetry-complete facet set, so a non-zero E requires an assumed shape "
    "asymmetry that nothing in this pipeline constrains. E therefore carries "
    "strictly more uncertainty than delta_D: different and less well measured "
    "couplings, times an unconstrained geometric prefactor.")


def channel_coupling_sensitivity(eps_cubic: np.ndarray, frame: np.ndarray,
                                 params, rel_step: float = 0.05) -> dict:
    """
    Which spin-strain couplings each channel actually depends on, measured.

    Perturbs each coupling by `rel_step` and records the fractional response of
    delta_D and of E. Computed rather than asserted, so it stays true if the
    Hamiltonian is ever changed.

    Returns per-coupling {"d_rel": .., "e_rel": ..}, plus "hydrostatic_combo"
    (the value of 2*h41 + h43, the only combination a hydrostatic experiment
    fixes) and "e_is_identically_zero".
    """
    import dataclasses

    base = nv_spin_strain.nv_observables(eps_cubic, frame, params)
    d0, e0 = base["delta_D_mhz"], base["E_mhz"]
    out = {"couplings": {}}
    for name in ("h41", "h43", "h15", "h16", "h25", "h26"):
        pert = dataclasses.replace(params, **{name: getattr(params, name) * (1.0 + rel_step)})
        obs = nv_spin_strain.nv_observables(eps_cubic, frame, pert)
        out["couplings"][name] = {
            "d_rel": (None if abs(d0) < 1e-12
                      else (obs["delta_D_mhz"] - d0) / abs(d0)),
            "e_rel": (None if abs(e0) < 1e-12
                      else (obs["E_mhz"] - e0) / abs(e0)),
        }
    out["hydrostatic_combo_2h41_plus_h43_mhz_per_strain"] = (
        2.0 * params.h41 + params.h43)
    out["e_is_identically_zero"] = abs(e0) < 1e-12
    out["rel_step"] = rel_step
    return out


def facet_lattice_directions(taus: dict, elastic, a0_angstrom: float,
                             radius_nm: float = 1.5) -> list:
    """
    Per-family lattice-strain direction and da/d(1/R) slope.

    The DIRECTION is the most robust thing this module produces. It survives
    two separate uncertainties at once:

      * no spin-strain coupling enters it, so the ~1.65x coupling spread that
        dominates every MHz number is absent;
      * it does not depend on the absolute tau scale either -- multiplying
        every tau by a common positive factor rescales the magnitude and
        leaves the sign untouched. Only the SIGN of each facet's mean f
        matters, and (100) has the opposite sign to (110) and (111).

    So a {100}-dominated particle contracts while {110}- or {111}-dominated
    ones expand, and a size-resolved lattice-parameter measurement reads off
    which facet family dominates from the direction of the shift alone. The
    slope then gives the magnitude for a quantitative comparison.
    """
    rows = []
    for family in sorted(taus):
        res = evaluate(taus, {family: 1.0}, radius_nm, elastic,
                       [nv_spin_strain.UDVARHELYI_DFT],
                       shape_label=family, a0_angstrom=a0_angstrom)
        strain = res.lattice_strain
        rows.append({
            "family": family,
            "f_mean_n_per_m": 0.5 * (taus[family]["tau_xx"]
                                     + taus[family]["tau_yy"]),
            "lattice_strain_percent": 100.0 * strain,
            # everything is exactly 1/R, so one point fixes the slope
            "da_d_inverse_radius_angstrom_nm": a0_angstrom * strain * radius_nm,
            "direction": "expand" if strain > 0 else "contract",
        })
    return rows


def stability_weighted_fractions(gammas: dict, scale_j_m2: float = None) -> dict:
    """
    Facet area fractions ordered by surface stability, from gamma alone.

    WHY NOT A WULFF CONSTRUCTION
    ----------------------------
    Wulff needs positive gamma: it places facet f at a distance proportional to
    gamma_f from the centre, so a negative gamma has no geometric reading and
    the equilibrium shape is unbounded. For H-terminated diamond referenced to
    H2 the energies are negative, and — this is the part that matters — they
    are negative across the WHOLE physically allowed range of the hydrogen
    chemical potential, not merely at one point. dgamma/dmu_H > 0 for all three
    facets, and mu_H <= E(H2)/2 is the physical bound, so gamma is at its
    maximum at the H-rich limit and only becomes more negative going H-poor. At
    that maximum (110) still needs delta_mu_H > +0.233 eV and (111) > +0.337 eV
    to turn positive, both beyond the bound. There is no chemical potential at
    which the equilibrium shape is defined, which is why `--shape wulff`
    refuses rather than offering a default.

    WHAT THIS DOES INSTEAD
    ----------------------
    The energies still fix the stability ORDER, and the order is offset-free
    even though the absolute values are reference-dependent. This uses a
    Boltzmann-like weight

        x_i  ∝  exp(-gamma_i / s)

    which is invariant under gamma_i -> gamma_i + c (a constant shift cancels
    in the normalisation), so it depends only on the gamma DIFFERENCES — the
    part of the energetics that does not depend on the H2 reference. `s`
    defaults to the spread max(gamma) - min(gamma), which sets "one unit of
    stability difference" to the full range actually present.

    THIS IS NOT AN EQUILIBRIUM SHAPE and the output says so. It is a
    transparent, reproducible way to weight the three measured facets by their
    own computed stabilities instead of by taste. The fractions are a modelling
    choice; `s` is a convention. Vary it and the mixture moves, which is why
    the report emits the sensitivity.
    """
    if not gammas:
        raise ParticleStrainError("no surface energies supplied")
    values = list(gammas.values())
    spread = max(values) - min(values)
    if scale_j_m2 is None:
        scale_j_m2 = spread
    if scale_j_m2 <= 0:
        raise ParticleStrainError(
            f"stability scale must be positive, got {scale_j_m2}; all facets "
            "have the same surface energy so no ordering exists")
    weights = {f: math.exp(-g / scale_j_m2) for f, g in gammas.items()}
    total = sum(weights.values())
    return {f: w / total for f, w in sorted(weights.items())}


def size_sweep(taus: dict, family_fractions: dict, radii_nm, elastic,
               param_sets, facet_overrides: dict = None,
               shape_label: str = "mixture", a0_angstrom: float = None) -> list:
    """
    Lattice strain and ZFS shifts versus particle radius.

    Everything here scales as 1/R exactly: the interior stress is
    -(3/R) sum_f x_f tau_f, the strain is linear in it, and the ZFS shifts are
    linear in the strain. So this sweep is not new physics, it is the SHAPE of
    the prediction — a straight line through the origin in 1/R — which is what
    a size-resolved measurement actually tests.

    The point of emitting it is the lattice-strain column: nanodiamond lattice
    parameter versus size is a standard PXRD experiment, and it probes the
    interior strain WITHOUT any spin-strain coupling constant. A measured
    a(1/R) slope either matches the tau-derived prediction or it does not, and
    that comparison is unaffected by the ~1.65x coupling uncertainty that
    dominates every MHz number in this module.

    Returns one row per (radius, param_set); the lattice columns are identical
    across param sets by construction, which is the property being advertised.
    """
    rows = []
    for radius_nm in radii_nm:
        res = evaluate(taus, family_fractions, radius_nm, elastic, param_sets,
                       facet_overrides=facet_overrides,
                       shape_label=shape_label, a0_angstrom=a0_angstrom)
        a_strained = res.lattice_parameter_angstrom
        for params in res.param_sets:
            sub = res.rows_for(params)
            rows.append({
                "shape": shape_label,
                "radius_nm": radius_nm,
                "inverse_radius_per_nm": 1.0 / radius_nm,
                "diameter_nm": 2.0 * radius_nm,
                # --- coupling-independent block -----------------------------
                "lattice_strain_percent": res.lattice_strain_percent,
                "lattice_parameter_angstrom": a_strained,
                "delta_a_angstrom": (None if a_strained is None
                                     else a_strained - res.a0_angstrom),
                "pressure_gpa": res.pressure_gpa,
                # --- coupling-dependent block -------------------------------
                "spin_strain_param_set": params,
                "delta_D_mhz": sub[0]["delta_D_mhz"],
                "delta_D_spread_over_axes_mhz": (
                    max(r["delta_D_mhz"] for r in sub)
                    - min(r["delta_D_mhz"] for r in sub)),
                "E_max_mhz": max(abs(r["E_mhz"]) for r in sub),
            })
    return rows


SWEEP_FIELDS = ["shape", "radius_nm", "diameter_nm", "inverse_radius_per_nm",
                "lattice_strain_percent", "lattice_parameter_angstrom",
                "delta_a_angstrom", "pressure_gpa",
                "spin_strain_param_set", "delta_D_mhz",
                "delta_D_spread_over_axes_mhz", "E_max_mhz",
                "coupling_independent_columns", "epistemic_level"]

COUPLING_FREE_COLUMNS = ("lattice_strain_percent lattice_parameter_angstrom "
                         "delta_a_angstrom pressure_gpa")


def sweep_rows_for_csv(rows) -> list:
    out = []
    for r in rows:
        out.append({
            "shape": r["shape"],
            "radius_nm": _v(r["radius_nm"], 4),
            "diameter_nm": _v(r["diameter_nm"], 4),
            "inverse_radius_per_nm": _v(r["inverse_radius_per_nm"], 6),
            "lattice_strain_percent": _v(r["lattice_strain_percent"], 6),
            "lattice_parameter_angstrom": _v(r["lattice_parameter_angstrom"], 6),
            "delta_a_angstrom": _v(r["delta_a_angstrom"], 6),
            "pressure_gpa": _v(r["pressure_gpa"], 5),
            "spin_strain_param_set": r["spin_strain_param_set"],
            "delta_D_mhz": _v(r["delta_D_mhz"], 4),
            "delta_D_spread_over_axes_mhz": _v(r["delta_D_spread_over_axes_mhz"], 6),
            "E_max_mhz": _v(r["E_max_mhz"], 6),
            "coupling_independent_columns": COUPLING_FREE_COLUMNS,
            "epistemic_level": EPISTEMIC_LEVEL,
        })
    return out


# ================================================================ mu_H scan
# The equilibrium shape is a function of the hydrogen chemical potential, and
# so therefore is the interior stress it produces. Because the (100) facets
# carry a net TENSILE surface stress while (111) carries a compressive one,
# growing the {100} fraction can drive the interior pressure through zero. That
# sign change is the experimentally interesting prediction: it says an ODMR
# shift should INVERT under annealing, which is a much sharper claim than any
# single magnitude in this module.
#
# The scan is reported against (T, p_H2) rather than delta_mu, because delta_mu
# is an abstract axis and "is this reachable?" is the whole question.
# ============================================================================
def surface_engine_resid(surface_energy, root, row):
    """Ideal-gas fit residual converted to kelvin, for one pressure."""
    return surface_energy.temperature_uncertainty_k(root, row["p_h2_pa"])


def scan_mu_h(taus: dict, surface_energy_path, radius_nm: float, elastic,
              param_sets, mu_range=(0.0, 3.0), n_points: int = 121) -> dict:
    """Sweep delta_mu_H, rebuilding the Wulff shape at each step."""
    import surface_energy

    _g, _cfg, prov = load_surface_energies(surface_energy_path, 0.0)
    available_from = prov["wulff_available_from_delta_mu_ev"]
    if available_from is None:
        raise ParticleStrainError(
            "no delta_mu_H makes every gamma positive, so there is no Wulff "
            "shape to scan at any hydrogen chemical potential")

    lo = max(mu_range[0], available_from + 1e-6)
    hi = mu_range[1]
    if hi <= lo:
        raise ParticleStrainError(
            f"scan range ({mu_range[0]}, {mu_range[1]}) eV lies entirely below "
            f"delta_mu = {available_from:.4f} eV, where the Wulff construction "
            f"first exists")

    def evaluate_at(delta_mu):
        gammas, _c, _p = load_surface_energies(surface_energy_path, delta_mu)
        fractions = wulff_area_fractions(gammas)
        return evaluate(taus, fractions, radius_nm, elastic, param_sets,
                        shape_label="wulff"), gammas, fractions

    rows = []
    for i in range(n_points):
        d = lo + (hi - lo) * i / (n_points - 1)
        result, gammas, fractions = evaluate_at(d)
        d_lo, d_hi = result.delta_d_band_mhz()
        rows.append({
            "delta_mu_h_ev": d,
            **{f"gamma_{k}_j_m2": gammas[k] for k in sorted(gammas)},
            **{f"area_fraction_{k}": fractions.get(k, 0.0)
               for k in sorted(gammas)},
            "pressure_gpa": result.pressure_gpa,
            "delta_D_band_lo_mhz": d_lo,
            "delta_D_band_hi_mhz": d_hi,
            "e_max_mhz": result.e_max_mhz,
        })

    # Locate every sign change of the interior pressure and refine it.
    sign_changes = []
    for a, b in zip(rows, rows[1:]):
        pa, pb = a["pressure_gpa"], b["pressure_gpa"]
        if pa == 0.0 or pa * pb >= 0:
            continue
        x_lo, x_hi = a["delta_mu_h_ev"], b["delta_mu_h_ev"]
        f_lo = pa
        for _ in range(60):
            mid = 0.5 * (x_lo + x_hi)
            p_mid = evaluate_at(mid)[0].pressure_gpa
            if p_mid == 0.0:
                x_lo = x_hi = mid
                break
            if (p_mid < 0) == (f_lo < 0):
                x_lo, f_lo = mid, p_mid
            else:
                x_hi = mid
        root = 0.5 * (x_lo + x_hi)
        result, gammas, fractions = evaluate_at(root)
        curve = surface_energy.tp_curve(root)
        sign_changes.append({
            "delta_mu_h_ev": root,
            "direction": ("tension_to_compression" if pa < 0 else
                          "compression_to_tension"),
            "area_fractions": fractions,
            "gammas_j_m2": gammas,
            "pressure_gpa_at_root": result.pressure_gpa,
            "tp_curve": curve,
            # The ideal-gas fit residual: small, and NOT the dominant term.
            "temperature_fit_residual_k": {
                r["p_h2_bar"]: surface_engine_resid(surface_energy, root, r)
                for r in curve},
            # The missing zero-point energy: the dominant systematic, roughly
            # thirty times larger, and one-directional.
            "zpe_systematic": {
                r["p_h2_bar"]: surface_energy.zpe_systematic_on_temperature(
                    root, r["p_h2_pa"])
                for r in curve},
        })

    return {
        "rows": rows, "sign_changes": sign_changes,
        "wulff_available_from_ev": available_from,
        "scan_range_ev": [lo, hi],
        "rrho_validation": surface_energy.rrho_validation(),
    }


SCAN_TP_FIELDS = ["delta_mu_h_ev", "direction", "p_h2_bar", "p_h2_pa",
                  "p_h2_torr", "temperature_k", "temperature_c",
                  "temperature_fit_residual_k",
                  "zpe_delta_ev", "temperature_with_zpe_k",
                  "temperature_with_zpe_c", "zpe_shift_k",
                  "dominant_uncertainty", "reachable",
                  "area_fraction_100", "area_fraction_111",
                  "pressure_gpa_at_root", "what_is_computed",
                  "what_is_ideal_gas", "epistemic_level"]

DOMINANT_UNCERTAINTY = (
    "missing zero-point energy: DELTA_ZPE = ZPE_ads(per H) - ZPE(H2)/2 ~ +0.20 "
    "eV shifts the delta_mu axis rigidly, moving the crossing temperature by "
    "of order 100-200 K (pressure dependent) TOWARDS lower temperature, i.e. "
    "MORE accessible. The ideal-gas fit residual (temperature_fit_residual_k) "
    "is ~30x smaller and is not the limiting term. ZPE is quoted, not applied, "
    "because applying the H2 side alone would be unbalanced; "
    "analyze_h_phonons.py computes the adsorbed side properly.")

WHAT_IS_COMPUTED = ("DFT (this project): E(H2), slab total energies, hence "
                    "gamma(H-rich) and its slope N_H/2A; tau per facet; the "
                    "continuum interior stress and its sign")
WHAT_IS_IDEAL_GAS = ("ideal-gas statistical thermodynamics with literature H2 "
                     "spectroscopic constants: the entire T and p dependence "
                     "(rigid rotor with explicit level sum, harmonic "
                     "oscillator, no ZPE, no anharmonicity)")


def sign_change_tp_rows(scan: dict) -> list:
    rows = []
    for sc in scan["sign_changes"]:
        for r in sc["tp_curve"]:
            z = sc["zpe_systematic"].get(r["p_h2_bar"], {})
            t_zpe = z.get("temperature_with_zpe_k")
            rows.append({
                "delta_mu_h_ev": f"{sc['delta_mu_h_ev']:.6f}",
                "direction": sc["direction"],
                "p_h2_bar": f"{r['p_h2_bar']:.3e}",
                "p_h2_pa": f"{r['p_h2_pa']:.3e}",
                "p_h2_torr": f"{r['p_h2_torr']:.3e}",
                "temperature_k": ("" if r["temperature_k"] is None
                                  else f"{r['temperature_k']:.1f}"),
                "temperature_c": ("" if r["temperature_c"] is None
                                  else f"{r['temperature_c']:.1f}"),
                "temperature_fit_residual_k": _v(
                    sc["temperature_fit_residual_k"].get(r["p_h2_bar"],
                                                         float("nan")), 1),
                "zpe_delta_ev": _v(z.get("delta_zpe_ev", float("nan")), 4),
                "temperature_with_zpe_k": ("" if t_zpe is None
                                           else f"{t_zpe:.1f}"),
                "temperature_with_zpe_c": ("" if t_zpe is None
                                           else f"{t_zpe - 273.15:.1f}"),
                "zpe_shift_k": ("" if z.get("shift_k") is None
                                else f"{z['shift_k']:.1f}"),
                "dominant_uncertainty": DOMINANT_UNCERTAINTY,
                "reachable": r["reachable"],
                "area_fraction_100": f"{sc['area_fractions'].get('100', 0.0):.4f}",
                "area_fraction_111": f"{sc['area_fractions'].get('111', 0.0):.4f}",
                "pressure_gpa_at_root": f"{sc['pressure_gpa_at_root']:.6f}",
                "what_is_computed": WHAT_IS_COMPUTED,
                "what_is_ideal_gas": WHAT_IS_IDEAL_GAS,
                "epistemic_level": EPISTEMIC_LEVEL,
            })
    return rows


def render_scan_report(scan: dict, meta: dict) -> str:
    import surface_energy as surface_energy_module
    L = ["# Interior-pressure sign change versus hydrogen chemical potential",
         "",
         f"**Epistemic level: {EPISTEMIC_LEVEL}.** Continuum model, literature "
         "elastic constants, no explicit NV defect. See "
         "`particle_strain_report_*.md` for the full caveat list.", "",
         "## The prediction", "",
         "The equilibrium (Wulff) habit is a function of the hydrogen chemical "
         "potential. Lowering mu_H raises every surface energy, but at "
         "facet-dependent rates, so the {100} area fraction grows. (100)-H "
         "carries a net TENSILE surface stress (f = +2.0 N/m) while (111)-H "
         "carries a compressive one (f = -0.48 N/m), so growing the {100} "
         "fraction drives the particle interior from tension towards "
         "compression -- and through zero.", "",
         "At the crossing the ODMR shift changes sign: Delta D passes through "
         "zero and reverses. That is a far sharper experimental signature than "
         "any single magnitude this model produces, because it does not depend "
         "on the spin-strain coupling set, on the particle radius, or on the "
         "absolute size of tau -- only on the shape at which the two facet "
         "contributions balance.", ""]

    if not scan["sign_changes"]:
        L += ["## Result: no sign change in the scanned window", "",
              f"The interior pressure does not change sign for delta_mu_H "
              f"between {scan['scan_range_ev'][0]:.3f} and "
              f"{scan['scan_range_ev'][1]:.3f} eV. Either the window is too "
              f"narrow or the effect does not occur; widen it with "
              f"`--scan-mu-h LO HI` before concluding the latter.", ""]
        return "\n".join(L) + "\n"

    L += ["## Where it happens", ""]
    for sc in scan["sign_changes"]:
        L += [f"### delta_mu_H = {sc['delta_mu_h_ev']:.3f} eV "
              f"({sc['direction'].replace('_', ' ')})", "",
              "Equilibrium shape at the crossing: "
              + ", ".join(f"{{{k}}} {v:.3f}"
                          for k, v in sorted(sc["area_fractions"].items())),
              "",
              "| p_H2 (bar) | p_H2 (Torr) | T (K) | T (C) | ZPE-corrected T (C) "
              "| ZPE shift (K) | ideal-gas resid (K) |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in sc["tp_curve"]:
            resid = sc["temperature_fit_residual_k"].get(r["p_h2_bar"],
                                                        float("nan"))
            resid_txt = "" if resid != resid else f"{resid:.0f}"
            z = sc["zpe_systematic"].get(r["p_h2_bar"], {})
            t_zpe = z.get("temperature_with_zpe_k")
            shift = z.get("shift_k")
            zc = "" if t_zpe is None else f"{t_zpe - 273.15:.0f}"
            zs = "" if shift is None else f"{shift:+.0f}"
            if r["temperature_k"] is None:
                L.append(f"| {r['p_h2_bar']:.0e} | {r['p_h2_torr']:.1e} | "
                         f"not reached below 4000 K | | {zc} | {zs} | "
                         f"{resid_txt} |")
            else:
                L.append(f"| {r['p_h2_bar']:.0e} | {r['p_h2_torr']:.1e} | "
                         f"{r['temperature_k']:.0f} | "
                         f"{r['temperature_c']:.0f} | {zc} | {zs} | "
                         f"{resid_txt} |")
        L += ["", "**The ZPE column is the honest number, and the ZPE shift is "
              "the dominant uncertainty** -- see below. The ideal-gas residual "
              "is roughly thirty times smaller and is not the limiting term.",
              ""]

    L += ["## What is computed and what is thermodynamics", "",
          f"* **Computed here (DFT):** {WHAT_IS_COMPUTED}.",
          f"* **Not computed (textbook):** {WHAT_IS_IDEAL_GAS}.", ""]
    val = scan["rrho_validation"]
    dz = surface_energy_module.delta_zpe_ev()
    L += [f"The ideal-gas model reproduces the NIST-JANAF tabulation for H2 to "
          f"{val['max_abs_deviation_ev'] * 1000:.0f} meV over 298-2000 K, which "
          f"is the `ideal-gas resid` column above (a few K). That is NOT the "
          f"dominant uncertainty.", "",
          "## The dominant uncertainty: missing zero-point energy", "",
          f"gamma was built from DFT total energies with no vibrational term on "
          f"either side. Restoring zero-point energy consistently adds "
          f"N_H*ZPE_ads to E_slab and ZPE(H2)/2 to mu_H, which is algebraically "
          f"a rigid shift of the delta_mu axis by",
          "",
          f"    DELTA_ZPE = ZPE_ads(per H) - ZPE(H2)/2 ~ {dz:.3f} eV",
          "",
          f"using literature monohydride C-H frequencies (stretch ~"
          f"{surface_energy_module.ZPE_CH_STRETCH_CM1:.0f} cm^-1, two bends ~"
          f"{surface_energy_module.ZPE_CH_BEND_CM1:.0f} cm^-1) against "
          f"ZPE(H2)/2 = {surface_energy_module.zpe_h2_per_h_ev():.3f} eV. "
          f"Because the C-H zero-point energy is nearly facet-independent, it "
          f"enters through N_H/2A exactly as mu_H does: the SHAPE at a given "
          f"delta_mu is unchanged, but the (T, p) needed to reach it moves.",
          "",
          "DELTA_ZPE is positive, so the correction moves every crossing to "
          "LOWER temperature -- the effect is more accessible than the "
          "uncorrected numbers say, not less. The shift is pressure dependent "
          "(d(delta_mu)/dT carries a (k/2)ln(p0/p) term), which is why the "
          "column above varies with pressure.",
          "",
          "It is QUOTED, NOT APPLIED. Applying the H2 side alone would be an "
          "unbalanced correction, and the adsorbed-H side needs a phonon "
          "calculation. `make_h_phonons.py` generates that campaign and "
          "`analyze_h_phonons.py` turns it into a computed DELTA_ZPE, at which "
          "point this systematic becomes a number rather than an estimate.",
          "", "The DFT surface energies and tau carry their own uncertainties, "
          "not included in any column here; (100)'s gamma alone scatters by "
          "0.024 J/m^2.", "",
          "## What would falsify this, and what could stop it happening", "",
          "* **The H-terminated surface is assumed stable at every mu_H.** Only "
          "H-terminated facets were calculated, so nothing in this model stops "
          "gamma_H rising indefinitely. In reality the surface dehydrogenates "
          "or reconstructs once gamma_H exceeds the bare or reconstructed "
          "surface energy, and that bound CANNOT be computed from this data "
          "set -- it needs bare and reconstructed facet calculations. This is "
          "the largest single caveat, and it bites hardest at exactly the "
          "hydrogen-poor end where the crossing sits.",
          "* **The CH4 / graphite+H2 bound on mu_H is still missing** and is "
          "deliberately not invented here.",
          "* **Kinetics are absent.** Nanodiamond surfaces graphitize on "
          "vacuum annealing. A (T, p) point being thermodynamically reachable "
          "does not mean the H-terminated particle survives the anneal, and "
          "graphitization is not in this picture at all.",
          "* **The particle is assumed to re-equilibrate its shape.** A real "
          "particle whose habit is kinetically frozen will not follow the "
          "Wulff locus, in which case the sign change tracks whatever the "
          "actual facet fractions are rather than the equilibrium ones.",
          "* The (100) surface energy is L1 with 0.024 J/m^2 of scatter "
          "(`surface_energy.py`), and the crossing position depends on it.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- reporting
FACET_FIELDS = ["family", "miller", "orbit", "area_fraction",
                "nx", "ny", "nz", "surface_x_in_cubic", "surface_y_in_cubic",
                "tau_xx_surface_n_per_m", "tau_yy_surface_n_per_m",
                "tau_cubic_xx", "tau_cubic_yy", "tau_cubic_zz",
                "tau_cubic_yz", "tau_cubic_xz", "tau_cubic_xy",
                "epistemic_level"]

NV_FIELDS = ["shape", "radius_nm", "spin_strain_param_set", "nv_axis",
             # coupling-INDEPENDENT: identical across param sets by
             # construction. These are the columns a PXRD measurement tests.
             "lattice_strain_percent", "lattice_parameter_angstrom",
             "eps_axial", "eps_transverse_xx", "eps_transverse_yy",
             "eps_transverse_xy",
             "delta_D_mhz", "delta_D_band_lo_mhz", "delta_D_band_hi_mhz",
             "E_mhz", "E_band_lo_mhz", "E_band_hi_mhz",
             "f_plus_mhz", "f_minus_mhz",
             "pressure_gpa", "epistemic_level", "elastic_source_type",
             "spin_strain_param_sets_emitted", "coupling_spread_factor",
             "spin_strain_param_set_reference", "sign_convention",
             "coupling_uncertainty_note", "channel_constraint_note"]

COUPLING_NOTE = (
    "The spin-strain coupling set is the dominant uncertainty in these MHz "
    "values: the two published sets differ by ~1.65x on the axial channel for "
    "identical strain. delta_D_band_lo/hi span ALL emitted sets for this NV "
    "axis and are the honest quantity; any single delta_D_mhz is one corner of "
    "that band.")


def _v(x, nd=6):
    if isinstance(x, float) and not math.isfinite(x):
        return ""
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def facet_rows(result: ParticleResult) -> list:
    rows = []
    for f in result.facets:
        t = f.tau_cubic
        rows.append({
            "family": f"({f.family})", "miller": f.miller, "orbit": f.orbit,
            "area_fraction": _v(f.area_fraction),
            "nx": _v(float(f.normal[0]), 6), "ny": _v(float(f.normal[1]), 6),
            "nz": _v(float(f.normal[2]), 6),
            "surface_x_in_cubic": " ".join(f"{c:+.4f}" for c in f.frame[0]),
            "surface_y_in_cubic": " ".join(f"{c:+.4f}" for c in f.frame[1]),
            "tau_xx_surface_n_per_m": _v(float(f.tau_surface[0, 0])),
            "tau_yy_surface_n_per_m": _v(float(f.tau_surface[1, 1])),
            "tau_cubic_xx": _v(float(t[0, 0])), "tau_cubic_yy": _v(float(t[1, 1])),
            "tau_cubic_zz": _v(float(t[2, 2])), "tau_cubic_yz": _v(float(t[1, 2])),
            "tau_cubic_xz": _v(float(t[0, 2])), "tau_cubic_xy": _v(float(t[0, 1])),
            "epistemic_level": EPISTEMIC_LEVEL,
        })
    return rows


def nv_rows_for_csv(result: ParticleResult, meta: dict) -> list:
    rows = []
    for r in result.nv_rows:
        d_lo, d_hi = result.delta_d_band_mhz(r["nv_axis"])
        e_lo, e_hi = result.e_band_mhz(r["nv_axis"])
        rows.append({
            "shape": result.shape, "radius_nm": _v(result.radius_nm, 4),
            "spin_strain_param_set": r["param_set"],
            "nv_axis": r["nv_axis"],
            "lattice_strain_percent": _v(result.lattice_strain_percent, 6),
            "lattice_parameter_angstrom": _v(result.lattice_parameter_angstrom, 6),
            "eps_axial": _v(r["eps_axial"], 9),
            "eps_transverse_xx": _v(r["eps_transverse_xx"], 9),
            "eps_transverse_yy": _v(r["eps_transverse_yy"], 9),
            "eps_transverse_xy": _v(r["eps_transverse_xy"], 9),
            "delta_D_mhz": _v(r["delta_D_mhz"], 4),
            "delta_D_band_lo_mhz": _v(d_lo, 4),
            "delta_D_band_hi_mhz": _v(d_hi, 4),
            "E_mhz": _v(r["E_mhz"], 6),
            "E_band_lo_mhz": _v(e_lo, 6),
            "E_band_hi_mhz": _v(e_hi, 6),
            "f_plus_mhz": _v(r["f_plus_mhz"], 3),
            "f_minus_mhz": _v(r["f_minus_mhz"], 3),
            "pressure_gpa": _v(result.pressure_gpa, 5),
            "epistemic_level": EPISTEMIC_LEVEL,
            "elastic_source_type": meta["elastic_source_type"],
            "spin_strain_param_sets_emitted": " ".join(result.param_sets),
            "coupling_spread_factor": _v(result.coupling_spread_factor, 3),
            "spin_strain_param_set_reference": r["param_set_reference"],
            "sign_convention": SIGN_NOTE,
            "coupling_uncertainty_note": COUPLING_NOTE,
            "channel_constraint_note": CHANNEL_NOTE,
        })
    return rows


def write_csv(rows, fields, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def render_report(result: ParticleResult, meta: dict, taus: dict) -> str:
    sig = result.sigma_mech_pa / GPA
    eps = result.eps_cubic
    L = ["# Particle interior strain and NV zero-field splitting", "",
         f"**Epistemic level: {EPISTEMIC_LEVEL}** (CLAUDE.md sec 4). This is a "
         "continuum model with LITERATURE cubic elastic constants "
         f"(C11 = {meta['C11']:.1f}, C12 = {meta['C12']:.1f}, "
         f"C44 = {meta['C44']:.1f} GPa, source "
         f"`{meta['elastic_source_type']}`), no explicit NV defect, no depth "
         "model, and published spin-strain parameter sets "
         f"({', '.join('`' + p + '`' for p in result.param_sets)}). The "
         "defensible outputs are the ranking across shapes and the D/E channel "
         "separation, not the absolute MHz values.", "",
         "## Sign conventions", "", f"{SIGN_NOTE}", "",
         "## Coupling-set uncertainty", "", COUPLING_NOTE, "",
         f"Emitted parameter sets: "
         + "; ".join(f"`{p}`" for p in result.param_sets)
         + f". Ratio of the largest to the smallest |Delta D| across sets: "
           f"**{result.coupling_spread_factor:.2f}x**.", ""]
    if len(result.param_sets) < 2:
        L += ["**WARNING**: only one parameter set was emitted, so the "
              "dominant uncertainty in every MHz value below is invisible in "
              "this report. Rerun without `--params` to get the band.", ""]
    # The coupling-free result goes FIRST, ahead of every MHz number, because
    # it is the only line in this report that a measurement can test without
    # adopting a spin-strain parameter set.
    a_str = result.lattice_parameter_angstrom
    dirs = meta.get("facet_lattice_directions")

    L += ["## The testable prediction: which way the lattice moves", ""]
    if dirs:
        L += ["A {100}-dominated particle **contracts**; {110}- and "
              "{111}-dominated particles **expand**. The direction alone "
              "identifies the dominant facet family.", "",
              "| dominant facet | mean f (N/m) | lattice strain at R=1.5 nm | "
              "direction | da/d(1/R) (A per nm^-1) |",
              "| --- | ---: | ---: | :---: | ---: |"]
        for d in dirs:
            L.append(f"| ({d['family']}) | {d['f_mean_n_per_m']:+.3f} | "
                     f"{d['lattice_strain_percent']:+.4f} % | "
                     f"**{d['direction']}** | "
                     f"{d['da_d_inverse_radius_angstrom_nm']:+.6f} |")
        L += ["",
              "This is the most directly testable output of the whole "
              "pipeline, and it is robust twice over:", "",
              "1. **No spin-strain coupling enters it.** The ~1.65x spread "
              "between published coupling sets, which dominates every MHz "
              "number below, is simply absent here.",
              "2. **The direction does not depend on the absolute tau scale.** "
              "Scaling every tau by a common positive factor changes the "
              "magnitude and not the sign. Only the sign of each family's mean "
              "f matters, and (100) has the opposite sign to (110) and (111).",
              "",
              "Nanodiamond lattice parameter versus particle size is a standard "
              "powder-XRD measurement. A size-resolved series therefore reads "
              "off the dominant facet family from the SIGN of the shift, and "
              "then tests the magnitude against the 1/R slope above. Both "
              "comparisons are independent of the couplings.", ""]

    L += [f"### This shape ({result.shape})", "",
          f"* **Lattice strain = {result.lattice_strain_percent:+.4f} %** "
          f"(linear, tr(eps)/3) at R = {result.radius_nm:.3f} nm "
          f"-> **{'expands' if result.lattice_strain > 0 else 'contracts'}**",
          (f"* **Lattice parameter a = {a_str:.6f} A** vs unstrained "
           f"a0 = {result.a0_angstrom:.6f} A "
           f"(delta_a = {a_str - result.a0_angstrom:+.6f} A)"
           if a_str is not None else "* Lattice parameter: a0 not supplied"),
          f"* Interior pressure = {result.pressure_gpa:+.5f} GPa "
          "(compressive-positive)", "",
          "Every quantity here scales as 1/R exactly; the companion "
          "`..._size_sweep_*.csv` tabulates that curve.", ""]

    L += ["## Inputs", "",
          f"* Shape: **{result.shape}**, facet area fractions "
          + ", ".join(f"({k}) {v:.4f}" for k, v in sorted(result.family_fractions.items())),
          f"* Effective radius R = 3V/A_total = {result.radius_nm:.3f} nm",
          f"* Surface stress from `{meta['tau_csv']}` "
          f"(epistemic level {meta['tau_epistemic_level']}, "
          f"fitted layers {meta['tau_layers_used']}, "
          f"excluded {meta['tau_layers_excluded']})"]
    if meta.get("surface_energy_provenance"):
        sp = meta["surface_energy_provenance"]
        L.append(f"* Surface energies from `{sp['surface_energy_config']}` "
                 f"(source_type `{sp['surface_energy_source_type']}`) evaluated "
                 f"at delta_mu_H = {sp['delta_mu_h_ev']:.4f} eV below the "
                 f"H-rich limit: "
                 + ", ".join(f"({k}) {v:+.4f}"
                             for k, v in sorted(sp['gamma_at_mu_h_j_m2'].items()))
                 + " J/m^2")
    L.append("")
    L += ["| family | tau_xx (N/m) | tau_yy (N/m) | surface x | surface y |",
          "| --- | ---: | ---: | --- | --- |"]
    for fam in sorted(result.family_fractions):
        t = taus[fam]
        L.append(f"| ({fam}) | {t['tau_xx']:+.4f} | {t['tau_yy']:+.4f} | "
                 f"{t['axis_x']} | {t['axis_y']} |")

    sens = meta.get("channel_coupling_sensitivity")
    if sens:
        combo = sens["hydrostatic_combo_2h41_plus_h43_mhz_per_strain"]
        step = 100.0 * sens["rel_step"]
        L += ["", "## D and E are not equally constrained", "",
              "Measured by perturbing each coupling by "
              f"{step:.0f}% and recording the response "
              "(computed here, not asserted):", "",
              "| coupling | delta_D response | E response |",
              "| --- | ---: | ---: |"]
        for name, v in sens["couplings"].items():
            d = "--" if v["d_rel"] is None else f"{100 * v['d_rel']:+.2f} %"
            e = "--" if v["e_rel"] is None else f"{100 * v['e_rel']:+.2f} %"
            L.append(f"| `{name}` | {d} | {e} |")
        L += ["",
              "The two channels use **disjoint** couplings: `h41`/`h43` set "
              "delta_D, `h15`/`h16` set E, and `h25`/`h26` enter only at second "
              "order.", "",
              f"For the hydrostatic interior stress of a symmetry-complete "
              f"particle, delta_D collapses onto the single combination "
              f"**2*h41 + h43 = {combo:.1f} MHz/strain** — which is precisely "
              "what a hydrostatic-pressure ODMR experiment measures. That is "
              "why the axial channel, for all its 1.65x spread between "
              "parameter sets, rests on a directly measured quantity.", "",
              "E does not. `h15`/`h16` are untouched by any hydrostatic "
              "measurement, and E carries a second, larger problem:"]
        if sens["e_is_identically_zero"]:
            L += ["", "> **E is identically zero for this shape.** Every "
                  "symmetry-complete facet set gives a hydrostatic interior "
                  "stress, and a hydrostatic strain produces no transverse "
                  "splitting for *any* values of the couplings. The E reported "
                  "below is zero by symmetry, not by cancellation.", ""]
        else:
            L += ["", "This shape carries a deliberate facet asymmetry, so E "
                  "is non-zero here. Note that its magnitude is proportional "
                  "to that asymmetry, which is an input, not a result.", ""]

        L += ["**Consequence for the (100) anisotropy result.** The (100) "
              "surface stress anisotropy is real and is the largest in the "
              "set, but this model cannot presently turn it into a predicted "
              "E: the volume average it computes has no deviatoric part for a "
              "symmetric particle. The ranking of surfaces by anisotropy is "
              "the defensible output; a predicted E in MHz is not.", "",
              "### Open question: is the volume average the right object for E?",
              "",
              "**Hypothesis, not a result.** The volume-averaged interior "
              "stress this module computes is exact for its TRACE — that is a "
              "consequence of the divergence theorem and needs no assumption "
              "about how stress is distributed inside the particle. The trace "
              "is what delta_D responds to, which is why the axial channel is "
              "on firm ground.",
              "",
              "E responds to the DEVIATORIC part, and the volume average of "
              "the deviatoric stress is not the deviatoric stress anywhere in "
              "particular. Inside a faceted particle the deviatoric field is "
              "not uniform: it varies with position, and near a facet it "
              "reflects that facet's own anisotropic tau rather than the "
              "orientation average. In a 3 nm particle every NV sits within "
              "~1.5 nm of a surface, so no NV samples the average.",
              "",
              "It is therefore possible that solving the elasticity "
              "boundary-value problem for the actual polyhedron and evaluating "
              "the strain at realistic NV depths yields a non-zero E from "
              "computed geometry, with no assumed shape asymmetry anywhere. "
              "That would put E on the same footing as delta_D instead of "
              "resting on a free parameter.",
              "",
              "This is untested. It is recorded here as a question to settle, "
              "not as a claim: the boundary-value problem has not been solved, "
              "the depth dependence has not been computed, and it is not "
              "established that the result is non-zero. Until it is, this "
              "module reports E = 0 for symmetric particles and says why.", ""]

    L += ["", "## Validation: Laplace limit", "",
          "A sphere with isotropic tau must give a compressive pressure "
          "P = 2*tau/R. If it does not, the geometric factors are wrong and "
          "nothing below is trustworthy.", "",
          f"* model {result.laplace['pressure_model_gpa']:.6f} GPa vs Laplace "
          f"{result.laplace['pressure_laplace_gpa']:.6f} GPa "
          f"(relative error {result.laplace['relative_error']:.2e}, "
          f"{result.laplace['n_directions']} directions)",
          f"* residual anisotropy of the sphere result: "
          f"{result.laplace['max_diagonal_anisotropy_pa'] / GPA:.2e} GPa "
          f"diagonal, {result.laplace['max_off_diagonal_pa'] / GPA:.2e} GPa "
          f"off-diagonal", "",
          "## Interior stress and strain", "",
          "Volume-averaged interior stress, continuum convention "
          "(positive = tension), GPa:", "", "```"]
    for r in sig:
        L.append("  " + "  ".join(f"{c:+10.5f}" for c in r))
    L += ["```", "",
          f"Hydrostatic pressure (compressive-positive): "
          f"**{result.pressure_gpa:+.5f} GPa**",
          f"In the project/QE convention (positive = compressed) the same "
          f"tensor is diag("
          + ", ".join(f"{-c:+.5f}" for c in np.diag(sig)) + ") GPa.", "",
          "Interior strain (dimensionless, negative = compression):", "", "```"]
    for r in eps:
        L.append("  " + "  ".join(f"{c:+12.3e}" for c in r))
    L += ["```", "", "## NV zero-field splitting", "",
          "Delta D and E are given as a BAND across the emitted spin-strain "
          "parameter sets. The band, not either endpoint, is the result.", "",
          "| NV axis | eps_axial | Delta D band (MHz) | E band (MHz) |",
          "| --- | ---: | ---: | ---: |"]
    for axis in [a for a in dict.fromkeys(r["nv_axis"] for r in result.nv_rows)]:
        row = next(r for r in result.nv_rows if r["nv_axis"] == axis)
        d_lo, d_hi = result.delta_d_band_mhz(axis)
        e_lo, e_hi = result.e_band_mhz(axis)
        L.append(f"| {axis} | {row['eps_axial']:+.3e} | "
                 f"{d_lo:+.3f} .. {d_hi:+.3f} | {e_lo:.4f} .. {e_hi:.4f} |")
    L += ["", "### Per parameter set", "",
          "| param set | NV axis | Delta D (MHz) | E (MHz) | f+ (MHz) | f- (MHz) |",
          "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in result.nv_rows:
        L.append(f"| {r['param_set']} | {r['nv_axis']} | "
                 f"{r['delta_D_mhz']:+.3f} | {r['E_mhz']:.4f} | "
                 f"{r['f_plus_mhz']:.2f} | {r['f_minus_mhz']:.2f} |")
    d_lo, d_hi = result.delta_d_band_mhz()
    L += ["", f"* Delta D over all axes and all parameter sets: "
          f"**{d_lo:+.3f} .. {d_hi:+.3f} MHz** "
          f"({result.coupling_spread_factor:.2f}x from the coupling set alone)",
          f"* Largest |E| across the four NV orientations and all parameter "
          f"sets: **{result.e_max_mhz:.4f} MHz**",
          f"* Largest spread in Delta D across orientations within one "
          f"parameter set: {result.delta_d_spread_mhz:.4f} MHz", ""]
    for name in result.param_sets:
        ref = next(r["param_set_reference"] for r in result.nv_rows
                   if r["param_set"] == name)
        L.append(f"* `{name}`: {ref}")
    L.append("")

    L += ["## Symmetry check", ""]
    if result.shape_is_symmetric and result.e_max_mhz < E_ZERO_TOL_MHZ:
        L.append(
            "* Every facet of every family carries equal area, so the shape has "
            "the full Td symmetry of the crystal. Summing an anisotropic facet "
            "tau over a full Td orbit gives a Td-invariant tensor, and the only "
            "Td-invariant symmetric rank-2 tensor is isotropic. The interior "
            "stress is therefore purely hydrostatic and **E = 0 exactly** -- as "
            "it must be. Note this holds for ANY number of facet families, not "
            "only one: mixing (100) and (111) does not by itself produce E.")
        L.append(
            "* Nonzero E requires genuinely broken symmetry: unequal areas among "
            "the facets of a family (a non-equilibrium shape, specified here "
            "with a per-facet `--fraction` override), or an NV shallow enough to "
            "see local rather than volume-averaged strain -- which this model "
            "does not treat at all.")
    elif result.shape_is_symmetric:
        L.append(f"* **PROBLEM**: the shape has equal facet areas throughout and "
                 f"must give E = 0 exactly by symmetry, but produced "
                 f"E = {result.e_max_mhz:.6f} MHz. The model has a bug; do not "
                 f"use these numbers.")
    else:
        L.append(f"* Facet areas within a family are unequal (per-facet "
                 f"overrides {sorted(result.facet_overrides)}), so the shape "
                 f"does not have the full crystal symmetry and a nonzero E is "
                 f"allowed. Observed max |E| = {result.e_max_mhz:.4f} MHz.")
    L += ["", "## What this cannot support", "",
          "* No explicit NV defect is present anywhere in this chain. D and E "
          "come from a published spin-strain tensor applied to a "
          "volume-averaged continuum strain.",
          "* The elastic constants are literature values, not fitted from this "
          "project's DFT (CLAUDE.md sec 3). Every number above inherits that.",
          "* Facet tau values are infinite-plane quantities. Edges, corners, "
          "and facet-size effects are not modelled and are not obviously small "
          "for a particle of this size.",
          "* `layer_profile.py` shows the surface perturbation decays within a "
          "few Angstrom, which is what licenses a uniform interior strain for "
          "a particle this size. It does not license applying that average to "
          "a near-surface NV.", ""]
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tau-csv", default=DEFAULT_TAU_CSV)
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--shape", default="octahedron",
                    choices=sorted(list(SINGLE_FAMILY_SHAPES) + ["wulff", "mixture"]))
    ap.add_argument("--fraction", action="append", metavar="KEY=VALUE",
                    help="area fraction for --shape mixture; KEY is a family "
                         "(100/110/111) or a signed Miller index for a "
                         "per-facet weight override")
    ap.add_argument("--radius-nm", type=float, default=DEFAULT_RADIUS_NM,
                    help="effective radius R = 3V/A_total (default: %(default)s)")
    ap.add_argument("--surface-energies", default=DEFAULT_SURFACE_ENERGIES)
    ap.add_argument("--mu-h-offset", type=float, default=0.0, metavar="EV",
                    help="hydrogen chemical potential below the H-rich limit, "
                         "delta_mu_H = mu_H(H-rich) - mu_H in eV. Every gamma "
                         "rises with it at a facet-dependent rate, so this sets "
                         "both the signs and the ordering of the surface "
                         "energies, and hence the Wulff shape. See "
                         "surface_energy.py. (default: %(default)s, the H-rich "
                         "limit, where the Wulff construction does not exist)")
    ap.add_argument("--wulff-gamma-offset", type=float, default=0.0,
                    help="bare additive shift of every gamma (J/m^2), for "
                         "testing the Wulff geometry only. Prefer --mu-h-offset, "
                         "which is the physical control.")
    ap.add_argument("--sweep-radii-nm", type=float, nargs="+",
                    default=list(DEFAULT_SWEEP_RADII_NM),
                    metavar="R",
                    help="particle radii (nm) for the size sweep, which is "
                         "what a size-resolved PXRD series is compared "
                         "against. Everything scales as 1/R.")
    ap.add_argument("--scan-mu-h", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="sweep delta_mu_H over this range (eV), rebuilding the "
                         "Wulff shape at each step, and report where the "
                         "interior pressure changes sign -- as a (T, p_H2) "
                         "curve, not a number in eV. Implies --shape wulff.")
    ap.add_argument("--scan-points", type=int, default=121)
    ap.add_argument("--no-frame-check", action="store_true",
                    help="skip verifying the production cells against "
                         "geometry._frame (output is marked unverified)")
    ap.add_argument("--params", nargs="+", metavar="SET",
                    choices=sorted(nv_spin_strain.PARAM_SETS),
                    default=["dft", "barson"],
                    help="spin-strain coupling sets to emit. BOTH are emitted "
                         "by default because they differ by ~1.65x on the axial "
                         "channel, which is the dominant uncertainty in the MHz "
                         "values; requesting a single set hides it.")
    # Mirrors nv_spin_strain's elastic-provenance CLI so the same resolution
    # and provenance record is reused rather than duplicated.
    ap.add_argument("--reference-config", default=DEFAULT_REFERENCE)
    ap.add_argument("--elastic-source", choices=["config", "legacy"],
                    default="config")
    ap.add_argument("--C11", type=float, default=None, metavar="GPA")
    ap.add_argument("--C12", type=float, default=None, metavar="GPA")
    ap.add_argument("--C44", type=float, default=None, metavar="GPA")
    ap.add_argument("--warning-threshold-pct", type=float,
                    default=elastic_reference.DEFAULT_WARNING_THRESHOLD_PCT)
    args = ap.parse_args(argv)

    try:
        ref = elastic_reference.load_elastic_reference(args.reference_config)
        a0 = ref.bulk.a0_angstrom
        _verify_slab_frames(a0)
        if args.no_frame_check:
            cell_check = {"status": "SKIPPED (--no-frame-check)"}
            print("WARNING: surface frames were NOT verified against the "
                  "production cells; the crystal-frame tau below is "
                  "unverified.", file=sys.stderr)
        else:
            verified = verify_production_cells(args.runs_dir, a0)
            cell_check = {face: {"supercell": [v[0], v[1]], "run": v[2]}
                          for face, v in verified.items()}

        taus = load_taus(args.tau_csv)
        elastic, prov = nv_spin_strain.resolve_elastic_context(args)
        param_sets = [nv_spin_strain.PARAM_SETS[p] for p in args.params]
        # Deduplicate: "dft" and "udvarhelyi2018_dft" are the same object.
        seen, deduped = set(), []
        for p in param_sets:
            if p.name not in seen:
                seen.add(p.name)
                deduped.append(p)
        param_sets = deduped
        if len(param_sets) < 2:
            print("WARNING: only one spin-strain parameter set requested. The "
                  "two published sets differ by ~1.65x on the axial channel, "
                  "and that is the dominant uncertainty in every Delta D "
                  "below; it is now invisible in this output.", file=sys.stderr)

        if args.scan_mu_h:
            import surface_energy
            scan = scan_mu_h(taus, args.surface_energies, args.radius_nm,
                             elastic, param_sets, tuple(args.scan_mu_h),
                             args.scan_points)
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            write_csv([{k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()} for r in scan["rows"]],
                      list(scan["rows"][0].keys()),
                      out_dir / "particle_strain_mu_h_scan.csv")
            tp_rows = sign_change_tp_rows(scan)
            if tp_rows:
                write_csv(tp_rows, SCAN_TP_FIELDS,
                          out_dir / "particle_strain_sign_change_tp.csv")
            (out_dir / "particle_strain_sign_change_report.md").write_text(
                render_scan_report(scan, {}))
            print(f"# interior-pressure sign change vs mu_H   "
                  f"[{EPISTEMIC_LEVEL}]")
            print(f"# Wulff shape exists from delta_mu_H >= "
                  f"{scan['wulff_available_from_ev']:.4f} eV; scanned "
                  f"{scan['scan_range_ev'][0]:.3f}..{scan['scan_range_ev'][1]:.3f} eV")
            print(f"# computed here: {WHAT_IS_COMPUTED}")
            print(f"# NOT computed: {WHAT_IS_IDEAL_GAS}")
            if not scan["sign_changes"]:
                print("no interior-pressure sign change in the scanned window")
            for sc in scan["sign_changes"]:
                print(f"\nsign change at delta_mu_H = "
                      f"{sc['delta_mu_h_ev']:.4f} eV ({sc['direction']}), "
                      f"shape " + ", ".join(f"{{{k}}} {v:.3f}" for k, v
                                            in sorted(sc["area_fractions"].items())))
                print(f"{'p_H2 (bar)':>12s} {'p_H2 (Torr)':>12s} "
                      f"{'T (K)':>9s} {'T (C)':>9s} {'ZPE T (C)':>10s} "
                      f"{'ZPE dT':>8s} {'gasresid':>9s}")
                for r in sc["tp_curve"]:
                    resid = sc["temperature_fit_residual_k"].get(
                        r["p_h2_bar"], float("nan"))
                    z = sc["zpe_systematic"].get(r["p_h2_bar"], {})
                    tz = z.get("temperature_with_zpe_k")
                    sh = z.get("shift_k")
                    zc = "" if tz is None else f"{tz - 273.15:10.0f}"
                    zs = "" if sh is None else f"{sh:+8.0f}"
                    if r["temperature_k"] is None:
                        print(f"{r['p_h2_bar']:12.0e} {r['p_h2_torr']:12.1e} "
                              f"{'>4000':>9s} {'':>9s} {zc:>10s} {zs:>8s} "
                              f"{resid:9.0f}")
                    else:
                        print(f"{r['p_h2_bar']:12.0e} {r['p_h2_torr']:12.1e} "
                              f"{r['temperature_k']:9.0f} "
                              f"{r['temperature_c']:9.0f} {zc:>10s} {zs:>8s} "
                              f"{resid:9.0f}")
                print("  ZPE T = crossing temperature once the missing "
                      "zero-point energy is restored (DOMINANT systematic, "
                      f"DELTA_ZPE ~ {surface_energy.delta_zpe_ev():.3f} eV, "
                      "estimated); gasresid = ideal-gas fit residual, ~30x "
                      "smaller")
            for name in ("particle_strain_mu_h_scan.csv",
                         "particle_strain_sign_change_tp.csv",
                         "particle_strain_sign_change_report.md"):
                if (out_dir / name).exists():
                    print(f"wrote {os.path.join(args.out_dir, name)}")
            return 0

        wulff_note = ""
        overrides = {}
        se_provenance = None
        if args.shape == "wulff":
            gammas, gcfg, se_provenance = load_surface_energies(
                args.surface_energies, args.mu_h_offset)
            fractions = wulff_area_fractions(
                gammas, args.wulff_gamma_offset,
                available_from=se_provenance["wulff_available_from_delta_mu_ev"])
            wulff_note = (
                f"surface energies from {args.surface_energies} "
                f"(source_type {gcfg.get('source_type')}) at delta_mu_H = "
                f"{args.mu_h_offset:.4f} eV below the H-rich limit: "
                + ", ".join(f"({k}) {v:+.4f}" for k, v in sorted(gammas.items()))
                + " J/m^2")
            if args.wulff_gamma_offset:
                wulff_note += (f"; PLUS a bare gamma offset "
                               f"{args.wulff_gamma_offset:+.4f} J/m^2 -- NOT the "
                               f"equilibrium shape")
        elif args.shape == "mixture":
            if args.fraction:
                fractions, overrides = parse_fraction_spec(args.fraction)
            else:
                # No --fraction given: derive the mixture from the surface
                # energies rather than refusing. A Wulff construction is
                # unavailable at every physically allowed mu_H (see
                # stability_weighted_fractions), so this weights the three
                # measured facets by their own computed stabilities using an
                # offset-invariant rule, and labels the result as a modelling
                # choice rather than an equilibrium shape.
                gammas, gcfg, se_provenance = load_surface_energies(
                    args.surface_energies, args.mu_h_offset)
                fractions = stability_weighted_fractions(gammas)
                scale = max(gammas.values()) - min(gammas.values())
                wulff_note = (
                    f"facet fractions DERIVED from {args.surface_energies} "
                    f"(source_type {gcfg.get('source_type')}) at delta_mu_H = "
                    f"{args.mu_h_offset:.4f} eV: "
                    + ", ".join(f"({k}) gamma={v:+.4f}"
                                for k, v in sorted(gammas.items()))
                    + f" J/m^2, stability-weighted x_i ~ exp(-gamma_i/s) with "
                      f"s = {scale:.4f} J/m^2 (the gamma spread). "
                      "Offset-invariant: uses only gamma DIFFERENCES, not the "
                      "H2-referenced absolute values. NOT an equilibrium shape "
                      "-- Wulff is undefined here because gamma < 0 across the "
                      "whole allowed mu_H range.")
        else:
            fractions = dict(SINGLE_FAMILY_SHAPES[args.shape])

        result = evaluate(taus, fractions, args.radius_nm, elastic, param_sets,
                          overrides, shape_label=args.shape, a0_angstrom=a0)
        sweep = size_sweep(taus, fractions, args.sweep_radii_nm, elastic,
                           param_sets, facet_overrides=overrides,
                           shape_label=args.shape, a0_angstrom=a0)
    except (ParticleStrainError, elastic_reference.ElasticReferenceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    tau_meta = next(iter(taus.values()))
    meta = {
        "epistemic_level": EPISTEMIC_LEVEL,
        "shape": args.shape,
        "wulff_note": wulff_note,
        "radius_nm": args.radius_nm,
        "family_fractions": fractions,
        "facet_overrides": overrides,
        "tau_csv": args.tau_csv,
        "tau_epistemic_level": tau_meta["epistemic_level"],
        "tau_layers_used": tau_meta["layers_used"],
        "tau_layers_excluded": tau_meta["layers_excluded"],
        "a0_angstrom": a0,
        "C11": elastic.C11, "C12": elastic.C12, "C44": elastic.C44,
        "param_sets": [p.name for p in param_sets],
        "param_set_references": {p.name: p.reference for p in param_sets},
        "coupling_spread_factor": result.coupling_spread_factor,
        "coupling_uncertainty_note": COUPLING_NOTE,
        "delta_mu_h_ev": args.mu_h_offset,
        "surface_energy_provenance": se_provenance,
        "surface_frame_check": cell_check,
        "laplace_validation": result.laplace,
        "sign_convention": SIGN_NOTE,
        "channel_constraint_note": CHANNEL_NOTE,
        "facet_lattice_directions": facet_lattice_directions(taus, elastic, a0),
        "channel_coupling_sensitivity": channel_coupling_sensitivity(
            result.eps_cubic, nv_spin_strain.nv_frames()[0][1], param_sets[0]),
        "coupling_independent_observable": {
            "quantity": "lattice_strain_percent / lattice_parameter_angstrom",
            "value_percent": result.lattice_strain_percent,
            "lattice_parameter_angstrom": result.lattice_parameter_angstrom,
            "a0_unstrained_angstrom": a0,
            "why": ("interior strain BEFORE any spin-strain coupling is "
                    "applied, so none of the ~1.65x coupling uncertainty "
                    "enters. Directly measurable as the cubic lattice "
                    "parameter vs particle size by powder XRD; see "
                    f"particle_strain_size_sweep_{args.shape}.csv for the 1/R "
                    "curve to compare a size-resolved series against."),
        },
        **{k: prov[k] for k in ("elastic_source_type", "elastic_citation",
                                "elastic_config_path", "cli_override_used")},
    }

    out_dir = Path(args.out_dir)
    tag = args.shape
    write_csv(facet_rows(result), FACET_FIELDS,
              out_dir / f"particle_strain_facets_{tag}.csv")
    write_csv(nv_rows_for_csv(result, meta), NV_FIELDS,
              out_dir / f"particle_strain_nv_{tag}.csv")
    write_csv(sweep_rows_for_csv(sweep), SWEEP_FIELDS,
              out_dir / f"particle_strain_size_sweep_{tag}.csv")
    (out_dir / f"particle_strain_meta_{tag}.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n")
    (out_dir / f"particle_strain_report_{tag}.md").write_text(
        render_report(result, meta, taus))

    print(f"# particle interior strain -> NV ZFS   [{EPISTEMIC_LEVEL}]")
    print(f"# {SIGN_NOTE}")
    print(f"# elastic: {prov['elastic_source_type']} "
          f"(C11={elastic.C11:.1f}, C12={elastic.C12:.1f}, "
          f"C44={elastic.C44:.1f} GPa); spin-strain sets: "
          f"{', '.join(result.param_sets)}")
    print(f"# surface-frame check: {cell_check}")
    print(f"# Laplace validation: model "
          f"{result.laplace['pressure_model_gpa']:.6f} GPa vs 2*tau/R "
          f"{result.laplace['pressure_laplace_gpa']:.6f} GPa "
          f"(rel. err {result.laplace['relative_error']:.2e})")
    print(f"shape = {args.shape}, R = {args.radius_nm:.3f} nm, fractions = "
          + ", ".join(f"({k}) {v:.4f}" for k, v in sorted(fractions.items())))
    if wulff_note:
        print(f"  {wulff_note}")
    print(f"interior pressure (compressive-positive) = "
          f"{result.pressure_gpa:+.5f} GPa")
    print(f"{'param set':22s} {'NV axis':10s} {'eps_axial':>12s} "
          f"{'dD (MHz)':>10s} {'E (MHz)':>10s} {'f+ (MHz)':>10s} "
          f"{'f- (MHz)':>10s}")
    for r in result.nv_rows:
        print(f"{r['param_set']:22s} {r['nv_axis']:10s} "
              f"{r['eps_axial']:12.4e} "
              f"{r['delta_D_mhz']:+10.3f} {r['E_mhz']:10.5f} "
              f"{r['f_plus_mhz']:10.2f} {r['f_minus_mhz']:10.2f}")
    d_lo, d_hi = result.delta_d_band_mhz()
    print(f"Delta D BAND over all axes and coupling sets: "
          f"{d_lo:+.3f} .. {d_hi:+.3f} MHz "
          f"({result.coupling_spread_factor:.2f}x from the coupling set alone)")
    print(f"max |E| = {result.e_max_mhz:.6f} MHz; "
          f"Delta D spread across orientations (within a set) = "
          f"{result.delta_d_spread_mhz:.6f} MHz")
    if result.shape_is_symmetric and result.e_max_mhz > E_ZERO_TOL_MHZ:
        print("ERROR: a shape with equal facet areas throughout must give "
              f"E = 0 exactly by symmetry, but produced {result.e_max_mhz:.6g} "
              "MHz. The model has a bug; the numbers above are not usable.",
              file=sys.stderr)
        return 1
    for name in (f"particle_strain_facets_{tag}.csv",
                 f"particle_strain_nv_{tag}.csv",
                 f"particle_strain_meta_{tag}.json",
                 f"particle_strain_report_{tag}.md"):
        print(f"wrote {os.path.join(args.out_dir, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
