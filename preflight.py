#!/usr/bin/env python3
"""
preflight.py - Structural gate that must pass before any QE job is submitted.

Implements CLAUDE.md section 7. This project has already lost 45 completed DFT
calculations to a silent geometry error: slabs intended as symmetric H/H were
generated with H on the top face and a bare bottom face. Nothing crashed, SCF
converged, BFGS converged, JOB DONE was reached. The structures were simply not
the ones the science claimed.

Every check here reads the geometry. None of them trusts a folder name, an
intent, or the script that generated the structure. Where a name makes a claim,
the claim is checked against the coordinates and a disagreement is a failure.

Exit status is 0 only if every structure passes every check, so this can gate
submission directly:

    python3 preflight.py results/production/thick_a0corr~C100_6L || exit 1

Checks
------
  finite_coordinates        no NaN/inf in any coordinate or cell vector
  vacuum_present            adequate vacuum gap along the slab normal
  no_periodic_overlap       no atom within bonding range of its own image
  bond_lengths              C-C, C-H, C-F within physically reasonable bounds
  carbon_coordination       no undercoordinated carbon on a terminated face
  face_adsorbate_balance    equal adsorbate counts on both faces (if symmetric)
  inversion_symmetry        inversion symmetry holds (if symmetric)
  pseudopotentials_exist    every declared UPF is actually present
  name_matches_geometry     folder-name tokens agree with the coordinates
  dipole_correction         asymmetric slabs carry a dipole correction

A note on counting neighbours in small cells
--------------------------------------------
In a 1x1 surface cell an atom's four tetrahedral neighbours are largely
*periodic images of the same one or two atoms*. Taking one minimum-image
distance per atom pair therefore counts four neighbours as two, and reports
every interior carbon as undercoordinated. Neighbours are counted per
(atom, lattice image) pair for that reason. A plain minimum-image convention is
also invalid for the hexagonal (111) cell, so images are enumerated explicitly
rather than obtained by rounding fractional coordinates.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from pathlib import Path

import numpy as np

import parse_convergence

# ── Physical bounds ───────────────────────────────────────────────────────────
# Bonded if within the cutoff. C-C is 1.545 in bulk diamond and ~1.60 across a
# (100) 2x1 dimer, so 1.85 separates bonded from the 2.53 second shell.
BOND_CUTOFF = {
    frozenset(("C", "C")): 1.85,
    frozenset(("C", "H")): 1.35,
    frozenset(("C", "F")): 1.65,
    frozenset(("C", "O")): 1.60,
    frozenset(("O", "H")): 1.20,
}

# A bonded pair shorter than this is unphysical, not merely strained.
BOND_MIN = {
    frozenset(("C", "C")): 1.30,
    frozenset(("C", "H")): 0.90,
    frozenset(("C", "F")): 1.20,
    frozenset(("C", "O")): 1.15,
    frozenset(("O", "H")): 0.85,
}

ADSORBATES = ("H", "F", "O")
FULL_CARBON_COORDINATION = 4

# Any two atoms closer than this are overlapping, whatever the species.
HARD_MIN_SEPARATION = 0.70

DEFAULT_MIN_VACUUM = 5.0
INVERSION_TOL = 0.10

IMAGE_SHIFTS = [np.array(s, dtype=float)
                for s in itertools.product((-1, 0, 1), repeat=3)]

ORIENTATION_TOKENS = ("C100", "C110", "C111")


class PreflightError(Exception):
    """Structure could not be read well enough to check."""


# ── Structure ─────────────────────────────────────────────────────────────────

class Structure:
    """A QE input read as geometry, not as text."""

    def __init__(self, path, species, positions, cell, pseudos, control, system):
        self.path = Path(path)
        self.name = self.path.name
        self.species = list(species)
        self.positions = np.asarray(positions, dtype=float)
        self.cell = np.asarray(cell, dtype=float)
        self.pseudos = dict(pseudos)
        self.control = dict(control)
        self.system = dict(system)

    # -- derived geometry --------------------------------------------------
    @property
    def n_atoms(self):
        return len(self.species)

    def indices(self, *elements):
        return [i for i, s in enumerate(self.species) if s in elements]

    @property
    def carbon_z(self):
        return self.positions[self.indices("C"), 2]

    @property
    def adsorbate_indices(self):
        return self.indices(*ADSORBATES)

    def slab_span(self):
        """(z_min, z_max) over every atom."""
        z = self.positions[:, 2]
        return float(z.min()), float(z.max())

    def vacuum(self):
        lo, hi = self.slab_span()
        return float(self.cell[2, 2]) - (hi - lo)

    def face_adsorbate_counts(self):
        """
        Adsorbates above the topmost carbon vs below the bottommost carbon.

        Defined against the carbon slab rather than the cell midpoint, because
        the failure this exists to catch is exactly a slab that has adsorbates
        on one carbon face and nothing on the other.
        """
        cz = self.carbon_z
        if len(cz) == 0:
            return 0, 0
        top_c, bot_c = float(cz.max()), float(cz.min())
        top = sum(1 for i in self.adsorbate_indices if self.positions[i, 2] > top_c)
        bot = sum(1 for i in self.adsorbate_indices if self.positions[i, 2] < bot_c)
        return top, bot

    def neighbours(self, index):
        """[(other_index, distance)] over all periodic images within cutoff."""
        out = []
        ri = self.positions[index]
        for j in range(self.n_atoms):
            cutoff = BOND_CUTOFF.get(frozenset((self.species[index], self.species[j])), 0.0)
            if cutoff <= 0:
                continue
            for shift in IMAGE_SHIFTS:
                d = float(np.linalg.norm(self.positions[j] + shift @ self.cell - ri))
                if 1e-6 < d <= cutoff:
                    out.append((j, d))
        return out

    def coordination(self, index):
        return len(self.neighbours(index))

    def claims_symmetric(self):
        return "_sym" in self.name

    def is_geometrically_symmetric(self, tol=INVERSION_TOL):
        """
        Are the two faces equivalent, i.e. is there no net dipole along z?

        This deliberately does NOT require an inversion centre. A slab's faces
        can be related by a mirror or glide with no inversion centre present —
        diamond (111) at some layer counts is exactly that — and such a slab
        still has no net dipole and needs no dipole correction. What matters is
        that the z-profile of every species is symmetric about the slab
        midplane, which is what distinguishes H/H from H/bare.
        """
        top, bot = self.face_adsorbate_counts()
        if top != bot:
            return False
        return self.z_profile_mismatch(tol) <= tol

    def z_profile_mismatch(self, tol=INVERSION_TOL):
        """
        Worst deviation from mirror symmetry of the per-species z-profile.

        For each species the sorted z values must map onto themselves under
        z -> (z_min + z_max) - z, taken over all atoms.
        """
        z_all = self.positions[:, 2]
        midpoint = 0.5 * (float(z_all.min()) + float(z_all.max()))
        worst = 0.0
        for element in sorted(set(self.species)):
            zs = sorted(float(self.positions[i, 2]) for i in self.indices(element))
            mirrored = sorted(2.0 * midpoint - z for z in zs)
            if len(zs) != len(mirrored):
                return math.inf
            for a, b in zip(zs, mirrored):
                worst = max(worst, abs(a - b))
        return worst

    def find_inversion_centre(self, tol=INVERSION_TOL):
        """
        An inversion centre for the structure, or None.

        The centre is NOT assumed to be the atomic centroid: for the (100) 2x1
        and (110) cells the centroid is not an inversion centre even though the
        slab is genuinely symmetric, so a centroid-only test rejects correct
        structures. If a centre exists, atom 0 must map onto some same-species
        atom j (in some lattice image), so the centre must be the midpoint of
        that pair. Enumerating those midpoints is therefore a complete search.
        """
        r0 = self.positions[0]
        for j in range(self.n_atoms):
            if self.species[j] != self.species[0]:
                continue
            for shift in IMAGE_SHIFTS:
                centre = 0.5 * (r0 + self.positions[j] + shift @ self.cell)
                if self.inversion_mismatch(centre, tol)[0] == 0:
                    return centre
        return None

    def inversion_mismatch(self, centre, tol=INVERSION_TOL):
        """
        (n_unmatched, worst_distance) for inversion through `centre`.

        For each atom, look for an atom of the same species at 2*centre - r,
        allowing any lattice image.
        """
        centre = np.asarray(centre, dtype=float)
        worst = 0.0
        unmatched = 0
        for i in range(self.n_atoms):
            target = 2.0 * centre - self.positions[i]
            best = math.inf
            for j in range(self.n_atoms):
                if self.species[j] != self.species[i]:
                    continue
                for shift in IMAGE_SHIFTS:
                    d = float(np.linalg.norm(self.positions[j] + shift @ self.cell - target))
                    best = min(best, d)
                    if best <= tol:
                        break
                if best <= tol:
                    break
            if best > tol:
                unmatched += 1
            worst = max(worst, best if best < math.inf else 0.0)
        return unmatched, worst


def _parse_namelist(text, name):
    m = re.search(rf"&{name}(.*?)^\s*/", text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        line = line.split("!")[0].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().lower()] = v.strip().rstrip(",").strip().strip("'\"")
    return out


def load_structure(directory, filename="pw.in"):
    """Read a QE input into a Structure. Refuses to guess units."""
    directory = Path(directory)
    path = directory / filename
    if not path.exists():
        raise PreflightError(f"{directory}: no {filename}")
    text = path.read_text()
    lines = text.splitlines()

    cell = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            if "angstrom" not in line.lower():
                raise PreflightError(
                    f"{directory}: CELL_PARAMETERS unit in {line.strip()!r} is not "
                    "angstrom; refusing to guess")
            cell = [[float(v) for v in lines[i + k].split()[:3]] for k in (1, 2, 3)]
            break
    if cell is None:
        raise PreflightError(f"{directory}: no CELL_PARAMETERS block")

    species, positions = [], []
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("ATOMIC_POSITIONS"):
            if "angstrom" not in line.lower():
                raise PreflightError(
                    f"{directory}: ATOMIC_POSITIONS unit in {line.strip()!r} is "
                    "not angstrom; refusing to guess")
            for row in lines[i + 1:]:
                parts = row.split()
                if len(parts) < 4 or row.strip().startswith("!"):
                    break
                try:
                    xyz = [float(v) for v in parts[1:4]]
                except ValueError:
                    break
                species.append(parts[0])
                positions.append(xyz)
            break
    if not species:
        raise PreflightError(f"{directory}: no ATOMIC_POSITIONS block")

    pseudos = {}
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("ATOMIC_SPECIES"):
            for row in lines[i + 1:]:
                parts = row.split()
                if len(parts) != 3 or row.strip().startswith("!"):
                    break
                pseudos[parts[0]] = parts[2]
            break

    return Structure(directory, species, positions, cell, pseudos,
                     _parse_namelist(text, "CONTROL"),
                     _parse_namelist(text, "SYSTEM"))


# ── Checks ────────────────────────────────────────────────────────────────────
#
# Each returns a list of violation strings. Empty means the check passed. Every
# message names the structure and the specific violation, so a failing gate
# tells you what to fix without opening the file.

def check_finite_coordinates(s):
    bad = []
    for i, r in enumerate(s.positions):
        if not np.all(np.isfinite(r)):
            bad.append(f"{s.name}: atom {i} ({s.species[i]}) has non-finite "
                       f"coordinate {tuple(r)}")
    if not np.all(np.isfinite(s.cell)):
        bad.append(f"{s.name}: cell contains a non-finite component")
    return bad


def check_vacuum_present(s, min_vacuum=DEFAULT_MIN_VACUUM):
    vac = s.vacuum()
    if not np.isfinite(vac):
        return [f"{s.name}: vacuum gap is not finite"]
    if vac < min_vacuum:
        lo, hi = s.slab_span()
        return [f"{s.name}: vacuum gap {vac:.2f} A is below the {min_vacuum:.2f} A "
                f"minimum (cell height {s.cell[2, 2]:.2f} A, atoms span "
                f"{lo:.2f}-{hi:.2f} A)"]
    return []


def check_no_periodic_overlap(s):
    """No atom within bonding range of its OWN image, and nothing overlapping."""
    bad = []
    for i in range(s.n_atoms):
        for shift in IMAGE_SHIFTS:
            if not shift.any():
                continue
            d = float(np.linalg.norm(shift @ s.cell))
            if d < HARD_MIN_SEPARATION:
                bad.append(f"{s.name}: lattice vector combination {tuple(int(v) for v in shift)} "
                           f"has length {d:.3f} A; the cell is degenerate")
                break
    for i in range(s.n_atoms):
        for j in range(i, s.n_atoms):
            for shift in IMAGE_SHIFTS:
                if i == j and not shift.any():
                    continue
                d = float(np.linalg.norm(s.positions[j] + shift @ s.cell - s.positions[i]))
                if d < HARD_MIN_SEPARATION:
                    what = "its own periodic image" if i == j else f"atom {j} ({s.species[j]})"
                    bad.append(f"{s.name}: atom {i} ({s.species[i]}) is {d:.3f} A from "
                               f"{what}, below the {HARD_MIN_SEPARATION:.2f} A overlap threshold")
                    break
    return bad


def check_bond_lengths(s):
    bad = []
    for i in range(s.n_atoms):
        for j, d in s.neighbours(i):
            if j < i:
                continue
            pair = frozenset((s.species[i], s.species[j]))
            lo = BOND_MIN.get(pair)
            if lo is not None and d < lo:
                a, b = sorted(pair) if len(pair) == 2 else (s.species[i], s.species[i])
                bad.append(f"{s.name}: {a}-{b} bond between atoms {i} and {j} is "
                           f"{d:.3f} A, below the {lo:.2f} A physical minimum")
    return bad


def check_carbon_coordination(s):
    """
    Every carbon must be fully coordinated.

    An interior carbon reaches 4 through C neighbours; a carbon on a terminated
    face reaches 4 as 3 C plus 1 adsorbate. A carbon on a *bare* face cannot,
    which is precisely the 45-calculation error: H on top, nothing underneath.
    """
    bad = []
    for i in s.indices("C"):
        n = s.coordination(i)
        if n < FULL_CARBON_COORDINATION:
            z = s.positions[i, 2]
            cz = s.carbon_z
            face = "bottom" if abs(z - cz.min()) < abs(z - cz.max()) else "top"
            bad.append(f"{s.name}: carbon atom {i} on the {face} face at z={z:.3f} A "
                       f"is {n}-coordinated, not {FULL_CARBON_COORDINATION}; that face "
                       f"carries dangling bonds")
    return bad


def check_face_adsorbate_balance(s):
    if not s.claims_symmetric():
        return []
    top, bot = s.face_adsorbate_counts()
    if top != bot:
        return [f"{s.name}: name claims symmetry (_sym) but the faces carry "
                f"different adsorbate counts: {top} above the top carbon, "
                f"{bot} below the bottom carbon"]
    return []


def check_inversion_symmetry(s, tol=INVERSION_TOL):
    """
    CLAUDE.md invariant 1: a *_sym structure must have inversion symmetry.

    Only applied where the name makes the claim ("where required"). The centre
    is searched for rather than assumed, and the fallback message reports the
    z-profile mirror deviation too, since a slab that is face-symmetric but has
    no inversion centre is a different (and milder) situation from one whose
    faces genuinely differ.
    """
    if not s.claims_symmetric():
        return []
    if s.find_inversion_centre(tol) is not None:
        return []
    mirror = s.z_profile_mismatch(tol)
    detail = (f"its z-profile is mirror-symmetric to {mirror:.3f} A, so the faces "
              f"match but no inversion centre exists"
              if mirror <= tol else
              f"and its z-profile is not mirror-symmetric either "
              f"(worst deviation {mirror:.3f} A)")
    return [f"{s.name}: name claims symmetry (_sym) but no inversion centre "
            f"exists to {tol:.2f} A tolerance; {detail}"]


def check_pseudopotentials_exist(s, extra_dirs=()):
    """
    Every UPF declared in ATOMIC_SPECIES must actually be readable.

    Searched: the run directory, its `pseudo_dir` (as QE would resolve it), and
    any --pseudo-dir given. The repo deliberately does not commit UPF files, so
    an *archived* results directory will legitimately fail this check; point
    --pseudo-dir at the staging directory, or --skip it when validating history
    rather than gating a submission.
    """
    bad = []
    pseudo_dir = s.control.get("pseudo_dir", "./")
    for element, filename in sorted(s.pseudos.items()):
        searched = [s.path / filename,
                    s.path / pseudo_dir / filename,
                    Path(pseudo_dir) / filename]
        searched += [Path(d) / filename for d in extra_dirs]
        if not any(c.exists() for c in searched):
            where = ", ".join(str(c.parent) for c in searched)
            bad.append(f"{s.name}: pseudopotential {filename!r} declared for {element} "
                       f"is not present (looked in {where})")
    return bad


def check_name_matches_geometry(s):
    """Folder-name tokens must describe the geometry actually present."""
    bad = []
    name = s.name

    declared = [t for t in ORIENTATION_TOKENS if t in name]
    if len(declared) > 1:
        bad.append(f"{s.name}: name declares more than one orientation {declared}")

    m = re.search(r"(\d+)L(?:\b|_)", name)
    if m:
        want = int(m.group(1))
        got = _count_carbon_layers(s)
        if got != want:
            bad.append(f"{s.name}: name claims {want} carbon layers but the "
                       f"coordinates contain {got}")

    for element in ADSORBATES:
        token = re.search(rf"_{element}_", name) or re.search(rf"_{element}$", name)
        if token and element not in s.species:
            bad.append(f"{s.name}: name claims {element} termination but no {element} "
                       f"atom is present")
    return bad


def _count_carbon_layers(s):
    """
    Number of distinct carbon layers, delegated to parse_convergence.

    DELIBERATELY NOT REIMPLEMENTED HERE. This gate previously carried its own
    copy that clustered on a fixed 0.25 A tolerance, which is not sufficient
    and cannot be made sufficient: (100)'s 0.292 A intra-layer buckling is
    wider than (111)'s 0.488 A inter-layer split is narrow, so no single
    distance classifies both. The consequence was 16 false failures across
    results/slabs/, every one reporting a 6-layer (100) slab as 8 layers.

    parse_convergence.count_layers was fixed by adding a counting invariant on
    top of the clustering (every layer of a slab holds the same number of
    symmetry-equivalent sites, so unequal cluster populations prove the
    clustering split a layer). That fix did not reach this file, which is the
    whole reason the two drifted. Importing it means they cannot drift again:
    there is one implementation and one set of tests for it.

    A gate that emits false failures on good structures is worse than no gate,
    because the failures get waived and then a real one is waived with them.
    """
    return parse_convergence.count_layers(
        [float(v) for v in s.carbon_z]) or 0


DIPOLE_KEYS = ("dipfield", "tefield", "assume_isolated")


def check_dipole_correction(s):
    """
    An asymmetric slab has a net dipole and needs the correction (invariant 4).

    Asymmetry is decided from the coordinates, never from the name, so a slab
    that is asymmetric while claiming to be symmetric is still caught.
    """
    if s.is_geometrically_symmetric():
        return []
    active = []
    for key in DIPOLE_KEYS:
        val = s.control.get(key) or s.system.get(key)
        if val is None:
            continue
        if key == "assume_isolated":
            if val.lower() not in ("none", ""):
                active.append(f"{key}={val}")
        elif val.lower().strip(".") in ("true", "t"):
            active.append(f"{key}={val}")
    if active:
        return []
    top, bot = s.face_adsorbate_counts()
    return [f"{s.name}: structure is geometrically asymmetric "
            f"({top} adsorbate(s) on the top face, {bot} on the bottom) but no "
            f"dipole correction is active; expected one of {', '.join(DIPOLE_KEYS)} "
            f"in &CONTROL or &SYSTEM"]


CHECKS = (
    ("finite_coordinates", check_finite_coordinates),
    ("vacuum_present", check_vacuum_present),
    ("no_periodic_overlap", check_no_periodic_overlap),
    ("bond_lengths", check_bond_lengths),
    ("carbon_coordination", check_carbon_coordination),
    ("face_adsorbate_balance", check_face_adsorbate_balance),
    ("inversion_symmetry", check_inversion_symmetry),
    ("pseudopotentials_exist", check_pseudopotentials_exist),
    ("name_matches_geometry", check_name_matches_geometry),
    ("dipole_correction", check_dipole_correction),
)


def run_checks(structure, min_vacuum=DEFAULT_MIN_VACUUM, skip=(), pseudo_dirs=()):
    """Return {check_name: [violations]} for one structure."""
    results = {}
    for name, fn in CHECKS:
        if name in skip:
            continue
        if name == "vacuum_present":
            results[name] = fn(structure, min_vacuum=min_vacuum)
        elif name == "pseudopotentials_exist":
            results[name] = fn(structure, extra_dirs=pseudo_dirs)
        else:
            results[name] = fn(structure)
    return results


def check_directory(directory, min_vacuum=DEFAULT_MIN_VACUUM, skip=(),
                    filename="pw.in", pseudo_dirs=()):
    """(passed, {check: [violations]}) for one run directory."""
    structure = load_structure(directory, filename=filename)
    results = run_checks(structure, min_vacuum=min_vacuum, skip=skip,
                         pseudo_dirs=pseudo_dirs)
    return not any(results.values()), results


# ── CLI ───────────────────────────────────────────────────────────────────────

def iter_targets(paths, filename="pw.in"):
    """Each path is a run directory, or a parent whose children are run dirs."""
    for p in paths:
        p = Path(p)
        if (p / filename).exists():
            yield p
        elif p.is_dir():
            children = sorted(c for c in p.iterdir() if (c / filename).exists())
            if not children:
                raise PreflightError(f"{p}: no {filename} here or in any subdirectory")
            yield from children
        else:
            raise PreflightError(f"{p}: not a directory")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Structural pre-flight gate (CLAUDE.md section 7). "
                    "Exits nonzero if any structure fails any check.")
    ap.add_argument("paths", nargs="+",
                    help="run directories, or parents of run directories")
    ap.add_argument("--min-vacuum", type=float, default=DEFAULT_MIN_VACUUM,
                    help=f"minimum vacuum gap in Angstrom (default {DEFAULT_MIN_VACUUM})")
    ap.add_argument("--skip", action="append", default=[],
                    help="skip a named check (repeatable)")
    ap.add_argument("--pseudo-dir", action="append", default=[],
                    help="extra directory to search for declared UPF files "
                         "(repeatable). The repo does not commit UPFs, so "
                         "archived results directories need this or "
                         "--skip pseudopotentials_exist.")
    ap.add_argument("--input-name", default="pw.in")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="print only failures")
    args = ap.parse_args(argv)

    unknown = set(args.skip) - {n for n, _ in CHECKS}
    if unknown:
        print(f"unknown check(s) to skip: {sorted(unknown)}", file=sys.stderr)
        return 2

    try:
        targets = list(iter_targets(args.paths, args.input_name))
    except PreflightError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    n_pass = 0
    failures = []
    for directory in targets:
        try:
            passed, results = check_directory(
                directory, min_vacuum=args.min_vacuum, skip=set(args.skip),
                filename=args.input_name, pseudo_dirs=args.pseudo_dir)
        except PreflightError as exc:
            failures.append(str(directory))
            print(f"FAIL {directory}\n  unreadable: {exc}")
            continue
        if passed:
            n_pass += 1
            if not args.quiet:
                print(f"PASS {directory}")
        else:
            failures.append(str(directory))
            print(f"FAIL {directory}")
            for check, violations in results.items():
                for v in violations:
                    print(f"  [{check}] {v}")

    print(f"\n{n_pass}/{len(targets)} structures passed.")
    if failures:
        print(f"{len(failures)} FAILED. Do not submit:")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
