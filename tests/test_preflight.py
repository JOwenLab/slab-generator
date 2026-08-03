"""
Tests for the structural pre-flight gate (preflight.py, CLAUDE.md section 7).

The gate's whole purpose is to fail on structures that look fine. So the tests
that matter most are the two end-to-end ones: it must reject every archived
asymmetric slab (H on top, bare bottom — the 45-calculation error) and accept
every production slab. Everything else here pins a specific check, usually by
taking a known-good structure and breaking exactly one thing about it.
"""

import re
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import preflight

ARCHIVE = REPO_ROOT / "results" / "archive_asymmetric_6L"
PRODUCTION = REPO_ROOT / "results" / "production"
GOOD = PRODUCTION / "thick_a0corr~C111_6L"          # small, symmetric, H/H
GOOD_100 = PRODUCTION / "thick_a0corr~C100_6L"

# UPFs are deliberately not committed (CLAUDE.md), so structure-only tests skip
# the pseudopotential existence check rather than staging binaries.
NO_PSEUDO = {"pseudopotentials_exist"}


def _archive_dirs():
    if not ARCHIVE.is_dir():
        return []
    return sorted(d for d in ARCHIVE.iterdir() if (d / "pw.in").exists())


def _production_dirs():
    if not PRODUCTION.is_dir():
        return []
    return sorted(d for d in PRODUCTION.glob("thick_a0corr~*") if (d / "pw.in").exists())


# ── Acceptance: the two ends of the gate ─────────────────────────────────────

@pytest.mark.skipif(not _archive_dirs(), reason="archive fixture not present")
@pytest.mark.parametrize("directory", _archive_dirs(), ids=lambda p: p.name)
def test_archive_asymmetric_slabs_are_rejected(directory):
    """
    Every archived slab is known-bad by construction: adsorbate on one face,
    bare carbon on the other, and no dipole correction. If any of these passes,
    the gate is broken regardless of how clean the code looks.
    """
    passed, results = preflight.check_directory(directory, skip=NO_PSEUDO)
    assert not passed, f"{directory.name} passed the gate but is known-bad"

    violated = {k for k, v in results.items() if v}
    assert "carbon_coordination" in violated, (
        f"{directory.name}: the bare face's dangling bonds were not detected")
    assert "dipole_correction" in violated, (
        f"{directory.name}: asymmetry without a dipole correction was not detected")


@pytest.mark.skipif(not _production_dirs(), reason="production fixture not present")
@pytest.mark.parametrize("directory", _production_dirs(), ids=lambda p: p.name)
def test_production_slabs_pass(directory):
    passed, results = preflight.check_directory(directory, skip=NO_PSEUDO)
    assert passed, (
        f"{directory.name} failed the gate: "
        + "; ".join(v for vs in results.values() for v in vs))


ARCHIVE_ROOTS = ["C100_2x1_H_6L", "C110_1x1_H_6L_SSSP", "C111_1x1_H_6L"]


@pytest.mark.skipif(not _archive_dirs(), reason="archive fixture not present")
@pytest.mark.parametrize("root", ARCHIVE_ROOTS)
def test_sym_claim_on_real_archive_geometry_fails_on_face_count(tmp_path, root):
    """
    Positive evidence for the branch that would have caught the original error.

    No archived directory actually carries a `_sym` token, so the acceptance
    sweep never exercises `face_adsorbate_balance` — it fails those structures
    on coordination and dipole correction instead. That leaves the one check
    aimed squarely at "symmetric slab that isn't" untested against real data.

    Here a genuine archive structure (H on top, bare bottom) is renamed to claim
    symmetry, which is exactly the 45-calculation error as it appeared on disk:
    a folder asserting H/H containing H/bare. The gate must fail it *on the face
    count*, naming both faces.
    """
    src = ARCHIVE / root
    if not (src / "pw.in").exists():
        pytest.skip(f"{root} not present")

    dest = tmp_path / f"{root}_sym"
    dest.mkdir()
    shutil.copy2(src / "pw.in", dest / "pw.in")

    s = preflight.load_structure(dest)
    assert s.claims_symmetric(), "fixture does not claim symmetry"

    top, bot = s.face_adsorbate_counts()
    assert top > 0 and bot == 0, (
        f"{root}: expected adsorbates on one face only, got {top}/{bot}")

    violations = preflight.check_face_adsorbate_balance(s)
    assert violations, (
        f"{root}_sym: the face-count check did NOT fire on a structure claiming "
        f"symmetry with {top} adsorbate(s) on top and {bot} on the bottom. This "
        f"is the branch that would have caught the original 45-run error.")
    assert "different adsorbate counts" in violations[0]
    assert f"{top} above the top carbon" in violations[0]
    assert f"{bot} below the bottom carbon" in violations[0]

    # and the gate as a whole rejects it
    passed, results = preflight.check_directory(dest, skip=NO_PSEUDO)
    assert not passed
    assert results["face_adsorbate_balance"], "face-count violation lost by the runner"


@pytest.mark.skipif(not _archive_dirs() or not GOOD.exists(),
                    reason="fixtures not present")
def test_face_count_check_is_silent_on_a_genuinely_symmetric_sym_slab(tmp_path):
    """The same check must not fire on a real H/H slab renamed to claim _sym."""
    dest = tmp_path / "thick_a0corr~C111_6L_sym"
    dest.mkdir()
    shutil.copy2(GOOD / "pw.in", dest / "pw.in")

    s = preflight.load_structure(dest)
    assert s.claims_symmetric()
    assert s.face_adsorbate_counts() == (1, 1)
    assert preflight.check_face_adsorbate_balance(s) == []


@pytest.mark.skipif(not _production_dirs(), reason="production fixture not present")
def test_cli_exit_codes():
    """Nonzero exit is what makes this usable as a submission gate."""
    assert preflight.main([str(GOOD), "--skip", "pseudopotentials_exist", "-q"]) == 0
    bad = _archive_dirs()[0]
    assert preflight.main([str(bad), "--skip", "pseudopotentials_exist", "-q"]) == 1
    # an unknown --skip must not silently disable nothing and pass
    assert preflight.main([str(GOOD), "--skip", "no_such_check"]) == 2


# ── Geometry primitives that earlier drafts got wrong ────────────────────────

@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_interior_carbon_is_four_coordinate_in_a_1x1_cell():
    """
    In a 1x1 surface cell an atom's four neighbours are largely periodic images
    of the same one or two atoms. Counting one minimum-image distance per atom
    pair reports interior carbon as 2-coordinate and fails every good slab.
    """
    s = preflight.load_structure(GOOD)
    carbons = s.indices("C")
    interior = [i for i in carbons
                if s.carbon_z.min() < s.positions[i, 2] < s.carbon_z.max()]
    assert interior, "fixture has no interior carbon to test"
    for i in interior:
        assert s.coordination(i) == 4, (
            f"interior carbon {i} counted as {s.coordination(i)}-coordinate")


@pytest.mark.skipif(not GOOD_100.exists(), reason="production fixture not present")
def test_inversion_centre_is_searched_not_assumed_to_be_the_centroid():
    """
    The (100) 2x1 and (110) slabs are genuinely symmetric but their atomic
    centroid is not an inversion centre. Testing only the centroid rejects
    correct structures.
    """
    s = preflight.load_structure(GOOD_100)
    centroid = s.positions.mean(axis=0)
    assert s.inversion_mismatch(centroid)[0] > 0, "fixture no longer exercises this"
    # a real centre exists elsewhere, or the faces are at least mirror-related
    assert (s.find_inversion_centre() is not None
            or s.z_profile_mismatch() <= preflight.INVERSION_TOL)


@pytest.mark.skipif(not _production_dirs() or not _archive_dirs(),
                    reason="fixtures not present")
def test_face_symmetry_is_decided_by_the_z_profile_not_by_inversion():
    """
    The dipole check asks whether the two faces are equivalent, i.e. whether
    there is a net dipole along z. That is the z-profile mirror test, not an
    inversion test: a slab whose faces are related by a mirror or glide with no
    inversion centre still has no dipole, and demanding inversion there would
    require a dipole correction for a correct structure.
    """
    good = preflight.load_structure(PRODUCTION / "thick_a0corr~C111_16L")
    assert good.z_profile_mismatch() <= preflight.INVERSION_TOL
    assert good.is_geometrically_symmetric()
    assert preflight.check_dipole_correction(good) == []

    bare = preflight.load_structure(ARCHIVE / "C111_1x1_H_6L")
    assert bare.z_profile_mismatch() > 1.0, "bare-bottom slab looks mirror-symmetric"
    assert not bare.is_geometrically_symmetric()
    assert preflight.check_dipole_correction(bare)


# ── One broken thing at a time ───────────────────────────────────────────────

def _copy(tmp_path, src=GOOD, name=None):
    dest = tmp_path / (name or src.name)
    dest.mkdir(parents=True)
    shutil.copy2(src / "pw.in", dest / "pw.in")
    return dest


def _edit(directory, fn):
    p = directory / "pw.in"
    p.write_text(fn(p.read_text()))
    return directory


def _violations(directory, check, **kw):
    s = preflight.load_structure(directory)
    return dict(preflight.CHECKS)[check](s, **kw)


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_nan_coordinate_is_caught(tmp_path):
    def nan_first_atom(text):
        lines = text.splitlines()
        i = next(j for j, l in enumerate(lines)
                 if l.strip().upper().startswith("ATOMIC_POSITIONS"))
        p = lines[i + 1].split()
        lines[i + 1] = f"  {p[0]}  {p[1]}  {p[2]}  NaN"
        return "\n".join(lines) + "\n"

    d = _edit(_copy(tmp_path), nan_first_atom)
    out = _violations(d, "finite_coordinates")
    assert out and "non-finite" in out[0]


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_insufficient_vacuum_is_caught(tmp_path):
    d = _copy(tmp_path)
    s = preflight.load_structure(d)
    assert preflight.check_vacuum_present(s, min_vacuum=1.0) == []
    out = preflight.check_vacuum_present(s, min_vacuum=99.0)
    assert out and "below the" in out[0]


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_overlapping_atoms_are_caught(tmp_path):
    """Move one H onto a carbon."""
    def clobber(text):
        lines = text.splitlines()
        i = next(j for j, l in enumerate(lines)
                 if l.strip().upper().startswith("ATOMIC_POSITIONS"))
        carbon = lines[i + 1].split()
        for j in range(i + 1, len(lines)):
            if lines[j].split() and lines[j].split()[0] == "H":
                lines[j] = f"H  {carbon[1]}  {carbon[2]}  {carbon[3]}"
                break
        return "\n".join(lines) + "\n"

    d = _edit(_copy(tmp_path), clobber)
    out = _violations(d, "no_periodic_overlap")
    assert out and "overlap threshold" in out[0]


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_short_bond_is_caught(tmp_path):
    """Pull an H to 0.5 A above its carbon: bonded, but unphysically short."""
    def shorten(text):
        lines = text.splitlines()
        i = next(j for j, l in enumerate(lines)
                 if l.strip().upper().startswith("ATOMIC_POSITIONS"))
        for j in range(i + 1, len(lines)):
            p = lines[j].split()
            if p and p[0] == "C":
                cz = float(p[3])
                cx, cy = p[1], p[2]
                break
        for j in range(i + 1, len(lines)):
            p = lines[j].split()
            if p and p[0] == "H":
                lines[j] = f"H  {cx}  {cy}  {cz + 0.5:.10f}"
                break
        return "\n".join(lines) + "\n"

    d = _edit(_copy(tmp_path), shorten)
    out = _violations(d, "bond_lengths")
    assert out and "below the" in out[0]


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_sym_name_with_one_bare_face_is_caught(tmp_path):
    """
    The exact 45-calculation error: a structure named *_sym with the bottom
    adsorbate deleted. Must fail on the face balance, not merely on symmetry.
    """
    def strip_bottom_h(text):
        lines = text.splitlines()
        i = next(j for j, l in enumerate(lines)
                 if l.strip().upper().startswith("ATOMIC_POSITIONS"))
        body, lowest, lowest_j = [], None, None
        for j in range(i + 1, len(lines)):
            p = lines[j].split()
            if len(p) < 4:
                break
            if p[0] == "H" and (lowest is None or float(p[3]) < lowest):
                lowest, lowest_j = float(p[3]), j
        del lines[lowest_j]
        return re.sub(r"nat\s*=\s*\d+",
                      lambda m: f"nat      = {int(m.group(0).split('=')[1]) - 1}",
                      "\n".join(lines) + "\n")

    d = _edit(_copy(tmp_path, name="C111_1x1_H_6L_sym"), strip_bottom_h)
    s = preflight.load_structure(d)

    assert s.claims_symmetric()
    assert preflight.check_face_adsorbate_balance(s), "bare face not detected"
    assert preflight.check_carbon_coordination(s), "dangling bonds not detected"
    assert preflight.check_dipole_correction(s), "missing dipole correction not detected"
    passed, _ = preflight.check_directory(d, skip=NO_PSEUDO)
    assert not passed


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_dipole_correction_satisfies_the_asymmetric_case(tmp_path):
    """An asymmetric slab that DOES declare a dipole correction must not fail."""
    def strip_and_correct(text):
        lines = text.splitlines()
        i = next(j for j, l in enumerate(lines)
                 if l.strip().upper().startswith("ATOMIC_POSITIONS"))
        lowest, lowest_j = None, None
        for j in range(i + 1, len(lines)):
            p = lines[j].split()
            if len(p) < 4:
                break
            if p[0] == "H" and (lowest is None or float(p[3]) < lowest):
                lowest, lowest_j = float(p[3]), j
        del lines[lowest_j]
        text = "\n".join(lines) + "\n"
        return text.replace("&CONTROL", "&CONTROL\n  dipfield = .true.", 1)

    d = _edit(_copy(tmp_path), strip_and_correct)
    s = preflight.load_structure(d)
    assert not s.is_geometrically_symmetric()
    assert preflight.check_dipole_correction(s) == []


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_missing_pseudopotential_is_caught(tmp_path):
    d = _copy(tmp_path)
    s = preflight.load_structure(d)
    out = preflight.check_pseudopotentials_exist(s)
    assert out and "not present" in out[0]
    # and is satisfied once the file is staged
    (d / "C.pbe-n-kjpaw_psl.1.0.0.UPF").write_text("stub")
    (d / "H_ONCV_PBE-1.0.oncvpsp.upf").write_text("stub")
    assert preflight.check_pseudopotentials_exist(preflight.load_structure(d)) == []


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_layer_count_in_name_must_match_geometry(tmp_path):
    """Invariant 3: a folder called 12L must contain twelve carbon layers."""
    d = _copy(tmp_path, name="thick_a0corr~C111_12L")   # geometry is really 6L
    out = _violations(d, "name_matches_geometry")
    assert out and "claims 12 carbon layers" in out[0]


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_layer_counting_resolves_the_111_bilayer(tmp_path):
    """
    The (111) bilayer halves are ~0.49 A apart and the (100) dimer buckles by
    ~0.08 A. A tolerance that merges the bilayer halves reports every (111)
    slab as half its true thickness.
    """
    s = preflight.load_structure(GOOD)
    assert preflight._count_carbon_layers(s) == 6
    s16 = preflight.load_structure(PRODUCTION / "thick_a0corr~C111_16L")
    assert preflight._count_carbon_layers(s16) == 16


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_crystal_coordinates_are_converted_not_refused(tmp_path):
    """
    A crystal block must give the SAME geometry as the angstrom one it was
    converted from. The gate used to refuse crystal outright, which left all 15
    vcrelax~* runs -- the evidence for the tau sign correction -- unchecked.
    """
    import numpy as np

    ref = preflight.load_structure(GOOD)
    inv = np.linalg.inv(ref.cell)

    def to_crystal(text):
        lines = text.splitlines()
        i = next(j for j, l in enumerate(lines)
                 if l.strip().upper().startswith("ATOMIC_POSITIONS"))
        out = lines[:i] + ["ATOMIC_POSITIONS crystal"]
        for k, (sp, r) in enumerate(zip(ref.species, ref.positions)):
            f = r @ inv
            out.append(f"  {sp}  {f[0]:.12f}  {f[1]:.12f}  {f[2]:.12f}")
        out += lines[i + 1 + len(ref.species):]
        return "\n".join(out) + "\n"

    got = preflight.load_structure(_edit(_copy(tmp_path), to_crystal))
    assert got.species == ref.species
    assert np.allclose(got.positions, ref.positions, atol=1e-9)
    # and the physics survives: coordination is what a wrong transpose breaks
    for i in got.indices("C"):
        assert got.coordination(i) == ref.coordination(i)


def test_bohr_coordinates_are_converted(tmp_path):
    import numpy as np
    ref = preflight.load_structure(GOOD)

    def to_bohr(text):
        lines = text.splitlines()
        i = next(j for j, l in enumerate(lines)
                 if l.strip().upper().startswith("ATOMIC_POSITIONS"))
        out = lines[:i] + ["ATOMIC_POSITIONS bohr"]
        for sp, r in zip(ref.species, ref.positions):
            b = r / preflight.BOHR_TO_ANGSTROM
            out.append(f"  {sp}  {b[0]:.12f}  {b[1]:.12f}  {b[2]:.12f}")
        out += lines[i + 1 + len(ref.species):]
        return "\n".join(out) + "\n"

    got = preflight.load_structure(_edit(_copy(tmp_path), to_bohr))
    assert np.allclose(got.positions, ref.positions, atol=1e-9)


def test_crystal_conversion_uses_rows_not_columns():
    """
    Regression guard on the transpose. r = f . A with A's rows the lattice
    vectors. Using columns instead would not raise -- it would silently shear
    any non-orthogonal cell -- so this checks against a hexagonal one, where
    the two differ.
    """
    import numpy as np
    cell = np.array([[2.52, 0.0, 0.0],
                     [1.26, 2.18, 0.0],
                     [0.0, 0.0, 20.0]])
    frac = [[0.5, 0.5, 0.25]]
    got = preflight._to_cartesian_angstrom(frac, "crystal", cell)
    expected = np.array(frac) @ cell           # rows
    wrong = np.array(frac) @ cell.T            # columns
    assert np.allclose(got, expected)
    assert not np.allclose(expected, wrong), "fixture no longer distinguishes them"


@pytest.mark.skipif(not GOOD.exists(), reason="production fixture not present")
def test_alat_and_unknown_units_are_still_refused(tmp_path):
    """
    Converting what can be converted must not turn into guessing at what
    cannot. alat needs celldm(1), which these ibrav=0 inputs do not carry.
    """
    for unit in ("alat", "crystal_sg", "furlongs"):
        d = _edit(_copy(tmp_path, name=f"u_{unit}"),
                  lambda t, u=unit: t.replace("ATOMIC_POSITIONS angstrom",
                                              f"ATOMIC_POSITIONS {u}"))
        with pytest.raises(preflight.PreflightError, match="Refusing to guess"):
            preflight.load_structure(d)


@pytest.mark.skipif(not (PRODUCTION / "vcrelax~C111_16L" / "pw.in").exists(),
                    reason="vc-relax fixture not present")
@pytest.mark.parametrize("name", ["vcrelax~C100_16L", "vcrelax~C110_16L",
                                  "vcrelax~C111_16L"])
def test_the_vcrelax_evidence_is_inside_the_gate(name):
    """
    These are the runs that settled the tau sign convention. They must be
    readable AND pass, not merely be skipped.
    """
    d = PRODUCTION / name
    s = preflight.load_structure(d)
    assert all(s.coordination(i) == 4 for i in s.indices("C")), \
        "conversion produced a geometry with dangling bonds"
    passed, results = preflight.check_directory(d, skip=NO_PSEUDO)
    assert passed, "; ".join(v for vs in results.values() for v in vs)
