"""
Tests for make_slab_stress_scf.py.

The two defects covered here are both silent-wrong-answer defects: neither
crashed, and both produced QE inputs that looked entirely ordinary.

  (a) The cell was always copied from the source pw.in. Correct for a
      fixed-cell relax, but for a vc-relax it paired the relaxed positions with
      the unrelaxed cell.
  (b) A hardcoded list of three run names from the 6L campaign meant the script
      silently operated on the wrong set, or failed, on any later campaign.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import make_slab_stress_scf as mss

# Cell in the *input*: the pre-relaxation guess.
INPUT_CELL_Z = "14.500000000"
# Cell in the *output*: where a vc-relax actually finished.
RELAXED_CELL_Z = "39.575908610"


def pw_in_text(calculation, cell_z=INPUT_CELL_Z):
    return f"""! generated from results/test/src
&CONTROL
  calculation   = '{calculation}'
  prefix        = 'slab'
  pseudo_dir    = './'
/

&SYSTEM
  ibrav    = 0
  nat      = 2
  ntyp     = 2
  ecutwfc  = 90.0
/

&ELECTRONS
  conv_thr    = 1.0d-8
/

ATOMIC_SPECIES
  C   12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF
  H    1.008  H_ONCV_PBE-1.0.oncvpsp.upf

CELL_PARAMETERS angstrom
   2.527050962   0.000000000   0.000000000
   1.263525476   2.188490333   0.000000000
   0.000000000   0.000000000  {cell_z}

ATOMIC_POSITIONS (angstrom)
C                0.0000000000       0.0000000000       5.0000000000
H                0.0000000000       0.0000000000       6.1000000000

K_POINTS automatic
  9 9 1  0 0 0
"""


def pw_out_text(with_final_cell, positions_z="7.7777777777"):
    cell_block = (
        f"""     new unit-cell volume =   1477.01916 a.u.^3
     density =      2.74905 g/cm^3

CELL_PARAMETERS (angstrom)
   2.527050962   0.000000000   0.000000000
   1.263525476   2.188490333   0.000000000
   0.000000000   0.000000000  {RELAXED_CELL_Z}

"""
        if with_final_cell
        else ""
    )
    return f"""     Program PWSCF v.7.3.1 starts

Begin final coordinates
{cell_block}ATOMIC_POSITIONS (angstrom)
C                0.0000000000       0.0000000000       5.5555555555
H                0.0000000000       0.0000000000       {positions_z}
End final coordinates

     JOB DONE.
"""


def make_src(tmp_path, calculation, with_final_cell, name="src"):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "pw.in").write_text(pw_in_text(calculation))
    (d / "pw.out").write_text(pw_out_text(with_final_cell))
    return d


# ── (a) cell provenance ───────────────────────────────────────────────────────

def test_vc_relax_takes_cell_from_pw_out_not_pw_in(tmp_path):
    """The defect: a vc-relax source must not inherit the pw.in cell."""
    src = make_src(tmp_path, "vc-relax", with_final_cell=True)
    dest = tmp_path / "out"

    text = mss.make_one(src, dest).read_text()

    assert RELAXED_CELL_Z in text, "relaxed cell from pw.out is missing"
    assert INPUT_CELL_Z not in text, (
        "unrelaxed pw.in cell leaked into a vc-relax-derived input; this is the "
        "defect — relaxed positions paired with the pre-relaxation cell"
    )
    # positions must still be the relaxed ones
    assert "5.5555555555" in text
    assert "pw.out final coordinates" in text


def test_fixed_cell_relax_keeps_input_cell(tmp_path):
    """A fixed-cell relax never moved the cell, so pw.in remains correct."""
    src = make_src(tmp_path, "relax", with_final_cell=False)
    dest = tmp_path / "out"

    text = mss.make_one(src, dest).read_text()

    assert INPUT_CELL_Z in text
    assert RELAXED_CELL_Z not in text
    assert "5.5555555555" in text
    assert "fixed-cell source calculation=relax" in text


def test_vc_relax_without_final_cell_raises_rather_than_falling_back(tmp_path):
    """
    An unfinished vc-relax must be a hard error. Falling back to the pw.in cell
    here is exactly the silent pairing the fix exists to prevent.
    """
    src = make_src(tmp_path, "vc-relax", with_final_cell=False)

    with pytest.raises(ValueError, match="vc-relax"):
        mss.make_one(src, tmp_path / "out")


def test_extract_final_cell_returns_none_for_fixed_cell_relax():
    """Absence of a final cell is the expected result, not a parse failure."""
    assert mss.extract_final_cell(pw_out_text(with_final_cell=False)) is None
    got = mss.extract_final_cell(pw_out_text(with_final_cell=True))
    assert got is not None and RELAXED_CELL_Z in got


def test_extract_calculation_reads_control_block():
    assert mss.extract_calculation(pw_in_text("vc-relax")) == "vc-relax"
    assert mss.extract_calculation(pw_in_text("relax")) == "relax"
    assert mss.extract_calculation("no control block here") is None


# ── (b) discovery instead of a hardcoded list ────────────────────────────────

def test_discovery_finds_relaxations_and_skips_scf_only(tmp_path):
    """
    Selection is by declared calculation, not folder name, so the *_stress_scf
    outputs this script writes are not re-consumed as if they were relaxations.
    """
    root = tmp_path / "results"
    make_src(root, "relax", True, name="slab_a")
    make_src(root, "vc-relax", True, name="slab_b")
    make_src(root, "scf", True, name="slab_a_stress_scf")   # our own output
    (root / "not_a_run").mkdir()                            # no pw.in/pw.out

    assert mss.discover_relaxations(root) == ["slab_a", "slab_b"]


def test_discovery_is_not_the_old_hardcoded_campaign_list(tmp_path):
    """
    Regression guard: the old default was a fixed list of three 6L run names,
    so a results root containing none of them yielded nothing usable.
    """
    root = tmp_path / "results"
    make_src(root, "relax", True, name="thick_a0corr~C100_10L")

    found = mss.discover_relaxations(root)

    assert found == ["thick_a0corr~C100_10L"]
    assert not hasattr(mss, "DEFAULT_ONLY"), (
        "the hardcoded campaign list is back; --only-less runs will silently "
        "target the wrong campaign"
    )


def test_discovery_raises_on_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        mss.discover_relaxations(tmp_path / "nope")


def test_cli_without_only_walks_results_root(tmp_path, monkeypatch, capsys):
    root = tmp_path / "results"
    make_src(root, "relax", True, name="slab_a")
    runs = tmp_path / "runs"

    monkeypatch.setattr(sys, "argv", [
        "make_slab_stress_scf.py",
        "--results-root", str(root),
        "--runs-root", str(runs),
    ])
    mss.main()

    assert (runs / "slab_a_stress_scf" / "pw.in").exists()
    assert "discovered 1 relaxation" in capsys.readouterr().out
