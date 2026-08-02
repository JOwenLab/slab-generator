"""Tests for make_bare_slabs.py and the dehydrogenation bound."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import make_bare_slabs as mb
import preflight
import surface_energy as se

A0 = 3.572997470394866


def test_cutoff_patch_is_asserted_not_assumed():
    """A silent no-op here would generate a whole campaign at 60/480 Ry."""
    good = "&SYSTEM\n  ecutwfc  = 60.0\n  ecutrho  = 480.0\n/\n"
    out = mb.patch_production_settings(good)
    assert "90.0" in out and "720.0" in out and "60.0" not in out
    with pytest.raises(mb.BareSlabError, match="template has changed"):
        mb.patch_production_settings("&SYSTEM\n  nat = 4\n/\n")


def test_generated_slabs_are_bare_symmetric_and_production_settings(tmp_path):
    result = mb.run(tmp_path, A0, 8.0, include_pandey=False, ladder=(6, 8))
    assert len(result["runs"]) == 6            # 3 motifs x 2 thicknesses
    assert result["n_failed"] == 0
    for m in result["runs"]:
        assert "H" not in m["counts"]          # bare means bare
        text = (tmp_path / m["run"] / "pw.in").read_text()
        assert "ecutwfc  = 90.0" in text
        assert "ecutrho  = 720.0" in text
        assert "tefield" not in text           # symmetric: no dipole correction
        prov = json.loads((tmp_path / m["run"] / "provenance.json").read_text())
        assert prov["symmetric"] is True
        assert prov["top_termination"] == prov["bottom_termination"] == "bare"
        assert prov["not_submitted"] is True
        assert prov["a0_angstrom"] == A0
        assert prov["vacuum_angstrom"] == 8.0


def test_kmesh_matches_the_h_terminated_production_runs(tmp_path):
    result = mb.run(tmp_path, A0, 8.0, include_pandey=False, ladder=(8,))
    meshes = {m["run"].split("~")[1].split("_")[0]: m["kpoints"]
              for m in result["runs"]}
    assert meshes == {"C100": "9 5 1", "C110": "9 7 1", "C111": "9 9 1"}


def test_only_the_coordination_check_is_skipped():
    """Bare faces carry dangling bonds by construction; every OTHER structural
    invariant must still be enforced."""
    assert mb.PREFLIGHT_SKIP == ("carbon_coordination",)
    names = {n for n, _ in preflight.CHECKS}
    assert "inversion_symmetry" in names - set(mb.PREFLIGHT_SKIP)
    assert "no_periodic_overlap" in names - set(mb.PREFLIGHT_SKIP)
    assert "name_matches_geometry" in names - set(mb.PREFLIGHT_SKIP)


def test_pandey_uses_its_own_thicker_ladder(tmp_path):
    result = mb.run(tmp_path, A0, 8.0, include_pandey=True, ladder=(6,))
    pandey = [m for m in result["runs"] if "pandey" in m["run"]]
    assert {int(m["run"].rsplit("_", 1)[1][:-1]) for m in pandey} \
        == set(mb.PANDEY_LADDER)
    assert all(m["passed"] for m in pandey)


# ------------------------------------------------------ dehydrogenation bound
def _fit(surface, gamma, slope, n_H=2):
    return se.GammaFit(
        surface=surface, orientation=f"({surface[1:4]})", area_angstrom2=10.0,
        n_H=n_H, layers_used=[8, 10, 12, 16], layers_excluded=[6],
        mu_c_fit_ry=-18.0, mu_c_bulk_ry=-18.0, fit_rms_ry=0.0,
        gamma_h_rich_j_m2=gamma, slope_j_m2_per_ev=slope,
        coverage_per_a2=slope / se.EV_PER_A2_TO_J_PER_M2,
        per_slab_gamma_j_m2={}, gamma_scatter_j_m2=0.0)


def test_bound_is_where_gamma_h_reaches_gamma_bare():
    h = _fit("C111", -0.98, 2.9)
    bare = _fit("C111", 5.0, 0.0, n_H=0)
    out = se.dehydrogenation_bound([h], [bare])
    assert out["available"]
    expected = (5.0 - (-0.98)) / 2.9
    assert out["delta_mu_max_ev"] == pytest.approx(expected)
    assert h.gamma_at(out["delta_mu_max_ev"]) == pytest.approx(5.0)


def test_bound_takes_the_lowest_bare_variant():
    """Using the unreconstructed (111) alone gives a bound that is too
    permissive; the Pandey reconstruction is lower and must win."""
    h = _fit("C111", -0.98, 2.9)
    unrecon = _fit("C111", 6.0, 0.0, n_H=0)
    pandey = _fit("C111pandey", 4.0, 0.0, n_H=0)
    out = se.dehydrogenation_bound([h], [unrecon, pandey])
    assert out["per_surface"]["C111"]["bare_variant"] == "C111pandey"
    assert out["delta_mu_max_ev"] == pytest.approx((4.0 + 0.98) / 2.9)
    # ... and it really is the tighter of the two.
    loose = se.dehydrogenation_bound([h], [unrecon])
    assert out["delta_mu_max_ev"] < loose["delta_mu_max_ev"]


def test_binding_facet_is_the_first_to_dehydrogenate():
    fits = [_fit("C111", -0.98, 2.9), _fit("C110", -0.83, 3.55)]
    bare = [_fit("C111", 4.0, 0.0, n_H=0), _fit("C110", 3.0, 0.0, n_H=0)]
    out = se.dehydrogenation_bound(fits, bare)
    ceilings = {s: e["delta_mu_max_ev"] for s, e in out["per_surface"].items()}
    assert out["binding_surface"] == min(ceilings, key=ceilings.get)
    assert out["delta_mu_max_ev"] == min(ceilings.values())


def test_missing_bare_ladder_reports_the_gap_rather_than_a_number():
    out = se.dehydrogenation_bound([_fit("C111", -0.98, 2.9)], [])
    assert not out["available"]
    assert out["delta_mu_max_ev"] is None
    assert "make_bare_slabs.py" in out["reason"]
    assert "unbounded" in out["reason"]


def test_a_bare_fit_containing_hydrogen_is_refused():
    with pytest.raises(se.SurfaceEnergyError, match="no hydrogen"):
        se.lowest_bare_gamma([_fit("C111", 4.0, 0.0, n_H=2)])


def test_bare_surface_tag_parsing():
    assert se.bare_base_surface("C111") == "C111"
    assert se.bare_base_surface("C111pandey") == "C111"
    with pytest.raises(se.SurfaceEnergyError):
        se.bare_base_surface("nonsense")


def test_run_reports_the_missing_bound_as_a_warning(tmp_path):
    """Until the campaign runs, the gap must be loud."""
    result = se.run("results/production", "results/reference_90_720", tmp_path,
                    se.DEFAULT_PREFIX, [6], (0.0, 3.0), 21,
                    se.DEFAULT_MU_C_TOL_MRY, bare_runs_dir=None)
    assert not result["dehydrogenation_bound"]["available"]
    assert any("dehydrogenation bound MISSING" in w
               for w in result["meta"]["warnings"])
