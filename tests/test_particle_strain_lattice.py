"""
Tests for the coupling-independent observable, the size sweep, the
energy-derived mixture, and the D-vs-E channel constraint analysis.

The lattice strain matters because it is the one prediction in this chain that
a measurement can test without adopting a spin-strain parameter set: the two
published sets differ by ~1.65x on the axial channel, and that factor
multiplies every MHz number but none of the lattice numbers.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nv_spin_strain
import particle_strain as ps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU_CSV = os.path.join(REPO, "results", "production", "tau_infinity.csv")
SURFACE_ENERGIES = os.path.join(REPO, "config", "surface_energies_h.json")
A0 = 3.572997470394866
ELASTIC = nv_spin_strain.ElasticConstants()
DFT = nv_spin_strain.UDVARHELYI_DFT
BARSON = nv_spin_strain.BARSON_SCALED

TAU = {   # continuum f, tension positive (what evaluate() consumes)
    "100": {"tau_xx": -1.0323, "tau_yy": 5.0241},
    "110": {"tau_xx": -2.1445, "tau_yy": -4.3975},
    "111": {"tau_xx": -0.4833, "tau_yy": -0.4833},
}


def _eval(fractions, radius_nm=1.5, param_sets=(DFT,), overrides=None):
    return ps.evaluate(TAU, fractions, radius_nm, ELASTIC, list(param_sets),
                       overrides, shape_label="test", a0_angstrom=A0)


# ── the coupling-independent claim ───────────────────────────────────────────

def test_lattice_strain_is_identical_across_coupling_sets():
    """
    The whole point. The two parameter sets change delta_D by 1.65x; if they
    changed the lattice strain at all, it would not be coupling-independent.
    """
    r = _eval({"111": 1.0}, param_sets=(DFT, BARSON))
    dft = [x["delta_D_mhz"] for x in r.rows_for(DFT.name)]
    bar = [x["delta_D_mhz"] for x in r.rows_for(BARSON.name)]
    assert bar[0] / dft[0] == pytest.approx(1.654, rel=1e-3)  # couplings do move D

    # ... and the lattice quantities are properties of the strain alone
    assert r.lattice_strain == pytest.approx(float(np.trace(r.eps_cubic)) / 3.0)
    assert r.lattice_parameter_angstrom == pytest.approx(
        A0 * (1.0 + r.lattice_strain))


def test_lattice_strain_matches_the_pressure_and_bulk_modulus():
    """eps_lin = -P/(3B) to within the difference between the fitted B and
    (C11+2*C12)/3, which is the only reason the two are not identical."""
    r = _eval({"111": 1.0})
    B = (ELASTIC.C11 + 2.0 * ELASTIC.C12) / 3.0
    assert r.lattice_strain == pytest.approx(-r.pressure_gpa / (3.0 * B), rel=1e-9)


def test_lattice_parameter_is_none_without_a0():
    r = ps.evaluate(TAU, {"111": 1.0}, 1.5, ELASTIC, [DFT], shape_label="t")
    assert r.a0_angstrom is None
    assert r.lattice_parameter_angstrom is None
    assert math.isfinite(r.lattice_strain)      # strain still defined


def test_the_three_shapes_do_not_all_strain_the_lattice_the_same_way():
    """
    A {100}-dominated particle CONTRACTS while {110}/{111} EXPAND, because
    (100) is the one surface with net tensile f. That sign difference is a
    sharp, coupling-free PXRD signature and must not collapse.
    """
    oct_ = _eval({"111": 1.0}).lattice_strain_percent
    cube = _eval({"100": 1.0}).lattice_strain_percent
    rd = _eval({"110": 1.0}).lattice_strain_percent
    assert oct_ > 0 and rd > 0, "compressive surfaces must expand the lattice"
    assert cube < 0, "the tensile (100) surface must contract the lattice"
    assert oct_ == pytest.approx(0.0486, abs=5e-3)
    assert cube == pytest.approx(-0.2007, abs=5e-3)
    assert rd == pytest.approx(0.3289, abs=5e-3)


# ── size sweep ───────────────────────────────────────────────────────────────

def test_the_direction_survives_rescaling_every_tau():
    """
    The sign split is robust to the absolute tau scale, not just to the
    couplings. Multiplying every tau by a common positive factor must leave
    every direction unchanged — that is what makes the direction, rather than
    the magnitude, the primary experimental handle.
    """
    base = ps.facet_lattice_directions(TAU, ELASTIC, A0)
    for factor in (0.5, 2.0, 10.0):
        scaled = {f: {k: v * factor for k, v in t.items()}
                  for f, t in TAU.items()}
        got = ps.facet_lattice_directions(scaled, ELASTIC, A0)
        assert [d["direction"] for d in got] == [d["direction"] for d in base]
        for b, g in zip(base, got):
            assert g["lattice_strain_percent"] == pytest.approx(
                b["lattice_strain_percent"] * factor, rel=1e-12)


def test_facet_lattice_directions_reports_the_100_split():
    dirs = {d["family"]: d for d in ps.facet_lattice_directions(TAU, ELASTIC, A0)}
    assert dirs["100"]["direction"] == "contract"
    assert dirs["110"]["direction"] == "expand"
    assert dirs["111"]["direction"] == "expand"
    # slope is da/d(1/R); sign follows the direction
    assert dirs["100"]["da_d_inverse_radius_angstrom_nm"] < 0
    assert dirs["110"]["da_d_inverse_radius_angstrom_nm"] > 0
    assert dirs["111"]["da_d_inverse_radius_angstrom_nm"] > 0


@pytest.mark.skipif(not os.path.isfile(TAU_CSV), reason="tau_infinity.csv absent")
def test_prose_adjectives_match_the_committed_tau_signs():
    """
    The sign-change narrative names which facet is tensile and which is
    compressive. Those adjectives were inverted for several commits while the
    conclusion they supported stayed correct, so nothing caught it. Pin them to
    the data.
    """
    import csv as _csv
    proj = {}
    with open(TAU_CSV, newline="") as fh:
        for row in _csv.DictReader(fh):
            proj[row["surface"][1:]] = 0.5 * (float(row["tau_xx_inf_n_per_m"])
                                              + float(row["tau_yy_inf_n_per_m"]))
    # continuum f = -tau_project; positive f is tensile
    f = {k: -v for k, v in proj.items()}
    assert f["100"] > 0, "(100) must be TENSILE in the continuum sense"
    assert f["111"] < 0, "(111) must be COMPRESSIVE in the continuum sense"

    src = open(os.path.join(REPO, "particle_strain.py")).read()
    assert "(100) facets\n# carry a net TENSILE" in src or \
           "carry a net TENSILE surface stress while (111)" in src, \
        "the module comment no longer matches the data"
    assert "carries a net TENSILE surface stress (f = +2.0 N/m) while (111)-H" in src, \
        "the report prose no longer matches the data"
    assert "(100)-H carries a net COMPRESSIVE" not in src, \
        "the inverted adjective pair is back"


def test_size_sweep_is_exactly_linear_in_inverse_radius():
    """The prediction's SHAPE is what a size-resolved series tests."""
    radii = [1.0, 2.0, 4.0, 8.0]
    rows = [r for r in ps.size_sweep(TAU, {"111": 1.0}, radii, ELASTIC, [DFT],
                                     shape_label="t", a0_angstrom=A0)]
    slopes = [r["lattice_strain_percent"] * r["radius_nm"] for r in rows]
    assert all(s == pytest.approx(slopes[0], rel=1e-12) for s in slopes)
    # doubling R halves both the strain and the shift
    assert rows[1]["lattice_strain_percent"] == pytest.approx(
        rows[0]["lattice_strain_percent"] / 2.0, rel=1e-12)
    assert rows[1]["delta_D_mhz"] == pytest.approx(
        rows[0]["delta_D_mhz"] / 2.0, rel=1e-9)


def test_sweep_lattice_columns_do_not_depend_on_the_param_set():
    rows = ps.size_sweep(TAU, {"111": 1.0}, [1.5], ELASTIC, [DFT, BARSON],
                         shape_label="t", a0_angstrom=A0)
    assert len(rows) == 2
    assert rows[0]["lattice_strain_percent"] == rows[1]["lattice_strain_percent"]
    assert rows[0]["lattice_parameter_angstrom"] == rows[1]["lattice_parameter_angstrom"]
    assert rows[0]["delta_D_mhz"] != rows[1]["delta_D_mhz"]


# ── energy-derived mixture ───────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.isfile(SURFACE_ENERGIES),
                    reason="surface energy config not present")
def test_mixture_fractions_follow_the_stability_order():
    gammas, _cfg, _prov = ps.load_surface_energies(SURFACE_ENERGIES, 0.0)
    frac = ps.stability_weighted_fractions(gammas)
    assert sum(frac.values()) == pytest.approx(1.0)
    # most stable (most negative gamma) gets the largest area
    order_gamma = sorted(gammas, key=lambda k: gammas[k])
    order_area = sorted(frac, key=lambda k: -frac[k])
    assert order_gamma == order_area
    assert frac["111"] > frac["110"] > frac["100"]


def test_stability_weighting_is_invariant_under_a_gamma_offset():
    """
    The absolute gammas are H2-reference-dependent and negative; only their
    DIFFERENCES are meaningful. A weighting that moved when every gamma shifted
    by a constant would be reading the reference state, not the physics.
    """
    g = {"100": 0.05, "110": -0.83, "111": -0.98}
    base = ps.stability_weighted_fractions(g, scale_j_m2=1.0)
    for offset in (-5.0, +2.5, +100.0):
        shifted = ps.stability_weighted_fractions(
            {k: v + offset for k, v in g.items()}, scale_j_m2=1.0)
        for k in g:
            assert shifted[k] == pytest.approx(base[k], rel=1e-12)


def test_stability_weighting_rejects_a_degenerate_scale():
    with pytest.raises(ps.ParticleStrainError, match="same surface energy"):
        ps.stability_weighted_fractions({"100": 1.0, "111": 1.0})


@pytest.mark.skipif(not os.path.isfile(SURFACE_ENERGIES),
                    reason="surface energy config not present")
def test_wulff_is_undefined_across_the_whole_physical_mu_range():
    """
    Justifies deriving the mixture some other way. dgamma/dmu_H > 0 and
    mu_H <= E(H2)/2, so gamma is maximal at the H-rich limit; if it is still
    negative there, it is negative everywhere allowed.
    """
    cfg = json.loads(open(SURFACE_ENERGIES).read())["surface_energies"]
    for fam, entry in cfg.items():
        assert entry["dgamma_dmu_j_m2_per_ev"] > 0
    negative_at_h_rich = [f for f, e in cfg.items() if e["value"] < 0]
    assert negative_at_h_rich, "fixture no longer exercises the ill-posed case"
    for fam in negative_at_h_rich:
        # would need to go ABOVE the H-rich limit to turn positive
        assert cfg[fam]["zero_crossing_delta_mu_ev"] > 0


# ── D vs E channel constraint ────────────────────────────────────────────────

def test_delta_d_and_e_depend_on_disjoint_couplings():
    """
    The substantive claim behind the channel note: the axial and transverse
    channels share no parameter, so constraining one says nothing about the
    other.
    """
    r = _eval({"111": 0.5, "100": 0.5},
              overrides={"001": 2.0})            # break symmetry so E != 0
    assert r.e_max_mhz > 0
    sens = ps.channel_coupling_sensitivity(
        r.eps_cubic, nv_spin_strain.nv_frames()[0][1], DFT)
    c = sens["couplings"]
    assert not sens["e_is_identically_zero"]

    for axial in ("h41", "h43"):
        assert abs(c[axial]["d_rel"]) > 1e-3, f"{axial} must move delta_D"
        # Not exactly zero: h41/h43 shift the ms=0 <-> ms=+-1 energy
        # denominator, so they reach E through the second-order h25/h26
        # admixture. That leak is ~1e-9 relative against a ~4e-2 direct
        # effect -- seven orders down, and it vanishes identically when
        # h25 = h26 = 0 (asserted below).
        assert abs(c[axial]["e_rel"]) < 1e-6, f"{axial} must barely touch E"
    for transverse in ("h15", "h16"):
        assert abs(c[transverse]["e_rel"]) > 1e-3, f"{transverse} must move E"
        assert abs(c[transverse]["d_rel"]) < 1e-9, f"{transverse} must NOT move delta_D"
    for second_order in ("h25", "h26"):
        assert abs(c[second_order]["d_rel"]) < 1e-3
        assert abs(c[second_order]["e_rel"]) < 1e-3


def test_the_axial_leak_into_e_is_purely_the_second_order_admixture():
    """
    Pins the mechanism, not just the size. With the ms=0 mixing terms switched
    off, the axial couplings move E by EXACTLY zero -- so the channels are
    disjoint at first order with no residual, and the tiny leak seen otherwise
    is the h25/h26 admixture reacting to a shifted energy denominator.
    """
    import dataclasses
    r = _eval({"111": 0.5, "100": 0.5}, overrides={"001": 2.0})
    frame = nv_spin_strain.nv_frames()[0][1]
    no_mixing = dataclasses.replace(DFT, h25=0.0, h26=0.0)
    sens = ps.channel_coupling_sensitivity(r.eps_cubic, frame, no_mixing)
    assert sens["couplings"]["h41"]["e_rel"] == 0.0
    assert sens["couplings"]["h43"]["e_rel"] == 0.0


def test_hydrostatic_strain_probes_only_one_combination_of_the_axial_couplings():
    """
    delta_D under hydrostatic strain = (2*h41 + h43) * eps, exactly. That single
    number is what a hydrostatic-pressure experiment measures, and it is all
    that a symmetric particle's delta_D needs.
    """
    eps = np.eye(3) * 1.0e-3
    frame = nv_spin_strain.nv_frames()[0][1]
    obs = nv_spin_strain.nv_observables(eps, frame, DFT)
    assert obs["delta_D_mhz"] == pytest.approx((2 * DFT.h41 + DFT.h43) * 1.0e-3,
                                               rel=1e-9)
    assert obs["E_mhz"] == pytest.approx(0.0, abs=1e-12)


def test_e_is_identically_zero_for_any_symmetric_shape_whatever_the_couplings():
    """
    E's magnitude is not merely uncertain, it is unconstrained: for a
    symmetry-complete facet set it vanishes for every parameter set, so any
    non-zero prediction comes from an assumed shape asymmetry.
    """
    for fractions in ({"111": 1.0}, {"100": 1.0}, {"110": 1.0},
                      {"111": 0.45, "110": 0.39, "100": 0.16}):
        r = _eval(fractions, param_sets=(DFT, BARSON))
        assert r.shape_is_symmetric
        assert r.e_max_mhz == pytest.approx(0.0, abs=ps.E_ZERO_TOL_MHZ)


def test_channel_sensitivity_reports_undefined_rather_than_dividing_by_zero():
    r = _eval({"111": 1.0})
    sens = ps.channel_coupling_sensitivity(
        r.eps_cubic, nv_spin_strain.nv_frames()[0][1], DFT)
    assert sens["e_is_identically_zero"]
    assert all(v["e_rel"] is None for v in sens["couplings"].values())
    assert sens["couplings"]["h41"]["d_rel"] is not None
