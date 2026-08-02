"""Tests for surface_energy.py — gamma(mu_H) from the production ladder."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import surface_energy as se

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTION = os.path.join(REPO, "results", "production")
REFERENCE = os.path.join(REPO, "results", "reference_90_720")


# ------------------------------------------------------------------- units
def test_unit_conversion_chain():
    # 1 Ry = 13.605693 eV; 1 eV/A^2 = 16.02176634 J/m^2
    assert se.RY_PER_A2_TO_J_PER_M2 == pytest.approx(13.605693 * 16.02176634)
    # A quick independent anchor: 1 eV/A^2 in SI.
    assert se.EV_PER_A2_TO_J_PER_M2 == pytest.approx(1.602176634e-19 / 1e-20)


# ------------------------------------------------------------- the algebra
def _fake_ref(mu_c=-18.0, mu_h=-1.0):
    return se.Reference(
        mu_h_rich_ry=mu_h, e_h2_ry=2 * mu_h, mu_c_bulk_ry=mu_c,
        h2_run="fake", bulk_source="fake", pseudo_C="c.upf", pseudo_H="h.upf",
        ecutwfc=90.0, ecutrho=720.0)


def _fake_points(surface, gamma_ry_per_a2, area, n_H, mu_c, mu_h, layers,
                 per_layer=1):
    """Slabs built to satisfy E = N_C*mu_C + N_H*mu_H + 2*A*gamma exactly."""
    out = []
    for n in layers:
        n_C = n * per_layer
        E = n_C * mu_c + n_H * mu_h + 2 * area * gamma_ry_per_a2
        out.append(se.SlabEnergy(
            surface=surface, layers=n, run=f"x~{surface}_{n}L", n_C=n_C,
            n_H=n_H, area_angstrom2=area, energy_ry=E,
            pseudo_C="c.upf", pseudo_H="h.upf", ecutwfc=90.0, ecutrho=720.0))
    return out


def test_fit_recovers_a_known_gamma_and_mu_c_exactly():
    ref = _fake_ref()
    gamma_ry = -0.004477                      # about -0.976 J/m^2
    pts = _fake_points("C111", gamma_ry, 5.528, 2, ref.mu_c_bulk_ry,
                       ref.mu_h_rich_ry, [6, 8, 10, 12, 16])
    fit = se.fit_surface_energy("C111", pts, ref, exclude_layers=[6])
    assert fit.mu_c_fit_ry == pytest.approx(ref.mu_c_bulk_ry, abs=1e-12)
    assert fit.gamma_h_rich_j_m2 == pytest.approx(
        gamma_ry * se.RY_PER_A2_TO_J_PER_M2, abs=1e-9)
    assert fit.fit_rms_ry == pytest.approx(0.0, abs=1e-12)
    assert fit.gamma_scatter_j_m2 == pytest.approx(0.0, abs=1e-9)


def test_slope_is_hydrogen_coverage_per_area():
    ref = _fake_ref()
    pts = _fake_points("C110", -0.004, 9.0271, 4, ref.mu_c_bulk_ry,
                       ref.mu_h_rich_ry, [8, 10, 12, 16], per_layer=2)
    fit = se.fit_surface_energy("C110", pts, ref, exclude_layers=[])
    assert fit.coverage_per_a2 == pytest.approx(4 / (2 * 9.0271))
    assert fit.slope_j_m2_per_ev == pytest.approx(
        fit.coverage_per_a2 * se.EV_PER_A2_TO_J_PER_M2)


def test_gamma_at_is_linear_in_delta_mu():
    ref = _fake_ref()
    pts = _fake_points("C111", -0.004477, 5.528, 2, ref.mu_c_bulk_ry,
                       ref.mu_h_rich_ry, [8, 10, 12, 16])
    fit = se.fit_surface_energy("C111", pts, ref, exclude_layers=[])
    assert fit.gamma_at(0.0) == pytest.approx(fit.gamma_h_rich_j_m2)
    assert fit.gamma_at(2.0) - fit.gamma_at(1.0) == pytest.approx(
        fit.slope_j_m2_per_ev)
    z = fit.zero_crossing_ev()
    assert fit.gamma_at(z) == pytest.approx(0.0, abs=1e-9)


def test_fit_refuses_too_few_points():
    ref = _fake_ref()
    pts = _fake_points("C111", -0.004, 5.528, 2, ref.mu_c_bulk_ry,
                       ref.mu_h_rich_ry, [8, 10])
    with pytest.raises(se.SurfaceEnergyError, match="too few"):
        se.fit_surface_energy("C111", pts, ref, exclude_layers=[])


def test_fit_refuses_a_ladder_with_changing_area_or_termination():
    ref = _fake_ref()
    pts = _fake_points("C111", -0.004, 5.528, 2, ref.mu_c_bulk_ry,
                       ref.mu_h_rich_ry, [8, 10, 12, 16])
    bad = list(pts)
    bad[1] = se.SlabEnergy(**{**bad[1].__dict__, "area_angstrom2": 99.0})
    with pytest.raises(se.SurfaceEnergyError, match="cell area differs"):
        se.fit_surface_energy("C111", bad, ref, exclude_layers=[])

    bad = list(pts)
    bad[1] = se.SlabEnergy(**{**bad[1].__dict__, "n_H": 8})
    with pytest.raises(se.SurfaceEnergyError, match="hydrogen count differs"):
        se.fit_surface_energy("C111", bad, ref, exclude_layers=[])


# ------------------------------------------------------- reference hygiene
def test_mismatched_pseudopotential_is_refused():
    """A chemical-potential subtraction across different pseudopotentials
    produces a plausible, meaningless number (CLAUDE.md invariant 7)."""
    ref = _fake_ref()
    pts = _fake_points("C111", -0.004, 5.528, 2, ref.mu_c_bulk_ry,
                       ref.mu_h_rich_ry, [8, 10, 12])
    bad = [se.SlabEnergy(**{**pts[0].__dict__, "pseudo_C": "other.upf"})] + pts[1:]
    with pytest.raises(se.SurfaceEnergyError, match="pseudopotential"):
        se.check_reference_consistency(bad, ref)


def test_mismatched_cutoff_is_refused():
    ref = _fake_ref()
    pts = _fake_points("C111", -0.004, 5.528, 2, ref.mu_c_bulk_ry,
                       ref.mu_h_rich_ry, [8, 10, 12])
    bad = [se.SlabEnergy(**{**pts[0].__dict__, "ecutwfc": 60.0})] + pts[1:]
    with pytest.raises(se.SurfaceEnergyError, match="cutoffs"):
        se.check_reference_consistency(bad, ref)


# -------------------------------------------------------------- crossings
def _fit_with(surface, gamma, slope):
    f = se.GammaFit(
        surface=surface, orientation=f"({surface[1:]})", area_angstrom2=10.0,
        n_H=2, layers_used=[8, 10, 12, 16], layers_excluded=[6],
        mu_c_fit_ry=-18.0, mu_c_bulk_ry=-18.0, fit_rms_ry=0.0,
        gamma_h_rich_j_m2=gamma, slope_j_m2_per_ev=slope,
        coverage_per_a2=slope / se.EV_PER_A2_TO_J_PER_M2,
        per_slab_gamma_j_m2={}, gamma_scatter_j_m2=0.0)
    return f


def test_a_curve_starting_above_and_rising_faster_never_crosses():
    """The geometry that decides whether an ordering can invert at all."""
    low_slow = _fit_with("C111", -1.0, 2.9)
    high_fast = _fit_with("C110", -0.85, 3.5)
    crossings = [c for c in se.find_crossings([low_slow, high_fast], (0.0, 3.0))
                 if c.kind == "ordering"]
    assert len(crossings) == 1
    # It "crosses" only at negative delta_mu, i.e. never physically.
    assert crossings[0].delta_mu_ev < 0
    assert not crossings[0].physical
    # ... and the ordering is the same at both ends of the physical range.
    assert (se.ordering_at([low_slow, high_fast], 0.0)
            == se.ordering_at([low_slow, high_fast], 3.0))


def test_a_curve_starting_above_and_rising_slower_does_cross():
    low_fast = _fit_with("C110", -1.0, 3.5)
    high_slow = _fit_with("C100", -0.85, 2.5)
    crossings = [c for c in se.find_crossings([low_fast, high_slow], (0.0, 3.0))
                 if c.kind == "ordering"]
    assert crossings[0].physical
    assert crossings[0].delta_mu_ev == pytest.approx(0.15 / 1.0)
    assert se.ordering_at([low_fast, high_slow], 0.0)[0] == "C110"
    assert se.ordering_at([low_fast, high_slow], 3.0)[0] == "C100"


def test_wulff_available_from_is_the_last_gamma_to_turn_positive():
    fits = [_fit_with("C100", 0.05, 2.5),      # already positive
            _fit_with("C110", -0.83, 3.5),     # turns positive at 0.237
            _fit_with("C111", -0.98, 2.9)]     # turns positive at 0.338
    avail = se.wulff_available_from_ev(fits)
    assert avail == pytest.approx(0.98 / 2.9)
    for f in fits:
        assert f.gamma_at(avail) >= -1e-12
    assert any(f.gamma_at(avail - 0.01) < 0 for f in fits)


def test_wulff_never_available_if_a_gamma_never_rises():
    fits = [_fit_with("C111", -0.98, 0.0)]
    assert se.wulff_available_from_ev(fits) is None


# ------------------------------------------------------------- production
production = pytest.mark.skipif(
    not os.path.isfile(os.path.join(REFERENCE, "H2", "pw.out")),
    reason="production runs or H2 reference not present")


@pytest.fixture(scope="module")
def derived(tmp_path_factory):
    out = tmp_path_factory.mktemp("surface_energy")
    return se.run(PRODUCTION, REFERENCE, out, se.DEFAULT_PREFIX, [6],
                  (0.0, 3.0), 61, se.DEFAULT_MU_C_TOL_MRY,
                  config_out=out / "surface_energies_h.json"), out


@production
def test_chemical_potentials_are_read_not_assumed(derived):
    result, _ = derived
    ref = result["reference"]
    assert ref.e_h2_ry == pytest.approx(-2.3329149665, abs=1e-8)
    assert ref.mu_h_rich_ry == pytest.approx(ref.e_h2_ry / 2)
    # mu_C is the BM3 E0 per carbon, NOT the sampled eps=0 run (a0 = 3.567).
    assert ref.mu_c_bulk_ry == pytest.approx(-147.47261566492298 / 8, abs=1e-9)
    assert ref.mu_c_bulk_ry != pytest.approx(-147.47249987 / 8, abs=1e-9)


@production
def test_fitted_mu_c_agrees_with_the_bulk_reference_on_110_and_111(derived):
    """The check that says the slab interior really is bulk-like."""
    result, _ = derived
    fits = {f.surface: f for f in result["fits"]}
    for surface in ("C110", "C111"):
        assert abs(fits[surface].mu_c_drift_mry) < 0.02, surface
        assert fits[surface].epistemic_level == "L2"


@production
def test_c100_is_downgraded_and_warned_about(derived):
    """(100)'s parity oscillation shows up here exactly as it does in the tau
    fit: a drifting mu_C and a scattered per-slab gamma."""
    result, _ = derived
    fits = {f.surface: f for f in result["fits"]}
    assert fits["C100"].epistemic_level == "L1"
    assert abs(fits["C100"].mu_c_drift_mry) > 0.02
    assert fits["C100"].gamma_scatter_j_m2 > 0.01
    assert any("C100" in w for w in result["meta"]["warnings"])


@production
def test_derived_gammas_match_the_ad_hoc_campaign_values(derived):
    """The values that were computed ad hoc and never written down were
    -0.980 / -0.832 / +0.031 J/m^2. The derivation should land close, and the
    residual difference is itself informative: (100) is the one that moves."""
    result, _ = derived
    g = {f.surface: f.gamma_h_rich_j_m2 for f in result["fits"]}
    assert g["C111"] == pytest.approx(-0.980, abs=0.01)
    assert g["C110"] == pytest.approx(-0.832, abs=0.01)
    # (100) moves by more than its own scatter, consistent with its L1 status.
    assert g["C100"] == pytest.approx(0.031, abs=0.03)


@production
def test_coverage_coefficients_match_the_geometry(derived):
    """dgamma/d(-mu_H) = N_H / 2A, in Angstrom^-2."""
    result, _ = derived
    expected = {"C111": 0.181, "C110": 0.222, "C100": 0.157}
    for f in result["fits"]:
        assert f.coverage_per_a2 == pytest.approx(expected[f.surface], abs=5e-4)
        assert f.coverage_per_a2 == pytest.approx(
            f.n_H / (2 * f.area_angstrom2), abs=1e-12)


@production
def test_the_110_111_ordering_does_not_invert_at_physical_mu_h(derived):
    """(110) starts ABOVE (111) and rises FASTER, so the gap only widens.
    Their formal crossing sits at negative delta_mu, which would require mu_H
    above the H-rich limit."""
    result, _ = derived
    fits = {f.surface: f for f in result["fits"]}
    assert fits["C110"].gamma_h_rich_j_m2 > fits["C111"].gamma_h_rich_j_m2
    assert fits["C110"].slope_j_m2_per_ev > fits["C111"].slope_j_m2_per_ev
    cross = [c for c in result["crossings"]
             if c.kind == "ordering" and set(c.surfaces) == {"C110", "C111"}]
    assert len(cross) == 1 and not cross[0].physical
    for d in (0.0, 1.0, 2.0, 3.0):
        assert fits["C111"].gamma_at(d) < fits["C110"].gamma_at(d)


@production
def test_100_becomes_the_most_stable_facet_only_far_from_the_h_rich_limit(derived):
    result, _ = derived
    fits = result["fits"]
    assert se.ordering_at(fits, 0.0)[0] == "C111"
    cross = [c for c in result["crossings"]
             if c.kind == "ordering" and set(c.surfaces) == {"C100", "C111"}]
    assert len(cross) == 1 and cross[0].physical
    x = cross[0].delta_mu_ev
    assert x > 2.0
    assert se.ordering_at(fits, x - 0.1)[0] == "C111"
    assert se.ordering_at(fits, x + 0.1)[0] == "C100"


@production
def test_wulff_threshold_is_where_the_last_gamma_turns_positive(derived):
    result, _ = derived
    avail = result["meta"]["wulff_available_from_ev"]
    assert 0.3 < avail < 0.4
    fits = result["fits"]
    assert all(f.gamma_at(avail + 1e-6) > 0 for f in fits)
    assert any(f.gamma_at(avail - 0.01) < 0 for f in fits)
    # It is set by (111), the last one to turn positive.
    last = max(fits, key=lambda f: f.zero_crossing_ev() or -math.inf)
    assert last.surface == "C111"


@production
def test_generated_config_replaces_the_hand_entered_one(derived):
    """Item 3: config/surface_energies_h.json becomes generated output, so its
    provenance_gap closes."""
    _result, out = derived
    cfg = json.loads((out / "surface_energies_h.json").read_text())
    assert cfg["source_type"] == "project_dft_fit"
    assert "provenance_gap" not in cfg
    assert cfg["generated_by"] == "surface_energy.py"
    for family in ("100", "110", "111"):
        entry = cfg["surface_energies"][family]
        assert "dgamma_dmu_j_m2_per_ev" in entry
        assert "epistemic_level" in entry
        assert entry["layers_excluded"] == [6]
    cp = cfg["chemical_potentials"]
    assert cp["mu_H_rich_definition"].startswith("E(H2)/2")
    assert "Birch-Murnaghan" in cp["mu_C_definition"]
    assert "NOT COMPUTED" in cfg["mu_h_dependence"]["h_poor_limit"]


@production
def test_outputs_state_the_unbounded_h_poor_limit(derived):
    _result, out = derived
    report = (out / "surface_energy_report.md").read_text()
    assert "CONVENTION" in report or "convention" in report
    assert "CH4" in report
    assert "UNPHYSICAL" in report          # the non-crossing is called out
    scan = (out / "surface_energy_vs_mu_h.csv").read_text()
    assert "wulff_defined" in scan and "most_stable" in scan


# ============================================ H2 ideal-gas chemical potential
def test_rrho_reproduces_the_janaf_tabulation():
    """The load-bearing validation of the T-p mapping. If this drifts, every
    temperature the mapping reports drifts with it."""
    val = se.rrho_validation()
    assert val["passes"], val
    assert val["max_abs_deviation_ev"] < se.RRHO_VS_JANAF_MAX_DEV_EV
    for T, v in val["points"].items():
        assert v["model_ev"] == pytest.approx(v["janaf_ev"], abs=0.015), T


def test_h2_zero_point_energy_matches_the_spectroscopic_value():
    zpe = 0.5 * se.KB_EV * se.H2_THETA_VIB_K
    assert zpe == pytest.approx(0.273, abs=0.003)      # H2 ZPE ~ 0.27 eV
    with_zpe = se.h2_mu_shift_ev(1000.0, se.PA_PER_BAR, include_zpe=True)
    without = se.h2_mu_shift_ev(1000.0, se.PA_PER_BAR, include_zpe=False)
    assert with_zpe - without == pytest.approx(zpe, abs=1e-9)


def test_rotational_partition_function_approaches_the_high_t_limit():
    """The explicit sum is used because H2's theta_rot is large; it must still
    converge to T/(sigma*theta_rot) when T >> theta_rot."""
    for T in (2000.0, 4000.0):
        exact = se.h2_rotational_partition_function(T)
        high_t = T / (se.H2_SYMMETRY_NUMBER * se.H2_THETA_ROT_K)
        assert exact == pytest.approx(high_t, rel=0.02)
    # ... and must differ noticeably at room temperature, which is why the
    # high-T limit is not used.
    room = se.h2_rotational_partition_function(300.0)
    assert abs(room - 300.0 / (2 * se.H2_THETA_ROT_K)) / room > 0.05


def test_delta_mu_grows_with_temperature_and_with_falling_pressure():
    base = se.delta_mu_from_tp(600.0, se.PA_PER_BAR)
    assert se.delta_mu_from_tp(1200.0, se.PA_PER_BAR) > base
    assert se.delta_mu_from_tp(600.0, se.PA_PER_BAR * 1e-6) > base


def test_pressure_dependence_is_the_ideal_gas_logarithm():
    """delta_mu gains (kT/2)*ln(p0/p) exactly."""
    T = 900.0
    a = se.delta_mu_from_tp(T, se.PA_PER_BAR)
    b = se.delta_mu_from_tp(T, se.PA_PER_BAR * 1e-6)
    assert b - a == pytest.approx(0.5 * se.KB_EV * T * math.log(1e6), abs=1e-9)


def test_temperature_solver_inverts_the_mapping():
    for target in (0.34, 0.85, 1.3, 2.6):
        for p_bar in (1.0, 1e-6, 1e-12):
            T = se.temperature_for_delta_mu(target, p_bar * se.PA_PER_BAR)
            if T is None:
                continue
            assert se.delta_mu_from_tp(T, p_bar * se.PA_PER_BAR) == \
                pytest.approx(target, abs=1e-6)


def test_unreachable_condition_returns_none_rather_than_extrapolating():
    assert se.temperature_for_delta_mu(50.0, se.PA_PER_BAR) is None


def test_low_temperature_is_refused_not_extrapolated():
    """Below ~150 K the rigid-rotor treatment of H2 is simply wrong
    (ortho/para nuclear-spin statistics)."""
    with pytest.raises(se.SurfaceEnergyError, match="rigid-rotor"):
        se.delta_mu_from_tp(77.0, se.PA_PER_BAR)


def test_non_positive_pressure_is_refused():
    with pytest.raises(se.SurfaceEnergyError, match="pressure must be positive"):
        se.delta_mu_from_tp(500.0, 0.0)


def test_tp_curve_is_a_locus_of_constant_delta_mu():
    rows = se.tp_curve(1.3)
    assert len(rows) == len(se.DEFAULT_TP_PRESSURES_BAR)
    for r in rows:
        if r["temperature_k"] is None:
            continue
        assert se.delta_mu_from_tp(r["temperature_k"], r["p_h2_pa"]) == \
            pytest.approx(1.3, abs=1e-6)
        assert r["p_h2_torr"] == pytest.approx(r["p_h2_pa"] / se.PA_PER_TORR)
        assert r["temperature_c"] == pytest.approx(r["temperature_k"] - 273.15)
    # Lower pressure must always mean a lower temperature for the same mu_H.
    temps = [r["temperature_k"] for r in rows if r["temperature_k"]]
    assert temps == sorted(temps, reverse=True)


def test_temperature_uncertainty_is_small_but_nonzero():
    u = se.temperature_uncertainty_k(1.3, se.PA_PER_BAR * 1e-6)
    assert 0 < u < 50


@production
def test_tp_map_is_written_and_labels_its_provenance(derived):
    _result, out = derived
    text = (out / "surface_energy_tp_map.csv").read_text()
    assert "temperature_k" in text and "p_h2_torr" in text
    report = (out / "surface_energy_report.md").read_text()
    assert "ideal-gas chemical potential" in report
    assert "NIST-JANAF" in report
    assert "Zero-point energy is EXCLUDED" in report
    # The dominant caveat must be in the output, not just a comment.
    assert "remain the stable termination" in report
    assert "graphitize" in report
