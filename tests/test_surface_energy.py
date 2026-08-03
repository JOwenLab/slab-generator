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


# ================================================== mu_C drift: reject, not warn
def _ladder_from_per_slab_gammas(surface, gammas, area, ref, n_H=0,
                                 per_layer=1, match_ref=False, nspin=None):
    """A ladder whose per-slab gamma (with the BULK mu_C) is exactly `gammas`.

    Inverts gamma = [E - N_C*mu_C - N_H*mu_H] / 2A, so a ladder that is NOT on
    a single straight line can be written down directly -- which is the whole
    point: an unconverged thin end is exactly a ladder that is not.
    """
    out = []
    for n, g in sorted(gammas.items()):
        n_C = n * per_layer
        E = (g * 2 * area / se.RY_PER_A2_TO_J_PER_M2
             + n_C * ref.mu_c_bulk_ry + n_H * ref.mu_h_rich_ry)
        out.append(se.SlabEnergy(
            surface=surface, layers=n, run=f"bare~{surface}_{n}L", n_C=n_C,
            n_H=n_H, area_angstrom2=area, energy_ry=E,
            pseudo_C=ref.pseudo_C if match_ref else "c.upf",
            pseudo_H=ref.pseudo_H if match_ref else "h.upf",
            ecutwfc=ref.ecutwfc if match_ref else 90.0,
            ecutrho=ref.ecutrho if match_ref else 720.0,
            nspin_declared=(nspin or {}).get(n, 1),
            nspin_effective=(nspin or {}).get(n, 1),
            total_magnetization_bohr=(
                2.0 if (nspin or {}).get(n) == 2 else None)))
    return out


# A SYNTHETIC ladder with the SHAPE the bare (111) campaign showed: the two
# thinnest rungs sit outside the asymptotic regime because the two bare faces
# still interact, and the rest sit on a plateau.
#
# The plateau value is a fixture parameter, not a physical expectation. The
# real bare (111) energies are non-spin-polarised and therefore provisional
# (see the spin tests below), so no test here asserts a J/m^2 value for
# gamma_bare, a binding facet, or a delta_mu ceiling. What is asserted is that
# the machinery recovers whatever plateau it was handed, and refuses ladders it
# should refuse.
BARE_111_AREA = 5.5280
BARE_111_PLATEAU = 5.67
BARE_111_THIN_END = {6: -0.81, 8: -1.05}


def _bare_ladder_gammas(plateau=BARE_111_PLATEAU):
    """Unconverged 6L/8L, then a plateau at `plateau` for 10/12/16L."""
    return {**BARE_111_THIN_END,
            10: plateau + 0.01, 12: plateau + 0.01, 16: plateau}


def _bare_111(ref, exclude, reason="test", plateau=BARE_111_PLATEAU, **kw):
    pts = _ladder_from_per_slab_gammas(
        "C111", _bare_ladder_gammas(plateau), BARE_111_AREA, ref, **kw)
    return se.fit_surface_energy("C111", pts, ref, exclude, reason)


def test_gamma_bias_from_mu_c_drift_is_an_exact_identity():
    """The rejection criterion is not a heuristic: the bias is exactly the gap
    between the Boettger gamma and the mean per-slab gamma with the bulk mu_C."""
    ref = _fake_ref()
    for exclude in ([], [6], [6, 8]):
        f = _bare_111(ref, exclude)
        mean_per_slab = sum(f.per_slab_gamma_j_m2[n] for n in f.layers_used) \
            / len(f.layers_used)
        assert f.gamma_bias_from_mu_c_drift_j_m2 == pytest.approx(
            f.gamma_h_rich_j_m2 - mean_per_slab, abs=1e-9), exclude


def test_bare_111_with_the_thin_end_left_in_is_rejected_not_warned():
    """The failure this guard exists for. The unconverged 6L/8L points drag the
    fitted slope tens of mRy off bulk, which does not scatter gamma -- it moves
    it by many J/m^2, and to the WRONG SIGN."""
    ref = _fake_ref()
    bad = _bare_111(ref, [6])
    assert abs(bad.mu_c_drift_mry) > 10.0
    assert bad.gamma_h_rich_j_m2 < 0            # the artifact: negative gamma_bare
    assert abs(bad.gamma_bias_from_mu_c_drift_j_m2) > 1.0

    with pytest.raises(se.SurfaceEnergyError) as exc:
        se.grade_fits([bad], se.DEFAULT_MU_C_TOL_MRY,
                      se.DEFAULT_MU_C_REJECT_J_M2, "--bare-exclude-layers-for")
    msg = str(exc.value)
    assert "mu_C" in msg
    assert "biases gamma" in msg
    assert f"{se.DEFAULT_MU_C_REJECT_J_M2:.3f} J/m^2" in msg
    # It must say which thicknesses were in the fit and which were dropped.
    assert "8L" in msg and "16L" in msg
    assert "Excluded so far: 6L" in msg
    assert bad.rejected and bad.epistemic_level == "REJECTED"


def test_rejection_message_names_the_thicknesses_to_drop():
    """A rejection the user cannot act on is only marginally better than a
    warning. The per-slab gammas already identify the culprits."""
    ref = _fake_ref()
    with pytest.raises(se.SurfaceEnergyError) as exc:
        se.grade_fits([_bare_111(ref, [])], se.DEFAULT_MU_C_TOL_MRY,
                      se.DEFAULT_MU_C_REJECT_J_M2, "--bare-exclude-layers-for")
    assert "--bare-exclude-layers-for C111=6,8" in str(exc.value)


@pytest.mark.parametrize("plateau", [1.5, 5.67, 9.0])
def test_dropping_the_unconverged_thin_end_recovers_the_plateau(plateau):
    """Refit over the asymptotic rungs only: the slope comes back to bulk and
    gamma comes back to whatever plateau the ladder was built on.

    Parametrised deliberately. The assertion is that the machinery recovers the
    plateau it was handed -- not that bare (111) is any particular number. The
    real bare energies are non-spin-polarised and provisional."""
    ref = _fake_ref()
    good = _bare_111(ref, [6, 8], plateau=plateau)
    assert good.layers_used == [10, 12, 16]
    assert abs(good.mu_c_drift_mry) < 0.2
    # abs=0.05 because the fixture's plateau carries a 0.01 J/m^2 wobble, as
    # the real one does; a perfectly flat plateau would recover exactly.
    assert good.gamma_h_rich_j_m2 == pytest.approx(plateau, abs=0.05)
    # The residual wobble still trips the mRy warning tolerance and downgrades
    # to L1 -- which is the point of having two thresholds. It is nowhere near
    # the rejection budget, so the fit survives and is reported.
    se.grade_fits([good], se.DEFAULT_MU_C_TOL_MRY, se.DEFAULT_MU_C_REJECT_J_M2)
    assert not good.rejected
    assert abs(good.gamma_bias_from_mu_c_drift_j_m2) < se.DEFAULT_MU_C_REJECT_J_M2
    # The contaminated fit does not merely differ from this by noise: it lands
    # on the other side of zero, which is what made it bind as a ceiling.
    bad = _bare_111(ref, [6], plateau=plateau)
    assert bad.gamma_h_rich_j_m2 < 0 < good.gamma_h_rich_j_m2


def test_a_negative_gamma_bare_would_have_bound_the_ceiling():
    """Why the bad fit is worse than no fit: it does not merely add noise to
    the ceiling, it becomes the binding one and drags the bound negative."""
    h = _fit_with("C111", -0.9761, 2.8983)
    bad_bare = _fit_with("C111", -3.735, 0.0)
    good_bare = _fit_with("C111", +5.67, 0.0)
    for bare in (bad_bare, good_bare):
        bare.n_H = 0
    bad = se.dehydrogenation_bound([h], [bad_bare])
    good = se.dehydrogenation_bound([h], [good_bare])
    assert bad["delta_mu_max_ev"] < 0        # "already dehydrogenated at H-rich"
    assert good["delta_mu_max_ev"] > 2.0
    # ... and the two differ by far more than any margin downstream cares about.
    assert good["delta_mu_max_ev"] - bad["delta_mu_max_ev"] > 3.0


def test_a_rejected_bare_fit_can_never_reach_the_ceiling():
    """Silently dropping a rejected variant would raise the minimum over the
    survivors, making the ceiling too PERMISSIVE -- the one direction of error
    this bound must not fail in."""
    f = _fit_with("C111", -3.735, 0.0)
    f.n_H = 0
    f.rejected = True
    f.reject_reason = "mu_C drift"
    with pytest.raises(se.SurfaceEnergyError, match="rejected fit"):
        se.lowest_bare_gamma([f])


def test_a_loose_budget_still_catches_the_bare_111_failure():
    """The threshold is not balanced on a knife edge: the production ladders
    sit ~3 orders of magnitude below the failure."""
    ref = _fake_ref()
    bad = _bare_111(ref, [6])
    for budget in (0.1, 1.0):
        bad.rejected = False
        with pytest.raises(se.SurfaceEnergyError):
            se.grade_fits([bad], se.DEFAULT_MU_C_TOL_MRY, budget)


# ============================================== per-surface layer exclusion
def test_parse_layer_exclusions_round_trips():
    assert se.parse_layer_exclusions(["C111=6,8", "C100=6"]) == {
        "C111": [6, 8], "C100": [6]}
    assert se.parse_layer_exclusions(["C111=6L,8L"]) == {"C111": [6, 8]}
    assert se.parse_layer_exclusions(["C110="]) == {"C110": []}
    assert se.parse_layer_exclusions([]) == {}
    assert se.parse_layer_exclusions(None) == {}


def test_parse_layer_exclusions_refuses_ambiguity():
    with pytest.raises(se.SurfaceEnergyError, match="SURFACE=N"):
        se.parse_layer_exclusions(["C111"])
    with pytest.raises(se.SurfaceEnergyError, match="not a layer count"):
        se.parse_layer_exclusions(["C111=six"])
    with pytest.raises(se.SurfaceEnergyError, match="twice"):
        se.parse_layer_exclusions(["C111=6", "C111=8"])


def test_per_surface_exclusion_overrides_the_default_and_does_not_leak():
    per = {"C111": [6, 8]}
    layers, why = se.resolve_exclusions("C111", [6], per)
    assert layers == [6, 8] and "per-surface" in why and "C111" in why
    # A surface with no override keeps the campaign default, unchanged.
    layers, why = se.resolve_exclusions("C110", [6], per)
    assert layers == [6] and "default" in why
    # And an empty override really means "keep everything", not "use default".
    layers, why = se.resolve_exclusions("C110", [6], {"C110": []})
    assert layers == []


def test_a_misdirected_exclusion_is_refused_not_ignored():
    """Excluding a thickness the ladder does not have leaves the bad point in
    the fit while the report claims it was dropped."""
    ref = _fake_ref()
    pts = _ladder_from_per_slab_gammas("C111", _bare_ladder_gammas(),
                                       BARE_111_AREA, ref)
    with pytest.raises(se.SurfaceEnergyError, match="which the ladder does not"):
        se.fit_surface_energy("C111", pts, ref, [7])


def test_exclusion_reason_travels_with_the_fit():
    ref = _fake_ref()
    f = _bare_111(ref, [6, 8], "explicit per-surface exclusion for C111")
    assert f.layers_excluded == [6, 8]
    assert f.layers_used == [10, 12, 16]
    assert "C111" in f.exclusion_reason


@production
def test_production_exclusions_are_reported_per_surface(derived):
    result, out = derived
    meta = result["meta"]
    assert set(meta["exclusions"]) == {"C100", "C110", "C111"}
    for s, e in meta["exclusions"].items():
        assert e["excluded"] == [6]
        assert e["used"] == [8, 10, 12, 16]
        assert e["reason"]
    report = (out / "surface_energy_report.md").read_text()
    assert "Which thicknesses entered each fit" in report
    assert "gamma bias from mu_C drift" in report
    cfg = json.loads((out / "surface_energies_h.json").read_text())
    per = cfg["method"]["excluded_layers_per_surface"]
    assert per["C100"]["excluded"] == [6] and per["C100"]["used"] == [8, 10, 12, 16]


@production
def test_production_h_ladders_are_warned_about_but_never_rejected(derived):
    """(100) drifts enough to lose L2 and nowhere near enough to be rejected.
    If this ever flips, the budget moved, not the physics."""
    result, _ = derived
    for f in result["fits"]:
        assert not f.rejected
        assert abs(f.gamma_bias_from_mu_c_drift_j_m2) < se.DEFAULT_MU_C_REJECT_J_M2
    fits = {f.surface: f for f in result["fits"]}
    # The bias is what the budget is spent on; (100) uses about a tenth of it.
    assert fits["C100"].gamma_bias_from_mu_c_drift_j_m2 == pytest.approx(
        0.0103, abs=5e-4)
    for s in ("C110", "C111"):
        assert abs(fits[s].gamma_bias_from_mu_c_drift_j_m2) < 0.001


@production
def test_run_wires_the_bare_ladder_through_with_its_own_exclusions(
        tmp_path, monkeypatch):
    """End to end on the path that failed: a bare (111) ladder whose thin end
    is unconverged must be rejected under the H ladder's policy, and must give
    a physical ceiling once its own policy drops 6L and 8L."""
    ref_probe = se.load_references(REFERENCE)
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()

    real_loader = se.load_slab_energies

    def fake_loader(runs_dir, prefix):
        if str(runs_dir) != str(bare_dir):
            return real_loader(runs_dir, prefix)
        return {"C111": _ladder_from_per_slab_gammas(
            "C111", _bare_ladder_gammas(), BARE_111_AREA, ref_probe,
            match_ref=True)}

    monkeypatch.setattr(se, "load_slab_energies", fake_loader)

    def go(**kw):
        return se.run(PRODUCTION, REFERENCE, tmp_path / "out",
                      se.DEFAULT_PREFIX, [6], (0.0, 3.0), 21,
                      se.DEFAULT_MU_C_TOL_MRY, bare_runs_dir=bare_dir, **kw)

    # Inheriting the H ladder's policy (drop 6L only) leaves 8L in and the
    # bare fit is rejected rather than quietly becoming the binding ceiling.
    with pytest.raises(se.SurfaceEnergyError, match="biases gamma"):
        go()

    result = go(bare_exclude_layers_for={"C111": [6, 8]})
    bound = result["dehydrogenation_bound"]
    assert bound["available"]
    e = bound["per_surface"]["C111"]
    assert e["bare_layers_excluded"] == [6, 8]
    assert e["bare_layers_used"] == [10, 12, 16]
    # The ceiling is the algebra applied to whatever gamma_bare came out --
    # no J/m^2 or eV value is asserted, because the bare energies behind the
    # real campaign are non-spin-polarised and provisional.
    assert e["delta_mu_max_ev"] == pytest.approx(
        (e["gamma_bare_j_m2"] - e["gamma_h_rich_j_m2"]) / e["slope_j_m2_per_ev"])
    # The H ladders keep their own policy; the bare override does not leak.
    assert all(f.layers_excluded == [6] for f in result["fits"])
    assert result["meta"]["bare_exclusions"]["C111"]["excluded"] == [6, 8]

    report = (tmp_path / "out" / "surface_energy_report.md").read_text()
    assert "Binding facet C111" in report
    assert "10L 12L 16L" in report
    # The "unbounded above" caveat must not survive alongside a real ceiling.
    assert "ASSUMED to remain the stable termination at every mu_H" not in report
    # ... but the spin caveat must, because this ladder is non-spin-polarised.
    assert "nspin=1, UNTESTED" in report
    assert "too PERMISSIVE" in report
    assert any("NO spin-polarised run exists" in w
               for w in result["meta"]["warnings"])


# ==================================================== bare-facet spin state
# A bare face carries one unpaired electron per dangling bond. nspin = 2 is
# variational over nspin = 1, so a non-spin-polarised bare energy is an UPPER
# bound, gamma_bare is an upper bound, and the ceiling built from it is too
# PERMISSIVE -- it claims the H-terminated surface survives to
# hydrogen-poorer conditions than it does. That is the one direction of error
# this bound must not fail in, and no fit diagnostic can see it.
def _bare_points(surface, ref, nspin, layers=(10, 12, 16), plateau=5.67):
    return _ladder_from_per_slab_gammas(
        surface, {n: plateau for n in layers}, BARE_111_AREA, ref,
        nspin={n: nspin.get(n, 1) for n in layers} if isinstance(nspin, dict)
        else {n: nspin for n in layers})


def test_pw_in_and_pw_out_spin_fields_are_read(tmp_path):
    """nspin from the input; the magnetization from the OUTPUT, which is what
    actually ran (CLAUDE.md invariant 7)."""
    import parse_slab

    (tmp_path / "a.in").write_text(
        "&system\n nat=4, ntyp=1, ecutwfc=90, ecutrho=720,\n"
        " nspin=2, starting_magnetization(1)=0.5,\n/\n")
    pin = parse_slab.parse_pw_in(tmp_path / "a.in")
    assert pin["nspin"] == 2 and pin["starting_magnetization"] is True

    # QE defaults nspin to 1 when the card is absent; "absent" and "declared
    # closed-shell" must read identically.
    (tmp_path / "b.in").write_text("&system\n nat=4, ecutwfc=90,\n/\n")
    assert parse_slab.parse_pw_in(tmp_path / "b.in")["nspin"] == 1
    assert parse_slab.parse_pw_in(tmp_path / "b.in")["starting_magnetization"] is False

    (tmp_path / "a.out").write_text(
        "     total magnetization       =     2.00 Bohr mag/cell\n"
        "     absolute magnetization    =     2.14 Bohr mag/cell\n"
        "!    total energy              =    -100.5 Ry\n"
        "     JOB DONE.\n")
    pout = parse_slab.parse_pw_out(tmp_path / "a.out")
    assert pout["nspin_effective"] == 2
    assert pout["total_magnetization_bohr"] == pytest.approx(2.00)
    assert pout["absolute_magnetization_bohr"] == pytest.approx(2.14)

    (tmp_path / "b.out").write_text(
        "!    total energy              =    -100.5 Ry\n     JOB DONE.\n")
    pout = parse_slab.parse_pw_out(tmp_path / "b.out")
    assert pout["nspin_effective"] == 1
    assert pout["total_magnetization_bohr"] is None

    # An incomplete run must not be claimed as closed-shell -- it may simply
    # not have got far enough to print a magnetization.
    (tmp_path / "c.out").write_text("     Program PWSCF starts\n")
    assert parse_slab.parse_pw_out(tmp_path / "c.out")["nspin_effective"] is None


def test_spin_polarised_is_taken_from_the_output_not_the_input():
    ref = _fake_ref()
    declared_only = se.SlabEnergy(
        **{**_bare_points("C111", ref, 1)[0].__dict__,
           "nspin_declared": 2, "nspin_effective": 1})
    assert not declared_only.spin_polarised
    ran = se.SlabEnergy(**{**declared_only.__dict__, "nspin_effective": 2})
    assert ran.spin_polarised


def test_a_facet_with_a_mixed_spin_ladder_is_refused():
    """Once a spin-polarised result exists, the non-polarised energies on that
    facet are not unvalidated -- they are known to be the wrong state."""
    ref = _fake_ref()
    pts = _bare_points("C111", ref, {10: 1, 12: 1, 16: 2})
    with pytest.raises(se.SurfaceEnergyError) as exc:
        se.check_bare_spin_states({"C111": pts})
    msg = str(exc.value)
    assert "spin-polarised result exists" in msg
    assert "C111_16L" in msg and "C111_10L" in msg
    assert "2.00" in msg                       # the magnetization is quoted
    assert "--bare-exclude-layers-for" in msg  # and the way out is named


def test_a_reconstruction_variant_counts_as_evidence_about_its_facet():
    """C111pandey is the same dangling-bond chemistry as C111; a spin-polarised
    result on one is evidence about the other."""
    ref = _fake_ref()
    with pytest.raises(se.SurfaceEnergyError, match="spin-polarised result exists"):
        se.check_bare_spin_states({
            "C111": _bare_points("C111", ref, 1),
            "C111pandey": _bare_points("C111pandey", ref, 2,
                                       layers=(20, 22, 24))})


def test_a_wholly_unpolarised_facet_warns_loudly_rather_than_refusing():
    """Refusing outright would leave no ceiling at all, which is worse than a
    labelled one. The warning has to name the direction of the error."""
    ref = _fake_ref()
    warns = se.check_bare_spin_states({"C111": _bare_points("C111", ref, 1)})
    assert len(warns) == 1
    assert "NO spin-polarised run exists" in warns[0]
    assert "too PERMISSIVE" in warns[0]
    assert "C111" in warns[0]


def test_a_fully_polarised_facet_passes_without_comment():
    ref = _fake_ref()
    assert se.check_bare_spin_states({"C111": _bare_points("C111", ref, 2)}) == []


def test_declared_nspin_2_that_did_not_run_polarised_is_refused():
    """The input says one calculation, the output shows another."""
    ref = _fake_ref()
    pts = [se.SlabEnergy(**{**p.__dict__, "nspin_declared": 2,
                            "nspin_effective": 1,
                            "total_magnetization_bohr": None})
           for p in _bare_points("C111", ref, 1)]
    with pytest.raises(se.SurfaceEnergyError) as exc:
        se.check_bare_spin_states({"C111": pts})
    assert "declare nspin = 2" in str(exc.value)
    assert "invariant 7" in str(exc.value)


def test_an_unexpected_magnetization_warns():
    """2.00 Bohr mag/cell is one unpaired electron per dangling bond on each of
    two faces. Something else means the structure or the state is not what was
    assumed -- (100) quenching by dimerisation would read as 0.00, which is a
    legitimate answer, not an anomaly."""
    ref = _fake_ref()
    pts = _bare_points("C111", ref, 2)
    odd = [se.SlabEnergy(**{**p.__dict__, "total_magnetization_bohr": 1.37})
           for p in pts]
    warns = se.check_bare_spin_states({"C111": odd})
    assert len(warns) == 1 and "not the 2.0 expected" in warns[0]

    quenched = [se.SlabEnergy(**{**p.__dict__, "total_magnetization_bohr": 0.0})
                for p in pts]
    assert se.check_bare_spin_states({"C100": quenched}) == []


def test_spin_audit_reports_per_facet_not_per_variant():
    ref = _fake_ref()
    audit = se.spin_audit({
        "C111": _bare_points("C111", ref, 2),
        "C111pandey": _bare_points("C111pandey", ref, 2, layers=(20, 22, 24)),
        "C100": _bare_points("C100", ref, 1)})
    assert set(audit) == {"C111", "C100"}
    assert audit["C111"]["variants"] == ["C111", "C111pandey"]
    assert audit["C111"]["has_polarised_reference"]
    assert not audit["C111"]["mixed"]
    assert not audit["C100"]["has_polarised_reference"]
    assert len(audit["C111"]["polarised"]) == 6


# ============================================ E_s and the ceiling correction
def _spin_pair(surface, ref, layers, e_s_ev, n_db, mag, area=BARE_111_AREA,
               plateau=5.67, per_layer=1):
    """A non-polarised ladder and its polarised counterpart, built so that
    E(nspin=1) - E(nspin=2) = n_db * e_s_ev on every rung."""
    nsp = _ladder_from_per_slab_gammas(
        surface, {n: plateau for n in layers}, area, ref,
        per_layer=per_layer, nspin={n: 1 for n in layers})
    sp = [se.SlabEnergy(**{**p.__dict__,
                           "energy_ry": p.energy_ry - n_db * e_s_ev / se.RY_TO_EV,
                           "nspin_declared": 2, "nspin_effective": 2,
                           "total_magnetization_bohr": mag})
          for p in nsp]
    return nsp, sp


def test_e_s_is_the_pair_difference_over_the_dangling_bond_count():
    ref = _fake_ref()
    nsp, sp = _spin_pair("C111", ref, (12, 16), e_s_ev=0.31, n_db=2, mag=2.0)
    st = se.measure_spin_stabilisation(nsp, sp, 2)
    assert st.thicknesses == [12, 16]
    assert st.e_s_ev == pytest.approx(0.31, abs=1e-9)
    assert st.spread_ev == pytest.approx(0.0, abs=1e-9)
    assert st.identity_applies and not st.quenched


def test_the_ceiling_drops_by_exactly_e_s_on_every_facet():
    """The identity. Area, coverage and thickness all cancel, so a facet with
    twice the area and twice the coverage takes the SAME shift in eV."""
    E_S = 0.31
    facets = {                       # surface: (area, N_H per cell)
        "C100": (12.7663, 4), "C110": (9.0271, 4), "C111": (5.5280, 2)}
    shifts, d_gammas = {}, {}
    for s, (area, n_H) in facets.items():
        slope = n_H / (2 * area) * se.EV_PER_A2_TO_J_PER_M2
        h = _fit_with(s, -0.5, slope)
        n_db = n_H                                  # one H caps one dangling bond
        # gamma_bare falls by n_db*E_s over both faces, i.e. per face over 2A.
        d_gamma = n_db * E_S / (2 * area) * se.EV_PER_A2_TO_J_PER_M2
        d_gammas[s] = d_gamma
        bare_hi = _fit_with(s, 5.67, 0.0)
        bare_lo = _fit_with(s, 5.67 - d_gamma, 0.0)
        for b in (bare_hi, bare_lo):
            b.n_H = 0
        hi = se.dehydrogenation_bound([h], [bare_hi])["per_surface"][s]
        lo = se.dehydrogenation_bound([h], [bare_lo])["per_surface"][s]
        shifts[s] = hi["delta_mu_max_ev"] - lo["delta_mu_max_ev"]
    # The gamma_bare shifts differ between facets...
    assert len({round(v, 4) for v in d_gammas.values()}) == 3
    # ... and the ceiling shifts do not: every one is E_s.
    for s, v in shifts.items():
        assert v == pytest.approx(E_S, abs=1e-9), s


def test_apply_spin_correction_reports_both_ceilings():
    ref = _fake_ref()
    h = _fit_with("C111", -0.9761, 2.8983)
    bare = _fit_with("C111", 5.67, 0.0)
    bare.n_H = 0
    bound = se.dehydrogenation_bound([h], [bare])
    before = bound["per_surface"]["C111"]["delta_mu_max_ev"]
    nsp, sp = _spin_pair("C111", ref, (12, 16), e_s_ev=0.31, n_db=2, mag=2.0)
    st = se.measure_spin_stabilisation(nsp, sp, 2)
    se.apply_spin_correction(bound, {"C111": st})
    e = bound["per_surface"]["C111"]
    assert e["spin_correction_applied"]
    assert e["delta_mu_max_uncorrected_ev"] == pytest.approx(before)
    assert e["delta_mu_max_ev"] == pytest.approx(before - 0.31)
    assert bound["delta_mu_max_uncorrected_ev"] == pytest.approx(before)
    assert bound["delta_mu_max_ev"] == pytest.approx(before - 0.31)


def test_a_quenched_face_takes_no_correction():
    """M = 0 means the dangling bonds are gone -- the expected answer for a
    dimerised (100), and the decisive one for (110)."""
    ref = _fake_ref()
    nsp, sp = _spin_pair("C100", ref, (12, 16), e_s_ev=0.0, n_db=4, mag=0.0,
                         area=12.7663, per_layer=2)
    st = se.measure_spin_stabilisation(nsp, sp, 4)
    assert st.quenched and not st.identity_applies
    assert st.e_s_ev == pytest.approx(0.0, abs=1e-9)
    assert "quenched" in st.reason

    h = _fit_with("C100", 0.0501, 2.5100)
    bare = _fit_with("C100", 4.5, 0.0)
    bare.n_H = 0
    bound = se.dehydrogenation_bound([h], [bare])
    before = bound["per_surface"]["C100"]["delta_mu_max_ev"]
    se.apply_spin_correction(bound, {"C100": st})
    assert bound["per_surface"]["C100"]["delta_mu_max_ev"] == pytest.approx(before)


def test_a_partially_quenched_face_reports_e_s_but_withholds_it():
    """Between 0 and n_DB the cancellation is no longer exact, so the number is
    shown and not used."""
    ref = _fake_ref()
    nsp, sp = _spin_pair("C110", ref, (12, 16), e_s_ev=0.25, n_db=4, mag=1.7,
                         area=9.0271, per_layer=2)
    st = se.measure_spin_stabilisation(nsp, sp, 4)
    assert not st.identity_applies and not st.quenched
    assert st.e_s_ev == pytest.approx(0.25, abs=1e-9)
    assert "PARTIALLY" in st.reason

    h = _fit_with("C110", -0.8265, 3.5497)
    bare = _fit_with("C110", 5.162, 0.0)
    bare.n_H = 0
    bound = se.dehydrogenation_bound([h], [bare])
    before = bound["per_surface"]["C110"]["delta_mu_max_ev"]
    se.apply_spin_correction(bound, {"C110": st})
    e = bound["per_surface"]["C110"]
    assert not e["spin_correction_applied"]
    assert e["delta_mu_max_ev"] == pytest.approx(before)
    assert e["e_s_ev"] == pytest.approx(0.25)


def test_e_s_refuses_to_pair_across_thicknesses():
    """A cross-thickness difference folds the bulk term back in -- exactly what
    the pairing exists to cancel."""
    ref = _fake_ref()
    nsp, _ = _spin_pair("C111", ref, (10, 12), e_s_ev=0.3, n_db=2, mag=2.0)
    _, sp = _spin_pair("C111", ref, (16, 20), e_s_ev=0.3, n_db=2, mag=2.0)
    with pytest.raises(se.SurfaceEnergyError, match="cannot be taken across"):
        se.measure_spin_stabilisation(nsp, sp, 2)


def test_a_polarised_run_above_its_unpolarised_partner_is_refused():
    """nspin=2 is variational over nspin=1; the reverse cannot happen."""
    ref = _fake_ref()
    nsp, sp = _spin_pair("C111", ref, (12, 16), e_s_ev=-0.2, n_db=2, mag=2.0)
    with pytest.raises(se.SurfaceEnergyError, match="variational"):
        se.measure_spin_stabilisation(nsp, sp, 2)


def test_a_mismatched_pair_is_refused():
    ref = _fake_ref()
    nsp, sp = _spin_pair("C111", ref, (12, 16), e_s_ev=0.3, n_db=2, mag=2.0)
    sp = [se.SlabEnergy(**{**sp[0].__dict__, "n_C": 99})] + sp[1:]
    with pytest.raises(se.SurfaceEnergyError, match="differ in cell"):
        se.measure_spin_stabilisation(nsp, sp, 2)


@production
def test_run_measures_e_s_and_corrects_the_ceiling(tmp_path, monkeypatch):
    """End to end: a non-polarised bare ladder plus a polarised pair at two
    thicknesses gives a corrected ceiling exactly E_s below the raw one."""
    ref_probe = se.load_references(REFERENCE)
    bare_dir, spin_dir = tmp_path / "bare", tmp_path / "spin"
    bare_dir.mkdir()
    spin_dir.mkdir()
    E_S = 0.31

    nsp = _ladder_from_per_slab_gammas(
        "C111", {10: 5.68, 12: 5.68, 16: 5.67}, BARE_111_AREA, ref_probe,
        match_ref=True, nspin={10: 1, 12: 1, 16: 1})
    sp = [se.SlabEnergy(**{**p.__dict__,
                           "energy_ry": p.energy_ry - 2 * E_S / se.RY_TO_EV,
                           "nspin_declared": 2, "nspin_effective": 2,
                           "total_magnetization_bohr": 2.0})
          for p in nsp if p.layers in (12, 16)]

    real_loader = se.load_slab_energies

    def fake_loader(runs_dir, prefix):
        if str(runs_dir) == str(bare_dir):
            return {"C111": nsp}
        if str(runs_dir) == str(spin_dir):
            return {"C111": sp}
        return real_loader(runs_dir, prefix)

    monkeypatch.setattr(se, "load_slab_energies", fake_loader)
    result = se.run(PRODUCTION, REFERENCE, tmp_path / "out", se.DEFAULT_PREFIX,
                    [6], (0.0, 3.0), 21, se.DEFAULT_MU_C_TOL_MRY,
                    bare_runs_dir=bare_dir, bare_spin_runs_dir=spin_dir,
                    bare_exclude_layers=[])
    bound = result["dehydrogenation_bound"]
    e = bound["per_surface"]["C111"]
    assert e["spin_correction_applied"]
    assert e["e_s_ev"] == pytest.approx(E_S, abs=1e-6)
    assert e["e_s_n_db_per_cell"] == 2          # read from the H facet's N_H
    assert e["e_s_thicknesses"] == [12, 16]
    assert e["delta_mu_max_ev"] == pytest.approx(
        e["delta_mu_max_uncorrected_ev"] - E_S, abs=1e-6)

    report = (tmp_path / "out" / "surface_energy_report.md").read_text()
    assert "d(delta_mu_max) = E_s" in report
    assert "unreconstructed 1x1" in report
    assert "Pandey" in report and "dimerised (100)" in report
    assert "ceiling before (eV)" in report
    # The "too permissive, untested" caveat must go once E_s is measured.
    assert "nspin=1, UNTESTED" not in report
