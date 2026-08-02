"""Tests for fit_tau_infinity.py — the tau thickness extrapolation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import fit_tau_infinity as fti

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTION = os.path.join(REPO, "results", "production")

A0 = 3.572997470394866


def make_point(layers, sigma_xx, sigma_yy, Lz, thickness, area=10.0,
               n_C=None, surface="C111"):
    return fti.StressPoint(
        surface=surface, layers=layers, run=f"x~{surface}_{layers}L_stress_scf",
        n_C=n_C if n_C is not None else layers, n_H=2,
        area_angstrom2=area, Lz_angstrom=Lz,
        thickness_angstrom=thickness, thickness_geometric_angstrom=thickness - 1.5,
        sigma_xx_kbar=sigma_xx, sigma_yy_kbar=sigma_yy,
        pseudopotentials="C:c.upf; H:h.upf")


def synthetic_ladder(tau_xx, tau_yy, sigma_res, thicknesses, Lz_of_t=None):
    """Points that satisfy sigma*Lz = 2*tau + sigma_res*t exactly."""
    pts = []
    for i, t in enumerate(thicknesses):
        Lz = (Lz_of_t or (lambda x: x + 8.0))(t)
        y_xx = 2.0 * tau_xx / fti.KBAR_ANGSTROM_TO_N_PER_M + sigma_res * t
        y_yy = 2.0 * tau_yy / fti.KBAR_ANGSTROM_TO_N_PER_M + sigma_res * t
        pts.append(make_point(6 + 2 * i, y_xx / Lz, y_yy / Lz, Lz, t))
    return pts


# ------------------------------------------------------------ unit conversion
def test_bulk_volume_per_carbon_is_an_eighth_of_the_cubic_cell():
    assert fti.bulk_volume_per_carbon(A0) == pytest.approx(A0 ** 3 / 8.0)
    # a 3.573 A cell holding 8 atoms -> ~5.70 A^3 per carbon
    assert fti.bulk_volume_per_carbon(A0) == pytest.approx(5.70175, abs=1e-4)


def test_tau_conversion_matches_the_claude_md_anchor():
    """CLAUDE.md sec 2: sigma = 3.19 kbar with Lz ~ 14.8 A gives tau = 0.236 N/m.

    A dropped or inverted factor of 2 yields plausible numbers, so this pins it.
    """
    sigma, Lz = 3.19, 14.8
    tau = fti.KBAR_ANGSTROM_TO_N_PER_M * sigma * Lz / 2.0
    assert tau == pytest.approx(0.236, abs=5e-4)


# -------------------------------------------------------------------- the fit
def test_fit_recovers_known_tau_and_sigma_res_exactly():
    pts = synthetic_ladder(tau_xx=1.25, tau_yy=-3.5, sigma_res=-0.42,
                           thicknesses=[7.0, 9.0, 11.0, 14.0])
    fxx = fti.fit_component(pts, "xx")
    fyy = fti.fit_component(pts, "yy")
    assert fxx.tau_inf_n_per_m == pytest.approx(1.25, abs=1e-9)
    assert fyy.tau_inf_n_per_m == pytest.approx(-3.5, abs=1e-9)
    assert fxx.sigma_res_kbar == pytest.approx(-0.42, abs=1e-9)
    assert fxx.rms_kbar_angstrom == pytest.approx(0.0, abs=1e-9)


def test_thickness_offset_moves_the_intercept_but_not_the_slope():
    """The documented claim: choosing t geometrically instead of by atom count
    shifts tau_inf but leaves sigma_res untouched, because the two definitions
    differ by an additive constant."""
    offset = 1.37
    base = synthetic_ladder(2.0, 2.0, -0.5, [7.0, 9.0, 11.0, 14.0])
    shifted = [make_point(p.layers, p.sigma_xx_kbar, p.sigma_yy_kbar,
                          p.Lz_angstrom, p.thickness_angstrom - offset)
               for p in base]
    f0 = fti.fit_component(base, "xx")
    f1 = fti.fit_component(shifted, "xx")
    assert f1.sigma_res_kbar == pytest.approx(f0.sigma_res_kbar, abs=1e-9)
    expected_shift = fti.KBAR_ANGSTROM_TO_N_PER_M * f0.sigma_res_kbar * offset / 2.0
    assert f1.tau_inf_n_per_m == pytest.approx(
        f0.tau_inf_n_per_m + expected_shift, abs=1e-9)
    assert f1.tau_inf_n_per_m != pytest.approx(f0.tau_inf_n_per_m, abs=1e-6)


def test_fit_refuses_too_few_points():
    pts = synthetic_ladder(1.0, 1.0, -0.3, [7.0, 9.0])
    with pytest.raises(fti.TauFitError, match="too few"):
        fti.fit_component(pts, "xx")


def test_excluded_layers_are_kept_out_of_the_fit_and_still_reported():
    pts = synthetic_ladder(1.0, 1.0, -0.3, [5.0, 7.0, 9.0, 11.0, 14.0])
    # Corrupt the first (6L) point so its inclusion would be obvious.
    pts[0] = make_point(6, 99.0, 99.0, pts[0].Lz_angstrom, pts[0].thickness_angstrom)
    fit = fti.fit_surface("C111", pts, exclude_layers=[6])
    assert 6 not in fit.layers_used
    assert fit.layers_excluded == [6]
    assert fit.fit_xx.tau_inf_n_per_m == pytest.approx(1.0, abs=1e-9)
    # The exclusion is quantified rather than silent.
    assert abs(fit.excluded_residuals[6]["xx"]) > 1.0


def test_tau_mean_and_aniso_definitions():
    pts = synthetic_ladder(1.0, -5.0, -0.3, [7.0, 9.0, 11.0, 14.0])
    fit = fti.fit_surface("C100", pts, exclude_layers=[])
    assert fit.tau_mean_n_per_m == pytest.approx(-2.0, abs=1e-9)
    assert fit.tau_aniso_n_per_m == pytest.approx(6.0, abs=1e-9)


# ------------------------------------------------------- sigma_res consistency
def _fit_with_sigma_res(surface, sigma_res):
    pts = synthetic_ladder(1.0, 1.0, sigma_res, [7.0, 9.0, 11.0, 14.0])
    pts = [make_point(p.layers, p.sigma_xx_kbar, p.sigma_yy_kbar, p.Lz_angstrom,
                      p.thickness_angstrom, surface=surface) for p in pts]
    return fti.fit_surface(surface, pts, exclude_layers=[])


def test_consistent_sigma_res_produces_no_warning():
    fits = [_fit_with_sigma_res(s, r) for s, r in
            (("C100", -0.28), ("C110", -0.26), ("C111", -0.36))]
    assert fti.check_sigma_res_consistency(fits, 0.3) == []


def test_inconsistent_sigma_res_is_warned_about():
    fits = [_fit_with_sigma_res(s, r) for s, r in
            (("C100", -0.28), ("C110", -0.26), ("C111", -2.60))]
    warnings = fti.check_sigma_res_consistency(fits, 0.3)
    assert warnings and "sigma_res differs" in warnings[0]
    assert "lattice constant" in warnings[0]


# ---------------------------------------------------------------- invariants
def test_non_orthogonal_cell_is_rejected():
    """Lz and the in-plane sigma components are only meaningful if c is normal
    to the surface plane."""
    cell = [[2.5, 0, 0], [0, 2.5, 0], [0, 1.0, 20.0]]
    with pytest.raises(fti.TauFitError, match="not perpendicular"):
        fti._cell_invariants(cell, "fake_run")


def test_orthogonal_cell_gives_area_and_height():
    area, Lz = fti._cell_invariants([[2.0, 0, 0], [0, 3.0, 0], [0, 0, 20.0]], "r")
    assert area == pytest.approx(6.0)
    assert Lz == pytest.approx(20.0)


def test_ladder_with_varying_area_is_rejected():
    pts = synthetic_ladder(1.0, 1.0, -0.3, [7.0, 9.0, 11.0])
    pts[1] = make_point(pts[1].layers, 1.0, 1.0, pts[1].Lz_angstrom,
                        pts[1].thickness_angstrom, area=99.0)
    with pytest.raises(fti.TauFitError, match="cell area differs"):
        fti._check_ladder_consistency("C111", pts)


def test_ladder_with_inconsistent_layer_stoichiometry_is_rejected():
    """CLAUDE.md invariant 3: the folder name must describe the geometry."""
    pts = [make_point(8, 1.0, 1.0, 16.0, 8.0, n_C=8),
           make_point(10, 1.0, 1.0, 18.0, 10.0, n_C=10),
           make_point(12, 1.0, 1.0, 20.0, 12.0, n_C=36)]
    with pytest.raises(fti.TauFitError, match="atoms per layer differs"):
        fti._check_ladder_consistency("C111", pts)


def test_ladder_with_changing_termination_is_rejected():
    pts = synthetic_ladder(1.0, 1.0, -0.3, [7.0, 9.0, 11.0])
    bad = pts[1]
    pts[1] = fti.StressPoint(
        surface=bad.surface, layers=bad.layers, run=bad.run, n_C=bad.n_C,
        n_H=8, area_angstrom2=bad.area_angstrom2, Lz_angstrom=bad.Lz_angstrom,
        thickness_angstrom=bad.thickness_angstrom,
        thickness_geometric_angstrom=bad.thickness_geometric_angstrom,
        sigma_xx_kbar=bad.sigma_xx_kbar, sigma_yy_kbar=bad.sigma_yy_kbar,
        pseudopotentials=bad.pseudopotentials)
    with pytest.raises(fti.TauFitError, match="hydrogen count differs"):
        fti._check_ladder_consistency("C111", pts)


# -------------------------------------------------------- production regression
production = pytest.mark.skipif(
    not os.path.isdir(os.path.join(PRODUCTION, "thick_a0corr_stress~C111_16L_stress_scf")),
    reason="production stress SCFs not present")


@production
def test_production_ladder_reproduces_the_recorded_baseline(tmp_path):
    """Guards the numbers the rest of the project quotes.

    Two checks with different purposes:

    * a loose cross-check against the recorded production baseline (commit
      03fec70): (100) +1.04/-4.99, (110) +2.15/+4.39, (111) +0.485/+0.485 N/m.
      Those were computed with a GEOMETRIC slab thickness, so exact agreement
      is not expected; agreement to a few hundredths of an N/m is what confirms
      the thickness convention only moves the intercept slightly, as documented.
    * a tight regression lock on the atomic-thickness values this module
      actually produces, so a refactor cannot move them unnoticed.
    """
    result = fti.run(PRODUCTION, tmp_path, fti.DEFAULT_PREFIX, [6],
                     os.path.join(REPO, "config", "reference_pbe_sssp.json"),
                     fti.DEFAULT_SIGMA_RES_TOL_KBAR)
    fits = result["fits"]
    assert set(fits) == {"C100", "C110", "C111"}

    recorded = {"C100": (1.04, -4.99), "C110": (2.15, 4.39), "C111": (0.485, 0.485)}
    for surface, (xx, yy) in recorded.items():
        assert fits[surface].fit_xx.tau_inf_n_per_m == pytest.approx(xx, abs=0.05)
        assert fits[surface].fit_yy.tau_inf_n_per_m == pytest.approx(yy, abs=0.05)

    atomic = {"C100": (1.0323, -5.0241), "C110": (2.1445, 4.3975),
              "C111": (0.4833, 0.4833)}
    for surface, (xx, yy) in atomic.items():
        assert fits[surface].fit_xx.tau_inf_n_per_m == pytest.approx(xx, abs=5e-4)
        assert fits[surface].fit_yy.tau_inf_n_per_m == pytest.approx(yy, abs=5e-4)

    # (111) is in-plane isotropic by 3-fold symmetry: not approximately, exactly.
    assert fits["C111"].tau_aniso_n_per_m == pytest.approx(0.0, abs=1e-12)

    # sigma_res is a bulk property: same sign, small, and consistent.
    for f in fits.values():
        assert -0.6 < f.sigma_res_mean_kbar < 0.0
    assert result["warnings"] == []

    assert all(f.layers_used == [8, 10, 12, 16] for f in fits.values())
    assert all(f.layers_excluded == [6] for f in fits.values())


@production
def test_production_run_writes_all_outputs(tmp_path):
    fti.run(PRODUCTION, tmp_path, fti.DEFAULT_PREFIX, [6],
            os.path.join(REPO, "config", "reference_pbe_sssp.json"),
            fti.DEFAULT_SIGMA_RES_TOL_KBAR)
    for name in ("tau_infinity.csv", "tau_infinity_points.csv",
                 "tau_infinity_meta.json", "tau_infinity_report.md"):
        assert (tmp_path / name).exists(), name
    text = (tmp_path / "tau_infinity_report.md").read_text()
    assert "6L" in text and "Excluded" in text          # the exclusion is stated
    assert "L2" in text                                  # epistemic level stated
    assert "compressed" in text                          # sign convention stated


@production
def test_atomic_and_geometric_thickness_agree_on_sigma_res_in_real_data():
    """The comment in the module claims the thickness convention moves only the
    intercept. Check it against the real ladder, not just synthetic data."""
    omega = fti.bulk_volume_per_carbon(A0)
    by_surface = fti.discover_points(PRODUCTION, fti.DEFAULT_PREFIX, omega)
    for surface, pts in by_surface.items():
        used = [p for p in pts if p.layers != 6]
        atomic = fti.fit_component(used, "xx")
        geometric_pts = [make_point(p.layers, p.sigma_xx_kbar, p.sigma_yy_kbar,
                                    p.Lz_angstrom, p.thickness_geometric_angstrom)
                         for p in used]
        geometric = fti.fit_component(geometric_pts, "xx")
        # Slopes agree closely; the offset between the two definitions is not
        # perfectly constant across the ladder, but it is close enough that
        # sigma_res is unaffected at the reported precision.
        assert geometric.sigma_res_kbar == pytest.approx(
            atomic.sigma_res_kbar, abs=0.01), surface
