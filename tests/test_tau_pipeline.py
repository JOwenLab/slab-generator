import csv
import io
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import analyze_slab_stress as ass
import fit_slab_strain as fss


def test_tau_conversion_formula():
    row = {
        "stress_xx_kbar": "100.0", "stress_yy_kbar": "80.0",
        "stress_zz_kbar": "0.0", "stress_xy_kbar": "0.0",
        "cell_height_angstrom": "20.0",
        "complete": "True", "pseudo_consistency": "ok", "relax_converged": "",
        "folder_name": "x", "orientation": "(100)", "termination": "H", "formula": "C",
    }
    out = ass.analyze_row(row)
    factor = 20.0 * ass.KBAR_TO_GPA * ass.GPA_ANG_TO_N_PER_M / 2.0
    assert out["tau_xx_n_per_m"] == pytest.approx(100.0 * factor)
    assert out["tau_yy_n_per_m"] == pytest.approx(80.0 * factor)
    assert out["tau_mean_n_per_m"] == pytest.approx(90.0 * factor)
    assert out["tau_anisotropy_n_per_m"] == pytest.approx(20.0 * factor)
    assert out["tau_mean_n_per_m"] == pytest.approx(out["inplane_mean_stress_kbar"] * 20.0 * 0.005)


def _synthetic_items(m, b, lz, eps_list, ref_overrides=None):
    factor = lz * 0.005
    ref = {"orientation": "(100)", "termination": "H", "formula": "C", "warnings": ""}
    if ref_overrides:
        ref.update(ref_overrides)
    items = []
    for eps in eps_list:
        sigma = m * eps + b
        tau = sigma * factor
        items.append((eps, sigma, tau, ref))
    return items


def test_tau_fit_matches_scaled_sigma_fit():
    m, b, lz = -1000.0, -5.0, 15.0
    eps_list = [-0.01, -0.005, 0.0, 0.005, 0.01]
    items = _synthetic_items(m, b, lz, eps_list)

    fit = fss.analyze_group(("series1", "biaxial"), items)

    factor = lz * 0.005
    assert fit["tau_slope_n_per_m_per_eps"] == pytest.approx(m * factor)
    assert fit["tau_intercept_n_per_m"] == pytest.approx(b * factor)
    assert fit["stress_slope_kbar_per_eps"] == pytest.approx(m)
    assert fit["stress_intercept_kbar"] == pytest.approx(b)


def test_zero_crossing_equality_sigma_vs_tau():
    m, b, lz = -1000.0, -5.0, 15.0
    eps_list = [-0.01, -0.005, 0.0, 0.005, 0.01]
    items = _synthetic_items(m, b, lz, eps_list)

    fit = fss.analyze_group(("series1", "biaxial"), items)

    expected_eps0 = -b / m
    assert fit["zero_stress_epsilon"] == pytest.approx(expected_eps0)
    assert fit["zero_tau_strain"] == pytest.approx(expected_eps0)
    assert fit["zero_strain_discrepancy"] < 1e-9
    assert "zero_strain_mismatch_sigma_vs_tau" not in fit["warnings"]


def test_r2_unchanged_between_sigma_and_tau_fits():
    lz = 15.0
    factor = lz * 0.005
    eps_list = [-0.01, -0.005, 0.0, 0.005, 0.01]
    noisy_sigma = [-15.3, -10.1, -4.8, 0.2, 5.4]
    ref = {"orientation": "(111)", "termination": "H", "formula": "C", "warnings": ""}
    items = [(e, s, s * factor, ref) for e, s in zip(eps_list, noisy_sigma)]

    fit = fss.analyze_group(("series2", "biaxial"), items)

    assert fit["r2_tau_fit"] == pytest.approx(fit["r2_stress_fit"], abs=1e-12)


def test_mismatch_warning_raised_beyond_tolerance():
    # sigma implies zero-crossing eps0 = -(-5)/-1000 = -0.005; tau is deliberately
    # an unrelated line (as if from an inconsistent Lz) with zero-crossing -0.02.
    eps_list = [-0.01, -0.005, 0.0, 0.005, 0.01]
    ref = {"orientation": "(100)", "termination": "H", "formula": "C", "warnings": ""}
    items = []
    for eps in eps_list:
        sigma = -1000.0 * eps - 5.0
        tau = 500.0 * eps + 10.0
        items.append((eps, sigma, tau, ref))

    fit = fss.analyze_group(("series3", "biaxial"), items, zero_strain_tolerance=1e-6)

    assert fit["zero_stress_epsilon"] == pytest.approx(-0.005)
    assert fit["zero_tau_strain"] == pytest.approx(-0.02)
    assert fit["zero_strain_discrepancy"] > 1e-6
    assert "zero_strain_mismatch_sigma_vs_tau" in fit["warnings"]
    assert fit["fit_status"] == "warning"


def test_mismatch_not_flagged_when_tolerance_relaxed():
    m, b, lz = -1000.0, -5.0, 15.0
    eps_list = [-0.01, -0.005, 0.0, 0.005, 0.01]
    factor = lz * 0.005
    ref = {"orientation": "(100)", "termination": "H", "formula": "C", "warnings": ""}
    items = []
    for i, eps in enumerate(eps_list):
        sigma = m * eps + b
        tau = sigma * factor + (1e-8 if i == 0 else 0.0)
        items.append((eps, sigma, tau, ref))

    fit = fss.analyze_group(("series4", "biaxial"), items, zero_strain_tolerance=1.0)

    assert "zero_strain_mismatch_sigma_vs_tau" not in fit["warnings"]


def test_csv_schema_backward_compatible(tmp_path):
    m, b, lz = -1000.0, -5.0, 15.0
    eps_list = [-0.01, -0.005, 0.0, 0.005, 0.01]
    items = _synthetic_items(m, b, lz, eps_list, {"series": "s"})
    fit = fss.analyze_group(("series5", "biaxial"), items)

    legacy_fields = [
        "series", "orientation", "termination", "formula", "strain_mode",
        "n_points", "epsilon_min", "epsilon_max",
        "stress_slope_kbar_per_eps", "stress_intercept_kbar",
        "zero_stress_epsilon", "stress_at_zero_kbar",
        "tau_slope_n_per_m_per_eps", "tau_intercept_n_per_m", "tau_at_zero_n_per_m",
        "r2_stress_fit", "fit_status", "warnings",
    ]
    for field in legacy_fields:
        assert field in fit

    out_path = tmp_path / "fit.csv"
    fss.write_csv([fit], out_path)
    with out_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    for field in legacy_fields:
        assert field in rows[0]
    for field in ("zero_tau_strain", "r2_tau_fit", "n_tau_points", "zero_strain_discrepancy"):
        assert field in rows[0]
