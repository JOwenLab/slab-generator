import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import fit_bulk_reference as fbr

REFERENCE_SUMMARY_CSV = REPO_ROOT / "results" / "reference_diamond" / "reference_summary.csv"
REFERENCE_90_720_CSV = REPO_ROOT / "results" / "reference_90_720" / "reference_summary.csv"


# ── Birch-Murnaghan round-trip ────────────────────────────────────────────────
#
# These exist because an earlier draft of the B0' recovery dropped a dx/dV
# factor in the chain rule and returned B0' ≈ 222 for diamond while V0 and B0
# stayed correct to five significant figures. Nothing in the a0 or B outputs
# looked wrong. The bug was also invisible at B0' = 4 exactly, which is the one
# case where BM3's cubic term vanishes — so a round-trip that only tested
# B0' = 4 would have passed.

def bm3_energy(V, E0, V0, B0_gpa, B0_prime):
    """Reference 3rd-order Birch-Murnaghan E(V), written out longhand."""
    B0 = B0_gpa / fbr.RY_A3_TO_GPA          # Ry/Å³
    u = (V0 / V) ** (2.0 / 3.0)
    t = u - 1.0
    return E0 + (9.0 * V0 * B0 / 16.0) * (t ** 3 * B0_prime + t ** 2 * (6.0 - 4.0 * u))


@pytest.mark.parametrize("V0,B0_gpa,B0_prime", [
    (45.6, 433.0, 3.70),    # diamond-like; B0' != 4 is the case that caught the bug
    (45.6, 433.0, 4.00),    # degenerate case: cubic term vanishes
    (45.6, 433.0, 4.50),
    (40.0, 200.0, 5.50),    # softer, far from diamond
    (20.0, 900.0, 2.50),
])
def test_bm3_roundtrip_recovers_known_parameters(V0, B0_gpa, B0_prime):
    E0 = -147.4
    V = [V0 * (1.0 + s) for s in (-0.03, -0.015, 0.0, 0.015, 0.03)]
    E = [bm3_energy(v, E0, V0, B0_gpa, B0_prime) for v in V]

    got = fbr.birch_murnaghan_fit(V, E, order=3)

    assert got["V0_angstrom3"] == pytest.approx(V0, rel=1e-9)
    assert got["B0_gpa"] == pytest.approx(B0_gpa, rel=1e-8)
    assert got["B0_prime"] == pytest.approx(B0_prime, rel=1e-6)
    assert got["E0_ry"] == pytest.approx(E0, abs=1e-10)
    assert got["a0_angstrom"] == pytest.approx(V0 ** (1.0 / 3.0), rel=1e-9)
    assert got["rms_residual_ry"] < 1e-12


def test_bm2_fixes_b0_prime_at_four_and_recovers_bm2_data():
    """order=2 must return B0'=4 exactly and be exact on true BM2 data."""
    V0, B0_gpa = 45.6, 433.0
    V = [V0 * (1.0 + s) for s in (-0.03, -0.015, 0.0, 0.015, 0.03)]
    E = [bm3_energy(v, -147.4, V0, B0_gpa, 4.0) for v in V]

    got = fbr.birch_murnaghan_fit(V, E, order=2)

    assert got["B0_prime"] == 4.0
    assert got["V0_angstrom3"] == pytest.approx(V0, rel=1e-9)
    assert got["B0_gpa"] == pytest.approx(B0_gpa, rel=1e-8)


def test_bm3_rejects_bad_order():
    with pytest.raises(ValueError):
        fbr.birch_murnaghan_fit([45.0, 45.5, 46.0, 46.5, 47.0], [1.0] * 5, order=5)


def test_bm3_flags_extrapolated_minimum_on_monotonic_data():
    """
    E strictly decreasing in V has no equilibrium in the sampled window. It is
    still curved in x = V^(-2/3), so the cubic does have a stationary point and
    the fit returns *something* — the invariant that must hold is that the
    result is marked as lying outside the sampled range, and that run_analysis
    turns that into a warning rather than a silent number.
    """
    V = [45.0, 45.5, 46.0, 46.5, 47.0]
    got = fbr.birch_murnaghan_fit(V, [-1.0 * v for v in V], order=3)
    assert got["V0_in_sampled_range"] is False
    assert not (min(V) <= got["V0_angstrom3"] <= max(V))

    points = [{"epsilon_declared": None, "energy_ry": -v, "pressure_kbar": 0.0,
               "volume_angstrom3": v, "folder_name": f"s{i}"}
              for i, v in enumerate(V)]
    a_ref, warns = fbr.assign_epsilon_from_volume(points)
    _, all_warnings = fbr.run_analysis(points, a_ref=a_ref, prior_warnings=warns)
    assert any("outside the sampled volume range" in w for w in all_warnings)


# ── Curvature actually matters on the real series ────────────────────────────

@pytest.mark.skipif(not REFERENCE_90_720_CSV.exists(), reason="90/720 fixture not present")
def test_quadratic_pressure_fit_beats_linear_on_real_series():
    """
    The whole reason for this change: P(eps) is curved, so a linear fit is
    biased. Assert the curvature is real and large enough to matter.
    """
    points = fbr.load_data(REFERENCE_90_720_CSV, strain_type=None)
    a_ref, _ = fbr.assign_epsilon_from_volume(points)
    eps = [p["epsilon"] for p in points]
    P = [p["pressure_kbar"] for p in points]

    lin = fbr.pressure_poly_fit(eps, P, a_ref, degree=1)
    quad = fbr.pressure_poly_fit(eps, P, a_ref, degree=2)

    # the linear fit leaves >30x the residual of the quadratic
    assert lin["rms_residual_kbar"] > 30 * quad["rms_residual_kbar"]
    # and its zero crossing is biased by >2e-4 in strain
    assert abs(lin["epsilon0"] - quad["epsilon0"]) > 2e-4


# ── a0 must not depend on which point is called the strain reference ─────────

@pytest.mark.skipif(not REFERENCE_90_720_CSV.exists(), reason="90/720 fixture not present")
def test_a0_is_invariant_to_choice_of_strain_reference():
    """
    a0 = a_ref (1 + eps0) with eps = (V/V_ref)^(1/3) - 1. Shifting V_ref must
    rescale eps0 to compensate exactly, leaving a0 unchanged. If this ever
    fails, the reported a0 depends on an arbitrary labelling choice.
    """
    points = fbr.load_data(REFERENCE_90_720_CSV, strain_type=None)
    _, _ = fbr.assign_epsilon_from_volume(points)
    P = [p["pressure_kbar"] for p in points]
    V = [p["volume_angstrom3"] for p in points]

    a0s = []
    for V_ref in V:
        a_ref = V_ref ** (1.0 / 3.0)
        eps = [(v / V_ref) ** (1.0 / 3.0) - 1.0 for v in V]
        a0s.append(fbr.pressure_poly_fit(eps, P, a_ref, degree=2)["a0_angstrom"])

    assert max(a0s) - min(a0s) < 1e-9


# ── Known-series regression ──────────────────────────────────────────────────

@pytest.mark.skipif(not REFERENCE_90_720_CSV.exists(), reason="90/720 fixture not present")
def test_known_90_720_series_values():
    """
    Pins the production numbers. a0 3.5730 / B 433.7 supersede the previous
    3.573641 / 445.1, which came from a linear P(eps) and a linear P(V) slope.
    """
    points = fbr.load_data(REFERENCE_90_720_CSV, strain_type=None)
    a_ref, warns = fbr.assign_epsilon_from_volume(points)
    results, _ = fbr.run_analysis(points, a_ref=a_ref, prior_warnings=warns)

    assert results["fit_method"] == "birch_murnaghan_3rd_order_EV"
    assert results["a0_fit_angstrom"] == pytest.approx(3.572997, abs=5e-6)
    assert results["bulk_modulus_gpa"] == pytest.approx(433.66, abs=0.1)
    assert results["bm3_B0_prime"] == pytest.approx(3.744, abs=0.02)
    # cross-check agrees
    assert results["pquad_a0_angstrom"] == pytest.approx(3.572939, abs=5e-6)
    assert results["consistency_status"] == "consistent"
    # the superseded values are still recorded
    assert results["legacy_a0_energy_volume_quadratic_angstrom"] == pytest.approx(
        3.573638, abs=5e-6)
    assert results["legacy_bulk_modulus_pv_linear_gpa"] == pytest.approx(445.18, abs=0.1)


@pytest.mark.skipif(not REFERENCE_SUMMARY_CSV.exists(), reason="80/640 fixture not present")
def test_both_cutoffs_agree_on_a0():
    """80/640 and 90/720 must give the same a0; the EOS is cutoff-insensitive."""
    out = {}
    for key, path, st in (("80", REFERENCE_SUMMARY_CSV, "hydrostatic"),
                          ("90", REFERENCE_90_720_CSV, None)):
        if not path.exists():
            pytest.skip(f"{key} fixture not present")
        pts = fbr.load_data(path, strain_type=st)
        a_ref, warns = fbr.assign_epsilon_from_volume(pts)
        out[key], _ = fbr.run_analysis(pts, a_ref=a_ref, prior_warnings=warns)

    assert out["80"]["a0_fit_angstrom"] == pytest.approx(
        out["90"]["a0_fit_angstrom"], abs=1e-5)
    assert out["80"]["bulk_modulus_gpa"] == pytest.approx(
        out["90"]["bulk_modulus_gpa"], abs=0.5)


# ── Disagreement must be flagged, not resolved ───────────────────────────────

def test_method_disagreement_is_flagged():
    """
    Feed E(V) and P(V) that describe different materials. The script must come
    back 'inconsistent' with a warning, not silently pick one.
    """
    V0_energy, V0_pressure = 45.6, 46.4
    V = [V0_energy * (1.0 + s) for s in (-0.03, -0.015, 0.0, 0.015, 0.03)]
    E = [bm3_energy(v, -147.4, V0_energy, 433.0, 3.7) for v in V]
    # pressure consistent with a noticeably larger equilibrium volume
    P = [-433.0 * 10.0 * ((v / V0_pressure) - 1.0) for v in V]

    points = [{"epsilon_declared": None, "energy_ry": e, "pressure_kbar": p,
               "volume_angstrom3": v, "folder_name": f"synthetic_{i}"}
              for i, (v, e, p) in enumerate(zip(V, E, P))]
    a_ref, warns = fbr.assign_epsilon_from_volume(points)
    results, all_warnings = fbr.run_analysis(points, a_ref=a_ref, prior_warnings=warns)

    assert results["consistency_status"] == "inconsistent"
    assert results["delta_a0_methods_angstrom"] > 0.0005
    assert any("DISAGREEMENT" in w for w in all_warnings)


def test_unphysical_b0_prime_is_warned():
    """B0' outside [2,8] must produce a warning even if a0/B look plausible."""
    V0 = 45.6
    V = [V0 * (1.0 + s) for s in (-0.03, -0.015, 0.0, 0.015, 0.03)]
    E = [bm3_energy(v, -147.4, V0, 433.0, 40.0) for v in V]
    P = [-(433.0 * 10.0 / 3.0) * 3.0 * ((v / V0) ** (1 / 3) - 1.0) for v in V]

    points = [{"epsilon_declared": None, "energy_ry": e, "pressure_kbar": p,
               "volume_angstrom3": v, "folder_name": f"synthetic_{i}"}
              for i, (v, e, p) in enumerate(zip(V, E, P))]
    a_ref, warns = fbr.assign_epsilon_from_volume(points)
    _, all_warnings = fbr.run_analysis(points, a_ref=a_ref, prior_warnings=warns)

    assert any("B0'" in w for w in all_warnings)


# ── Geometry, not filenames ──────────────────────────────────────────────────

def test_declared_epsilon_disagreeing_with_volume_is_warned_and_overridden():
    """CLAUDE.md §1: read the geometry. A mislabelled folder must be caught."""
    V0 = 45.6
    V = [V0 * (1.0 + s) for s in (-0.03, -0.015, 0.0, 0.015, 0.03)]
    declared = [-0.01, -0.005, 0.0, 0.005, 0.01]      # wrong by ~3x
    points = [{"epsilon_declared": d, "energy_ry": 0.0, "pressure_kbar": 0.0,
               "volume_angstrom3": v, "folder_name": f"eps_{d}"}
              for v, d in zip(V, declared)]

    a_ref, warns = fbr.assign_epsilon_from_volume(points)

    assert a_ref == pytest.approx(V0 ** (1.0 / 3.0))
    assert any("disagrees with" in w for w in warns)
    # volume-derived value wins
    assert points[0]["epsilon"] == pytest.approx((1 - 0.03) ** (1 / 3) - 1, abs=1e-12)


def _base_config():
    return {
        "name": "reference_pbe_sssp",
        "bulk_reference": {
            "cell_type": "conventional",
            "nat": 8,
            "a0_input_angstrom": 3.567,
            "a0_fit_angstrom": 1.0,
            "bulk_modulus_gpa": 1.0,
        },
        "elastic_tensor": {
            "symmetry": "cubic", "units": "GPa",
            "C11": 1076.0, "C12": 125.0, "C44": 576.0,
            "source_type": "literature", "citation": "test citation",
        },
    }


@pytest.mark.skipif(not REFERENCE_SUMMARY_CSV.exists(), reason="reference fixture data not present")
def test_update_config_writes_a0_and_b_only(tmp_path):
    cfg_path = tmp_path / "ref.json"
    cfg_path.write_text(json.dumps(_base_config()))
    outdir = tmp_path / "out"

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "fit_bulk_reference.py"),
         "--input", str(REFERENCE_SUMMARY_CSV),
         "--outdir", str(outdir),
         "--no-plots",
         "--update-config", str(cfg_path)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    fit_summary = json.loads((outdir / "bulk_fit_summary.json").read_text())
    updated = json.loads(cfg_path.read_text())

    # a0 now comes from the Birch-Murnaghan fit, not the E(V) quadratic minimum
    assert updated["bulk_reference"]["a0_fit_angstrom"] == pytest.approx(
        fit_summary["bm3_a0_angstrom"])
    assert updated["bulk_reference"]["bulk_modulus_gpa"] == pytest.approx(
        fit_summary["bm3_B0_gpa"])
    assert updated["bulk_reference"]["fit_method"] == "birch_murnaghan_3rd_order_EV"
    assert updated["bulk_reference"]["source_type"] == "project_dft_fit"

    # the biased legacy values must NOT be what got written
    assert updated["bulk_reference"]["a0_fit_angstrom"] != pytest.approx(
        fit_summary["legacy_a0_energy_volume_quadratic_angstrom"])
    assert updated["bulk_reference"]["bulk_modulus_gpa"] != pytest.approx(
        fit_summary["legacy_bulk_modulus_pv_linear_gpa"])

    assert updated["bulk_reference"]["cell_type"] == "conventional"
    assert updated["bulk_reference"]["nat"] == 8
    assert updated["bulk_reference"]["a0_input_angstrom"] == 3.567

    assert updated["elastic_tensor"]["C11"] == 1076.0
    assert updated["elastic_tensor"]["C12"] == 125.0
    assert updated["elastic_tensor"]["C44"] == 576.0
    assert updated["elastic_tensor"]["citation"] == "test citation"


@pytest.mark.skipif(not REFERENCE_SUMMARY_CSV.exists(), reason="reference fixture data not present")
def test_supersede_reason_archives_previous_values(tmp_path):
    """History is kept: the old a0/B must survive under bulk_reference.superseded."""
    cfg_path = tmp_path / "ref.json"
    cfg = _base_config()
    cfg["bulk_reference"]["a0_fit_angstrom"] = 3.573641
    cfg["bulk_reference"]["bulk_modulus_gpa"] = 445.116
    cfg_path.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "fit_bulk_reference.py"),
         "--input", str(REFERENCE_SUMMARY_CSV),
         "--outdir", str(tmp_path / "out"),
         "--no-plots",
         "--update-config", str(cfg_path),
         "--supersede-reason", "linear P(eps) fit was biased"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    br = json.loads(cfg_path.read_text())["bulk_reference"]
    assert br["superseded"]["a0_fit_angstrom"] == 3.573641
    assert br["superseded"]["bulk_modulus_gpa"] == 445.116
    assert br["superseded"]["reason_superseded"] == "linear P(eps) fit was biased"
    # and the live values did move
    assert br["a0_fit_angstrom"] != 3.573641
    assert br["bulk_modulus_gpa"] != 445.116


@pytest.mark.skipif(not REFERENCE_SUMMARY_CSV.exists(), reason="reference fixture data not present")
def test_stale_keys_from_previous_fit_method_are_removed(tmp_path):
    """
    Outputs of the old fit that the new method does not produce must not
    survive next to the new numbers, where they would read as current.
    """
    cfg_path = tmp_path / "ref.json"
    cfg = _base_config()
    cfg["bulk_reference"].update({
        "a0_energy_fit_angstrom": 3.573394,
        "a0_pressure_fit_angstrom": 3.573882,
        "epsilon0_energy_fit": 0.0017856646270185638,
        "epsilon0_pressure_fit": 0.001922336232058496,
    })
    cfg_path.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "fit_bulk_reference.py"),
         "--input", str(REFERENCE_SUMMARY_CSV),
         "--outdir", str(tmp_path / "out"), "--no-plots",
         "--update-config", str(cfg_path),
         "--supersede-reason", "test"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    br = json.loads(cfg_path.read_text())["bulk_reference"]
    for key in ("a0_energy_fit_angstrom", "a0_pressure_fit_angstrom",
                "epsilon0_energy_fit", "epsilon0_pressure_fit"):
        assert key not in br, f"{key} survived into the live block"
        assert key in br["superseded"], f"{key} was dropped instead of archived"
    # identity fields are not stale and must survive
    assert br["cell_type"] == "conventional"
    assert br["a0_input_angstrom"] == 3.567


@pytest.mark.skipif(not REFERENCE_90_720_CSV.exists(), reason="90/720 fixture not present")
def test_fitted_qe_settings_are_read_from_the_data():
    """Provenance: the cutoffs recorded must be the ones pw.out reports."""
    points = fbr.load_data(REFERENCE_90_720_CSV, strain_type=None)
    settings, warns = fbr.collect_run_settings(points)
    assert settings["ecutwfc"] == 90.0
    assert settings["ecutrho"] == 720.0
    assert not warns


def test_mixed_cutoffs_across_a_series_are_flagged():
    """An EOS fitted across different cutoffs is not meaningful — say so."""
    points = [
        {"ecutwfc": "80.0", "ecutrho": "640.0", "kpoints": "8 8 8", "pseudo_C": "C.upf"},
        {"ecutwfc": "90.0", "ecutrho": "720.0", "kpoints": "8 8 8", "pseudo_C": "C.upf"},
    ]
    _, warns = fbr.collect_run_settings(points)
    assert any("MIXED ecutwfc" in w for w in warns)


@pytest.mark.skipif(not REFERENCE_SUMMARY_CSV.exists(), reason="reference fixture data not present")
def test_no_update_config_leaves_config_untouched(tmp_path):
    cfg_path = tmp_path / "ref.json"
    original_cfg = _base_config()
    cfg_path.write_text(json.dumps(original_cfg))

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "fit_bulk_reference.py"),
         "--input", str(REFERENCE_SUMMARY_CSV),
         "--outdir", str(tmp_path / "out"),
         "--no-plots"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    assert json.loads(cfg_path.read_text()) == original_cfg
