import argparse
import json
import subprocess
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import pytest

import nv_spin_strain as nvs
import elastic_reference as er

REPO_ROOT = Path(__file__).resolve().parent.parent


def _args(**overrides):
    defaults = dict(
        reference_config=str(REPO_ROOT / "config" / "reference_pbe_sssp.json"),
        elastic_source="config",
        C11=None, C12=None, C44=None,
        warning_threshold_pct=er.DEFAULT_WARNING_THRESHOLD_PCT,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_config_matches_legacy_literature_values():
    elastic, prov = nvs.resolve_elastic_context(_args())
    assert elastic.C11 == 1076.0
    assert elastic.C12 == 125.0
    assert elastic.C44 == 576.0
    assert prov["elastic_source_type"] == "literature"
    assert prov["cli_override_used"] is False
    assert prov["a0_angstrom"] is not None
    assert prov["bulk_modulus_fit_gpa"] is not None


def test_cli_override_takes_precedence_over_config():
    elastic, prov = nvs.resolve_elastic_context(_args(C11=1000.0, C12=130.0, C44=550.0))
    assert (elastic.C11, elastic.C12, elastic.C44) == (1000.0, 130.0, 550.0)
    assert prov["elastic_source_type"] == "cli_override"
    assert prov["cli_override_used"] is True
    # a0/B still available for comparison since config was still loaded
    assert prov["a0_angstrom"] is not None


def test_partial_override_is_rejected():
    with pytest.raises(SystemExit):
        nvs.resolve_elastic_context(_args(C11=1000.0))


def test_legacy_source_bypasses_config_with_warning(capsys):
    elastic, prov = nvs.resolve_elastic_context(_args(elastic_source="legacy"))
    assert (elastic.C11, elastic.C12, elastic.C44) == (1076.0, 125.0, 576.0)
    assert prov["elastic_source_type"] == "legacy_hardcoded"
    assert prov["a0_angstrom"] is None
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_missing_config_gives_actionable_error():
    with pytest.raises(SystemExit, match="Could not load elastic reference"):
        nvs.resolve_elastic_context(_args(reference_config="/nonexistent/ref.json"))


def test_report_labels_mixed_source():
    elastic, prov = nvs.resolve_elastic_context(_args())
    lines = nvs._format_elastic_reference_section(elastic, prov)
    text = "\n".join(lines)
    assert "mixed-source" in text
    assert "1076.0" in text
    assert "literature" in text


def test_bulk_modulus_comparison_matches_elastic_reference_module():
    ref = er.load_elastic_reference(REPO_ROOT / "config" / "reference_pbe_sssp.json")
    elastic, prov = nvs.resolve_elastic_context(_args())
    assert prov["bulk_modulus_tensor_gpa"] == pytest.approx(ref.b_tensor_gpa)
    assert prov["bulk_modulus_discrepancy_pct"] == pytest.approx(ref.b_discrepancy_pct)


@pytest.mark.skipif(
    not (REPO_ROOT / "results" / "slabs" / "slab_strain_fit_summary.csv").exists(),
    reason="strain-fit summary fixture not present")
def test_cli_end_to_end_matches_legacy_hardcoded_defaults(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "nv_spin_strain.py"), "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    meta = json.loads((tmp_path / "nv_predictions_meta.json").read_text())
    assert meta["elastic_source_type"] == "literature"
    assert meta["cli_override_used"] is False

    csv_path = tmp_path / "nv_predictions.csv"
    rows = csv_path.read_text().splitlines()
    header = rows[0].split(",")
    assert "elastic_source_type" in header
    assert "elastic_C11_gpa" in header
