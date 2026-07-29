import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_SUMMARY_CSV = REPO_ROOT / "results" / "reference_diamond" / "reference_summary.csv"


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

    assert updated["bulk_reference"]["a0_fit_angstrom"] == pytest.approx(
        fit_summary["a0_energy_volume_fit_angstrom"])
    assert updated["bulk_reference"]["bulk_modulus_gpa"] == pytest.approx(
        fit_summary["bulk_modulus_gpa"])
    assert updated["bulk_reference"]["source_type"] == "project_dft_fit"

    assert updated["bulk_reference"]["cell_type"] == "conventional"
    assert updated["bulk_reference"]["nat"] == 8
    assert updated["bulk_reference"]["a0_input_angstrom"] == 3.567

    assert updated["elastic_tensor"]["C11"] == 1076.0
    assert updated["elastic_tensor"]["C12"] == 125.0
    assert updated["elastic_tensor"]["C44"] == 576.0
    assert updated["elastic_tensor"]["citation"] == "test citation"


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
