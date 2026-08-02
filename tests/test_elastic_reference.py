import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import elastic_reference as er


def _cfg():
    return {
        "bulk_reference": {"a0_fit_angstrom": 3.573641, "bulk_modulus_gpa": 445.116},
        "elastic_tensor": {
            "symmetry": "cubic", "units": "GPa",
            "C11": 1076.0, "C12": 125.0, "C44": 576.0,
            "source_type": "literature", "citation": "test citation",
        },
    }


def _write(tmp_path, cfg, name="ref.json"):
    p = tmp_path / name
    p.write_text(json.dumps(cfg))
    return p


def test_valid_cubic_tensor_loads(tmp_path):
    ref = er.load_elastic_reference(_write(tmp_path, _cfg()))
    assert ref.tensor.C11 == 1076.0
    assert ref.tensor.C12 == 125.0
    assert ref.tensor.C44 == 576.0
    assert ref.bulk.a0_angstrom == 3.573641
    assert ref.bulk.bulk_modulus_gpa == 445.116


def test_missing_c44_fails_clearly(tmp_path):
    cfg = _cfg()
    del cfg["elastic_tensor"]["C44"]
    with pytest.raises(er.ElasticReferenceError, match="C44"):
        er.load_elastic_reference(_write(tmp_path, cfg))


def test_nonnumeric_value_fails_clearly(tmp_path):
    cfg = _cfg()
    cfg["elastic_tensor"]["C11"] = "not-a-number"
    with pytest.raises(er.ElasticReferenceError, match="numeric"):
        er.load_elastic_reference(_write(tmp_path, cfg))


def test_unsupported_units_fails_clearly(tmp_path):
    cfg = _cfg()
    cfg["elastic_tensor"]["units"] = "kbar"
    with pytest.raises(er.ElasticReferenceError, match="units"):
        er.load_elastic_reference(_write(tmp_path, cfg))


def test_unsupported_symmetry_fails_clearly(tmp_path):
    cfg = _cfg()
    cfg["elastic_tensor"]["symmetry"] = "hexagonal"
    with pytest.raises(er.ElasticReferenceError, match="symmetry"):
        er.load_elastic_reference(_write(tmp_path, cfg))


def test_missing_file_fails_clearly(tmp_path):
    with pytest.raises(er.ElasticReferenceError, match="not found"):
        er.load_elastic_reference(tmp_path / "nope.json")


def test_malformed_json_fails_clearly(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    with pytest.raises(er.ElasticReferenceError, match="malformed JSON"):
        er.load_elastic_reference(p)


def test_c11_not_greater_than_c12_fails(tmp_path):
    cfg = _cfg()
    cfg["elastic_tensor"]["C11"] = 100.0
    cfg["elastic_tensor"]["C12"] = 125.0
    with pytest.raises(er.ElasticReferenceError, match="C11 - C12"):
        er.load_elastic_reference(_write(tmp_path, cfg))


def test_negative_c44_fails(tmp_path):
    cfg = _cfg()
    cfg["elastic_tensor"]["C44"] = -10.0
    with pytest.raises(er.ElasticReferenceError, match="C44"):
        er.load_elastic_reference(_write(tmp_path, cfg))


def test_negative_c11_plus_2c12_fails(tmp_path):
    cfg = _cfg()
    cfg["elastic_tensor"]["C11"] = 100.0
    cfg["elastic_tensor"]["C12"] = -60.0
    with pytest.raises(er.ElasticReferenceError, match=r"C11 \+ 2\*C12"):
        er.load_elastic_reference(_write(tmp_path, cfg))


def test_bulk_modulus_tensor_computed_correctly(tmp_path):
    ref = er.load_elastic_reference(_write(tmp_path, _cfg()))
    assert ref.b_tensor_gpa == pytest.approx((1076.0 + 2 * 125.0) / 3.0)


def test_percent_discrepancy_computed_correctly(tmp_path):
    ref = er.load_elastic_reference(_write(tmp_path, _cfg()))
    b_tensor = (1076.0 + 2 * 125.0) / 3.0
    assert ref.b_discrepancy_gpa == pytest.approx(b_tensor - 445.116)
    assert ref.b_discrepancy_pct == pytest.approx(100.0 * (b_tensor - 445.116) / 445.116)


def test_warning_threshold_toggles_consistency(tmp_path):
    cfg = _cfg()
    cfg["bulk_reference"]["bulk_modulus_gpa"] = 300.0
    p = _write(tmp_path, cfg)
    assert er.load_elastic_reference(p, warning_threshold_pct=1.0).consistent is False
    assert er.load_elastic_reference(p, warning_threshold_pct=90.0).consistent is True


def test_project_fitted_and_literature_sources_are_distinct(tmp_path):
    ref = er.load_elastic_reference(_write(tmp_path, _cfg()))
    assert ref.bulk.source_type == "project_dft_fit"
    assert ref.tensor.source_type == "literature"


def test_update_bulk_reference_changes_only_a0_and_b(tmp_path):
    cfg = _cfg()
    cfg["bulk_reference"]["cell_type"] = "conventional"
    cfg["bulk_reference"]["nat"] = 8
    cfg["extra_top_level"] = "keep me"
    p = _write(tmp_path, cfg)

    er.update_bulk_reference(p, a0_angstrom=3.6, bulk_modulus_gpa=450.0,
                              source_summary_json="results/x.json")

    updated = json.loads(p.read_text())
    assert updated["bulk_reference"]["a0_fit_angstrom"] == 3.6
    assert updated["bulk_reference"]["bulk_modulus_gpa"] == 450.0
    assert updated["bulk_reference"]["cell_type"] == "conventional"
    assert updated["bulk_reference"]["nat"] == 8
    assert updated["bulk_reference"]["source_type"] == "project_dft_fit"
    assert updated["extra_top_level"] == "keep me"
    assert updated["elastic_tensor"]["C11"] == 1076.0
    assert updated["elastic_tensor"]["citation"] == "test citation"


def test_update_bulk_reference_missing_config_raises(tmp_path):
    with pytest.raises(er.ElasticReferenceError, match="not found"):
        er.update_bulk_reference(tmp_path / "nope.json", 3.6, 450.0, "x.json")


def test_update_bulk_reference_malformed_config_not_overwritten(tmp_path):
    p = tmp_path / "bad.json"
    original = "{not valid json"
    p.write_text(original)
    with pytest.raises(er.ElasticReferenceError, match="malformed JSON"):
        er.update_bulk_reference(p, 3.6, 450.0, "x.json")
    assert p.read_text() == original
