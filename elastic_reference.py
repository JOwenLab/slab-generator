#!/usr/bin/env python3
"""Load, validate, and update the project's bulk/elastic reference config."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WARNING_THRESHOLD_PCT = 7.5


class ElasticReferenceError(Exception):
    pass


@dataclass(frozen=True)
class BulkReference:
    a0_angstrom: float
    bulk_modulus_gpa: float
    source_type: str
    source_summary_json: str = ""


@dataclass(frozen=True)
class ElasticTensor:
    symmetry: str
    units: str
    C11: float
    C12: float
    C44: float
    source_type: str
    citation: str
    notes: str

    def bulk_modulus_gpa(self) -> float:
        return (self.C11 + 2.0 * self.C12) / 3.0


@dataclass(frozen=True)
class ElasticReference:
    name: str
    config_path: str
    bulk: BulkReference
    tensor: ElasticTensor
    b_tensor_gpa: float
    b_discrepancy_gpa: float
    b_discrepancy_pct: float
    warning_threshold_pct: float
    consistent: bool


def _require(d, key, ctx):
    if key not in d:
        raise ElasticReferenceError(f"missing required field '{key}' in {ctx}")
    return d[key]


def _require_number(d, key, ctx, positive=False):
    v = _require(d, key, ctx)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ElasticReferenceError(
            f"field '{key}' in {ctx} must be numeric, got {type(v).__name__}: {v!r}")
    v = float(v)
    if positive and v <= 0:
        raise ElasticReferenceError(f"field '{key}' in {ctx} must be positive, got {v}")
    return v


def validate_cubic_tensor(C11: float, C12: float, C44: float, ctx: str = "elastic tensor") -> None:
    if C11 - C12 <= 0:
        raise ElasticReferenceError(
            f"mechanically unstable cubic tensor in {ctx}: C11 - C12 = {C11 - C12} <= 0")
    if C11 + 2 * C12 <= 0:
        raise ElasticReferenceError(
            f"mechanically unstable cubic tensor in {ctx}: C11 + 2*C12 = {C11 + 2 * C12} <= 0")
    if C44 <= 0:
        raise ElasticReferenceError(f"mechanically unstable cubic tensor in {ctx}: C44 = {C44} <= 0")


def load_elastic_reference(config_path,
                            warning_threshold_pct: float = DEFAULT_WARNING_THRESHOLD_PCT
                            ) -> ElasticReference:
    path = Path(config_path)
    if not path.exists():
        raise ElasticReferenceError(f"reference config not found: {path}")
    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ElasticReferenceError(f"malformed JSON in {path}: {exc}") from exc

    br_ctx = f"{path}:bulk_reference"
    br = _require(cfg, "bulk_reference", path)
    bulk = BulkReference(
        a0_angstrom=_require_number(br, "a0_fit_angstrom", br_ctx, positive=True),
        bulk_modulus_gpa=_require_number(br, "bulk_modulus_gpa", br_ctx, positive=True),
        source_type=br.get("source_type", "project_dft_fit"),
        source_summary_json=br.get("source_summary_json", ""),
    )

    et_ctx = f"{path}:elastic_tensor"
    et = _require(cfg, "elastic_tensor", path)
    symmetry = _require(et, "symmetry", et_ctx)
    if symmetry != "cubic":
        raise ElasticReferenceError(
            f"unsupported elastic symmetry '{symmetry}' in {et_ctx}; only 'cubic' is implemented")
    units = _require(et, "units", et_ctx)
    if units != "GPa":
        raise ElasticReferenceError(
            f"unsupported elastic units '{units}' in {et_ctx}; only 'GPa' is implemented")
    C11 = _require_number(et, "C11", et_ctx, positive=True)
    C12 = _require_number(et, "C12", et_ctx)
    C44 = _require_number(et, "C44", et_ctx, positive=True)
    validate_cubic_tensor(C11, C12, C44, et_ctx)

    tensor = ElasticTensor(
        symmetry=symmetry, units=units, C11=C11, C12=C12, C44=C44,
        source_type=et.get("source_type", "literature"),
        citation=et.get("citation", ""),
        notes=et.get("notes", ""),
    )

    b_tensor = tensor.bulk_modulus_gpa()
    diff_gpa = b_tensor - bulk.bulk_modulus_gpa
    diff_pct = 100.0 * diff_gpa / bulk.bulk_modulus_gpa

    return ElasticReference(
        name=cfg.get("name", path.stem),
        config_path=str(path),
        bulk=bulk,
        tensor=tensor,
        b_tensor_gpa=b_tensor,
        b_discrepancy_gpa=diff_gpa,
        b_discrepancy_pct=diff_pct,
        warning_threshold_pct=warning_threshold_pct,
        consistent=abs(diff_pct) <= warning_threshold_pct,
    )


def update_bulk_reference(config_path, a0_angstrom: float, bulk_modulus_gpa: float,
                           source_summary_json: str, extra_fields: dict | None = None) -> None:
    path = Path(config_path)
    if not path.exists():
        raise ElasticReferenceError(f"reference config not found: {path}")
    try:
        cfg = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ElasticReferenceError(f"malformed JSON in {path}: {exc}") from exc

    br = cfg.setdefault("bulk_reference", {})
    br["a0_fit_angstrom"] = a0_angstrom
    br["bulk_modulus_gpa"] = bulk_modulus_gpa
    br["source_type"] = "project_dft_fit"
    br["source_summary_json"] = source_summary_json
    if extra_fields:
        br.update(extra_fields)

    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(cfg, indent=2) + "\n")
    tmp_path.replace(path)
