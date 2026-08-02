"""
Pins the project stress sign convention against the one measurement that fixes
it beyond argument.

CLAUDE.md section 2: positive sigma means the cell is COMPRESSED and wants to
expand; negative means it is in TENSION. QE pressure P = +(1/3) tr(sigma).

The anchor is bulk diamond at -1% hydrostatic strain, which is unambiguously
compression by construction: the cell was made smaller. QE returns
sigma_xx = sigma_yy = sigma_zz = P = +162.86 kbar. Any chain of reasoning that
ends with compression giving negative sigma contradicts a number sitting in
results/reference_90_720/, and is wrong.

Everything derived from sigma inherits that sign:
  - effective_inplane_pressure_kbar = +mean(sigma_xx, sigma_yy)
  - approximate_pressure_proxy_kbar = +residual_mean_stress
  - tau = sigma * Lz / 2, so tau > 0 is COMPRESSIVE surface stress and equals
    MINUS the continuum surface stress f
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import analyze_slab_stress as ass

BULK_MINUS_1PCT = REPO_ROOT / "results" / "reference_90_720" / "bulk90~eps_-0.010" / "pw.out"
EXPECTED_KBAR = 162.86


@pytest.mark.skipif(not BULK_MINUS_1PCT.exists(), reason="bulk reference not present")
def test_bulk_anchor_compression_gives_positive_sigma_and_positive_pressure():
    """
    The load-bearing measurement. -1% strain is compression by construction.
    """
    text = BULK_MINUS_1PCT.read_text()

    m = re.search(r"total\s+stress.*?P=\s*(-?[0-9.]+)", text)
    assert m, "no stress block in the bulk reference output"
    pressure = float(m.group(1))

    diag = []
    rows = text[m.end():].splitlines()[1:4]
    for i, row in enumerate(rows):
        diag.append(float(row.split()[3 + i]))

    assert pressure == pytest.approx(EXPECTED_KBAR, abs=0.01), (
        "compression must give POSITIVE pressure under P = +(1/3)tr(sigma)")
    for i, s in enumerate(diag):
        assert s == pytest.approx(EXPECTED_KBAR, abs=0.01), (
            f"sigma_{'xyz'[i]}{'xyz'[i]} = {s}; compression must give positive sigma")

    # P = +(1/3) tr(sigma), not -(1/3) tr(sigma)
    assert pressure == pytest.approx(sum(diag) / 3.0, abs=0.01)


def test_effective_inplane_pressure_shares_sigmas_sign():
    """
    A field named "pressure" must be positive when the cell is compressed.
    It was negated until 2026-08, making it anti-correlated with pressure.
    """
    compressed = ass.analyze_row({
        "stress_xx_kbar": "40.0", "stress_yy_kbar": "30.0",
        "stress_zz_kbar": "0.0", "cell_height_angstrom": "20.0"})
    assert compressed["effective_inplane_pressure_kbar"] == pytest.approx(35.0)
    assert compressed["stress_interpretation"] == "compressive_inplane_pressure_like"

    tension = ass.analyze_row({
        "stress_xx_kbar": "-40.0", "stress_yy_kbar": "-30.0",
        "stress_zz_kbar": "0.0", "cell_height_angstrom": "20.0"})
    assert tension["effective_inplane_pressure_kbar"] == pytest.approx(-35.0)
    assert tension["stress_interpretation"] == "tensile_inplane_stress_like"

    near_zero = ass.analyze_row({
        "stress_xx_kbar": "0.4", "stress_yy_kbar": "-0.2",
        "stress_zz_kbar": "0.0", "cell_height_angstrom": "20.0"})
    assert near_zero["stress_interpretation"] == "near_zero_inplane_mean_stress"


def test_tau_shares_sigmas_sign_so_positive_tau_is_compressive():
    """
    tau = sigma * Lz / 2 carries sigma's sign, so positive tau is COMPRESSIVE
    and equals minus the continuum surface stress f.

    Measured, not asserted: free 2D cell relaxations (cell_dofree='2Dxy') at 16
    layers expanded the cell on all six axes carrying positive tau, and
    contracted the one axis carrying negative tau. A tensile surface pulls the
    cell inward, so positive tau cannot be tensile.
    """
    row = ass.analyze_row({
        "stress_xx_kbar": "40.0", "stress_yy_kbar": "30.0",
        "stress_zz_kbar": "0.0", "cell_height_angstrom": "20.0"})
    # tau = sigma * Lz * 0.005
    assert row["tau_mean_n_per_m"] == pytest.approx(35.0 * 20.0 * 0.005)
    assert row["tau_mean_n_per_m"] > 0, "positive sigma must give positive tau"


@pytest.mark.skipif(not (REPO_ROOT / "results/production/tau_infinity.csv").exists(),
                    reason="tau_infinity.csv not present")
def test_particle_strain_negates_tau_exactly_once():
    """
    The continuum mechanics wants f = -tau_project. Guard against the negation
    being dropped (every Delta D flips) or applied twice (silently back to the
    original error).
    """
    import particle_strain as ps

    csv_path = REPO_ROOT / "results/production/tau_infinity.csv"
    loaded = ps.load_taus(csv_path)

    import csv as _csv
    with csv_path.open(newline="") as fh:
        rows = {r["surface"][1:]: r for r in _csv.DictReader(fh)}

    assert set(loaded) == set(rows)
    for fam, raw in rows.items():
        assert loaded[fam]["tau_xx"] == pytest.approx(
            -float(raw["tau_xx_inf_n_per_m"]), abs=1e-12)

    # (111) is isotropic and positive in the CSV, so f must come out negative:
    # a compressive surface, which expands the particle.
    assert loaded["111"]["tau_xx"] < 0
