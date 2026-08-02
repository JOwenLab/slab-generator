"""Tests for make_h_phonons.py and analyze_h_phonons.py.

The analysis is exercised end-to-end against a SYNTHETIC harmonic force field
written into fake pw.out files, so the whole chain -- force parsing, central
differences, sign convention, mass weighting, unit conversion -- is checked
against a known answer rather than against itself.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import analyze_h_phonons as ah
import make_h_phonons as mh
import surface_energy as se

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTION = os.path.join(REPO, "results", "production")
REFERENCE = os.path.join(REPO, "results", "reference_90_720")


# ------------------------------------------------------------- unit anchors
def test_frequency_conversion_matches_a_known_force_constant():
    """A 500 N/m bond on a 1 amu mass sits near the C-H stretch."""
    k_ev_per_a2 = 500.0 * 1e-20 / 1.602176634e-19      # N/m -> eV/A^2
    freq = ah.FREQ_FACTOR_CM1 * math.sqrt(k_ev_per_a2 / 1.008)
    assert 2800 < freq < 3050


def test_frequency_factor_value():
    assert ah.FREQ_FACTOR_CM1 == pytest.approx(521.47, rel=1e-3)


def test_ry_per_bohr2_conversion():
    assert ah.RY_PER_BOHR2_TO_EV_PER_A2 == pytest.approx(
        13.605693 / 0.529177210903 ** 2)


# -------------------------------------------------- synthetic force campaign
def _write_pw_out(path, forces_ry_bohr):
    path.mkdir(parents=True, exist_ok=True)
    lines = ["     Forces acting on atoms (cartesian axes, Ry/au):", ""]
    for i, f in enumerate(forces_ry_bohr, start=1):
        lines.append(f"     atom {i:4d} type  1   force = "
                     f"{f[0]:16.8f} {f[1]:16.8f} {f[2]:16.8f}")
    lines += ["", "     Total force =     0.000000", "", "   JOB DONE."]
    (path / "pw.out").write_text("\n".join(lines) + "\n")


def _synthetic_campaign(tmp_path, tag, k_matrix_ev_per_a2, n_atoms, displaced,
                        delta=0.015):
    """Fake a campaign whose exact Hessian is k_matrix (eV/A^2, over `displaced`).

    Forces are F = -K u, converted into the Ry/bohr units QE prints.
    """
    to_ry_bohr = 1.0 / (ah.RY_TO_EV / ah.BOHR_TO_ANG)
    n = len(displaced)
    _write_pw_out(tmp_path / f"hphon~{tag}~ref", np.zeros((n_atoms, 3)))
    for j in range(n):
        for bx, axis in enumerate(mh.AXES):
            for sign, label in ((+1.0, "p"), (-1.0, "m")):
                u = np.zeros(3 * n)
                u[3 * j + bx] = sign * delta
                f_flat = -(k_matrix_ev_per_a2 @ u)
                forces = np.zeros((n_atoms, 3))
                for i, atom in enumerate(displaced):
                    forces[atom] = f_flat[3 * i:3 * i + 3] * to_ry_bohr
                _write_pw_out(tmp_path / f"hphon~{tag}~H{j:02d}_{axis}{label}",
                              forces)
    prov = {"displacement_angstrom": delta,
            "displaced_atom_indices": list(displaced)}
    for child in tmp_path.iterdir():
        if child.name.startswith(f"hphon~{tag}~"):
            (child / "provenance.json").write_text(json.dumps(prov))


def test_analysis_recovers_a_known_hessian(tmp_path):
    """The load-bearing test: a diagonal force field with known constants must
    come back as the right frequencies."""
    k = np.diag([30.0, 8.0, 8.0])           # one stiff, two soft, eV/A^2
    _synthetic_campaign(tmp_path, "SYNTH", k, n_atoms=3, displaced=[2])
    ms = ah.analyse_set(tmp_path, "SYNTH")
    expected = sorted(ah.FREQ_FACTOR_CM1 * math.sqrt(v / ah.MASS_H_AMU)
                      for v in (30.0, 8.0, 8.0))
    # Tolerance is set by QE's 8-decimal force printing, not by the algebra:
    # see test_force_print_precision_limits_the_hessian.
    assert np.allclose(ms.frequencies_cm1, expected, rtol=1e-4)
    assert len(ms.imaginary_cm1) == 0
    assert ms.hessian_asymmetry == pytest.approx(0.0, abs=1e-4)
    assert ms.zpe_ev == pytest.approx(
        0.5 * sum(expected) * ah.CM1_TO_EV, rel=1e-4)


def test_analysis_is_insensitive_to_displacement_size_for_a_harmonic_field(tmp_path):
    k = np.diag([30.0, 8.0, 8.0])
    for delta in (0.005, 0.03):
        d = tmp_path / f"d{delta}"
        d.mkdir()
        _synthetic_campaign(d, "SYNTH", k, 3, [2], delta=delta)
        ms = ah.analyse_set(d, "SYNTH")
        assert ms.frequencies_cm1[-1] == pytest.approx(
            ah.FREQ_FACTOR_CM1 * math.sqrt(30.0 / ah.MASS_H_AMU), rel=1e-3)


def test_off_diagonal_coupling_is_recovered(tmp_path):
    k = np.array([[25.0, 3.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 12.0]])
    _synthetic_campaign(tmp_path, "SYNTH", k, n_atoms=1, displaced=[0])
    ms = ah.analyse_set(tmp_path, "SYNTH")
    expected = sorted(ah.FREQ_FACTOR_CM1 * math.sqrt(v / ah.MASS_H_AMU)
                      for v in np.linalg.eigvalsh(k))
    assert np.allclose(ms.frequencies_cm1, expected, rtol=1e-4)


def test_imaginary_modes_are_reported_not_hidden(tmp_path):
    k = np.diag([30.0, 8.0, -5.0])          # a saddle point
    _synthetic_campaign(tmp_path, "SYNTH", k, n_atoms=1, displaced=[0])
    ms = ah.analyse_set(tmp_path, "SYNTH")
    assert len(ms.imaginary_cm1) == 1
    assert len(ms.frequencies_cm1) == 2
    # An imaginary mode must not contribute a negative ZPE.
    assert ms.zpe_ev > 0


def test_missing_displacement_is_refused(tmp_path):
    k = np.diag([30.0, 8.0, 8.0])
    _synthetic_campaign(tmp_path, "SYNTH", k, n_atoms=1, displaced=[0])
    import shutil
    shutil.rmtree(tmp_path / "hphon~SYNTH~H00_zp")
    with pytest.raises(ah.HPhononAnalysisError):
        ah.analyse_set(tmp_path, "SYNTH")


def test_missing_provenance_displacement_is_refused(tmp_path):
    k = np.diag([30.0, 8.0, 8.0])
    _synthetic_campaign(tmp_path, "SYNTH", k, n_atoms=1, displaced=[0])
    for child in tmp_path.iterdir():
        (child / "provenance.json").write_text(json.dumps({}))
    with pytest.raises(ah.HPhononAnalysisError, match="displacement magnitude"):
        ah.analyse_set(tmp_path, "SYNTH")


def test_forces_without_a_force_block_are_refused(tmp_path):
    d = tmp_path / "hphon~X~ref"
    d.mkdir(parents=True)
    (d / "pw.out").write_text("no forces here\n   JOB DONE.\n")
    with pytest.raises(ah.HPhononAnalysisError, match="no force block"):
        ah.parse_forces(d / "pw.out")


def test_unfinished_run_is_refused(tmp_path):
    d = tmp_path / "hphon~X~ref"
    d.mkdir(parents=True)
    (d / "pw.out").write_text("Forces acting on atoms\n")
    with pytest.raises(ah.HPhononAnalysisError, match="JOB DONE"):
        ah.parse_forces(d / "pw.out")


# --------------------------------------------------------- the H2 self-check
def test_h2_zero_modes_gate_catches_a_broken_campaign(tmp_path):
    """If the five translation/rotation modes are not near zero, something in
    the chain is wrong and the module must refuse rather than report a ZPE."""
    k = np.eye(6) * 40.0                    # nonsense: no acoustic modes
    _synthetic_campaign(tmp_path, "H2", k, n_atoms=2, displaced=[0, 1])
    with pytest.raises(ah.HPhononAnalysisError, match="near zero"):
        ah.analyse_h2(tmp_path)


def test_h2_with_a_physical_hessian_passes_and_gives_the_stretch(tmp_path):
    """A real H2 force field: stiff along the bond, zero for rigid motions."""
    k_bond = 36.0                            # eV/A^2, ~H2 stretch
    e = np.array([0.0, 0.0, 1.0])
    block = k_bond * np.outer(e, e)
    k = np.zeros((6, 6))
    k[:3, :3] = block
    k[3:, 3:] = block
    k[:3, 3:] = -block
    k[3:, :3] = -block
    _synthetic_campaign(tmp_path, "H2", k, n_atoms=2, displaced=[0, 1])
    ms = ah.analyse_h2(tmp_path)
    # Five near-zero modes plus the stretch at sqrt(2k/m) (reduced mass m/2).
    assert len(ms.frequencies_cm1) + len(ms.imaginary_cm1) == 6
    expected = ah.FREQ_FACTOR_CM1 * math.sqrt(2 * k_bond / ah.MASS_H_AMU)
    assert ms.frequencies_cm1[-1] == pytest.approx(expected, rel=1e-4)
    assert ms.zpe_ev == pytest.approx(0.5 * expected * ah.CM1_TO_EV, rel=1e-4)


def test_force_print_precision_limits_the_hessian(tmp_path):
    """QE prints forces to 8 decimal places. That, not the algebra, sets the
    accuracy of a finite-difference Hessian, and it is the reason the
    displacement must be large enough to lift the force difference well clear
    of the last printed digit."""
    k = np.diag([30.0, 8.0, 8.0])
    coarse = tmp_path / "coarse"
    coarse.mkdir()
    _synthetic_campaign(coarse, "SYNTH", k, 1, [0], delta=0.0005)
    fine = tmp_path / "fine"
    fine.mkdir()
    _synthetic_campaign(fine, "SYNTH", k, 1, [0], delta=0.02)
    exact = ah.FREQ_FACTOR_CM1 * math.sqrt(30.0 / ah.MASS_H_AMU)
    err_small = abs(ah.analyse_set(coarse, "SYNTH").frequencies_cm1[-1] - exact)
    err_large = abs(ah.analyse_set(fine, "SYNTH").frequencies_cm1[-1] - exact)
    # A displacement 40x smaller loses roughly that much precision.
    assert err_small > err_large
    assert err_large < 1.0        # cm^-1, negligible at the default setting


def test_delta_zpe_is_the_difference_per_hydrogen():
    class Fake:
        def __init__(self, tag, zpe, n):
            self.tag, self.zpe_ev, self.n_displaced = tag, zpe, n
            self.frequencies_cm1 = np.array([1000.0])
            self.imaginary_cm1 = np.array([])
        @property
        def zpe_per_h_ev(self):
            return self.zpe_ev / self.n_displaced
    h2 = Fake("H2", 0.27, 2)
    slab = Fake("C111_8L", 0.66, 2)          # 0.33 eV per H
    out = ah.delta_zpe([slab], h2)
    assert out["C111_8L"]["zpe_ads_per_h_ev"] == pytest.approx(0.33)
    assert out["C111_8L"]["zpe_h2_per_h_ev"] == pytest.approx(0.135)
    assert out["C111_8L"]["delta_zpe_ev"] == pytest.approx(0.195)


def test_analysis_refuses_to_run_without_the_h2_set(tmp_path):
    k = np.diag([30.0, 8.0, 8.0])
    _synthetic_campaign(tmp_path, "C111_8L", k, n_atoms=1, displaced=[0])
    with pytest.raises(ah.HPhononAnalysisError, match="no H2 set"):
        ah.run(tmp_path, tmp_path / "out")


# ----------------------------------------------------- the literature anchor
def test_the_estimate_the_campaign_will_replace():
    """surface_energy's DELTA_ZPE estimate, which this campaign supersedes."""
    assert se.zpe_adsorbed_h_ev() == pytest.approx(0.335, abs=0.005)
    assert se.zpe_h2_per_h_ev() == pytest.approx(0.136, abs=0.002)
    assert se.delta_zpe_ev() == pytest.approx(0.198, abs=0.005)
    # Positive: an adsorbed H is stiffer than half an H2.
    assert se.delta_zpe_ev() > 0


def test_zpe_shift_moves_the_crossing_to_lower_temperature():
    """Direction matters more than magnitude: the correction makes the
    prediction MORE accessible, not less."""
    out = se.zpe_systematic_on_temperature(1.295, se.PA_PER_BAR * 1e-9)
    assert out["temperature_with_zpe_k"] < out["temperature_no_zpe_k"]
    assert out["shift_k"] < 0
    assert 80 < abs(out["shift_k"]) < 260


def test_zpe_systematic_dwarfs_the_ideal_gas_residual():
    """The point of item 1: the quoted uncertainty was the wrong one."""
    p = se.PA_PER_BAR * 1e-9
    zpe = abs(se.zpe_systematic_on_temperature(1.295, p)["shift_k"])
    fit = se.temperature_uncertainty_k(1.295, p)
    assert zpe > 20 * fit


# ------------------------------------------------------------- generation
production = pytest.mark.skipif(
    not os.path.isfile(os.path.join(
        PRODUCTION, "thick_a0corr~C111_8L", "pw.out")),
    reason="production relaxations not present")


@production
def test_generator_displaces_exactly_one_hydrogen_per_run(tmp_path):
    import parse_slab
    mh.run(PRODUCTION, REFERENCE, tmp_path, mh.DEFAULT_PREFIX, [8], ["C111"],
           0.015, "top", include_h2=False)
    ref = parse_slab.parse_pw_in(tmp_path / "hphon~C111_8L~ref" / "pw.in")
    ref_pos = np.array([[a["x"], a["y"], a["z"]]
                        for a in ref["initial_positions_ang"]])
    n_disp = 0
    for child in sorted(tmp_path.iterdir()):
        if child.name.endswith("~ref"):
            continue
        pos = np.array([[a["x"], a["y"], a["z"]] for a in
                        parse_slab.parse_pw_in(child / "pw.in")["initial_positions_ang"]])
        diff = pos - ref_pos
        moved = np.nonzero(np.abs(diff).sum(axis=1) > 1e-9)[0]
        assert len(moved) == 1, child.name
        assert ref["initial_positions_ang"][moved[0]]["species"] == "H"
        assert abs(np.abs(diff[moved[0]]).max() - 0.015) < 1e-9
        n_disp += 1
    assert n_disp == 6            # one H on the top face of (111), 3 axes x 2


@production
def test_generator_writes_provenance_with_everything_the_analysis_needs(tmp_path):
    mh.run(PRODUCTION, REFERENCE, tmp_path, mh.DEFAULT_PREFIX, [8], ["C111"],
           0.015, "top", include_h2=False)
    prov = json.loads(
        (tmp_path / "hphon~C111_8L~H00_xp" / "provenance.json").read_text())
    assert prov["displacement_angstrom"] == 0.015
    assert prov["displaced_atom_indices"]
    assert prov["carbon_treatment"].startswith("held fixed")
    assert prov["not_submitted"] is True
    assert prov["ecutwfc"] == 90.0 and prov["ecutrho"] == 720.0
    assert prov["source_relaxed_geometry"].endswith("thick_a0corr~C111_8L")


@production
def test_top_face_selection_halves_the_work(tmp_path):
    import parse_slab
    both = mh.run(PRODUCTION, REFERENCE, tmp_path / "b", mh.DEFAULT_PREFIX,
                  [8], ["C100"], 0.015, "both", include_h2=False)
    top = mh.run(PRODUCTION, REFERENCE, tmp_path / "t", mh.DEFAULT_PREFIX,
                 [8], ["C100"], 0.015, "top", include_h2=False)
    assert both["sets"][0]["n_displaced"] == 2 * top["sets"][0]["n_displaced"]


@production
def test_h2_set_uses_the_same_cutoff_as_the_slabs(tmp_path):
    import parse_slab
    mh.run(PRODUCTION, REFERENCE, tmp_path, mh.DEFAULT_PREFIX, [8], ["C111"],
           0.015, "top", include_h2=True)
    slab = parse_slab.parse_pw_in(tmp_path / "hphon~C111_8L~ref" / "pw.in")
    h2 = parse_slab.parse_pw_in(tmp_path / "hphon~H2~ref" / "pw.in")
    assert h2["ecutwfc"] == slab["ecutwfc"]
    assert h2["ecutrho"] == slab["ecutrho"]
    # All six H2 degrees of freedom are displaced, for the zero-mode check.
    names = [c.name for c in tmp_path.iterdir() if c.name.startswith("hphon~H2~")]
    assert len(names) == 13       # 2 atoms x 3 axes x 2 signs, plus ref
