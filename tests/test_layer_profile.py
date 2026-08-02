"""Tests for layer_profile.py — depth profile of surface relaxation."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import layer_profile as lp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTION = os.path.join(REPO, "results", "production")
CONFIG = os.path.join(REPO, "config", "reference_pbe_sssp.json")

A0 = 3.572997470394866


# --------------------------------------------------------- ideal geometry
def test_ideal_spacings_match_the_lattice():
    assert lp.ideal_spacings("100", A0) == [pytest.approx(A0 / 4.0)]
    assert lp.ideal_spacings("110", A0) == [pytest.approx(A0 / (2 * math.sqrt(2)))]
    short, long = lp.ideal_spacings("111", A0)
    # The (111) bilayer repeat is the interplanar spacing a0/sqrt(3).
    assert short + long == pytest.approx(A0 / math.sqrt(3.0))
    assert long == pytest.approx(3.0 * short)


def test_bulk_bond_length():
    assert lp.bulk_bond_length(A0) == pytest.approx(math.sqrt(3) / 4 * A0)
    assert lp.bulk_bond_length(A0) == pytest.approx(1.54714, abs=1e-4)


def test_unknown_orientation_refuses():
    with pytest.raises(lp.LayerProfileError):
        lp.ideal_spacings("311", A0)


# ------------------------------------------------------------ layer sorting
def test_assign_layers_splits_on_gaps_not_on_counts():
    zs = [0.0, 0.05, 1.0, 1.02, 2.0]
    groups = lp.assign_layers(zs, tol=0.3)
    assert [len(g) for g in groups] == [2, 2, 1]


def test_assign_layers_tolerates_rumpling_below_the_split_tolerance():
    # 0.19 A of intralayer rumpling, as seen in the C100 subsurface.
    zs = [0.0, 0.19, 0.893, 1.083]
    groups = lp.assign_layers(zs, tol=0.4 * 0.893)
    assert [len(g) for g in groups] == [2, 2]


# ------------------------------------------------------------ gap classing
def test_classify_gaps_alternates_on_111():
    short, long = lp.ideal_spacings("111", A0)
    gaps = [short, long, short, long, short]
    assert lp._classify_gaps(gaps, [short, long], 0.25, "C111") == [0, 1, 0, 1, 0]


def test_classify_gaps_rejects_a_gap_far_from_every_class():
    short, long = lp.ideal_spacings("111", A0)
    with pytest.raises(lp.LayerProfileError, match="from the nearest ideal"):
        lp._classify_gaps([short, 1.05], [short, long], 0.10, "C111")


def test_classify_gaps_refuses_an_ambiguous_assignment():
    """Halfway between two classes means the alternation phase is a guess."""
    with pytest.raises(lp.LayerProfileError, match="ambiguous"):
        lp._classify_gaps([1.0], [0.9, 1.1], 0.5, "C111")


def test_classify_gaps_rejects_a_broken_alternation():
    short, long = lp.ideal_spacings("111", A0)
    with pytest.raises(lp.LayerProfileError, match="do not alternate"):
        lp._classify_gaps([short, short], [short, long], 0.25, "C111")


def test_classify_gaps_accepts_a_single_class_without_alternation():
    d = lp.ideal_spacings("110", A0)[0]
    assert lp._classify_gaps([d, d, d], [d], 0.25, "C110") == [0, 0, 0]


# ------------------------------------------------------------- decay fitting
def test_fit_exponential_recovers_a_known_decay_length():
    lam, amp = 0.75, 0.09
    depths = [0.3, 1.2, 2.1, 3.0]
    values = [amp * math.exp(-d / lam) for d in depths]
    fit = lp.fit_exponential(depths, values, "synthetic")
    assert fit.decay_length_angstrom == pytest.approx(lam, abs=1e-9)
    assert fit.amplitude == pytest.approx(amp, abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.status == "ok"


def test_fit_exponential_flags_a_non_decaying_profile():
    fit = lp.fit_exponential([0.3, 1.2, 2.1], [0.01, 0.02, 0.04], "growing")
    assert not math.isfinite(fit.decay_length_angstrom)
    assert "does not decay" in fit.status


def test_fit_exponential_flags_too_few_points():
    fit = lp.fit_exponential([0.3], [0.01], "one")
    assert math.isnan(fit.decay_length_angstrom)
    assert "too few points" in fit.status


# ------------------------------------------------------------- production
def _have(run):
    return os.path.isfile(os.path.join(PRODUCTION, run, "pw.out"))


production = pytest.mark.skipif(
    not all(_have(r) for r in lp.DEFAULT_RUNS.values()),
    reason="production relaxations not present")


@pytest.fixture(scope="module")
def profiles(tmp_path_factory):
    out = tmp_path_factory.mktemp("layer_profile")
    result = lp.run(PRODUCTION, out, dict(lp.DEFAULT_RUNS), CONFIG,
                    lp.DEFAULT_GAP_TOL, lp.DEFAULT_EPS_FLOOR,
                    lp.DEFAULT_PENETRATION_THRESHOLD)
    return {p.surface: p for p in result["profiles"]}, out


@production
def test_layer_counts_match_the_folder_names(profiles):
    profs, _ = profiles
    assert set(profs) == {"C100", "C110", "C111"}
    for p in profs.values():
        assert len(p.layers) == p.n_layers
        assert p.n_carbon == p.n_layers * p.atoms_per_layer
        assert sum(L.n_atoms for L in p.layers) == p.n_carbon
    assert profs["C111"].n_layers == 24        # the thickest C111 available
    assert profs["C100"].n_layers == 16
    assert profs["C110"].n_layers == 16


@production
def test_the_slabs_are_symmetric_top_to_bottom(profiles):
    """CLAUDE.md invariant 1. The folded profile is a free check on it: the two
    halves of a symmetric slab must give the same eps_zz."""
    profs, _ = profiles
    for surface, p in profs.items():
        assert p.fold_asymmetry < 1e-6, surface


@production
def test_relaxation_is_confined_to_the_first_few_angstrom(profiles):
    """The result that licenses the uniform-interior approximation in
    particle_strain.py."""
    profs, _ = profiles
    for surface, p in profs.items():
        assert 0.0 < p.fit_all.decay_length_angstrom < 2.0, surface
        assert 0.0 < p.penetration_depth_angstrom < 6.0, surface
        # ... and much smaller than a 3 nm particle's 15 A half-thickness.
        assert p.penetration_depth_angstrom < 15.0 / 3.0, surface


@production
def test_outermost_gap_contracts_on_every_surface(profiles):
    profs, _ = profiles
    for surface, p in profs.items():
        assert p.surface_eps_zz < 0, surface
        assert abs(p.surface_eps_zz) > 0.01, surface   # a real effect, > 1%


@production
def test_interior_gaps_return_to_the_bulk_spacing(profiles):
    """The interior must be bulk-like, or there is no bulk-like core to speak of."""
    profs, _ = profiles
    for surface, p in profs.items():
        gaps = [L for L in p.layers if L.eps_zz_below is not None]
        deepest = max(gaps, key=lambda L: L.gap_depth)
        assert abs(deepest.eps_zz_below) < 1e-3, surface


@production
def test_bond_lengths_stay_near_the_bulk_value(profiles):
    profs, _ = profiles
    d0 = lp.bulk_bond_length(A0)
    for surface, p in profs.items():
        for L in p.layers:
            assert L.n_bonds > 0, (surface, L.index)
            assert 0.9 * d0 < L.bond_mean < 1.15 * d0, (surface, L.index)
        assert abs(p.max_bond_dev_pct) < 6.0, surface


@production
def test_ch_bond_length_is_only_present_on_the_terminated_layers(profiles):
    profs, _ = profiles
    for surface, p in profs.items():
        with_h = [L for L in p.layers if L.ch_bond_mean is not None]
        assert len(with_h) == 2, surface           # one per face
        assert {L.index for L in with_h} == {0, p.n_layers - 1}, surface
        for L in with_h:
            assert 1.0 < L.ch_bond_mean < 1.2, surface


@production
def test_c111_reports_class_resolved_decay_and_others_do_not(profiles):
    """(111) has two alternating gap classes, so the single-exponential fit is
    an envelope and the class-resolved fits are reported alongside it."""
    profs, _ = profiles
    assert len(profs["C111"].fit_by_class) == 2
    assert profs["C100"].fit_by_class == []
    assert profs["C110"].fit_by_class == []


@production
def test_outputs_are_written_and_state_their_scope(profiles):
    _, out = profiles
    for surface in ("C100", "C110", "C111"):
        assert (out / f"layer_profile_{surface}.csv").exists()
    assert (out / "layer_profile_summary.csv").exists()
    assert (out / "layer_profile_meta.json").exists()
    text = (out / "layer_profile_report.md").read_text()
    assert "no depth-dependent in-plane strain" in text
    assert "L2" in text and "L1" in text


@production
def test_surface_mismatch_between_request_and_folder_is_rejected():
    """CLAUDE.md invariant 3: names describe geometry."""
    with pytest.raises(lp.LayerProfileError, match="folder names surface"):
        lp.analyse_run("C100",
                       __import__("pathlib").Path(PRODUCTION) / lp.DEFAULT_RUNS["C111"],
                       A0, lp.DEFAULT_GAP_TOL, lp.DEFAULT_EPS_FLOOR,
                       lp.DEFAULT_PENETRATION_THRESHOLD)
