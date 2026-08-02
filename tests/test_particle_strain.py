"""Tests for particle_strain.py — facet tau -> particle interior strain -> NV.

The two load-bearing checks here are the ones the module cannot be trusted
without:

  * the Laplace limit (P = 2*tau/R for a sphere with isotropic tau), which
    validates every geometric factor in the interior-stress formula;
  * E = 0 exactly for a symmetric particle, which validates the facet-orbit
    construction and the NV frame projection.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import geometry
import nv_spin_strain
import particle_strain as ps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTION = os.path.join(REPO, "results", "production")
TAU_CSV = os.path.join(PRODUCTION, "tau_infinity.csv")
SURFACE_ENERGIES = os.path.join(REPO, "config", "surface_energies_h.json")

A0 = 3.572997470394866
ELASTIC = nv_spin_strain.ElasticConstants()      # literature C11/C12/C44
PARAMS = nv_spin_strain.UDVARHELYI_DFT

# The production tau values (N/m), atomic-thickness convention.
#
# As stored in tau_infinity.csv, in the PROJECT convention: positive tau is
# COMPRESSIVE surface stress, because tau = sigma*Lz/2 inherits sigma's sign and
# positive sigma means the cell is compressed.
TAU_PROJECT = {
    "100": {"tau_xx": 1.0323, "tau_yy": -5.0241},
    "110": {"tau_xx": 2.1445, "tau_yy": 4.3975},
    "111": {"tau_xx": 0.4833, "tau_yy": 0.4833},
}

# What particle_strain's mechanics actually take: the continuum surface stress
# f, positive = TENSILE. load_taus() performs this negation on the way in, so
# these are the values the module works with, and feeding TAU_PROJECT straight
# into evaluate() would describe the opposite physical situation.
TAU = {fam: {"tau_xx": -v["tau_xx"], "tau_yy": -v["tau_yy"]}
       for fam, v in TAU_PROJECT.items()}


# =========================================================== REQUIRED CHECK 1
# A sphere with isotropic tau must reduce to the Laplace result P = 2*tau/R.
# If it does not, the geometric factors are wrong and nothing downstream is
# trustworthy.
# ============================================================================
@pytest.mark.parametrize("tau,radius_nm", [(1.0, 1.5), (0.4833, 1.5),
                                           (-2.0, 3.0), (5.0, 0.5)])
def test_laplace_limit_for_a_sphere(tau, radius_nm):
    R = radius_nm * 1e-9
    facets = ps.isotropic_facet_set(tau, n_directions=4000)
    sigma = ps.interior_stress_mech(facets, R)
    assert ps.pressure_gpa(sigma) == pytest.approx(
        2.0 * tau / R / ps.GPA, rel=1e-3)


def test_laplace_check_helper_reports_a_passing_validation():
    out = ps.laplace_check(tau_n_per_m=1.0, radius_m=1.5e-9, n_directions=4000)
    assert out["relative_error"] < 1e-3
    assert out["pressure_laplace_gpa"] == pytest.approx(1.0 / 1.5e-9 * 2 / ps.GPA)
    # A sphere is isotropic: no shear and no diagonal anisotropy should survive.
    assert out["max_off_diagonal_pa"] / ps.GPA < 1e-3
    assert out["max_diagonal_anisotropy_pa"] / ps.GPA < 1e-3


def test_laplace_limit_is_exact_for_a_cubically_symmetric_facet_set():
    """Any facet set with sum_f x_f n_f n_f = I/3 gives the Laplace result
    exactly, not just in the many-facet limit. The 6 cube normals do."""
    tau, R = 0.75, 2.0e-9
    facets = []
    for axis in range(3):
        for sign in (+1, -1):
            n = np.zeros(3)
            n[axis] = sign
            facets.append(ps.Facet(
                family="cube", normal=n, frame=np.eye(3),
                tau_surface=np.diag([tau, tau, 0.0]),
                tau_cubic=tau * (np.eye(3) - np.outer(n, n)),
                orbit="none", area_fraction=1.0 / 6.0))
    sigma = ps.interior_stress_mech(facets, R)
    assert ps.pressure_gpa(sigma) == pytest.approx(2.0 * tau / R / ps.GPA, rel=1e-14)


def test_interior_stress_rejects_a_non_positive_radius():
    with pytest.raises(ps.ParticleStrainError, match="radius must be positive"):
        ps.interior_stress_mech(ps.isotropic_facet_set(1.0, 10), 0.0)


def test_sign_of_the_interior_stress():
    """Tensile surface stress (tau > 0) must COMPRESS the interior."""
    facets = ps.isotropic_facet_set(1.0, 2000)
    sigma = ps.interior_stress_mech(facets, 1.5e-9)
    assert np.trace(sigma) < 0            # mechanics convention: tension positive
    assert ps.pressure_gpa(sigma) > 0     # compressive-positive
    # ... and a compressive surface stress puts the interior in tension.
    assert ps.pressure_gpa(ps.interior_stress_mech(
        ps.isotropic_facet_set(-1.0, 2000), 1.5e-9)) < 0


# ================================================================ point group
def test_td_has_24_operations_and_is_a_subgroup_of_oh():
    assert len(ps.oh_operations()) == 48
    assert len(ps.TD) == 24
    oh = [tuple(np.round(g.ravel(), 9)) for g in ps.oh_operations()]
    for g in ps.TD:
        assert tuple(np.round(g.ravel(), 9)) in oh


def test_td_excludes_inversion():
    """The whole point of using Td rather than Oh."""
    assert not any(np.allclose(g, -np.eye(3)) for g in ps.TD)


def test_facet_orbit_sizes():
    counts = {f: len(ps.facet_family(f, 1.0, 2.0)) for f in ("100", "110")}
    counts["111"] = len(ps.facet_family("111", 1.0, 1.0))
    assert counts == {"100": 6, "110": 12, "111": 8}


def test_only_the_111_family_needs_the_inversion_coset():
    """Td reaches every {100} and {110} facet; it reaches only 4 of 8 {111}."""
    for family in ("100", "110"):
        orbits = {f.orbit for f in ps.facet_family(family, 1.0, 2.0)}
        assert orbits == {"Td"}
    orbits = [f.orbit for f in ps.facet_family("111", 1.0, 1.0)]
    assert orbits.count("Td") == 4
    assert orbits.count("Td*inversion") == 4


def test_anisotropic_111_tau_is_refused():
    """A [111] normal's site symmetry within Td is C3v. Its 3-fold rotation
    admits only an in-plane isotropic tau, so a reconstructed anisotropic (111)
    surface has no well-defined crystal-frame tau and must be rejected rather
    than silently symmetry-averaged.

    This is also what makes giving the octahedron's four 'B' faces the
    reference tau safe: anything surviving this check is isotropic in-plane,
    and an isotropic in-plane tau is inversion-invariant.
    """
    with pytest.raises(ps.ParticleStrainError) as exc:
        ps.facet_family("111", 1.0, 2.0)
    assert "not invariant under its own site symmetry" in str(exc.value)
    assert "C3v" in str(exc.value)
    # An isotropic (111) tau passes, and the B faces then need no extra
    # assumption.
    assert len(ps.facet_family("111", 1.7, 1.7)) == 8


def test_opposite_100_facets_carry_perpendicular_dimer_directions():
    """A consequence of Td, not of Oh: the (001) and (00-1) faces of a diamond
    particle terminate on different sublattices."""
    facets = {f.miller: f for f in ps.facet_family("100", 1.0, -5.0)}
    up, down = facets["001"], facets["00-1"]
    # The in-plane surface x axes of the two faces are perpendicular.
    assert abs(float(np.dot(up.frame[0], down.frame[0]))) == pytest.approx(0.0, abs=1e-12)
    # Which shows up as an opposite-sign xy shear in the cubic frame.
    assert up.tau_cubic[0, 1] == pytest.approx(-down.tau_cubic[0, 1], abs=1e-12)
    assert abs(up.tau_cubic[0, 1]) > 1.0


def test_each_family_average_is_isotropic():
    """Summing an anisotropic facet tau over a full Td orbit must give a
    Td-invariant tensor, and the only such symmetric rank-2 tensor is
    isotropic. This is why a symmetric particle has E = 0."""
    for family, t in TAU.items():
        facets = ps.facet_family(family, t["tau_xx"], t["tau_yy"])
        avg = sum(f.tau_cubic for f in facets) / len(facets)
        assert np.allclose(avg, np.eye(3) * np.trace(avg) / 3.0, atol=1e-12), family
        assert np.trace(avg) / 3.0 == pytest.approx(
            2.0 / 3.0 * 0.5 * (t["tau_xx"] + t["tau_yy"]), abs=1e-12)


# =============================================================== frame checks
def test_nv_spin_strain_frames_still_match_geometry():
    """Two copies of the surface-frame convention are two chances to drift."""
    frames = ps._verify_slab_frames(A0)
    for face in ("100", "110", "111"):
        assert np.allclose(frames[face], geometry._frame(face, A0)[2])


def test_surface_frame_axes_are_the_expected_crystal_directions():
    """Pins the mapping used to rotate tau into the crystal frame."""
    s2 = 1.0 / math.sqrt(2.0)
    f100 = nv_spin_strain.slab_frame("100")
    assert np.allclose(f100[0], [s2, s2, 0])          # x = [110]
    assert np.allclose(f100[1], [-s2, s2, 0])         # y = [1-10] axis
    assert np.allclose(f100[2], [0, 0, 1])
    f110 = nv_spin_strain.slab_frame("110")
    assert np.allclose(f110[0], [s2, -s2, 0])         # x = [1-10]
    assert np.allclose(f110[1], [0, 0, -1])           # y = [001] axis
    f111 = nv_spin_strain.slab_frame("111")
    assert np.allclose(f111[0], [s2, -s2, 0])         # x = [1-10]
    assert np.allclose(f111[2], np.ones(3) / math.sqrt(3.0))


def test_swapping_the_100_surface_axes_flips_the_crystal_frame_anisotropy():
    """The failure mode the module exists to prevent. The (100) surface axes
    sit at 45 degrees to the cubic axes, so a diagonal surface-frame anisotropy
    is a pure xy shear in the crystal frame -- and swapping tau_xx with tau_yy
    inverts its sign while leaving every diagonal component untouched."""
    right = {f.miller: f for f in ps.facet_family("100", 1.0323, -5.0241)}["001"]
    wrong = {f.miller: f for f in ps.facet_family("100", -5.0241, 1.0323)}["001"]
    assert np.allclose(np.diag(right.tau_cubic), np.diag(wrong.tau_cubic))
    assert right.tau_cubic[0, 1] == pytest.approx(-wrong.tau_cubic[0, 1])
    assert right.tau_cubic[0, 1] == pytest.approx(
        0.5 * (1.0323 - (-5.0241)), abs=1e-9)


@pytest.mark.skipif(not os.path.isdir(PRODUCTION),
                    reason="production runs not present")
def test_production_cells_are_integer_multiples_of_the_geometry_frame():
    """Closes the loop from the tau numbers back to crystal directions."""
    found = ps.verify_production_cells(PRODUCTION, A0)
    assert set(found) == {"100", "110", "111"}
    # (100) is the 2x1 cell: doubled along A2, i.e. along the [1-10] axis.
    assert found["100"][:2] == (1, 2)
    assert found["110"][:2] == (1, 1)
    assert found["111"][:2] == (1, 1)


def test_missing_runs_directory_fails_loudly(tmp_path):
    with pytest.raises(ps.ParticleStrainError, match="cannot verify surface frames"):
        ps.verify_production_cells(tmp_path / "nope", A0)


# ================================================================= elasticity
def test_compliance_inverts_the_stiffness():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(3, 3)) * 1e9
    sigma = 0.5 * (m + m.T)
    eps = ps.compliance_strain(sigma, ELASTIC)
    C = ELASTIC.tensor() * ps.GPA
    assert np.allclose(np.einsum("ijkl,kl->ij", C, eps), sigma, rtol=1e-9)


def test_hydrostatic_stress_gives_hydrostatic_strain():
    """eps = sigma / (C11 + 2*C12) on each axis, with no shear."""
    p = -1.0e9                                   # 1 GPa of compression
    eps = ps.compliance_strain(np.eye(3) * p, ELASTIC)
    expected = p / ((ELASTIC.C11 + 2 * ELASTIC.C12) * ps.GPA)
    assert np.allclose(np.diag(eps), expected, rtol=1e-12)
    assert np.allclose(eps - np.diag(np.diag(eps)), 0.0, atol=1e-18)
    assert expected < 0                          # compression is negative strain


def test_shear_stress_uses_c44_without_a_factor_of_two_slip():
    sigma = np.zeros((3, 3))
    sigma[0, 1] = sigma[1, 0] = 1.0e9
    eps = ps.compliance_strain(sigma, ELASTIC)
    assert eps[0, 1] == pytest.approx(1.0e9 / (2 * ELASTIC.C44 * ps.GPA), rel=1e-12)


# =========================================================== REQUIRED CHECK 2
# A symmetric particle gives purely hydrostatic interior stress, hence a D
# shift with E EXACTLY zero. Nonzero E requires broken symmetry.
# ============================================================================
@pytest.mark.parametrize("fractions", [
    {"111": 1.0},                    # octahedron
    {"100": 1.0},                    # cube
    {"110": 1.0},                    # rhombic dodecahedron
    {"111": 0.6, "100": 0.4},        # symmetric mixture: still E = 0
    {"111": 0.4, "110": 0.3, "100": 0.3},
])
def test_symmetric_particle_gives_zero_transverse_splitting(fractions):
    result = ps.evaluate(TAU, fractions, 1.5, ELASTIC, PARAMS)
    assert result.shape_is_symmetric
    sigma = result.sigma_mech_pa
    assert np.allclose(sigma, np.eye(3) * np.trace(sigma) / 3.0, atol=1e-6)
    assert result.e_max_mhz < ps.E_ZERO_TOL_MHZ
    assert result.delta_d_spread_mhz < ps.E_ZERO_TOL_MHZ
    # All four NV orientations must be equivalent under a hydrostatic strain.
    assert len({round(r["delta_D_mhz"], 9) for r in result.nv_rows}) == 1


def test_broken_symmetry_is_required_for_nonzero_e():
    """Enlarging one {100} facet breaks the Td symmetry and switches E on."""
    result = ps.evaluate(TAU, {"100": 1.0}, 1.5, ELASTIC, PARAMS,
                         facet_overrides={"001": 3.0})
    assert not result.shape_is_symmetric
    assert result.e_max_mhz > 1.0
    assert result.delta_d_spread_mhz > 1.0


def test_area_fractions_must_sum_to_one():
    facets = ps.build_facets(TAU, {"111": 0.5, "100": 0.5})
    assert sum(f.area_fraction for f in facets) == pytest.approx(1.0)
    assert len(facets) == 8 + 6
    # Within a family the area is split equally when nothing is overridden.
    per_111 = {f.area_fraction for f in facets if f.family == "111"}
    assert len(per_111) == 1 and per_111.pop() == pytest.approx(0.5 / 8)


def test_negative_area_fraction_is_refused():
    with pytest.raises(ps.ParticleStrainError, match="negative area fraction"):
        ps.build_facets(TAU, {"111": -0.5, "100": 1.5})


def test_missing_tau_for_a_requested_family_is_refused():
    with pytest.raises(ps.ParticleStrainError, match="no tau available"):
        ps.build_facets({"111": TAU["111"]}, {"110": 1.0})


# ================================================================= scale check
def test_octahedron_scale_matches_the_expected_order_of_magnitude():
    """
    The real (111)-H octahedron: R = 1.5 nm, |f| ~ 0.48 N/m, P ~ 0.65 GPa and
    |Delta D| of order 5 MHz.

    The SIGN is the substantive claim. (111) carries tau_project = +0.483,
    i.e. a COMPRESSIVE surface, so f = -0.483 is compressive in the continuum
    sense too: the particle EXPANDS, its interior goes into TENSION, and D
    shifts DOWN. Measured directly: a free 2D vc-relax of the 16L (111) slab
    expanded by +0.052 % on both in-plane axes.
    """
    result = ps.evaluate({"111": {"tau_xx": -0.4833, "tau_yy": -0.4833}},
                         {"111": 1.0}, 1.5, ELASTIC, PARAMS)
    assert result.pressure_gpa == pytest.approx(-0.644, rel=0.02)
    row = result.nv_rows[0]
    dD = row["delta_D_mhz"]
    assert -20.0 < dD < -2.0               # negative, order 5-10 MHz
    assert dD < 0, "a compressive (111) surface must lower D, not raise it"
    assert row["eps_axial"] > 0, "the interior must be in tension, not compression"


def test_tensile_surface_stress_raises_d():
    """
    (100)-H is the one surface with a net TENSILE f, so it is the one that
    behaves the other way: a {100} cube contracts and its interior is
    compressed, raising D.

    f_mean = (-1.032 + 5.024)/2 = +2.0 N/m tensile. Confirmed by the free 2D
    vc-relax of the 16L (100) slab, whose [1-10] axis contracted by -0.664 %.
    """
    cube = ps.evaluate(TAU, {"100": 1.0}, 1.5, ELASTIC, PARAMS)
    assert cube.pressure_gpa > 0
    assert cube.nv_rows[0]["delta_D_mhz"] > 0
    assert cube.nv_rows[0]["eps_axial"] < 0


def test_the_three_surfaces_do_not_all_shift_d_the_same_way():
    """
    (100) shifts D up while (110) and (111) shift it down. If a future change
    makes all three agree, a sign has been applied globally somewhere.
    """
    signs = {}
    for fam in ("100", "110", "111"):
        r = ps.evaluate(TAU, {fam: 1.0}, 1.5, ELASTIC, PARAMS)
        signs[fam] = r.nv_rows[0]["delta_D_mhz"]
    assert signs["100"] > 0
    assert signs["110"] < 0
    assert signs["111"] < 0


def test_delta_d_scales_inversely_with_radius():
    small = ps.evaluate(TAU, {"111": 1.0}, 1.5, ELASTIC, PARAMS)
    big = ps.evaluate(TAU, {"111": 1.0}, 3.0, ELASTIC, PARAMS)
    assert big.pressure_gpa == pytest.approx(small.pressure_gpa / 2.0, rel=1e-12)
    assert big.nv_rows[0]["delta_D_mhz"] == pytest.approx(
        small.nv_rows[0]["delta_D_mhz"] / 2.0, rel=1e-3)


# ====================================================================== Wulff
def test_wulff_refuses_the_negative_h_rich_surface_energies():
    """At the H-rich limit two gammas are negative and no equilibrium shape
    exists. The refusal must say so AND name the mu_H at which it becomes
    available, so the user has an action rather than a dead end."""
    gammas, cfg, prov = ps.load_surface_energies(SURFACE_ENERGIES)
    assert gammas["111"] < 0 and gammas["110"] < 0
    assert cfg["source_type"] == "project_dft_fit"      # derived, not hand-entered
    with pytest.raises(ps.ParticleStrainError) as exc:
        ps.wulff_area_fractions(
            gammas, available_from=prov["wulff_available_from_delta_mu_ev"])
    message = str(exc.value)
    assert "undefined for non-positive surface energies" in message
    assert "unbounded" in message
    assert "not a bug in the energies" in message
    assert "--mu-h-offset" in message


def test_lowering_mu_h_raises_every_gamma_at_its_own_rate():
    """gamma(mu_H) = gamma_h_rich + (N_H/2A)*delta_mu. The rates differ per
    facet, which is why the ORDERING is a function of mu_H."""
    at0, _cfg, prov = ps.load_surface_energies(SURFACE_ENERGIES, 0.0)
    at1, _cfg, _p = ps.load_surface_energies(SURFACE_ENERGIES, 1.0)
    for fam in at0:
        assert at1[fam] > at0[fam]
        assert at1[fam] - at0[fam] == pytest.approx(
            prov["dgamma_dmu_j_m2_per_ev"][fam], abs=1e-9)
    # (110) has the highest hydrogen coverage per area and so rises fastest.
    rates = prov["dgamma_dmu_j_m2_per_ev"]
    assert max(rates, key=rates.get) == "110"


def test_hydrogen_richer_than_h2_is_refused():
    with pytest.raises(ps.ParticleStrainError, match="H2 condenses"):
        ps.load_surface_energies(SURFACE_ENERGIES, -0.5)


def test_wulff_becomes_available_below_the_stated_mu_h():
    _g, _cfg, prov = ps.load_surface_energies(SURFACE_ENERGIES)
    threshold = prov["wulff_available_from_delta_mu_ev"]
    assert threshold is not None and threshold > 0

    just_below, _c, _p = ps.load_surface_energies(SURFACE_ENERGIES,
                                                  threshold - 0.02)
    with pytest.raises(ps.ParticleStrainError):
        ps.wulff_area_fractions(just_below)

    just_above, _c, _p = ps.load_surface_energies(SURFACE_ENERGIES,
                                                  threshold + 0.02)
    assert all(v > 0 for v in just_above.values())
    fractions = ps.wulff_area_fractions(just_above)
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_wulff_shape_evolves_with_mu_h():
    """The equilibrium habit is not a fixed fact: as mu_H falls, the {100}
    fraction grows because gamma_100 rises most slowly."""
    prev = None
    for delta_mu in (0.40, 1.0, 2.0, 2.8):
        gammas, _c, _p = ps.load_surface_energies(SURFACE_ENERGIES, delta_mu)
        fractions = ps.wulff_area_fractions(gammas)
        x100 = fractions.get("100", 0.0)
        if prev is not None:
            assert x100 > prev, delta_mu
        prev = x100
    assert prev > 0.3          # {100} is a major facet by the H-poor end


def test_wulff_area_fractions_still_accepts_a_bare_offset():
    """Retained for testing the geometry independently of mu_H."""
    gammas, _cfg, _p = ps.load_surface_energies(SURFACE_ENERGIES)
    fractions = ps.wulff_area_fractions(gammas, offset=1.5)
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_wulff_gives_a_cube_when_100_is_much_cheaper():
    fractions = ps.wulff_area_fractions({"100": 1.0, "110": 9.0, "111": 9.0})
    assert set(fractions) == {"100"}
    assert fractions["100"] == pytest.approx(1.0)


def test_wulff_gives_an_octahedron_when_111_is_much_cheaper():
    fractions = ps.wulff_area_fractions({"100": 9.0, "110": 9.0, "111": 1.0})
    assert set(fractions) == {"111"}


def test_wulff_gives_a_mixed_shape_at_comparable_energies():
    fractions = ps.wulff_area_fractions({"100": 1.0, "110": 9.0, "111": 0.95})
    assert set(fractions) == {"100", "111"}
    assert all(v > 0.01 for v in fractions.values())
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_polygon_area_of_a_unit_square():
    pts = [np.array(p, dtype=float)
           for p in [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]]
    assert ps._polygon_area(pts, np.array([0.0, 0.0, 1.0])) == pytest.approx(1.0)


# ================================================================== tau intake
@pytest.mark.skipif(not os.path.isfile(TAU_CSV),
                    reason="run fit_tau_infinity.py first")
def test_taus_load_from_the_module_a_output():
    taus = ps.load_taus(TAU_CSV)
    assert set(taus) == {"100", "110", "111"}
    assert taus["111"]["tau_xx"] == pytest.approx(taus["111"]["tau_yy"])
    assert taus["100"]["axis_x"] == "[110]"
    assert taus["100"]["axis_y"] == "[1-10]"
    # NEGATED on the way in: the CSV holds tau_project = +1.0323 (compressive),
    # and the mechanics downstream want the continuum f = -1.0323. If this ever
    # comes back positive, every Delta D in results/production has flipped sign.
    assert taus["100"]["tau_xx"] == pytest.approx(-1.0323, abs=1e-3)
    assert taus["100"]["tau_yy"] == pytest.approx(+5.0241, abs=1e-3)


@pytest.mark.skipif(not os.path.isfile(TAU_CSV),
                    reason="run fit_tau_infinity.py first")
def test_loaded_taus_are_the_negation_of_the_csv():
    """
    Pins the sign flip against the file itself, so the two cannot drift.

    Settled against data, not argument: in free 2D vc-relax at 16 layers,
    positive tau_project EXPANDED the cell on 6 of 6 axes. A surface under
    tensile stress must pull the cell in, so positive tau_project is
    compressive and f = -tau_project.
    """
    import csv as _csv
    loaded = ps.load_taus(TAU_CSV)
    with open(TAU_CSV, newline="") as fh:
        for row in _csv.DictReader(fh):
            fam = row["surface"][1:]
            assert loaded[fam]["tau_xx"] == pytest.approx(
                -float(row["tau_xx_inf_n_per_m"]), abs=1e-12)
            assert loaded[fam]["tau_yy"] == pytest.approx(
                -float(row["tau_yy_inf_n_per_m"]), abs=1e-12)


def test_missing_tau_csv_points_at_module_a(tmp_path):
    with pytest.raises(ps.ParticleStrainError, match="fit_tau_infinity.py"):
        ps.load_taus(tmp_path / "absent.csv")


# ====================================================================== CLI
@pytest.mark.skipif(not os.path.isfile(TAU_CSV),
                    reason="run fit_tau_infinity.py first")
def test_cli_octahedron_writes_labelled_outputs(tmp_path):
    rc = ps.main(["--shape", "octahedron", "--radius-nm", "1.5",
                  "--tau-csv", TAU_CSV, "--runs-dir", PRODUCTION,
                  "--out-dir", str(tmp_path)])
    assert rc == 0
    report = (tmp_path / "particle_strain_report_octahedron.md").read_text()
    # Epistemic level and the literature dependency must be in the OUTPUT,
    # not only in a comment (CLAUDE.md sec 9).
    assert "Epistemic level: L1" in report
    assert "LITERATURE" in report
    assert "E = 0 exactly" in report
    assert "positive = COMPRESSED" in report
    nv = (tmp_path / "particle_strain_nv_octahedron.csv").read_text()
    assert "literature" in nv and "L1" in nv


@pytest.mark.skipif(not os.path.isfile(TAU_CSV),
                    reason="run fit_tau_infinity.py first")
def test_cli_wulff_exits_nonzero_with_an_explanation(capsys, tmp_path):
    rc = ps.main(["--shape", "wulff", "--tau-csv", TAU_CSV,
                  "--runs-dir", PRODUCTION, "--out-dir", str(tmp_path)])
    assert rc == 1
    assert "non-positive surface energies" in capsys.readouterr().err


# ==================================================== coupling-set uncertainty
def test_both_coupling_sets_are_emitted_by_default_from_the_cli(tmp_path):
    """Item 4: the coupling-set spread is the dominant uncertainty in the MHz
    values and must be visible in the output, not hidden behind a flag."""
    rc = ps.main(["--shape", "octahedron", "--tau-csv", TAU_CSV,
                  "--runs-dir", PRODUCTION, "--out-dir", str(tmp_path)])
    assert rc == 0
    nv = (tmp_path / "particle_strain_nv_octahedron.csv").read_text()
    assert "udvarhelyi2018_dft" in nv and "barson2017_scaled" in nv
    assert "delta_D_band_lo_mhz" in nv and "delta_D_band_hi_mhz" in nv
    report = (tmp_path / "particle_strain_report_octahedron.md").read_text()
    assert "Coupling-set uncertainty" in report
    assert "Delta D band" in report


def test_evaluate_emits_a_band_across_parameter_sets():
    result = ps.evaluate(TAU, {"111": 1.0}, 1.5, ELASTIC,
                         [nv_spin_strain.UDVARHELYI_DFT,
                          nv_spin_strain.BARSON_SCALED])
    assert result.param_sets == ["udvarhelyi2018_dft", "barson2017_scaled"]
    assert len(result.nv_rows) == 8            # 2 sets x 4 NV axes
    lo, hi = result.delta_d_band_mhz()
    assert lo < hi
    # The Barson set is the Udvarhelyi tensor scaled by 4.4/2.66 on the axial
    # channel, so a purely hydrostatic strain must reproduce that ratio.
    assert result.coupling_spread_factor == pytest.approx(4.4 / 2.66, rel=0.02)


def test_band_brackets_every_single_set_result():
    both = ps.evaluate(TAU, {"111": 1.0}, 1.5, ELASTIC,
                       [nv_spin_strain.UDVARHELYI_DFT, nv_spin_strain.BARSON_SCALED])
    for params in (nv_spin_strain.UDVARHELYI_DFT, nv_spin_strain.BARSON_SCALED):
        one = ps.evaluate(TAU, {"111": 1.0}, 1.5, ELASTIC, params)
        lo, hi = both.delta_d_band_mhz()
        assert lo - 1e-9 <= one.nv_rows[0]["delta_D_mhz"] <= hi + 1e-9


def test_single_parameter_set_still_works_and_is_flagged_in_the_report(tmp_path):
    rc = ps.main(["--shape", "octahedron", "--params", "dft",
                  "--tau-csv", TAU_CSV, "--runs-dir", PRODUCTION,
                  "--out-dir", str(tmp_path)])
    assert rc == 0
    report = (tmp_path / "particle_strain_report_octahedron.md").read_text()
    assert "only one parameter set was emitted" in report


def test_coupling_choice_does_not_move_the_strain():
    """The spread is a coupling-constant uncertainty, not a mechanics one: the
    interior stress and strain must be identical across parameter sets."""
    a = ps.evaluate(TAU, {"111": 1.0}, 1.5, ELASTIC, nv_spin_strain.UDVARHELYI_DFT)
    b = ps.evaluate(TAU, {"111": 1.0}, 1.5, ELASTIC, nv_spin_strain.BARSON_SCALED)
    assert np.allclose(a.eps_cubic, b.eps_cubic)
    assert a.pressure_gpa == pytest.approx(b.pressure_gpa)


def test_evaluate_refuses_an_empty_parameter_set_list():
    with pytest.raises(ps.ParticleStrainError, match="no spin-strain parameter"):
        ps.evaluate(TAU, {"111": 1.0}, 1.5, ELASTIC, [])
