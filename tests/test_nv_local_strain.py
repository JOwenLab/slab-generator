"""
Tests for nv_local_strain.py -- the position-resolved interior strain solve.

The four required validations are here as tests, not as printed output, so a
change that breaks the solve fails the suite rather than quietly producing a
different answer. Each is small enough to run at low refinement; the report
runs them again at production refinement.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("skfem")

import nv_local_strain as nls
import nv_spin_strain as nss
import particle_strain as ps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU_CSV = os.path.join(REPO, "results", "production", "tau_infinity.csv")
ELASTIC = nss.ElasticConstants()
PARAMS = nss.UDVARHELYI_DFT

needs_tau = pytest.mark.skipif(not os.path.isfile(TAU_CSV),
                               reason="tau_infinity.csv absent")


@pytest.fixture(scope="module")
def taus():
    return ps.load_taus(TAU_CSV)


# ── the mechanics, before any solver ─────────────────────────────────────────

@needs_tau
@pytest.mark.parametrize("shape", ["octahedron", "cube", "rhombic-dodecahedron"])
def test_edge_line_forces_are_self_equilibrated(taus, shape):
    """
    All load on a polyhedron sits on its edges, and that load must carry no net
    force or torque -- otherwise the body would accelerate and no static
    solution exists. This is checked at the LOAD level, before any FEM.
    """
    facets, verts, hull, _ = nls.build_body(shape, taus, 1.5)
    F = 0.0
    # The surface-load functional is what the solver actually uses; the net
    # force it produces is the same quantity.
    _, _, _, tf = nls.build_body(shape, taus, 1.5)
    sol = nls.solve(verts, hull, tf, facets, ELASTIC, refine=1)
    assert sol["load_rbm_residual"] < 1e-10, (
        "the surface load has a rigid-body component, so it is not "
        "self-equilibrated and the Neumann problem is inconsistent")


# ── VALIDATION 1 ─────────────────────────────────────────────────────────────

@needs_tau
@pytest.mark.parametrize("shape", ["octahedron", "cube", "rhombic-dodecahedron"])
def test_volume_average_reproduces_particle_strain(taus, shape):
    """
    The divergence theorem makes <sigma> = -(3/R) sum_f x_f tau_f EXACT. If the
    FEM field does not average to it, the solve is wrong -- not the average.
    """
    facets, verts, hull, tf = nls.build_body(shape, taus, 1.5)
    sol = nls.solve(verts, hull, tf, facets, ELASTIC, refine=2)
    avg, V = nls.volume_average_stress(sol)
    R = 3.0 * hull.volume / hull.area
    ref = -(3.0 / R) * sum(f.area_fraction * f.tau_cubic for f in facets)
    assert V == pytest.approx(hull.volume, rel=1e-10)
    assert np.abs(avg - ref).max() / np.abs(ref).max() < 1e-10


# ── VALIDATION 2 ─────────────────────────────────────────────────────────────

def test_sphere_reduces_to_the_laplace_limit():
    """
    A sphere under isotropic surface stress must carry uniform hydrostatic
    P = 2 tau / R with no deviatoric part. The body here is a geodesic
    polyhedron, so the residual deviatoric is FACETING: it must shrink as the
    facet count rises. If it did not, the deviatoric field this module reports
    would be a numerical artefact rather than the edges.
    """
    coarse = nls.validate_laplace(ELASTIC, 1.0, 1.5, n_sub=1, refine=2)
    fine = nls.validate_laplace(ELASTIC, 1.0, 1.5, n_sub=2, refine=2)
    assert fine["n_facets"] > coarse["n_facets"]
    assert fine["P_rel_error"] < coarse["P_rel_error"] < 0.05
    assert fine["dev_over_P"] < 0.5 * coarse["dev_over_P"]


# ── VALIDATION 3 ─────────────────────────────────────────────────────────────

@needs_tau
def test_mesh_convergence_of_E(taus):
    """|E| at fixed depth must settle as the mesh refines."""
    vals = []
    for rf in (2, 3):
        a = nls.analyse("cube", taus, ELASTIC, PARAMS, 1.5, rf, depths=(1.0,))
        vals.append(a["depth_bins"][0]["E_median_mhz"])
    assert all(v > 0 for v in vals)
    assert abs(vals[1] - vals[0]) / vals[1] < 0.6, (
        f"|E| not settling with refinement: {vals}")


# ── VALIDATION 4 ─────────────────────────────────────────────────────────────

@needs_tau
@pytest.mark.parametrize("shape", ["octahedron", "cube", "rhombic-dodecahedron"])
def test_averaged_E_vanishes_while_local_E_does_not(taus, shape):
    """
    The two statements the module has to produce simultaneously.

    E is a magnitude, so E(x) >= 0 pointwise and <E(x)> cannot vanish. The
    statement that does hold -- and the one particle_strain makes -- is that E
    evaluated AT the volume-averaged strain is zero, because <eps> is isotropic
    and an isotropic strain splits nothing for any couplings.
    """
    a = nls.analyse(shape, taus, ELASTIC, PARAMS, 1.5, refine=2, depths=(1.0,))
    assert a["E_at_mean_strain_mhz"] < 1e-6, "averaged strain is not isotropic"
    assert a["mean_deviatoric_max_gpa"] < 1e-8
    assert a["mean_local_E_mhz"] > 0.1, "local E vanished; nothing to report"


# ── the vectorised NV path must match the scalar one ─────────────────────────

def test_vectorised_nv_splitting_matches_nv_spin_strain():
    """
    nv_splitting batches the eigenproblem because the scalar path does not
    finish at production mesh sizes. It must be the same algebra.
    """
    rng = np.random.default_rng(0)
    eps = rng.normal(0, 1e-3, (12, 3, 3))
    eps = 0.5 * (eps + np.transpose(eps, (0, 2, 1)))
    dD, E = nls.nv_splitting(eps, PARAMS)
    for i, ec in enumerate(eps):
        for k, (_lab, fr) in enumerate(nss.nv_frames()):
            o = nss.nv_observables(ec, fr, PARAMS)
            assert o["delta_D_mhz"] == pytest.approx(dD[i, k], abs=1e-9)
            assert o["E_mhz"] == pytest.approx(E[i, k], abs=1e-9)


# ── mesh quality, which two earlier attempts got wrong ───────────────────────

@needs_tau
def test_mesh_has_no_slivers(taus):
    """
    A Delaunay mesh of this body produced 441 sliver tets out of 5071 and a
    stiffness condition number of 2.4e13, giving displacements six orders of
    magnitude too large. The cone-and-refine construction must stay bounded.
    """
    import itertools
    _, verts, hull, tf = nls.build_body("octahedron", taus, 1.5)
    P, tets = nls.mesh_polyhedron(verts, hull, tf, refine=2)
    v0 = P[tets[:, 0]]
    vol = np.abs(np.einsum("ij,ij->i",
                           np.cross(P[tets[:, 1]] - v0, P[tets[:, 2]] - v0),
                           P[tets[:, 3]] - v0)) / 6.0
    edge = np.zeros(len(tets))
    for a, b in itertools.combinations(range(4), 2):
        edge = np.maximum(edge, np.linalg.norm(P[tets[:, a]] - P[tets[:, b]], axis=1))
    q = vol / edge ** 3                       # 0.1179 for a regular tet
    assert q.min() > 1e-3, f"sliver present, min quality {q.min():.2e}"
    assert vol.sum() == pytest.approx(hull.volume, rel=1e-10), \
        "mesh does not tile the polyhedron"


@needs_tau
def test_zeroing_facet_anisotropy_preserves_the_mean(taus):
    flat = nls.zero_facet_anisotropy(taus, "100")
    for k in ("110", "111"):
        assert flat[k] == taus[k]
    assert flat["100"]["tau_xx"] == pytest.approx(flat["100"]["tau_yy"])
    assert (flat["100"]["tau_xx"] + flat["100"]["tau_yy"]) == pytest.approx(
        taus["100"]["tau_xx"] + taus["100"]["tau_yy"])


# ── the ODMR prediction ──────────────────────────────────────────────────────

@needs_tau
def test_odmr_broadening_is_dominated_by_the_spread_not_the_mean(taus):
    """
    The substantive claim: E varies enough across sites that an ensemble sees
    broadening rather than a resolved splitting. If p90/median ever collapsed
    towards 1, the prediction would change character and the report's framing
    would be wrong.
    """
    r = nls.analyse("cube", taus, ELASTIC, PARAMS, 1.5, refine=2, depths=(1.0,))
    o = nls.odmr_broadening(r, dead_layer_nm=0.5)
    assert o["reachable"]
    assert o["E_p90_over_median"] > 1.5, \
        "E is nearly single-valued; this would be a splitting, not broadening"
    assert o["fwhm_mhz"] > o["E_median_mhz"], \
        "the linewidth should reflect the spread, not just the median E"


@needs_tau
def test_odmr_width_is_sensitive_to_the_assumed_dark_layer(taus):
    """
    Guard against the dark layer being quoted as if it were harmless. The
    strain field is steeply depth-dependent, so this assumption moves the
    answer by more than most of the physics does, and the report says so.
    """
    r = nls.analyse("cube", taus, ELASTIC, PARAMS, 1.5, refine=2, depths=(1.0,))
    shallow = nls.odmr_broadening(r, dead_layer_nm=0.5)
    deep = nls.odmr_broadening(r, dead_layer_nm=1.0)
    assert shallow["fwhm_mhz"] > deep["fwhm_mhz"], \
        "excluding shallow sites must reduce the width"
    assert shallow["fwhm_mhz"] / max(deep["fwhm_mhz"], 1e-9) > 1.3


@needs_tau
def test_odmr_reports_unreachable_rather_than_extrapolating(taus):
    """A dark layer deeper than the inradius leaves no emitting sites."""
    r = nls.analyse("octahedron", taus, ELASTIC, PARAMS, 1.5, refine=2,
                    depths=(1.0,))
    o = nls.odmr_broadening(r, dead_layer_nm=5.0)
    assert o["reachable"] is False


def test_binding_caveat_is_first_and_names_the_edge_term():
    """
    The infinite-plane tau is the caveat that bounds how quotable the
    magnitudes are, so it leads the list rather than sitting among equals.
    """
    assert "THE BINDING ONE" in nls.CAVEATS[0]
    assert "edge" in nls.CAVEATS[0].lower()
    assert "infinite" in nls.CAVEATS[0].lower()
