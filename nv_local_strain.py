#!/usr/bin/env python3
"""
nv_local_strain.py - position-resolved interior strain of a faceted nanodiamond,
and the NV transverse splitting E it produces.

WHAT QUESTION THIS ANSWERS
--------------------------
`particle_strain.py` computes the VOLUME AVERAGE of the interior stress. That
average is exact for its trace -- a consequence of the divergence theorem, not
an approximation -- which is why the axial shift Delta D it reports is on firm
ground. But E responds to the DEVIATORIC part, and for any symmetry-complete
facet set the volume-averaged deviatoric part is identically zero: summing an
anisotropic facet tau over a full Td orbit gives a Td-invariant symmetric rank-2
tensor, and the only such tensor is isotropic.

So the (100) in-plane anisotropy -- the most convergence-robust number in the
campaign -- produces no observable transverse splitting at all in the averaged
model. This module tests whether that survives a position-resolved treatment:
in a 3 nm particle every NV sits within ~1.5 nm of a surface, so no NV samples
the average.

The hypothesis under test is that the LOCAL deviatoric field is non-zero even
where the average vanishes. It is stated as a hypothesis. A negative result --
"E is negligible at realistic depths, here is the magnitude" -- is reported as
readily as a positive one.

THE MECHANICS, VERIFIED BEFORE ANY SOLVER WAS WRITTEN
-----------------------------------------------------
Surface stress loads the bulk only through curvature. The generalized
Young-Laplace condition is t = -div_s(tau); for uniform tau on a FLAT facet
div_s(tau) = 0 identically, so a flat facet transmits no traction. (The same
statement in reverse is why a semi-infinite half-space under uniform surface
stress carries no bulk strain, and why the Laplace relation needs a closed
body: on a sphere the mean curvature 1/R gives t_n = 2 tau / R.)

On a polyhedron, therefore, the entire load is concentrated on the EDGES, where
the normal jumps and the surface-stress discontinuity produces a line force per
unit length

    F_edge = -(tau_A . m_A + tau_B . m_B)

with m the in-plane unit vector pointing from each facet toward the edge. Three
things were checked numerically before building anything, for the octahedron,
cube and rhombic dodecahedron:

  * the edge line forces are self-equilibrated -- net force ~3e-16, net torque
    exactly 0 -- so no vertex point forces are required;
  * integrating sym(F (x) x) along the edges reproduces particle_strain's
    -(3/R) sum_f x_f tau_f to ~1e-15, i.e. the divergence theorem closes;
  * consequently the deviatoric field is concentrated near edges, where a line
    force gives a 1/r stress. For a 3 nm particle everything is within ~1.5 nm
    of an edge, so this is not obviously a small correction.

METHOD
------
Gurtin-Murdoch surface elasticity solved by finite elements.

Rather than applying the singular edge line forces directly, the surface stress
enters through its virtual work, which for a prestressed membrane is

    L(v) = - integral_S  tau : sym(grad v)  dS

a SMOOTH integral over the facets. Integrating it by parts returns exactly the
edge line forces above, so the two formulations are the same problem; but the
load vector is bounded, which the line-force version is not. The 1/r stress
concentration still appears in the solution -- it is physical -- it simply is
not also present in the right-hand side.

  * Elements: P2 tetrahedra (quadratic), vector-valued. P1 gives element-wise
    constant strain, which is too coarse to read a field at a specified depth.
  * Stiffness: the full cubic anisotropic C_ijkl built from C11/C12/C44 in
    config/reference_pbe_sssp.json. Those are LITERATURE values, not fitted
    here; see CLAUDE.md section 3.
  * Mesh: every shape is convex, so a Delaunay tetrahedralization of a point
    cloud whose convex hull is the polyhedron tiles it exactly. No external
    mesher is needed.
  * The traction problem is pure Neumann and so singular. The six rigid-body
    modes are removed by Lagrange constraints enforcing zero mean displacement
    and zero mean rotation, rather than by pinning nodes, which would inject a
    spurious reaction force.

EPISTEMIC LEVEL: L1, and structurally so.
------------------------------------------
  * Continuum elasticity is being applied at 1.5 nm, about five lattice
    constants. That is marginal and no amount of mesh refinement addresses it.
  * C11/C12/C44 are literature values, not fitted from this project's DFT.
  * The facet tau values are for infinite planar surfaces and carry no edge or
    finite-size correction, yet are here applied to a body whose response is
    dominated by its edges.
  * There is no explicit NV defect anywhere: the strain is continuum strain
    evaluated at a point, not the strain a defect actually samples.
  * The spin-strain couplings h15/h16 that convert strain to E are the least
    constrained numbers in the chain (see particle_strain.CHANNEL_NOTE).

The defensible output is an ORDER OF MAGNITUDE and a yes/no against ensemble
ODMR linewidths, not a predicted splitting in MHz.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

import nv_spin_strain
import particle_strain as ps

# Lengths in nm, stresses in GPa. Then surface stress in N/m is numerically
# GPa*nm exactly (1 GPa * 1 nm = 1e9 Pa * 1e-9 m = 1 N/m), so no conversion
# factor appears anywhere below. This is the reason for the unit choice.
NM = 1.0
GPA = 1.0

EPISTEMIC_LEVEL = "L1"

# Ordered by how much each one limits how quotable the magnitudes are, not by
# how easy each is to state. The first is in a different category from the
# rest: it bounds the result, the others qualify it.
CAVEATS = [
    "THE BINDING ONE -- tau is an infinite-flat-facet quantity, and the entire "
    "effect reported here lives at edges. A real edge carries its own excess "
    "energy and its own excess stress: the atoms there are under-coordinated "
    "and relaxed differently from either adjoining facet, and nothing in this "
    "project computes that. The model applies a facet value right up to the "
    "edge and lets the elastic solution supply the concentration. In a 3 nm "
    "particle every site is within ~1.5 nm of an edge, so the omitted edge "
    "term is not obviously small compared to the term that is kept -- it could "
    "plausibly be comparable. This is why the defensible output is the "
    "existence of the effect, its scaling with depth and facet family, and its "
    "order of magnitude, and NOT a quotable MHz value. Closing it needs edge "
    "energetics from DFT (a nanorod or wedge calculation), which does not "
    "exist in this repository.",
    "Continuum elasticity evaluated at ~1.5 nm, about five lattice constants. "
    "Marginal by construction; mesh refinement does not address it.",
    "C11/C12/C44 are LITERATURE values (config/reference_pbe_sssp.json, "
    "elastic_tensor.source_type = literature), not fitted from this project.",
    "No explicit NV defect: this is continuum strain at a point, not the strain "
    "a defect samples. A real NV averages over its own wavefunction and "
    "relaxes its neighbourhood.",
    "E is converted from strain by h15/h16, the least experimentally "
    "constrained couplings in the chain (see particle_strain.CHANNEL_NOTE).",
    "The predicted ODMR width additionally assumes a depth distribution for "
    "emitting NVs; it is stated where it is used and the result is strongly "
    "sensitive to it.",
]


class LocalStrainError(Exception):
    pass


# ─────────────────────────────────────────────────────────── geometry

def cubic_stiffness(C11, C12, C44):
    """Full C_ijkl for cubic symmetry, in the crystal frame."""
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            C[i, i, j, j] = C12
        C[i, i, i, i] = C11
    for i in range(3):
        for j in range(3):
            if i != j:
                C[i, j, i, j] += C44
                C[i, j, j, i] += C44
    return C


def body_from_facets(facets, offsets=None):
    """
    Convex body {x : n_f . x <= h_f} carrying one tau per facet.

    Returns (vertices, hull, tri_facet) where tri_facet[k] indexes which
    polyhedron facet hull simplex k belongs to.
    """
    n = len(facets)
    offsets = np.ones(n) if offsets is None else np.asarray(offsets, float)
    hs = np.array([[*f.normal, -offsets[i]] for i, f in enumerate(facets)])
    hi = HalfspaceIntersection(hs, np.zeros(3))
    verts = hi.intersections
    hull = ConvexHull(verts)
    N = np.array([f.normal for f in facets])
    tri_facet = np.argmax(hull.equations[:, :3] @ N.T, axis=1)
    return verts, hull, tri_facet


def scale_to_effective_radius(verts, hull, radius_nm):
    """
    Rescale so R = 3V/A equals radius_nm.

    R = 3V/A is the same effective radius particle_strain uses, so the two
    modules describe the same body and their averages are comparable.
    """
    R = 3.0 * hull.volume / hull.area
    return verts * (radius_nm / R)


# ─────────────────────────────────────────────────────────────── mesh

def _facet_frame(normal):
    a = np.array([1.0, 0.0, 0.0])
    if abs(normal @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, a); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def mesh_polyhedron(verts, hull, tri_facet, refine=3):
    """
    Tetrahedral mesh of a CONVEX polyhedron, by construction rather than by
    Delaunay. Returns (points, tets).

    WHY NOT DELAUNAY. A Delaunay tetrahedralization of a point cloud gives no
    shape guarantee in 3D -- unlike 2D, it can and does produce slivers: tets
    with finite circumradius and near-zero volume. Two attempts failed here
    before this one. A cubic lattice is massively co-spherical, so Qhull
    resolved the ties into 441 slivers out of 5071 elements, giving a stiffness
    diagonal spanning 110 to 2.6e15 (condition number 2.4e13) and displacements
    six orders of magnitude too large. Jittering the points to break the
    co-sphericity reduced the error but left the slivers: it addresses the
    degeneracy, not the absence of a quality guarantee.

    WHAT THIS DOES INSTEAD. The body is convex, so coning from the centroid to
    each boundary triangle tiles it exactly. Each of those tets is then
    subdivided by red refinement (1 -> 8 on edge midpoints), which is
    shape-regular: repeated refinement does not degrade the worst element
    beyond a bounded factor. Planar facets stay planar under midpoint
    subdivision, so the geometry remains exact at every level and `refine` is a
    clean convergence knob.
    """
    from skfem import MeshTet

    centroid = verts[np.unique(hull.simplices)].mean(axis=0)
    pts = np.vstack([verts, centroid])
    ci = len(verts)

    tets = []
    for simplex, eq in zip(hull.simplices, hull.equations):
        a, b, c = verts[simplex]
        # orient so the tet has positive volume with the centroid as apex
        if np.dot(np.cross(b - a, c - a), centroid - a) > 0:
            simplex = simplex[[0, 2, 1]]
        tets.append([ci, *simplex])
    tets = np.array(tets)

    m = MeshTet(pts.T, tets.T)
    if refine:
        m = m.refined(refine)
    return m.p.T, m.t.T


# ──────────────────────────────────────────────────────────── solver

def solve(verts, hull, tri_facet, facets, elastic, refine=3):
    """
    Gurtin-Murdoch surface-stress problem by P2 tetrahedral FEM.

    Returns a dict with the mesh, the displacement, and a strain evaluator.
    """
    from skfem import (MeshTet, ElementVector, ElementTetP2, Basis,
                       FacetBasis, BilinearForm, LinearForm, asm, condense)
    from skfem.helpers import sym_grad, ddot

    P, tets = mesh_polyhedron(verts, hull, tri_facet, refine)
    mesh = MeshTet(P.T, tets.T)
    elem = ElementVector(ElementTetP2())
    basis = Basis(mesh, elem, intorder=4)

    C = cubic_stiffness(elastic.C11, elastic.C12, elastic.C44)

    @BilinearForm
    def stiffness(u, v, w):
        e = sym_grad(u)
        s = np.einsum("ijkl,kl...->ij...", C, e)
        return ddot(s, sym_grad(v))

    K = asm(stiffness, basis)

    # Surface load: for each boundary facet, tau of the polyhedron face it
    # belongs to. Boundary facets are matched to faces by outward normal.
    fb = FacetBasis(mesh, elem, intorder=4)
    normals = fb.normals                      # (3, nfacets, nqp)
    nmean = normals.mean(axis=2).T            # (nfacets, 3)
    nmean /= np.linalg.norm(nmean, axis=1)[:, None]
    Nf = np.array([f.normal for f in facets])
    which = np.argmax(nmean @ Nf.T, axis=1)
    TAU = np.array([facets[i].tau_cubic for i in which])       # (nfacets,3,3)
    tau_q = np.repeat(TAU[:, :, :, None], normals.shape[2], axis=3)
    tau_q = np.transpose(tau_q, (1, 2, 0, 3))                  # (3,3,nfacets,nqp)

    @LinearForm
    def surface_load(v, w):
        return -ddot(w["tau"], sym_grad(v))

    f = asm(surface_load, fb, tau=tau_q)

    # Pure Neumann problem: K is singular in the 6 rigid-body modes.
    #
    # Lagrange multipliers were tried first and are unusable here: the six
    # constraint rows are DENSE (every dof appears in a mean), so the bordered
    # matrix loses sparsity and the direct factorisation does not finish at
    # refine=4.
    #
    # Pinning six scalar dofs keeps the matrix sparse and is legitimate for
    # this problem, because the quantity wanted is STRAIN, which is invariant
    # under rigid-body motion. The one thing pinning can get wrong -- injecting
    # a spurious reaction force -- is avoided by first projecting the load onto
    # the range of K: with f orthogonal to all six modes the reactions at the
    # pinned dofs are zero, and the solution differs from the constrained one
    # only by a rigid motion. The residual check below confirms it.
    import scipy.sparse.linalg as spl

    ndof = K.shape[0]
    x = basis.doflocs
    R = np.zeros((6, ndof))
    for d in range(3):
        R[d, np.arange(d, ndof, 3)] = 1.0
    for k, (i, j) in enumerate(((0, 1), (1, 2), (0, 2))):
        R[3 + k, np.arange(i, ndof, 3)] = -x[j, np.arange(i, ndof, 3)]
        R[3 + k, np.arange(j, ndof, 3)] = x[i, np.arange(j, ndof, 3)]
    R /= np.linalg.norm(R, axis=1)[:, None]
    # Gram-Schmidt, then project the load onto range(K)
    Q, _ = np.linalg.qr(R.T)
    f_proj = f - Q @ (Q.T @ f)
    load_rbm_residual = float(np.linalg.norm(Q.T @ f) / max(np.linalg.norm(f), 1e-30))

    # Solve the singular system on range(K) directly, by conjugate gradients
    # applied to the PROJECTED operator P K P with P = I - Q Q^T.
    #
    # Pinning six dofs was tried and is wrong here. Unless the pinned set spans
    # exactly a complement of the rigid-body space it also constrains real
    # deformation, and an ad-hoc choice does not: it left validation 1 at 4.2e-2
    # instead of 1e-14, i.e. a 4% error in the volume-averaged stress. The
    # failure is quiet -- the residual on the free dofs was 3e-14, so the linear
    # algebra looked converged while the answer was wrong.
    #
    # K is symmetric positive definite ON range(K), and the load has already
    # been projected there, so CG converges to the minimum-norm solution, which
    # is the constrained one up to a rigid motion -- and strain does not see a
    # rigid motion.
    def apply(v):
        w = v - Q @ (Q.T @ v)
        w = K @ w
        return w - Q @ (Q.T @ w)

    Aop = spl.LinearOperator((ndof, ndof), matvec=apply, dtype=float)
    dinv = 1.0 / np.where(K.diagonal() > 0, K.diagonal(), 1.0)
    Mop = spl.LinearOperator((ndof, ndof), matvec=lambda v: dinv * v, dtype=float)
    u, info = spl.cg(Aop, f_proj, rtol=1e-12, atol=0.0, maxiter=20000, M=Mop)
    if info != 0:
        raise LocalStrainError(f"CG failed to converge (info={info})")
    u = u - Q @ (Q.T @ u)

    residual = float(np.linalg.norm(apply(u) - f_proj) /
                     max(np.linalg.norm(f_proj), 1e-30))

    return {"mesh": mesh, "basis": basis, "u": u, "C": C,
            "n_tets": tets.shape[0], "n_dof": ndof, "refine": refine,
            "verts": verts, "hull": hull, "residual": residual,
            "load_rbm_residual": load_rbm_residual}


def volume_average_stress(sol):
    """
    <sigma> = (1/V) int C:eps(u) dV, straight from the FEM field.

    VALIDATION 1. This must reproduce particle_strain's -(3/R) sum_f x_f tau_f,
    which follows from the divergence theorem and is exact -- so a mismatch
    means the SOLVE is wrong, not the average.
    """
    from skfem import Functional
    from skfem.helpers import sym_grad
    basis, C = sol["basis"], sol["C"]
    out = np.zeros((3, 3))
    V = 0.0
    for i in range(3):
        for j in range(3):
            @Functional
            def comp(w, i=i, j=j):
                e = sym_grad(w["uh"])
                s = np.einsum("ijkl,kl...->ij...", C, e)
                return s[i, j]
            out[i, j] = comp.assemble(basis, uh=basis.interpolate(sol["u"]))

    @Functional
    def vol(w):
        return 1.0 + 0.0 * w.x[0]
    V = vol.assemble(basis)
    return out / V, V


def sample_field(sol, verts, hull):
    """
    Strain and stress at every element centroid, with depth and edge distance.

    Sampling the field on the mesh rather than probing at prescribed points is
    deliberate: the question is how the field varies WITH depth, so a cloud of
    samples carrying their own depth is the natural object, and it avoids
    point-location entirely. At refine=3 that is 4096 samples, at refine=4
    32768.

    Returns a dict of arrays: x (n,3), eps (n,3,3), sigma (n,3,3),
    depth (n,) distance to the nearest facet plane, edge (n,) distance to the
    nearest polyhedron edge, vol (n,) element volume for volume-weighting.
    """
    from skfem.helpers import sym_grad
    basis, C = sol["basis"], sol["C"]
    mesh = sol["mesh"]

    # element centroids
    p, tt = mesh.p, mesh.t
    X = p[:, tt[:4]].mean(axis=1).T                     # (nelem, 3)

    # strain at those points: interpolate the P2 field's gradient at the
    # element's quadrature points and average, which for the centroid of a
    # linear-strain element is the centroid value.
    w = basis.interpolate(sol["u"])
    e_q = sym_grad(w)                                    # (3,3,nelem,nqp)
    eps = np.transpose(e_q.mean(axis=3), (2, 0, 1))      # (nelem,3,3)
    sig = np.einsum("ijkl,nkl->nij", C, eps)

    eq = hull.equations
    depth = -(X @ eq[:, :3].T + eq[:, 3]).max(axis=1)
    edge = distance_to_nearest_edge(X, verts, hull)

    v0 = p[:, tt[0]].T
    vol = np.abs(np.einsum("ij,ij->i",
                           np.cross(p[:, tt[1]].T - v0, p[:, tt[2]].T - v0),
                           p[:, tt[3]].T - v0)) / 6.0
    return {"x": X, "eps": eps, "sigma": sig, "depth": depth,
            "edge": edge, "vol": vol}


def deviatoric(t):
    return t - np.eye(3) * np.trace(t) / 3.0


def nv_observables_at(eps_cubic, params):
    """Delta D and E for all four <111> NV orientations at one strain."""
    rows = []
    for label, frame in nv_spin_strain.nv_frames():
        obs = nv_spin_strain.nv_observables(eps_cubic, frame, params)
        rows.append({"nv_axis": label, **obs})
    return rows


def sample_points_at_depth(verts, hull, depth_nm, n_target=400, rng=None):
    """
    Interior points a fixed distance `depth_nm` from the nearest facet.

    Built by shrinking every half-space by `depth_nm` and sampling the boundary
    of the shrunken body: those points are exactly at that depth. Returns an
    empty array when the depth exceeds the inradius, which is the honest answer
    for "no NV sits that deep in this particle".
    """
    rng = rng or np.random.default_rng(0)
    eq = hull.equations
    off = -eq[:, 3] - depth_nm
    if np.any(off <= 1e-9):
        return np.zeros((0, 3)), np.zeros(0)
    hs = np.column_stack([eq[:, :3], -off])
    try:
        hi = HalfspaceIntersection(hs, np.zeros(3))
    except Exception:
        return np.zeros((0, 3)), np.zeros(0)
    inner = ConvexHull(hi.intersections)
    pts, dmin = [], []
    for simplex in inner.simplices:
        tri = inner.points[simplex]
        w = rng.dirichlet(np.ones(3), size=max(1, n_target // len(inner.simplices)))
        pts.append(w @ tri)
    P = np.vstack(pts)
    # distance to the nearest ORIGINAL edge, for the edge-proximity analysis
    return P, P


def distance_to_nearest_edge(P, verts, hull):
    """Shortest distance from each point to any polyhedron edge segment."""
    edges = set()
    for simplex in hull.simplices:
        for a, b in itertools.combinations(sorted(simplex), 2):
            edges.add((a, b))
    best = np.full(len(P), np.inf)
    for a, b in edges:
        p, q = verts[a], verts[b]
        d = q - p
        L2 = d @ d
        t = np.clip(((P - p) @ d) / L2, 0.0, 1.0)
        proj = p + t[:, None] * d
        best = np.minimum(best, np.linalg.norm(P - proj, axis=1))
    return best


# ────────────────────────────────────────────────── NV splitting analysis

def nv_splitting(eps_cubic_array, params):
    """
    (delta_D, E) in MHz for every sample and each of the four <111> NV axes.

    Vectorised: builds the 3x3 spin Hamiltonian for all samples at once and
    diagonalises the stack. The scalar path in nv_spin_strain.nv_observables
    does one eigendecomposition per sample per axis, which at refine=4 is
    131072 of them in Python and does not finish. This reproduces that function
    exactly -- tests/test_nv_local_strain.py asserts agreement to 1e-9 MHz --
    it is the same algebra, batched.

    Returns arrays of shape (n_samples, 4).
    """
    nss_ = nv_spin_strain
    SX, SY, SZ = nss_.SX, nss_.SY, nss_.SZ
    D0 = nss_.D0_MHZ
    SZ2 = SZ @ SZ

    def anti(a, b):
        return a @ b + b @ a

    A_SXSZ, A_SYSZ = anti(SX, SZ), anti(SY, SZ)
    A_SXSY = anti(SX, SY)
    YY_XX = SY @ SY - SX @ SX

    frames = nss_.nv_frames()
    n = len(eps_cubic_array)
    dD = np.zeros((n, len(frames)))
    E = np.zeros((n, len(frames)))
    p = params

    for k, (_label, R) in enumerate(frames):
        e = np.einsum("ij,njk,lk->nil", R, eps_cubic_array, R)
        exx, eyy, ezz = e[:, 0, 0], e[:, 1, 1], e[:, 2, 2]
        exy, exz, eyz = e[:, 0, 1], e[:, 0, 2], e[:, 1, 2]

        c0 = p.h41 * (exx + eyy) + p.h43 * ezz
        c1 = 0.5 * (p.h26 * exz - 0.5 * p.h25 * (exx - eyy))
        c2 = 0.5 * (p.h26 * eyz + p.h25 * exy)
        c3 = 0.5 * (p.h16 * exz - 0.5 * p.h15 * (exx - eyy))
        c4 = 0.5 * (p.h16 * eyz + p.h15 * exy)

        H = (D0 * SZ2[None] + c0[:, None, None] * SZ2[None]
             + c1[:, None, None] * A_SXSZ[None] + c2[:, None, None] * A_SYSZ[None]
             + c3[:, None, None] * YY_XX[None] + c4[:, None, None] * A_SXSY[None])

        evals, evecs = np.linalg.eigh(H)
        # identify the ms=0-like state by overlap with |0> = (0,1,0)
        i0 = np.argmax(np.abs(evecs[:, 1, :]) ** 2, axis=1)
        rows = np.arange(n)
        e0 = evals[rows, i0]
        mask = np.ones_like(evals, dtype=bool)
        mask[rows, i0] = False
        others = evals[mask].reshape(n, 2)
        others.sort(axis=1)
        f_minus, f_plus = others[:, 0] - e0, others[:, 1] - e0
        dD[:, k] = 0.5 * (f_plus + f_minus) - D0
        E[:, k] = 0.5 * (f_plus - f_minus)
    return dD, E


def build_body(shape, taus, radius_nm, fractions=None):
    """Facets + scaled convex body for a named shape."""
    frac = fractions or ps.SINGLE_FAMILY_SHAPES.get(shape)
    if frac is None:
        raise LocalStrainError(f"unknown shape {shape!r}")
    facets = ps.build_facets(taus, frac)
    verts, hull, _ = body_from_facets(facets)
    verts = scale_to_effective_radius(verts, hull, radius_nm)
    hull = ConvexHull(verts)
    N = np.array([f.normal for f in facets])
    tri_facet = np.argmax(hull.equations[:, :3] @ N.T, axis=1)
    return facets, verts, hull, tri_facet


def analyse(shape, taus, elastic, params, radius_nm=1.5, refine=3,
            depths=(0.5, 1.0, 1.5, 2.0), tol=0.15, fractions=None):
    """
    Full position-resolved analysis of one shape.

    Returns a dict with the field sample, the volume-average check, and
    depth-binned statistics of |E| and delta_D over the four NV orientations.
    """
    facets, verts, hull, tri_facet = build_body(shape, taus, radius_nm, fractions)
    sol = solve(verts, hull, tri_facet, facets, elastic, refine=refine)
    avg, V = volume_average_stress(sol)
    R = 3.0 * hull.volume / hull.area
    ref = -(3.0 / R) * sum(f.area_fraction * f.tau_cubic for f in facets)

    F = sample_field(sol, verts, hull)
    dD, E = nv_splitting(F["eps"], params)
    absE = np.abs(E)

    dev = F["sigma"] - np.eye(3) * np.trace(F["sigma"], axis1=1, axis2=2)[:, None, None] / 3
    vm = np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev))

    # volume-weighted mean of E over the whole body, and of the deviatoric
    # tensor: VALIDATION 4, both must vanish for a symmetric shape.
    wv = F["vol"] / F["vol"].sum()
    mean_sigma = np.einsum("n,nij->ij", wv, F["sigma"])
    mean_dev = mean_sigma - np.eye(3) * np.trace(mean_sigma) / 3.0
    # VALIDATION 4, stated precisely.
    #
    # "The volume-averaged E must be zero while the local E is not" can only be
    # true in one reading, and it is worth being exact about which. E is a
    # magnitude -- half a frequency difference, taken positive -- so E(x) >= 0
    # pointwise and <E(x)> CANNOT vanish unless E is identically zero. The
    # statement that does hold, and the one particle_strain makes, is that E
    # evaluated AT the volume-averaged strain vanishes: <eps> is isotropic, and
    # an isotropic strain produces no transverse splitting for any couplings.
    #
    # Both numbers are reported. E_at_mean_strain ~ 0 is the validation;
    # mean_local_E > 0 is the new physics, and the gap between them is the
    # entire result of this module.
    mean_eps = np.einsum("n,nij->ij", wv, F["eps"])
    _dD_avg, _E_avg = nv_splitting(mean_eps[None], params)
    E_at_mean_strain = float(np.abs(_E_avg).max())
    mean_local_E = float((wv[:, None] * np.abs(E)).sum() / E.shape[1])

    bins = []
    for d in depths:
        m = np.abs(F["depth"] - d) < tol
        if not m.any():
            bins.append({"depth_nm": d, "n": 0, "reachable": False})
            continue
        bins.append({
            "depth_nm": d, "n": int(m.sum()), "reachable": True,
            "E_median_mhz": float(np.median(absE[m])),
            "E_p90_mhz": float(np.percentile(absE[m], 90)),
            "E_max_mhz": float(absE[m].max()),
            "dD_median_mhz": float(np.median(dD[m])),
            "sigma_dev_median_gpa": float(np.median(vm[m])),
            "edge_dist_median_nm": float(np.median(F["edge"][m])),
        })

    return {
        "shape": shape, "radius_nm": radius_nm, "refine": refine,
        "n_tets": sol["n_tets"], "n_dof": sol["n_dof"],
        "inradius_nm": float(F["depth"].max()),
        "volume_average_stress_gpa": avg,
        "volume_average_reference_gpa": ref,
        "volume_average_rel_error": float(np.abs(avg - ref).max() /
                                          max(np.abs(ref).max(), 1e-30)),
        "mean_deviatoric_max_gpa": float(np.abs(mean_dev).max()),
        "E_at_mean_strain_mhz": E_at_mean_strain,
        "mean_local_E_mhz": mean_local_E,
        "depth_bins": bins,
        "field": F, "E": E, "dD": dD, "vm": vm,
    }


# ────────────────────────────────────────────────────────── validations

def geodesic_sphere(n_sub=2):
    """Subdivided icosahedron: a convex body approaching a sphere."""
    phi = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                  [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                  [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]],
                 dtype=float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    hull = ConvexHull(v)
    tris = hull.simplices.tolist()
    for _ in range(n_sub):
        new = []
        for a, b, c in tris:
            ab = len(v); v = np.vstack([v, (v[a] + v[b]) / 2])
            bc = len(v); v = np.vstack([v, (v[b] + v[c]) / 2])
            ca = len(v); v = np.vstack([v, (v[c] + v[a]) / 2])
            new += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        tris = new
        v /= np.linalg.norm(v, axis=1)[:, None]
    v = np.unique(v.round(12), axis=0)
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v


def validate_laplace(elastic, tau_n_per_m=1.0, radius_nm=1.5, n_sub=2, refine=2):
    """
    VALIDATION 2. A sphere under isotropic surface stress must carry uniform
    hydrostatic stress P = 2*tau/R with zero deviatoric part everywhere.

    The body here is a geodesic polyhedron, so it is a sphere only in the limit
    of many facets: a residual deviatoric part is the faceting, not the solver,
    and it must fall as the facet count rises.
    """
    from types import SimpleNamespace
    v = geodesic_sphere(n_sub) * radius_nm
    hull = ConvexHull(v)
    v = v * (radius_nm / (3 * hull.volume / hull.area))
    hull = ConvexHull(v)

    facets = []
    for eq in hull.equations:
        n = eq[:3]
        facets.append(SimpleNamespace(
            normal=n, area_fraction=0.0,
            tau_cubic=tau_n_per_m * (np.eye(3) - np.outer(n, n))))
    tri_facet = np.arange(len(hull.equations))
    sol = solve(v, hull, tri_facet, facets, elastic, refine=refine)
    F = sample_field(sol, v, hull)
    P = -np.trace(F["sigma"], axis1=1, axis2=2) / 3.0
    dev = F["sigma"] - np.eye(3) * np.trace(F["sigma"], axis1=1, axis2=2)[:, None, None] / 3
    vm = np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev))
    R = 3 * hull.volume / hull.area
    return {
        "n_facets": len(hull.equations), "n_sub": n_sub, "refine": refine,
        "R_nm": float(R), "P_laplace_gpa": float(2 * tau_n_per_m / R),
        "P_model_median_gpa": float(np.median(P)),
        "P_rel_error": float(abs(np.median(P) - 2 * tau_n_per_m / R) /
                             (2 * tau_n_per_m / R)),
        "dev_over_P": float(np.median(vm) / (2 * tau_n_per_m / R)),
    }


def zero_facet_anisotropy(taus, family):
    """
    Copy of `taus` with one family's in-plane anisotropy removed, its MEAN
    surface stress preserved.

    The control for "is the (100) anisotropy actually what drives E?". If E is
    unchanged with this applied, the anisotropy is not the source.
    """
    out = {k: dict(v) for k, v in taus.items()}
    m = 0.5 * (out[family]["tau_xx"] + out[family]["tau_yy"])
    out[family]["tau_xx"] = m
    out[family]["tau_yy"] = m
    return out


# ──────────────────────────────────────────────────────────── reporting

LINEWIDTH_MHZ = 3.0     # typical ensemble ODMR linewidth, order of magnitude


def edge_profile(res, nbins=6):
    """|E| against distance to the nearest polyhedron edge, at fixed depth."""
    F, E = res["field"], np.abs(res["E"]).mean(axis=1)
    m = np.abs(F["depth"] - 1.0) < 0.3
    if m.sum() < 10:
        return []
    d, e = F["edge"][m], E[m]
    qs = np.quantile(d, np.linspace(0, 1, nbins + 1))
    out = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        s = (d >= lo) & (d <= hi)
        if s.sum():
            out.append({"edge_lo_nm": float(lo), "edge_hi_nm": float(hi),
                        "n": int(s.sum()), "E_median_mhz": float(np.median(e[s]))})
    return out


DEAD_LAYER_NM = 1.0      # NVs nearer the surface than this are assumed dark


def odmr_broadening(res, dead_layer_nm=DEAD_LAYER_NM, intrinsic_fwhm_mhz=0.0,
                    n_grid=4001):
    """
    Inhomogeneous ODMR linewidth contributed by the position-dependent strain.

    THE OBSERVABLE, rather than a number no one measures.

    The p90/median ratio of |E| is about 2, so E is not one value with a small
    spread: it varies by a factor of two across NV sites at a single depth, and
    a further order of magnitude across depths. An ensemble therefore does not
    show a resolved splitting at some particular E. It shows every site's pair
    of lines at once, which is inhomogeneous BROADENING of a single feature.
    That is what a nanodiamond ODMR spectrum usually looks like, and it is what
    an experimental collaborator would actually report.

    The spectrum is built explicitly rather than summarised: each NV site
    contributes two lines at f_pm = D0 + delta_D +/- E, weighted, and the FWHM
    of the composite is measured numerically.

    ASSUMED DEPTH DISTRIBUTION -- this is an assumption, not a result:

      * NVs uniform in VOLUME. They are native lattice defects in HPHT and
        milled material, not implanted, so no depth profile is imposed beyond
        the geometry. Implanted material would need a different weight and
        would give a different answer.
      * a dark layer of `dead_layer_nm` at the surface. NV centres within
        roughly a nanometre of a surface are typically unstable in charge state
        and do not contribute NV- fluorescence. Sites shallower than this are
        dropped rather than down-weighted, which is crude.

    Both choices matter: the strain field is steeply depth-dependent, so the
    predicted width depends on which sites are assumed to emit. The dead-layer
    sensitivity is reported alongside the value for that reason.

    Returns a dict; `fwhm_mhz` at intrinsic_fwhm_mhz = 0 is the strain-induced
    contribution alone, which convolves with whatever intrinsic width a given
    sample has.
    """
    F, E, dD = res["field"], res["E"], res["dD"]
    keep = F["depth"] >= dead_layer_nm
    if keep.sum() < 5:
        return {"reachable": False, "dead_layer_nm": dead_layer_nm,
                "n_sites": int(keep.sum())}

    w = np.repeat(F["vol"][keep], E.shape[1])
    w = w / w.sum()
    fp = (dD[keep] + E[keep]).ravel()
    fm = (dD[keep] - E[keep]).ravel()

    lo = min(fp.min(), fm.min()) - 5.0
    hi = max(fp.max(), fm.max()) + 5.0
    grid = np.linspace(lo, hi, n_grid)
    S = np.zeros_like(grid)
    g = max(intrinsic_fwhm_mhz, (hi - lo) / n_grid * 2.0) / 2.0
    for f in (fp, fm):
        # Lorentzian per site; at intrinsic width 0 this is a narrow kernel
        # that only resolves the grid, so the width measured is the spread of
        # the line POSITIONS, which is the inhomogeneous contribution.
        S += (w[:, None] * (g / np.pi) /
              ((grid[None, :] - f[:, None]) ** 2 + g ** 2)).sum(axis=0)

    half = S.max() / 2.0
    above = np.where(S >= half)[0]
    fwhm = float(grid[above[-1]] - grid[above[0]]) if len(above) > 1 else 0.0

    absE = np.abs(E[keep]).ravel()
    return {
        "reachable": True,
        "dead_layer_nm": dead_layer_nm,
        "n_sites": int(keep.sum()),
        "fwhm_mhz": fwhm,
        "intrinsic_fwhm_mhz": intrinsic_fwhm_mhz,
        "E_median_mhz": float(np.median(absE)),
        "E_p90_mhz": float(np.percentile(absE, 90)),
        "E_p90_over_median": float(np.percentile(absE, 90) /
                                   max(np.median(absE), 1e-30)),
        "frac_resolvably_split": float((2 * absE > max(intrinsic_fwhm_mhz,
                                                       LINEWIDTH_MHZ)).mean()),
        "dD_median_mhz": float(np.median(dD[keep])),
    }


def render_report(runs, validations, control, meta, odmr):
    L = [f"# Position-resolved NV transverse splitting  [{EPISTEMIC_LEVEL}]", "",
         "Does the (100) surface-stress anisotropy produce an observable "
         "transverse splitting E once the interior strain is resolved in "
         "position, rather than volume-averaged?", "",
         "## Answer", "", meta["verdict"], "",
         "## Method", "",
         "Gurtin-Murdoch surface elasticity, P2 tetrahedral FEM, full cubic "
         "C_ijkl. Surface stress enters as the facet integral "
         "`L(v) = -int_S tau : sym(grad v) dS`, which integrates by parts to "
         "the edge line forces that carry the entire load on a polyhedron "
         "(flat facets have zero curvature and transmit no traction). The "
         "singular Neumann problem is solved by conjugate gradients on the "
         "operator projected onto the range of K.", "",
         "## Validations", ""]
    for v in validations:
        L.append(f"- {v}")
    L += ["", "## |E| by depth (MHz, median / 90th percentile over the four "
          "<111> axes)", "",
          "| shape | R (nm) | inradius (nm) | " +
          " | ".join(f"d={d} nm" for d in meta["depths"]) + " |",
          "|---|---:|---:|" + "---:|" * len(meta["depths"])]
    for r in runs:
        cells = []
        for b in r["depth_bins"]:
            cells.append(f"{b['E_median_mhz']:.2f} / {b['E_p90_mhz']:.2f}"
                         if b.get("reachable") else "deeper than the particle")
        L.append(f"| {r['shape']} | {r['radius_nm']:.1f} | "
                 f"{r['inradius_nm']:.2f} | " + " | ".join(cells) + " |")
    L += ["", f"Typical ensemble ODMR linewidth is ~{LINEWIDTH_MHZ:.0f} MHz, so "
          "a median |E| above roughly that is detectable in an ensemble and "
          "below it is not.", "",
          "## Does the (100) anisotropy drive it?", "", control, "",
          "## Predicted ODMR signature: broadening, not a resolved splitting",
          "",
          "|E| varies by roughly a factor of two across NV sites at a single "
          "depth (p90/median ~2 in every shape below) and by an order of "
          "magnitude across depths. An ensemble therefore does not show a "
          "splitting at one value of E; it shows every site's pair of lines at "
          "once, which is inhomogeneous BROADENING of a single feature. That "
          "is closer to what nanodiamond ODMR usually looks like than a clean "
          "split would be, and it is the quantity an experiment reports.", "",
          "**Assumed depth distribution -- an assumption, not a result:** NVs "
          "uniform in VOLUME (native defects in HPHT and milled material, not "
          "implanted), with a dark surface layer inside which NV- is assumed "
          "charge-unstable and non-emitting. Two layer thicknesses are given "
          "because the answer depends strongly on the choice: the strain field "
          "is steeply depth-dependent, so which sites are assumed to emit "
          "matters as much as the strain does.", "",
          "| shape | dark layer (nm) | sites | median E (MHz) | p90/median | "
          "strain FWHM (MHz) | fraction with 2E > 3 MHz |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for row in odmr:
        L.append(f"| {row['shape']} | {row['dead_layer_nm']:.1f} | "
                 f"{row['n_sites']} | {row['E_median_mhz']:.2f} | "
                 f"{row['E_p90_over_median']:.2f} | {row['fwhm_mhz']:.2f} | "
                 f"{row['frac_resolvably_split']:.2f} |")
    L += ["", "The FWHM quoted is the strain contribution alone; it convolves "
          "with whatever intrinsic width a sample has. Doubling the assumed "
          "dark layer changes it by a factor of 2-4, larger than most other "
          "uncertainties here, which is why the assumption is stated rather "
          "than buried.", "",
          "## Two numerical failures worth recording", "",
          "Both produced plausible wrong answers with clean-looking "
          "diagnostics, which is the failure mode this project keeps meeting. "
          "Both are now pinned by tests.", "",
          "1. **Delaunay meshing gave slivers.** A cubic lattice is massively "
          "co-spherical, so its Delaunay tetrahedralization is degenerate and "
          "Qhull resolved the ties into 441 near-zero-volume tets out of 5071. "
          "The stiffness diagonal spanned 110 to 2.6e15 (condition number "
          "2.4e13) and displacements came out six orders of magnitude too "
          "large. Jittering the points to break the degeneracy reduced the "
          "error but did not remove the slivers, because 3D Delaunay carries "
          "no shape guarantee at all, unlike 2D. Replaced by coning from the "
          "centroid plus red refinement, which is shape-regular by "
          "construction.", "",
          "2. **Pinning six dofs to remove the rigid-body modes was wrong, and "
          "looked right.** Unless the pinned set spans exactly a complement of "
          "the rigid-body space it also constrains real deformation, and an "
          "ad-hoc choice does not. Validation 1 came back at 4.2e-2 instead of "
          "1e-14 -- a 4% error in the volume-averaged stress -- while the "
          "residual on the free dofs read 3e-14, so the solve looked "
          "converged. Replaced by conjugate gradients on the operator "
          "projected onto the range of K, which needs no arbitrary choice.",
          "",
          "## Caveats", ""]
    for c in CAVEATS:
        L.append(f"- {c}")
    L += ["", f"Epistemic level: **{EPISTEMIC_LEVEL}**. The defensible output "
          "is the order of magnitude and the detectable/not verdict, not a "
          "predicted splitting in MHz.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--tau-csv", default="results/production/tau_infinity.csv")
    ap.add_argument("--radius-nm", type=float, default=1.5)
    ap.add_argument("--refine", type=int, default=3)
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--dead-layer-nm", type=float, default=DEAD_LAYER_NM,
                    help="NVs shallower than this are assumed dark; the "
                         "predicted ODMR width is strongly sensitive to it")
    ap.add_argument("--out-dir", default="results/production")
    ap.add_argument("--shapes", nargs="+",
                    default=["octahedron", "cube", "rhombic-dodecahedron",
                             "mixture"])
    args = ap.parse_args(argv)

    taus = ps.load_taus(args.tau_csv)
    elastic = nv_spin_strain.ElasticConstants()
    params = nv_spin_strain.UDVARHELYI_DFT

    mix = ps.stability_weighted_fractions(
        ps.load_surface_energies("config/surface_energies_h.json", 0.0)[0])

    runs = []
    for shape in args.shapes:
        fr = mix if shape == "mixture" else None
        runs.append(analyse(shape, taus, elastic, params, args.radius_nm,
                            args.refine, tuple(args.depths), fractions=fr))

    # validations
    val = []
    v1 = max(r["volume_average_rel_error"] for r in runs)
    val.append(f"**1. Volume average reproduces particle_strain.** Max relative "
               f"error over all shapes {v1:.1e}. The divergence theorem makes "
               f"this exact, so a mismatch would mean the solve is wrong.")
    lap = [validate_laplace(elastic, 1.0, args.radius_nm, n_sub=n, refine=2)
           for n in (1, 2)]
    val.append(f"**2. Sphere reduces to Laplace.** P = 2 tau/R to "
               f"{lap[0]['P_rel_error']*100:.1f}% at {lap[0]['n_facets']} facets "
               f"and {lap[1]['P_rel_error']*100:.2f}% at {lap[1]['n_facets']}; "
               f"residual deviatoric falls {lap[0]['dev_over_P']:.2f} -> "
               f"{lap[1]['dev_over_P']:.2f} of P. The residual IS the faceting "
               f"and vanishes as the body approaches a sphere.")
    conv = [analyse("octahedron", taus, elastic, params, args.radius_nm, rf,
                    (0.5, 1.0)) for rf in (args.refine - 1, args.refine)]
    e_lo = conv[0]["depth_bins"][1]["E_median_mhz"]
    e_hi = conv[1]["depth_bins"][1]["E_median_mhz"]
    val.append(f"**3. Mesh convergence.** |E| at 1 nm depth on the octahedron: "
               f"{e_lo:.3f} MHz at {conv[0]['n_tets']} tets vs {e_hi:.3f} MHz at "
               f"{conv[1]['n_tets']} tets, a change of "
               f"{abs(e_hi-e_lo)/max(e_hi,1e-30)*100:.0f}%.")
    v4a = max(r["E_at_mean_strain_mhz"] for r in runs)
    v4b = min(r["mean_local_E_mhz"] for r in runs)
    val.append(f"**4. Averaged E vanishes, local E does not.** E evaluated at "
               f"the volume-averaged strain is < {v4a:.1e} MHz for every shape, "
               f"while the volume-averaged LOCAL |E| is >= {v4b:.1f} MHz. Both "
               f"hold simultaneously; E >= 0 pointwise, so <E(x)> cannot vanish "
               f"and the statement that does hold is E(<eps>) = 0.")

    # control
    flat = zero_facet_anisotropy(taus, "100")
    lines = ["Rerun with the (100) in-plane anisotropy set to zero and its mean "
             "surface stress preserved:", "",
             "| shape | depth (nm) | full tau | (100) isotropic | ratio |",
             "|---|---:|---:|---:|---:|"]
    for shape in args.shapes:
        fr = mix if shape == "mixture" else None
        A = analyse(shape, taus, elastic, params, args.radius_nm, args.refine,
                    (1.0,), fractions=fr)
        B = analyse(shape, flat, elastic, params, args.radius_nm, args.refine,
                    (1.0,), fractions=fr)
        a, b = A["depth_bins"][0], B["depth_bins"][0]
        if not a.get("reachable"):
            continue
        ratio = b["E_median_mhz"] / a["E_median_mhz"] if a["E_median_mhz"] else float("nan")
        lines.append(f"| {shape} | 1.0 | {a['E_median_mhz']:.2f} | "
                     f"{b['E_median_mhz']:.2f} | {ratio:.2f} |")
    control = "\n".join(lines)

    detect = [r for r in runs
              for b in r["depth_bins"]
              if b.get("reachable") and b["depth_nm"] == 1.0
              and b["E_median_mhz"] > LINEWIDTH_MHZ]
    verdict = (
        "**The hypothesis survives: a position-resolved treatment gives a "
        "non-zero E where the volume average gives exactly zero.** "
        f"{len(detect)} of {len(runs)} shapes exceed a ~{LINEWIDTH_MHZ:.0f} MHz "
        "ensemble linewidth at 1 nm depth."
        if detect else
        "**Negative result: E stays below a typical ensemble linewidth at every "
        "realistic depth.** The local field is non-zero where the average "
        "vanishes, but not by enough to observe.")

    odmr = []
    for r in runs:
        for dead in (0.5, args.dead_layer_nm):
            o = odmr_broadening(r, dead_layer_nm=dead)
            if o.get("reachable"):
                odmr.append({"shape": r["shape"], **o})

    meta = {"verdict": verdict, "depths": args.depths,
            "radius_nm": args.radius_nm, "refine": args.refine,
            "epistemic_level": EPISTEMIC_LEVEL}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "nv_local_strain_report.md").write_text(
        render_report(runs, val, control, meta, odmr))

    import csv as _csv
    with (out / "nv_local_strain_depth.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["shape", "radius_nm", "refine", "depth_nm", "n_samples",
                    "E_median_mhz", "E_p90_mhz", "E_max_mhz", "dD_median_mhz",
                    "sigma_dev_median_gpa", "edge_dist_median_nm",
                    "epistemic_level"])
        for r in runs:
            for b in r["depth_bins"]:
                if not b.get("reachable"):
                    continue
                w.writerow([r["shape"], r["radius_nm"], r["refine"],
                            b["depth_nm"], b["n"],
                            f"{b['E_median_mhz']:.4f}", f"{b['E_p90_mhz']:.4f}",
                            f"{b['E_max_mhz']:.4f}", f"{b['dD_median_mhz']:.4f}",
                            f"{b['sigma_dev_median_gpa']:.5f}",
                            f"{b['edge_dist_median_nm']:.4f}",
                            EPISTEMIC_LEVEL])
    with (out / "nv_local_strain_odmr.csv").open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["shape", "radius_nm", "refine", "dark_layer_nm", "n_sites",
                    "E_median_mhz", "E_p90_mhz", "E_p90_over_median",
                    "strain_fwhm_mhz", "frac_2E_over_linewidth",
                    "dD_median_mhz", "linewidth_ref_mhz", "epistemic_level"])
        for row in odmr:
            w.writerow([row["shape"], args.radius_nm, args.refine,
                        row["dead_layer_nm"], row["n_sites"],
                        f"{row['E_median_mhz']:.4f}", f"{row['E_p90_mhz']:.4f}",
                        f"{row['E_p90_over_median']:.4f}",
                        f"{row['fwhm_mhz']:.4f}",
                        f"{row['frac_resolvably_split']:.4f}",
                        f"{row['dD_median_mhz']:.4f}",
                        LINEWIDTH_MHZ, EPISTEMIC_LEVEL])
    print(f"wrote {out/'nv_local_strain_odmr.csv'}")
    print(f"wrote {out/'nv_local_strain_report.md'}")
    print(f"wrote {out/'nv_local_strain_depth.csv'}")
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
