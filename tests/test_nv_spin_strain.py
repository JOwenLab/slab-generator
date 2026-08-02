"""Unit tests for nv_spin_strain.py.

Validation targets:
  * zero strain -> no shift, no splitting
  * hydrostatic strain -> Delta D = (2 h41 + h43) eps, E = 0, all four
    orientations identical
  * hydrostatic pressure via the elasticity solver -> dD/dP consistent with
    the paper's own a1 = (2 g41 + g43)/3 = -2.66 MHz/GPa  (i.e. Delta D =
    -3 a1 P ~ +7.98 MHz/GPa of compression)
  * (100)-normal biaxial strain -> eps_zz = -2 C12/C11 * eps_par (cubic
    analytic result)
  * (111) biaxial: on-axis NV has E = 0 (pure D shift); the three inclined
    NVs are degenerate with equal, nonzero E
  * in-plane frame invariance for equal-biaxial loading
"""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from nv_spin_strain import (D0_MHZ, ElasticConstants, UDVARHELYI_DFT,
                            nv_frames, nv_observables, predict_for_slab,
                            slab_equilibrium_strain, slab_frame)

P = UDVARHELYI_DFT
EL = ElasticConstants()


def test_zero_strain_is_silent():
    eps = np.zeros((3, 3))
    for _, frame in nv_frames():
        o = nv_observables(eps, frame, P)
        assert abs(o["delta_D_mhz"]) < 1e-9
        assert abs(o["E_mhz"]) < 1e-9
        assert abs(o["f_plus_mhz"] - D0_MHZ) < 1e-9


def test_hydrostatic_strain_analytic():
    e = -1e-3  # 0.1 % compression
    eps = e * np.eye(3)
    expect = (2 * P.h41 + P.h43) * e
    for _, frame in nv_frames():
        o = nv_observables(eps, frame, P)
        assert o["delta_D_mhz"] == pytest.approx(expect, rel=1e-6)
        assert abs(o["E_mhz"]) < 1e-6


def test_hydrostatic_pressure_vs_paper_a1():
    # eps = -P/(3B) per axis under pressure P; B = (C11 + 2 C12)/3
    Pgpa = 1.0
    B = (EL.C11 + 2 * EL.C12) / 3.0
    eps = (-Pgpa / (3 * B)) * np.eye(3)
    o = nv_observables(eps, nv_frames()[0][1], P)
    dD_per_GPa = o["delta_D_mhz"] / Pgpa
    # paper: a1 = (2 g41 + g43)/3 = -2.66 MHz/GPa; Delta D = -3 a1 P
    assert dD_per_GPa == pytest.approx(3 * 2.66, rel=0.02)


def test_100_biaxial_poisson_analytic():
    e_par = -0.01
    eps_c = slab_equilibrium_strain("100", e_par, e_par, elastic=EL)
    R = slab_frame("100")
    ezz = (R @ eps_c @ R.T)[2, 2]
    assert ezz == pytest.approx(-2 * EL.C12 / EL.C11 * e_par, rel=1e-6)


def test_111_biaxial_signature():
    preds = predict_for_slab("111", -0.01, -0.01, params=P)
    on_axis = [p for p in preds if p["nv_axis"] == "[111]"]
    inclined = [p for p in preds if p["nv_axis"] != "[111]"]
    assert len(on_axis) == 1 and len(inclined) == 3
    # surface-normal NV: pure D shift, no transverse splitting
    assert abs(on_axis[0]["E_mhz"]) < 1e-6
    assert abs(on_axis[0]["delta_D_mhz"]) > 1.0
    # inclined NVs: degenerate, split
    Es = [p["E_mhz"] for p in inclined]
    Ds = [p["delta_D_mhz"] for p in inclined]
    assert max(Es) - min(Es) < 1e-6
    assert min(Es) > 0.1
    assert max(Ds) - min(Ds) < 1e-6


def test_biaxial_inplane_frame_invariance():
    """Equal-biaxial strain: results can't depend on the in-plane axis
    choice.  Compare the geometry-convention (111) frame against a rotated
    in-plane frame built directly."""
    e = -0.005
    eps_c = slab_equilibrium_strain("111", e, e, elastic=EL)
    z = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
    x = np.array([1.0, 0.0, -1.0]) / math.sqrt(2)  # different in-plane axis
    y = np.cross(z, x)
    R2 = np.array([x, y, z])
    eps_slab2 = np.diag([e, e, 0.0])
    # solve traction-free condition in this rotated frame via the same solver
    # trick: equal-biaxial in-plane is isotropic in the plane, so the cubic
    # tensor produced must be identical.
    C = EL.tensor()
    Cp = np.einsum("mi,nj,ok,pl,ijkl->mnop", R2, R2, R2, R2, C)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for i in range(3):
        A[i, 0] = Cp[i, 2, 0, 2] + Cp[i, 2, 2, 0]
        A[i, 1] = Cp[i, 2, 1, 2] + Cp[i, 2, 2, 1]
        A[i, 2] = Cp[i, 2, 2, 2]
        b[i] = -np.einsum("kl,kl->", Cp[i, 2], eps_slab2)
    u = np.linalg.solve(A, b)
    eps_slab2[0, 2] = eps_slab2[2, 0] = u[0]
    eps_slab2[1, 2] = eps_slab2[2, 1] = u[1]
    eps_slab2[2, 2] = u[2]
    eps_c2 = R2.T @ eps_slab2 @ R2
    assert np.allclose(eps_c, eps_c2, atol=1e-12)
