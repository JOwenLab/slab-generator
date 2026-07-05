"""Tests for inversion-symmetric slab construction (geometry.invert_symmetrize
and the slabgen symmetric literature/TDB paths)."""
import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import geometry as G
from slabgen import generate

A0 = 3.573641


def _assert_inversion_symmetric(slab, tol=0.02):
    """Every atom must have a same-species inversion partner (min image)."""
    # center: use z midpoint; find in-plane center by testing the pairing
    imgs = G._images(slab)
    z = slab.pos[:, 2]
    zc = 0.5 * (z.max() + z.min())
    # candidate in-plane centers: midpoints of atom pairs straddling zc
    for i in range(slab.n):
        for j in range(slab.n):
            c = 0.5 * (slab.pos[i] + slab.pos[j])
            if abs(c[2] - zc) > 0.05:
                continue
            ok = True
            for k in range(slab.n):
                target = 2.0 * c - slab.pos[k]
                d = slab.pos[None, :, :] + imgs[:, None, :] - target
                dist = np.linalg.norm(d, axis=2)
                hits = np.argwhere(dist < tol)
                if not any(slab.el[m] == slab.el[k] for (_, m) in hits):
                    ok = False
                    break
            if ok:
                return
    raise AssertionError("no valid inversion center found")


def test_pandey_symmetric_20L():
    s, _ = generate("C111_2x1_pandey", 20, A0, symmetric=True)
    assert s.el.count("C") == 40
    _assert_inversion_symmetric(s)


def test_seiwatz_symmetric_20L():
    s, _ = generate("C111_3db_2x1_seiwatz", 20, A0, symmetric=True)
    _assert_inversion_symmetric(s)


def test_unachievable_layer_count_raises():
    with pytest.raises(ValueError, match="achievable"):
        generate("C111_2x1_pandey", 10, A0, symmetric=True)


def test_coordination_sane_pandey_sym():
    """Interior atoms 4-coordinated; surface chain atoms 3-coordinated on
    BOTH faces (identical surfaces)."""
    s, _ = generate("C111_2x1_pandey", 20, A0, symmetric=True)
    bonds, _ = G.neighbors(s)
    counts = np.array([len(b) for b in bonds])
    assert set(counts) <= {3, 4}
    z = s.pos[:, 2]
    zc = 0.5 * (z.max() + z.min())
    n3_top = sum(1 for i in range(s.n) if counts[i] == 3 and z[i] > zc)
    n3_bot = sum(1 for i in range(s.n) if counts[i] == 3 and z[i] < zc)
    assert n3_top == n3_bot > 0
