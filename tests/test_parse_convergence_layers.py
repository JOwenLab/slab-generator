"""
count_layers must survive the fact that intra-layer buckling on one surface is
wider than inter-layer separation on another.

Measured carbon z-gaps in this project (parse_convergence.py header):

    (100)  0.076 A   buckling within a 2x1 dimer row      -> SAME layer
    (100)  0.292 A   buckling within a 2x2 interior layer -> SAME layer
    (100)  0.816 A   adjacent layers                      -> distinct
    (110)  0.000 A   two atoms per layer, degenerate      -> SAME layer
    (110)  1.235 A   adjacent layers                      -> distinct
    (111)  0.488 A   the two halves of a (111) bilayer    -> DISTINCT layers
    (111)  1.554 A   between bilayers                     -> distinct

0.292 > 0.488 is false but 0.292 is uncomfortably close, and the two windows
overlap once tolerance is allowed: no single distance classifies both. These
tests pin the cases that a distance-only rule gets wrong.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from parse_convergence import count_layers


def test_empty_is_none():
    assert count_layers([]) is None


def test_c111_bilayer_halves_are_distinct_layers():
    """0.488 A splits a (111) bilayer into two layers; it must not be merged."""
    z, zs = 0.0, []
    for i in range(6):
        zs.append(z)
        z += 0.488 if i % 2 == 0 else 1.554
    assert count_layers(zs) == 6


def test_c110_degenerate_pairs_are_one_layer():
    """Two atoms per layer at identical z."""
    zs = [i * 1.235 for i in range(6) for _ in range(2)]
    assert count_layers(zs) == 6


def test_c100_2x1_dimer_buckling_is_one_layer():
    """0.076 A buckling within a 2x1 row is intra-layer."""
    zs = [i * 0.816 + d for i in range(8) for d in (0.0, 0.076)]
    assert count_layers(zs) == 8


def test_c100_2x2_interior_buckling_is_one_layer():
    """
    Regression: the real 6-layer (100) 2x2 slab, reported as 8 layers.

    Its two interior layers buckle by 0.292 A, just above the 0.25 A clustering
    tolerance, so each was split into a pair of 2-atom clusters while the outer
    layers stayed whole at 4 atoms. The populations 4,4,2,2,2,2,4,4 are what
    give the split away, and the count must come back to 6.
    """
    zs = ([5.105] * 4 + [5.910] * 4
          + [6.660] * 2 + [6.952] * 2
          + [7.549] * 2 + [7.841] * 2
          + [8.591] * 4 + [9.397] * 4)
    assert len(zs) == 24
    assert count_layers(zs) == 6


@pytest.mark.parametrize("zs", [
    [5.105] * 4 + [5.910] * 4 + [6.660] * 2 + [6.952] * 2
    + [7.549] * 2 + [7.841] * 2 + [8.591] * 4 + [9.397] * 4,
    [i * 1.235 for i in range(6) for _ in range(2)],
    [i * 0.816 + d for i in range(8) for d in (0.0, 0.076)],
])
def test_layer_count_always_divides_atom_count(zs):
    """
    The invariant the repair enforces: layers are symmetry-equivalent sites, so
    every layer holds the same number of atoms. A count that does not divide
    the atom count would imply a fractional layer.
    """
    n = count_layers(zs)
    assert n is not None
    assert len(zs) % n == 0


def test_uneven_split_is_repaired_but_uniform_split_is_not():
    """
    The boundary of what the population invariant can repair, asserted so it
    cannot be mistaken for robustness it does not have.

    An uneven split leaves unequal populations and is repaired. A split that
    hits every layer identically leaves them equal, is invisible to the repair,
    and comes back a whole multiple too large -- which is why `tol` must stay
    below the smallest genuine interlayer gap, and why `analyze_run` still
    cross-checks this count against the folder name.
    """
    layers, buckle = 6, 0.09

    uneven = []
    for i in range(layers):
        uneven += [i * 1.235, i * 1.235 + (buckle if i in (2, 3) else 0.0)]
    assert count_layers(uneven, tol=0.01) == layers

    uniform = [i * 1.235 + d for i in range(layers) for d in (0.0, buckle)]
    assert count_layers(uniform, tol=0.01) == 2 * layers      # documented limit
    assert count_layers(uniform, tol=0.25) == layers          # correct tolerance
