"""Documents a diagnosed BLAS-backend artifact so it is not rediscovered.

On macOS arm64 with numpy 2.0.2 linked against Apple's Accelerate framework,
matmul on arrays above a row-count threshold sets spurious
divide-by-zero/overflow/invalid-value FPE flags that numpy surfaces as
RuntimeWarnings -- with no effect on the actual output, which remains exactly
finite and exactly correct. This was traced from geometry.py:bulk_slab's
lattice-rotation and in-plane-wrap matmuls (see CLAUDE.md sec 6 and
geometry._require_finite). Reproduced here with all-ones input times the
identity matrix, which cannot itself be invalid, to show the warnings are
shape-triggered rather than data-triggered.

This is BLAS-backend-specific: it is NOT expected to reproduce on the Linux
cluster's OpenBLAS build, hence the skip guard below.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


def _blas_name():
    try:
        return np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:
        return None


@pytest.mark.skipif(
    _blas_name() != "accelerate",
    reason="spurious matmul FPE warnings are an Accelerate-specific artifact; "
           "not expected on other BLAS backends (e.g. OpenBLAS on Linux)",
)
def test_accelerate_matmul_spurious_warning_is_harmless():
    a = np.ones((1000, 3))
    with pytest.warns(RuntimeWarning):
        out = a @ np.eye(3).T
    assert np.all(np.isfinite(out))
    assert np.array_equal(out, a)
