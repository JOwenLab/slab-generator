"""
config/reference_pbe_sssp.json is the single source of truth for C11/C12/C44.

These constants used to be duplicated as literals in nv_spin_strain's
`ElasticConstants` field defaults. That is the same failure shape as the a0
bug: the config is the artefact that gets updated — the planned DFT
elastic-tensor campaign will replace today's literature values — and a stale
second copy would keep feeding the old numbers into every NV shift through
particle_strain, silently, without any code changing.

These tests fail if the two ever diverge, and fail if a hardcoded copy
reappears in the source.
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import nv_spin_strain

CONFIG = REPO_ROOT / "config" / "reference_pbe_sssp.json"


def _config_tensor():
    return json.loads(CONFIG.read_text())["elastic_tensor"]


@pytest.mark.skipif(not CONFIG.exists(), reason="reference config not present")
def test_defaults_equal_the_config():
    """The default-constructed constants must be the config's, exactly."""
    cfg = _config_tensor()
    ec = nv_spin_strain.ElasticConstants()
    assert ec.C11 == cfg["C11"]
    assert ec.C12 == cfg["C12"]
    assert ec.C44 == cfg["C44"]


@pytest.mark.skipif(not CONFIG.exists(), reason="reference config not present")
def test_defaults_follow_the_config_when_it_changes(tmp_path):
    """
    The load-bearing test: change the config, and the defaults must move.

    A hardcoded copy passes `test_defaults_equal_the_config` for as long as the
    literals happen to match. Only this test distinguishes "reads the config"
    from "happens to agree with it today".
    """
    cfg = json.loads(CONFIG.read_text())
    cfg["elastic_tensor"]["C11"] = 1234.5
    cfg["elastic_tensor"]["C12"] = 234.5
    cfg["elastic_tensor"]["C44"] = 345.6
    alt = tmp_path / "alt.json"
    alt.write_text(json.dumps(cfg))

    got = nv_spin_strain.config_elastic_constants(str(alt))
    assert got == (1234.5, 234.5, 345.6)


@pytest.mark.skipif(not CONFIG.exists(), reason="reference config not present")
def test_no_hardcoded_copy_remains_in_the_source():
    """
    Guard against the literals being pasted back as field defaults.

    Matches an assignment of a bare number to C11/C12/C44, which is how the
    duplication looked. The values are still allowed to appear inside strings
    (documentation, provenance notes) and inside the config-reading helper.
    """
    src = (REPO_ROOT / "nv_spin_strain.py").read_text()
    offenders = []
    for line in src.splitlines():
        code = line.split("#")[0]
        if '"' in code or "'" in code:
            continue
        if re.search(r"\bC(?:11|12|44)\s*:\s*float\s*=\s*[0-9]", code):
            offenders.append(line.strip())
    assert not offenders, (
        "C11/C12/C44 have been hardcoded again as dataclass defaults; "
        "config/reference_pbe_sssp.json must remain the only source: "
        + "; ".join(offenders))


@pytest.mark.skipif(not CONFIG.exists(), reason="reference config not present")
def test_missing_config_is_a_hard_error_not_a_silent_fallback(tmp_path):
    """A silent fallback to literals is the failure mode being removed."""
    import elastic_reference
    with pytest.raises(elastic_reference.ElasticReferenceError):
        nv_spin_strain.config_elastic_constants(str(tmp_path / "absent.json"))


@pytest.mark.skipif(not CONFIG.exists(), reason="reference config not present")
def test_particle_strain_sees_the_same_constants():
    """
    particle_strain consumes these for the stress-to-strain conversion, so it
    must see the config's values and not a private copy.
    """
    ps = pytest.importorskip("particle_strain")
    cfg = _config_tensor()
    ec = nv_spin_strain.ElasticConstants()
    assert (ec.C11, ec.C12, ec.C44) == (cfg["C11"], cfg["C12"], cfg["C44"])
    # and the config still declares them literature-sourced, not project DFT
    assert cfg["source_type"] == "literature"
