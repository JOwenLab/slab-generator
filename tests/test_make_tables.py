"""
Tests for make_tables.py -- the T1-T4 result tables.

Two classes of failure are worth a test here, and they are not the arithmetic.

The first is TYPESETTING that succeeds. LaTeX will happily render "median
|E(x)|" as "median --E(x)--" and "cube {100}" as "cube 100", producing a
plausible-looking table that says something different from the data. Nothing
errors, so only a test catches it.

The second is a MISSING INPUT quietly becoming a number. The tables are meant
to state gaps; a change that makes them interpolate, carry a stale value
forward, or print "nan" would be invisible in the output.
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import make_tables as mt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def clean_gaps():
    """Gaps accumulate in a module global; no test may see another's."""
    mt._gaps.clear()
    yield
    mt._gaps.clear()


# ── the escapes that fail silently ───────────────────────────────────────────

@pytest.mark.parametrize("raw,must_contain", [
    ("|E(x)|", "$|$"),          # text-mode | is an em dash
    ("2|E| > 3", "$>$"),        # text-mode > is an inverted question mark
    ("a < b", "$<$"),           # text-mode < is an inverted exclamation
    ("cube {100}", r"\{100\}"),  # braces would be eaten as a group
    ("tau_xx", r"tau\_xx"),
    ("50%", r"50\%"),
])
def test_tex_escape_covers_the_characters_that_render_wrong(raw, must_contain):
    assert must_contain in mt.tex_escape(raw)


def test_tex_escape_does_not_double_escape_the_math_it_introduces():
    """
    The |, < and > rules expand into math mode, so they must run AFTER the "$"
    rule. If the order were reversed the output would contain a literal \\$.
    """
    out = mt.tex_escape("|E| > 0")
    assert r"\$" not in out


def test_headers_are_escaped_when_no_latex_header_row_is_given():
    t = mt.Table("k", 9, "t", ["quantity", "cube {100}"], [["a", "b"]], "cap")
    assert r"\{100\}" in mt.render_tex(t)


def test_explicit_latex_headers_pass_through_unescaped():
    """A header given as real LaTeX is markup, not data, and must survive."""
    t = mt.Table("k", 9, "t", ["tau"], [["1"]], "cap",
                 headers_tex=[r"$\tau_{xx}$"])
    assert r"$\tau_{xx}$" in mt.render_tex(t)


def test_body_cells_cannot_inject_markup():
    """A cell is data. If it could inject LaTeX, a CSV would control the page."""
    t = mt.Table("k", 9, "t", ["a"], [[r"\bottomrule \end{tabular}"]], "cap")
    out = mt.render_tex(t)
    assert out.count(r"\bottomrule") == 1
    assert out.count(r"\end{tabular}") == 1


def test_caption_contains_no_line_break_command():
    """
    \\\\ inside \\caption{} raises "There's no line here to end" under several
    document classes. Notes are separated with bullets instead.
    """
    t = mt.Table("k", 9, "t", ["a"], [["1"]], "cap", notes=["n1", "n2"])
    caption = mt.render_tex(t).split(r"\caption{")[1].split("\n")[0]
    assert r"\\" not in caption
    assert caption.count(r"\textbullet") == 2


def test_label_has_no_underscore():
    """\\label with an underscore breaks under hyperref."""
    t = mt.Table("t2_tau_infinity", 2, "t", ["a"], [["1"]], "cap")
    assert r"\label{tab:t2-tau-infinity}" in mt.render_tex(t)


def test_markdown_escapes_the_pipe_that_would_split_a_cell():
    t = mt.Table("k", 9, "t", ["a"], [["|E|"]], "cap")
    body = [ln for ln in mt.render_md(t).splitlines() if "E" in ln][-1]
    assert body.count("|") == 2 + 2  # two cell delimiters, two escaped
    assert r"\|E\|" in body


def test_markdown_supplies_the_table_number_and_latex_does_not():
    """LaTeX numbers its own captions; a stored prefix would give 'Table 4: Table T4'."""
    t = mt.Table("k", 4, "t", ["a"], [["1"]], "some caption")
    assert "**Table T4.** some caption" in mt.render_md(t)
    assert "Table T4." not in mt.render_tex(t)


# ── a missing input must become a gap, never a value ─────────────────────────

def test_absent_inputs_produce_no_table_and_a_stated_gap():
    assert mt.table_t2(None, None) is None
    assert mt.table_t3(None, None) is None
    assert mt.table_t1({}, None, None, None) is None
    assert len(mt._gaps) == 3


def test_a_facet_missing_from_the_csv_is_named_not_skipped_silently():
    rows = [{"surface": "C100", "orientation": "(100)", "epistemic_level": "L2",
             "axis_x": "[110]", "axis_y": "[1-10]",
             "tau_xx_inf_n_per_m": "1.0", "tau_yy_inf_n_per_m": "-5.0",
             "tau_mean_n_per_m": "-2.0", "tau_aniso_n_per_m": "6.0",
             "sigma_res_mean_kbar": "-0.3",
             "rms_xx_as_tau_n_per_m": "0.01", "rms_yy_as_tau_n_per_m": "0.02"}]
    t = mt.table_t2(rows, None)
    assert len(t.rows) == 1
    assert any("C110" in g for g in mt._gaps)
    assert any("C111" in g for g in mt._gaps)


def test_disagreeing_production_runs_report_a_gap_rather_than_picking_one():
    assert mt.unique([90.0, 80.0], "ecutwfc") is None
    assert mt._gaps and "disagree" in mt._gaps[0]
    mt._gaps.clear()
    assert mt.unique([90.0, 90.0, None], "ecutwfc") == 90.0
    assert not mt._gaps


# ── the k-point anchoring, which is a claim about what was demonstrated ──────

class _FakePC:
    """Stands in for plot_convergence with a hand-made sweep."""

    def __init__(self, points):
        self._points = points

    def sweep_rows(self, rows, sweep, surface=None):
        return self._points

    def series(self, rows, xkey, ykey):
        return [p[0] for p in rows], [p[1] for p in rows]

    def finite(self, xs, ys):
        return xs, ys


def _conv(points):
    return {"_pc": _FakePC(points), "_rows": []}


def test_kpoint_anchor_is_the_densest_swept_point_no_finer_than_production():
    """
    Convergence shown at a COARSER sampling implies it at production. Anchoring
    on a FINER swept point would claim something the data does not support.
    """
    ev = mt.kpoint_evidence(_conv([(15.0, 3.0), (20.0, 1.0), (30.0, 0.5),
                                   (50.0, 0.0)]), "C100", 22.7)
    assert ev["anchor"] == 20.0
    assert ev["residual"] == pytest.approx(1.0)
    assert ev["exact"] is False


def test_kpoint_anchor_tolerates_the_lattice_constant_correction():
    """
    The sweeps ran at the superseded a0 and production at the corrected one, so
    the same mesh on the same slab differs in density by 0.02%. Exact matching
    would drop the matching point and quote a coarser one, understating the
    evidence.
    """
    ev = mt.kpoint_evidence(_conv([(15.16, 1.0), (22.742512, 0.03),
                                   (45.5, 0.0)]), "C111", 22.738)
    assert ev["anchor"] == pytest.approx(22.742512)
    assert ev["exact"] is True
    assert ev["residual"] == pytest.approx(0.03)


def test_kpoint_reports_unsupported_when_production_is_coarser_than_any_sweep():
    ev = mt.kpoint_evidence(_conv([(30.0, 1.0), (50.0, 0.0)]), "C100", 12.0)
    assert ev["unsupported"] is True
    assert ev["coarsest_swept"] == 30.0


def test_no_sweep_for_a_surface_yields_no_evidence_rather_than_a_default():
    assert mt.kpoint_evidence(_conv([]), "C110", 22.7) is None
    assert mt.kpoint_evidence(None, "C110", 22.7) is None


# ── the committed output ─────────────────────────────────────────────────────

TABLES = os.path.join(REPO, "results", "tables")
needs_tables = pytest.mark.skipif(
    not os.path.isdir(TABLES), reason="results/tables not generated")


def md_cells(line):
    """
    Split a markdown row on its UNESCAPED pipes.

    Cells legitimately contain "\\|" -- "median |E(x)|" is one of them -- so a
    bare split on "|" tears a cell in half and silently changes what is being
    asserted about.
    """
    parts, cur, i = [], "", 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cur += "|"
            i += 2
        elif line[i] == "|":
            parts.append(cur.strip())
            cur = ""
            i += 1
        else:
            cur += line[i]
            i += 1
    parts.append(cur.strip())
    return [p for p in parts[1:-1]]


def md_rows(text):
    """Body rows of the markdown table: no header, no alignment rule."""
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    return [md_cells(ln) for ln in lines[2:]]


@needs_tables
@pytest.mark.parametrize("name", sorted(os.listdir(TABLES))
                         if os.path.isdir(TABLES) else [])
def test_no_committed_cell_is_empty_or_nan(name):
    """
    Checked per CELL, not per file: the prose legitimately contains "Nano
    Lett." and "3 nm", and a substring search over the whole file would either
    fire on those or be loosened until it caught nothing.
    """
    if not name.endswith(".md"):
        pytest.skip("cell structure is checked on the markdown rendering")
    for row in md_rows(open(os.path.join(TABLES, name)).read()):
        for cell in row:
            assert cell != "", f"{name}: empty cell in {row}"
            assert cell.lower() not in ("nan", "none", "inf", "-inf"), \
                f"{name}: non-value {cell!r} in {row}"


@needs_tables
def test_t4_states_both_that_averaged_E_vanishes_and_that_local_E_does_not():
    """
    The substantive result of the whole NV chain. If a refactor ever made these
    two rows agree, the table would be asserting something false in whichever
    direction it moved.
    """
    rows = md_rows(open(os.path.join(TABLES, "T4_predicted_D_and_E.md")).read())
    means = next(r for r in rows if "volume-averaged strain" in r[0])
    locals_ = next(r for r in rows if "at depth 0.5 nm" in r[0])
    assert all(float(c) == 0.0 for c in means[1:]), \
        "averaged strain is not isotropic"
    assert all(float(c) > 0.0 for c in locals_[1:]), \
        "local E vanished; nothing to report"


@needs_tables
def test_t4_names_the_literature_dependence_at_the_point_of_use():
    text = open(os.path.join(TABLES, "T4_predicted_D_and_E.md")).read()
    assert "LITERATURE" in text and "not fitted in this project" in text
    assert "coupling set" in text


@needs_tables
@pytest.mark.parametrize("name,level", [
    ("T2_tau_infinity.md", "L2"),
    ("T4_predicted_D_and_E.md", "L1"),
])
def test_every_table_carries_an_epistemic_level(name, level):
    assert level in open(os.path.join(TABLES, name)).read()


@needs_tables
def test_the_missing_dehydrogenation_ceiling_is_stated_not_filled():
    text = open(os.path.join(TABLES, "T3_surface_energies.md")).read()
    assert "NOT COMPUTED" in text
    assert "extrapolation with no stop" in text


@needs_tables
def test_tables_regenerate_byte_identically():
    """
    Nothing in these tables may depend on the wall clock, the working directory
    or dictionary iteration order -- otherwise regenerate.sh --check reports
    them as changed on every run and the reader learns to ignore it.
    """
    before = {n: open(os.path.join(TABLES, n)).read()
              for n in sorted(os.listdir(TABLES))}
    subprocess.run([sys.executable, "make_tables.py"], cwd=REPO,
                   check=True, capture_output=True)
    after = {n: open(os.path.join(TABLES, n)).read()
             for n in sorted(os.listdir(TABLES))}
    assert before == after
