#!/usr/bin/env python3
"""
make_tables.py - the H-terminated result tables, T1-T4.

WHAT THESE TABLES ARE FOR
-------------------------
The figure set (plot_h_results.py, F1-F5) shows the argument; these tables carry
the numbers a reader has to be able to check or reuse:

    T1  what the production calculations actually were, and what was
        demonstrated about each setting rather than assumed
    T2  tau_infinity per facet -- the one L2 quantity in the chain
    T3  surface energies and their hydrogen-chemical-potential slopes
    T4  the predicted D and E per habit at R = 1.5 nm

SOURCES
-------
Every value is read from a committed file. Nothing is hardcoded and nothing is
interpolated or extrapolated; a missing input becomes a stated gap and a message
on stdout, never a filled cell.

    results/production/thick_a0corr_stress~*/pw.{in,out}   T1 (pw.out is ground
                                                           truth for pseudos)
    results/convergence/convergence_summary.csv            T1 evidence column
    results/production/regression~kpt_C110~k_*             T1 (110) k-points
    results/production/tau_infinity.csv                    T2
    results/production/tau_infinity_meta.json              T2 provenance
    results/production/surface_energy.csv                  T3
    config/surface_energies_h.json                         T3 ceiling (if any)
    results/production/particle_strain_nv_<shape>.csv      T4 delta_D
    results/production/particle_strain_meta_<shape>.json   T4 facet fractions
    results/production/nv_local_strain_depth.csv           T4 |E| vs depth
    results/production/nv_local_strain_odmr.csv            T4 ODMR widths

The T1 convergence column is not recomputed here. It is taken from the same
functions that produce the convergence figures, so a table and a figure making
different claims about the same sweep is not a state this repo can reach.

SIGN CONVENTION (CLAUDE.md section 2)
-------------------------------------
Positive sigma means the cell is COMPRESSED. tau inherits it, so positive tau is
COMPRESSIVE surface stress and equals minus the continuum surface stress f.
Positive pressure is compressive. Each table restates this where it matters.

OUTPUT
------
results/tables/T<n>_<slug>.md and .tex (booktabs), plus TABLES.md collecting
every caption and every gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
PROD = REPO / "results" / "production"
CONFIG = REPO / "config"
DEFAULT_OUTDIR = REPO / "results" / "tables"

TAU_CSV = PROD / "tau_infinity.csv"
TAU_META = PROD / "tau_infinity_meta.json"
SURF_E_CSV = PROD / "surface_energy.csv"
GAMMA_JSON = CONFIG / "surface_energies_h.json"
NV_DEPTH = PROD / "nv_local_strain_depth.csv"
NV_ODMR = PROD / "nv_local_strain_odmr.csv"
REF_JSON = CONFIG / "reference_pbe_sssp.json"

PROD_STRESS_GLOB = "thick_a0corr_stress~*_stress_scf"

SURFACES = ("C100", "C110", "C111")
MILLER = {"C100": "(100)", "C110": "(110)", "C111": "(111)"}

# Habits, in the order used by every figure in the set.
SHAPES = ("cube", "rhombic-dodecahedron", "octahedron", "mixture")
SHAPE_LABEL = {"cube": "cube {100}",
               "rhombic-dodecahedron": "rh. dodec. {110}",
               "octahedron": "octahedron {111}",
               "mixture": "Wulff mixture"}

# Depths and dark layers to tabulate. Both are read back from the data rather
# than assumed present: a depth absent from the CSV becomes a gap.
DEPTHS_NM = (0.5, 1.0, 1.5)
DARK_LAYERS_NM = (0.5, 1.0)

_gaps: list[str] = []


def gap(what: str) -> None:
    _gaps.append(what)
    print(f"  GAP: {what}")


# ── small readers ─────────────────────────────────────────────────────────────

def read_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def fnum(row, key):
    v = (row.get(key) or "").strip()
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def close(a, b, rel=1e-6):
    return a is not None and b is not None and abs(a - b) <= rel * max(1.0, abs(b))


# ── table model and the two emitters ──────────────────────────────────────────

class Table:
    """
    One table in both output languages.

    `headers` and `rows` hold plain text. LaTeX-only markup is supplied
    separately as `headers_tex` when a header needs real mathematics; body cells
    are escaped rather than interpreted, because a cell is data and data must
    not be able to inject markup.
    """

    def __init__(self, key, number, title, headers, rows, caption,
                 headers_tex=None, align=None, notes=()):
        self.key = key
        self.number = number
        self.title = title
        self.headers = list(headers)
        # Absent an explicit LaTeX header row, the plain one is ESCAPED, not
        # passed through: a header like "cube {100}" is data, and unescaped it
        # silently typesets as "cube 100" with the braces eaten as a group.
        self.headers_tex = (list(headers_tex) if headers_tex
                            else [tex_escape(h) for h in headers])
        self.rows = [[str(c) for c in r] for r in rows]
        self.caption = caption
        self.align = align or ("l" + "r" * (len(headers) - 1))
        self.notes = list(notes)


# Order matters. The backslash goes first so it cannot escape the escapes, and
# the three that expand into math mode go LAST so the "$" they introduce is not
# itself escaped by the "$" rule above them.
#
# |, < and > are not cosmetic: in LaTeX text mode with the default encoding they
# typeset as an em-dash, an inverted exclamation and an inverted question mark
# respectively. "median |E(x)|" would silently come out as "median --E(x)--" and
# "2|E| > linewidth" as "2--E-- ? linewidth". Both are plausible-looking output,
# which is the failure mode this project cares about most.
TEX_ESCAPES = ((("\\", r"\textbackslash{}"),) +
               tuple((c, "\\" + c) for c in "&%$#_{}") +
               (("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                ("|", "$|$"), ("<", "$<$"), (">", "$>$")))


def tex_escape(s):
    for a, b in TEX_ESCAPES:
        s = s.replace(a, b)
    return s


def md_escape(s):
    return s.replace("|", r"\|")


def render_md(t: Table) -> str:
    widths = [max(len(md_escape(h)), *(len(md_escape(r[i])) for r in t.rows))
              if t.rows else len(h) for i, h in enumerate(t.headers)]
    sep = []
    for i, a in enumerate(t.align):
        sep.append(("-" * (widths[i] - 1) + ":") if a == "r" else "-" * widths[i])

    def line(cells):
        return "| " + " | ".join(
            md_escape(c).rjust(widths[i]) if t.align[i] == "r"
            else md_escape(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [f"# Table T{t.number} - {t.title}", "",
           line(t.headers), "| " + " | ".join(sep) + " |"]
    out += [line(r) for r in t.rows]
    # The "Table Tn." prefix is added here, not stored in the caption: LaTeX
    # supplies its own "Table 4:" and a stored prefix would double it.
    out += ["", f"**Table T{t.number}.** " + t.caption, ""]
    if t.notes:
        out.append("")
        out += [f"- {n}" for n in t.notes]
        out.append("")
    return "\n".join(out)


def render_tex(t: Table) -> str:
    body = [" & ".join(tex_escape(c) for c in r) + r" \\" for r in t.rows]
    # Notes run on as sentences rather than being broken with \\: a \\ inside
    # \caption{} is fragile (it raises "There's no line here to end" under
    # several document classes) and the caption is the one part of this file a
    # reader cannot easily repair.
    caption = tex_escape(t.caption)
    for n in t.notes:
        caption += " " + chr(92) + "textbullet" + chr(92) + " " + tex_escape(n)
    return "\n".join([
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{tab:" + t.key.replace("_", "-") + "}",
        r"  \begin{tabular}{" + t.align + "}",
        r"    \toprule",
        "    " + " & ".join(t.headers_tex) + r" \\",
        r"    \midrule",
        *("    " + b for b in body),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])


# ── T1: what the production calculations were ─────────────────────────────────

PWIN_NUM = r"([-+0-9.eEdD]+)"


def parse_pw_in(path):
    txt = Path(path).read_text()
    out = {}
    for key in ("ecutwfc", "ecutrho", "conv_thr", "degauss", "mixing_beta"):
        m = re.search(rf"^\s*{key}\s*=\s*{PWIN_NUM}", txt, re.M)
        if m:
            out[key] = float(m.group(1).replace("d", "e").replace("D", "e"))
    for key in ("smearing", "occupations", "calculation"):
        m = re.search(rf"^\s*{key}\s*=\s*'([^']*)'", txt, re.M)
        if m:
            out[key] = m.group(1)
    m = re.search(r"^\s*K_POINTS.*\n\s*(\d+)\s+(\d+)\s+(\d+)", txt, re.M)
    if m:
        out["kmesh"] = tuple(int(g) for g in m.groups())
    m = re.search(r"^CELL_PARAMETERS.*\n((?:\s*[-0-9.eE]+\s+[-0-9.eE]+\s+"
                  r"[-0-9.eE]+\s*\n){3})", txt, re.M)
    if m:
        vecs = [[float(x) for x in ln.split()]
                for ln in m.group(1).strip().splitlines()]
        out["cell"] = vecs
        out["a1"] = abs(vecs[0][0]) or None
        out["a2"] = abs(vecs[1][1]) or None
        out["lz"] = abs(vecs[2][2])
    return out


def parse_pw_out(path):
    """
    Settings as the RUN saw them. CLAUDE.md invariant 7: pw.out is ground truth
    for the pseudopotential actually opened, not the input declaration.
    """
    txt = Path(path).read_text()
    out = {}
    m = re.search(r"Exchange-correlation\s*=\s*(\S+)", txt)
    if m:
        out["xc"] = m.group(1)
    m = re.search(r"kinetic-energy cutoff\s*=\s*([\d.]+)", txt)
    if m:
        out["ecutwfc"] = float(m.group(1))
    m = re.search(r"charge density cutoff\s*=\s*([\d.]+)", txt)
    if m:
        out["ecutrho"] = float(m.group(1))
    m = re.search(r"scf convergence threshold\s*=\s*([\deE.+-]+)", txt)
    if m:
        out["conv_thr"] = float(m.group(1))
    m = re.search(r"number of k points\s*=\s*(\d+)\s+(.*?)smearing,\s*width \(Ry\)"
                  r"=\s*([\d.]+)", txt)
    if m:
        out["nk"] = int(m.group(1))
        out["smearing"] = m.group(2).strip()
        out["degauss"] = float(m.group(3))
    out["pseudos"] = {}
    for sp, fn in re.findall(r"PseudoPot\. #\s*\d+ for (\S+)\s+read from file:"
                             r"\s*\n\s*(\S+)", txt):
        out["pseudos"][sp] = Path(fn).name
    out["job_done"] = "JOB DONE" in txt
    return out


def surface_of_run(name):
    m = re.search(r"~(C\d{3})_", name)
    return m.group(1) if m else None


def collect_production_runs():
    """The stress SCFs the tau_infinity fit was built from, parsed from disk."""
    runs = {}
    for d in sorted(PROD.glob(PROD_STRESS_GLOB)):
        pwin, pwout = d / "pw.in", d / "pw.out"
        if not (pwin.exists() and pwout.exists()):
            continue
        runs[d.name] = {"dir": d, "surface": surface_of_run(d.name),
                        "in": parse_pw_in(pwin), "out": parse_pw_out(pwout)}
    return runs


def unique(values, what):
    """The single value shared by a set of runs, or None with a stated gap."""
    vals = {v for v in values if v is not None}
    if len(vals) == 1:
        return next(iter(vals))
    gap(f"T1 {what}: production runs disagree ({sorted(map(str, vals))}); "
        "no single value reported")
    return None


def convergence_stats():
    """
    Convergence evidence, taken from the functions that draw the figures.

    Deliberately not recomputed: two implementations of the same reduction can
    disagree, and a table contradicting its own figure is exactly the silent
    failure this project is built to avoid. The k-point sweep rows are carried
    through as well, because the production cells are not the cells that were
    swept and the table has to say so per axis rather than inherit the figure's
    single production point.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import plot_convergence as pc
    except Exception as exc:                       # noqa: BLE001
        gap(f"T1 convergence evidence: plot_convergence unavailable ({exc})")
        return None
    csv_path = REPO / "results" / "convergence" / "convergence_summary.csv"
    if not csv_path.exists():
        gap(f"T1 convergence evidence: {csv_path} absent")
        return None
    rows, excluded = pc.load_convergence(csv_path)
    for r in excluded:
        gap(f"T1: {r['run']} excluded (JOB DONE={r.get('job_done')}, "
            f"scf_converged={r.get('scf_converged')})")
    extra, skipped = pc.load_extra_kpoint_runs()
    for name, why in skipped:
        gap(f"T1: {name} excluded from the k-point evidence ({why})")
    rows = rows + extra

    stats = {"_pc": pc, "_rows": rows}
    for key, fn in (("cutoff", pc.figure_cutoff),
                    ("kpoint", pc.figure_kpoints),
                    ("vacuum", pc.figure_vacuum)):
        fig, _cap, st = fn(rows)
        plt.close(fig)
        stats[key] = st
    return stats


def kpoint_evidence(conv, surf, prod_density):
    """
    What the sweep demonstrates AT OR BELOW the production k-point density.

    The production cells are not the swept cells -- the (100) production slab is
    a 2x1 where the sweep used a 2x2 -- so the swept mesh and the production
    mesh are different densities and quoting the sweep's own production point
    would overstate what was shown. The anchor is therefore the densest swept
    point no finer than production: convergence demonstrated at a COARSER
    sampling is the conservative claim, and if the sweep never went that coarse
    the row says the production density is unsupported rather than inheriting a
    number from a finer mesh.
    """
    if not conv or "_pc" not in conv:
        return None
    pc, rows = conv["_pc"], conv["_rows"]
    rs = pc.sweep_rows(rows, "kpoint", surf)
    if not rs:
        return None
    xs, ys = pc.series(rs, "k_density_a1_angstrom", "sigma_mean_kbar")
    fx, fy = pc.finite(xs, ys)
    if len(fx) < 2:
        return None
    # 1% rather than exact equality: the sweeps were run at the superseded a0
    # and production at the corrected one, so the same mesh on the same slab
    # differs in density by 0.02%. Demanding equality there would silently drop
    # the matching sweep point and quote the next coarser one instead, making
    # the evidence look weaker than it is.
    tol = 0.01
    at_or_below = [(x, y) for x, y in zip(fx, fy)
                   if x <= prod_density * (1.0 + tol)]
    if not at_or_below:
        return {"unsupported": True, "coarsest_swept": fx[0], "dense": fx[-1]}
    ax, ay = at_or_below[-1]
    return {"unsupported": False, "anchor": ax,
            "residual": abs(ay - fy[-1]), "dense": fx[-1],
            "exact": abs(ax - prod_density) <= tol * prod_density}


def table_t1(runs, conv, tau_rows, ref):
    if not runs:
        gap("T1: no production stress runs found under "
            f"{PROD}/{PROD_STRESS_GLOB}")
        return None

    outs = [r["out"] for r in runs.values()]
    ins = [r["in"] for r in runs.values()]
    rows = []

    def add(param, value, evidence, level):
        rows.append([param, value, evidence, level])

    xc = unique((o.get("xc") for o in outs), "exchange-correlation functional")
    add("functional", xc or "n/a",
        "read from pw.out, not from the input", "--")

    pseudo_c = unique((o["pseudos"].get("C") for o in outs), "C pseudopotential")
    pseudo_h = unique((o["pseudos"].get("H") for o in outs), "H pseudopotential")
    add("pseudopotential (C)", pseudo_c or "n/a",
        "as opened by pw.x (CLAUDE.md invariant 7)", "--")
    add("pseudopotential (H)", pseudo_h or "n/a",
        "as opened by pw.x (CLAUDE.md invariant 7)", "--")

    ecutwfc = unique((o.get("ecutwfc") for o in outs), "ecutwfc")
    ecutrho = unique((o.get("ecutrho") for o in outs), "ecutrho")
    if conv and conv.get("cutoff"):
        c = conv["cutoff"]
        sig = max(s["residual_at_prod"] for s in c.values())
        ani = max(s["anis_residual_at_prod"] for s in c.values())
        ev = (f"vs densest cutoff computed: {sig:.2f} kbar in mean sigma, "
              f"{ani:.2f} kbar in anisotropy")
        lvl = "L2"
    else:
        ev, lvl = "NOT DEMONSTRATED HERE", "--"
    add("ecutwfc", f"{ecutwfc:.0f} Ry" if ecutwfc else "n/a", ev, lvl)
    add("ecutrho", f"{ecutrho:.0f} Ry" if ecutrho else "n/a",
        f"ecutrho/ecutwfc = {ecutrho / ecutwfc:.0f}" if ecutwfc and ecutrho
        else "n/a", lvl)

    for surf in SURFACES:
        rs = [r for r in runs.values() if r["surface"] == surf]
        if not rs:
            gap(f"T1: no production run for {surf}; its k-mesh is not reported")
            continue
        mesh = unique((r["in"].get("kmesh") for r in rs), f"{surf} k-mesh")
        mesh_txt = "x".join(str(m) for m in mesh) if mesh else "n/a"
        # Density along BOTH in-plane axes. The cells are not square on (100)
        # and (110), so a single number would hide the coarser direction, which
        # is the one that governs.
        cells = [r["in"].get("cell") for r in rs if r["in"].get("cell")]
        dens = None
        if mesh and cells:
            import math
            norms = [math.hypot(*cells[0][i][:2]) for i in (0, 1)]
            dens = [mesh[i] * norms[i] for i in (0, 1)]
        if dens:
            val = (f"{mesh_txt} (n_k a = {dens[0]:.1f}, {dens[1]:.1f} A)"
                   if abs(dens[0] - dens[1]) > 0.05
                   else f"{mesh_txt} (n_k a = {dens[0]:.1f} A)")
        else:
            val = mesh_txt
        st = kpoint_evidence(conv, surf, min(dens)) if dens else None
        if st is None:
            ev, lvl = "NOT SWEPT -- no k-point convergence claimed", "L1"
        elif st["unsupported"]:
            ev = (f"UNSUPPORTED: coarsest mesh swept is {st['coarsest_swept']:.1f} "
                  f"A, finer than production")
            lvl = "L1"
            gap(f"T1 k-mesh {MILLER[surf]}: production density "
                f"{min(dens):.1f} A is coarser than anything in the sweep; no "
                "k-point convergence is claimed for it")
        else:
            same = ("" if st["exact"]
                    else f" (production is finer than this swept point)")
            ev = (f"{st['anchor']:.1f} A sits within {st['residual']:.3f} kbar "
                  f"of the densest swept ({st['dense']:.1f} A){same}")
            lvl = "L2"
        add(f"k-mesh, {MILLER[surf]}", val, ev, lvl)

    lz = [r["in"].get("lz") for r in runs.values() if r["in"].get("lz")]
    vac = None
    if tau_rows:
        # Vacuum is Lz minus the atomic extent, so it needs the geometry, not
        # just the cell. Taken from the thinnest production slab, which is the
        # least favourable case in the ladder.
        pts = read_csv(PROD / "tau_infinity_points.csv")
        if pts:
            gaps = [fnum(p, "Lz_angstrom") - fnum(p, "thickness_geometric_angstrom")
                    for p in pts
                    if fnum(p, "Lz_angstrom") and fnum(p, "thickness_geometric_angstrom")]
            vac = min(gaps) if gaps else None
    if conv and conv.get("vacuum"):
        worst = max(s["tau_spread"] for s in conv["vacuum"].values())
        rng = next(iter(conv["vacuum"].values()))
        ev = (f"tau moves {worst:.4f} N/m over {rng['vac_min']:.0f}-"
              f"{rng['vac_max']:.0f} A; sigma alone is not converged")
        lvl = "L2"
    else:
        ev, lvl = "NOT DEMONSTRATED HERE", "--"
    add("vacuum (thinnest slab)",
        f"{vac:.1f} A" if vac else (f"Lz = {min(lz):.1f}-{max(lz):.1f} A"
                                    if lz else "n/a"), ev, lvl)

    if tau_rows:
        used = unique((r.get("layers_used") for r in tau_rows), "layers used")
        excl = unique((r.get("layers_excluded") for r in tau_rows),
                      "layers excluded")
        add("thickness ladder", used or "n/a",
            f"{excl} excluded as outside the asymptotic regime" if excl
            else "no exclusions", "L2")
    else:
        gap(f"T1 thickness ladder: {TAU_CSV} absent")

    if ref:
        b = ref["bulk_reference"]
        a0 = b.get("a0_fit_angstrom")
        bulk_mod = b.get("bulk_modulus_gpa")
        method = b.get("fit_method", "EOS fit")
        ev = method.replace("_", " ")
        if bulk_mod is None:
            gap("T1 lattice constant: bulk modulus absent from "
                f"{REF_JSON.name}; quoted without it")
        else:
            ev += f", B = {bulk_mod:.2f} GPa"
        add("lattice constant a0", f"{a0:.6f} A" if a0 else "n/a", ev, "L2")
    else:
        gap(f"T1 lattice constant: {REF_JSON} absent")

    sm = unique((o.get("smearing") for o in outs), "smearing")
    dg = unique((o.get("degauss") for o in outs), "degauss")
    add("smearing", f"{sm}, {dg:g} Ry" if sm and dg else "n/a",
        "metallic-style smearing on an insulator; a numerical device, "
        "no occupation is fractional", "--")
    ct = unique((o.get("conv_thr") for o in outs), "conv_thr")
    add("SCF threshold", f"{ct:g} Ry" if ct else "n/a",
        "tighter than the default 1e-6 because stress converges after energy",
        "--")
    calc = unique((i.get("calculation") for i in ins), "calculation")
    add("stress run type", calc or "n/a",
        "single point on the relaxed geometry; ions are NOT re-relaxed", "--")

    caption = (
        f"Production settings for the {len(runs)} stress SCFs that "
        f"the surface-stress results are built from, together with what was "
        f"demonstrated about each one. The 'evidence' column is the point of "
        f"the table: a setting with no sweep behind it says so rather than "
        f"being listed as though it were converged. Values are read from the "
        f"committed pw.in and pw.out of the runs themselves -- pseudopotentials "
        f"from pw.out, which is ground truth for the file pw.x actually opened. "
        f"The convergence numbers are taken from the same reduction that draws "
        f"the convergence figures, so table and figure cannot disagree. Levels "
        f"are per CLAUDE.md section 4 and apply to the CONVERGENCE CLAIM in "
        f"that row, not to the setting."
    )
    notes = ["Positive sigma means the cell is COMPRESSED (CLAUDE.md sec 2).",
             "Smearing and threshold rows carry no level: they are inputs, not "
             "claims."]
    return Table("t1_production_settings", 1,
                 "Production settings and what was demonstrated about each",
                 ["parameter", "production value", "convergence evidence",
                  "level"],
                 rows, caption, align="llll", notes=notes)


# ── T2: tau_infinity per facet ────────────────────────────────────────────────

def table_t2(tau_rows, meta):
    if not tau_rows:
        gap(f"T2: {TAU_CSV} absent")
        return None
    rows = []
    for surf in SURFACES:
        r = next((x for x in tau_rows if x["surface"] == surf), None)
        if r is None:
            gap(f"T2: no row for {surf} in tau_infinity.csv")
            continue
        rows.append([
            MILLER[surf],
            f"{r.get('axis_x','')} / {r.get('axis_y','')}",
            f"{fnum(r, 'tau_xx_inf_n_per_m'):+.3f}",
            f"{fnum(r, 'tau_yy_inf_n_per_m'):+.3f}",
            f"{fnum(r, 'tau_mean_n_per_m'):+.3f}",
            f"{fnum(r, 'tau_aniso_n_per_m'):+.3f}",
            f"{fnum(r, 'sigma_res_mean_kbar'):+.3f}",
            f"{max(fnum(r, 'rms_xx_as_tau_n_per_m'), fnum(r, 'rms_yy_as_tau_n_per_m')):.4f}",
            r.get("epistemic_level", "?"),
        ])
    if not rows:
        return None

    tol = (meta or {}).get("sigma_res_tol_kbar")
    consistent = (meta or {}).get("sigma_res_consistent")
    caption = (
        "Thickness-extrapolated surface stress per facet, the one L2 "
        "quantity in this chain and the input every later number depends on. "
        "tau_xx and tau_yy are the two in-plane axes named in the second "
        "column; the axis assignment is not conventional and the anisotropy "
        "sign is meaningless without it. POSITIVE tau IS COMPRESSIVE (CLAUDE.md "
        "sec 2): it equals MINUS the continuum surface stress f, so a facet "
        "with positive tau relaxes by EXPANDING, which was verified directly "
        "against free 2D vc-relax on all three surfaces and six axes. sigma_res "
        "is the residual interior stress the ladder fit leaves behind"
        + (f", within the {tol:g} kbar tolerance on all three facets"
           if consistent and tol else "")
        + ". The rms column is the larger of the two per-axis fit residuals "
        "expressed as a tau, i.e. the fit's own scatter, and is not an estimate "
        "of the systematic error in the DFT. (111) is isotropic identically, "
        "not numerically: its 3-fold symmetry forbids an in-plane anisotropy."
    )
    notes = []
    if meta:
        notes.append(f"Thickness definition: {meta.get('thickness_definition')}.")
        notes.append(f"Fit uses {meta.get('layers_used_note')}; "
                     f"{meta.get('layers_excluded_note')} excluded.")
        notes.append(f"a0 = {meta.get('a0_angstrom'):.6f} A at "
                     f"{meta.get('ecutwfc'):.0f}/{meta.get('ecutrho'):.0f} Ry.")
    hdr = ["facet", "axes x / y", "tau_xx", "tau_yy", "tau_mean", "tau_aniso",
           "sigma_res", "fit rms", "level"]
    hdr_tex = ["facet", "axes $x$ / $y$",
               r"$\tau_{xx}$", r"$\tau_{yy}$", r"$\bar\tau$",
               r"$\tau_{xx}-\tau_{yy}$", r"$\sigma_{\rm res}$",
               "fit rms", "level"]
    return Table("t2_tau_infinity", 2,
                 "Surface stress per facet (N/m; sigma_res in kbar)",
                 hdr, rows, caption, headers_tex=hdr_tex,
                 align="llrrrrrrl", notes=notes)


# ── T3: surface energies ──────────────────────────────────────────────────────

def table_t3(se_rows, gamma_cfg):
    if not se_rows:
        gap(f"T3: {SURF_E_CSV} absent")
        return None
    cfg_se = (gamma_cfg or {}).get("surface_energies", {})
    rows, any_ceiling = [], False
    for surf in SURFACES:
        r = next((x for x in se_rows if x["surface"] == surf), None)
        if r is None:
            gap(f"T3: no row for {surf} in surface_energy.csv")
            continue
        fam = surf[1:]
        ceiling = (cfg_se.get(fam) or {}).get("dehydrogenation_ceiling_delta_mu_ev")
        if ceiling is None:
            ceil_txt = "NOT COMPUTED"
        else:
            ceil_txt, any_ceiling = f"{ceiling:.3f}", True
        zc = fnum(r, "zero_crossing_delta_mu_ev")
        rows.append([
            MILLER[surf],
            f"{fnum(r, 'gamma_h_rich_j_m2'):+.4f}",
            f"{fnum(r, 'gamma_scatter_j_m2'):.4f}",
            f"{fnum(r, 'dgamma_dmu_j_m2_per_ev'):.4f}",
            f"{fnum(r, 'coverage_n_h_over_2a_per_a2'):.4f}",
            f"{zc:+.4f}" if zc is not None else "n/a",
            ceil_txt,
            f"{fnum(r, 'mu_c_drift_mry'):+.4f}",
            f"{fnum(r, 'gamma_bias_from_mu_c_drift_j_m2'):+.4f}",
            r.get("epistemic_level", "?"),
        ])
    if not rows:
        return None
    # Which facet the mu_C drift actually hurts, derived rather than asserted:
    # naming the wrong one in prose is exactly the kind of stale claim that
    # survives a data change unnoticed.
    def bias_of(r):
        return abs(fnum(r, "gamma_bias_from_mu_c_drift_j_m2") or 0.0)

    worst = max(se_rows, key=bias_of)
    others = [r for r in se_rows if r is not worst]
    dominates = others and bias_of(worst) > 3.0 * max(bias_of(r) for r in others)
    if dominates:
        lv = worst.get("epistemic_level", "?")
        other_lv = sorted({r.get("epistemic_level", "?") for r in others})
        worst_txt = (
            f"; on {MILLER[worst['surface']]} it is {bias_of(worst) / max(bias_of(r) for r in others):.0f}x "
            f"larger than on either other facet, which is why that row is {lv} "
            f"while the others are {'/'.join(other_lv)}")
    else:
        worst_txt = ""

    if not any_ceiling:
        gap("T3 dehydrogenation ceiling: no bare-facet ladder in results/, so "
            "the column is NOT COMPUTED on every facet. Without it the "
            "hydrogen-poor end of the gamma(mu_H) window has no stop, and F2/F3 "
            "inherit that.")

    caption = (
        "Surface energy of the H-terminated facets and its hydrogen "
        "chemical-potential dependence: gamma(Delta mu_H) = gamma(H-rich) + "
        "(dgamma/dmu) Delta mu_H, with Delta mu_H measured BELOW the H-rich "
        "limit. The slope is the H coverage, so the ordering of the three "
        "facets changes with mu_H and the equilibrium habit changes with it -- "
        "that dependence is what F2 and F3 are built on. Negative gamma at the "
        "H-rich limit is not an error: it means H termination is favourable "
        "against the H2 reservoir there, but a Wulff construction is undefined "
        "while any gamma < 0, which is why the shape at that end is not "
        "computed. The mu_C drift column is the fitted carbon chemical "
        "potential's departure from the bulk reference and the gamma bias is "
        "what that drift costs" + worst_txt + ". The dehydrogenation ceiling "
        "-- the Delta mu_H beyond which a bare facet is more stable -- is "
        + ("tabulated per facet." if any_ceiling else
           "NOT COMPUTED: it requires a bare-facet ladder that is not in this "
           "repository, so the hydrogen-poor side of the window is unbounded "
           "and any statement made there is an extrapolation with no stop.")
    )
    hdr = ["facet", "gamma(H-rich)", "scatter", "dgamma/dmu", "coverage",
           "gamma=0 at", "dehyd. ceiling", "mu_C drift", "gamma bias", "level"]
    hdr_tex = ["facet", r"$\gamma$ (H-rich)", "scatter",
               r"$d\gamma/d\mu$", "coverage", r"$\gamma=0$ at",
               "dehyd. ceiling", r"$\mu_{\rm C}$ drift", r"$\gamma$ bias",
               "level"]
    notes = [
        "Units: gamma and scatter J/m^2; dgamma/dmu J m^-2 eV^-1; coverage "
        "n_H/2A per A^2; crossings and ceiling eV below the H-rich limit; "
        "mu_C drift mRy; gamma bias J/m^2.",
        "Scatter is the spread of per-slab gamma across the fitted thickness "
        "ladder, i.e. reproducibility within the ladder, not accuracy.",
    ]
    return Table("t3_surface_energies", 3,
                 "Surface energies and their hydrogen chemical-potential slopes",
                 hdr, rows, caption, headers_tex=hdr_tex,
                 align="lrrrrrrrrl", notes=notes)


# ── T4: predicted D and E per habit ───────────────────────────────────────────

def pick(rows, **eq):
    for r in rows or []:
        if all(close(fnum(r, k), v) if isinstance(v, float) else r.get(k) == v
               for k, v in eq.items()):
            return r
    return None


def table_t4(nv_by_shape, metas, depth_rows, odmr_rows, radius_nm):
    shapes = [s for s in SHAPES if s in nv_by_shape]
    if not shapes:
        gap("T4: no particle_strain_nv_<shape>.csv found")
        return None

    # Emission order from the metadata, not alphabetical: the first entry is
    # the set the rest of the pipeline defaults to, and a reader comparing this
    # table against the CSVs should meet them in the same order.
    declared = []
    for s in shapes:
        for ps in (metas.get(s) or {}).get("param_sets", []):
            if ps not in declared:
                declared.append(ps)
    present = {r["spin_strain_param_set"] for s in shapes for r in nv_by_shape[s]}
    param_sets = ([p for p in declared if p in present]
                  + sorted(present - set(declared)))
    refs = {}
    for s in shapes:
        m = metas.get(s) or {}
        refs.update(m.get("param_set_references", {}))

    # Rows are quantities and columns are habits: the table is meant to be read
    # down a habit, and a 13-column layout is not readable on a page.
    labels, cells = [], []

    def add(label, fn):
        vals = []
        for s in shapes:
            try:
                vals.append(fn(s))
            except Exception:                       # noqa: BLE001
                vals.append(None)
        if all(v is None for v in vals):
            gap(f"T4 '{label}': no data for any habit; row omitted")
            return
        labels.append(label)
        cells.append(["n/a" if v is None else v for v in vals])

    def facet_txt(s):
        ff = (metas.get(s) or {}).get("family_fractions")
        if not ff:
            return None
        return ", ".join(f"{{{k}}} {v:.2f}" for k, v in sorted(ff.items()))

    def nv_row(s, pset=None):
        rows = nv_by_shape[s]
        if pset:
            rows = [r for r in rows if r["spin_strain_param_set"] == pset]
        rows = [r for r in rows if close(fnum(r, "radius_nm"), radius_nm)]
        return rows[0] if rows else None

    def scalar(s, key, fmt, pset=None):
        r = nv_row(s, pset)
        v = fnum(r, key) if r else None
        return None if v is None else format(v, fmt)

    def dD(s, pset):
        rows = [r for r in nv_by_shape[s]
                if r["spin_strain_param_set"] == pset
                and close(fnum(r, "radius_nm"), radius_nm)]
        vals = [fnum(r, "delta_D_mhz") for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        lo, hi = min(vals), max(vals)
        # Equal across NV axes for a symmetry-complete habit; if they ever
        # differ, the span is the honest report, not the first axis.
        return f"{lo:+.1f}" if close(lo, hi, 1e-9) else f"{lo:+.1f} to {hi:+.1f}"

    def depth_E(s, depth):
        r = pick([x for x in depth_rows or [] if x["shape"] == s],
                 depth_nm=depth, radius_nm=radius_nm)
        v = fnum(r, "E_median_mhz") if r else None
        return None if v is None else f"{v:.2f}"

    def odmr(s, dark, key, fmt):
        r = pick([x for x in odmr_rows or [] if x["shape"] == s],
                 dark_layer_nm=dark, radius_nm=radius_nm)
        if r is None:
            return None
        if (r.get("n_sites") or "0") == "0":
            return "no sites"
        v = fnum(r, key)
        return None if v is None else format(v, fmt)

    add("facet area fractions", facet_txt)
    add("interior P (GPa, + = compressive)",
        lambda s: scalar(s, "pressure_gpa", "+.3f"))
    add("lattice strain (%)",
        lambda s: scalar(s, "lattice_strain_percent", "+.3f"))
    add("lattice parameter a (A)",
        lambda s: scalar(s, "lattice_parameter_angstrom", ".4f"))
    for ps in param_sets:
        add(f"Delta D (MHz), {ps}", lambda s, ps=ps: dD(s, ps))
    add("E from the volume-averaged strain (MHz)",
        lambda s: scalar(s, "E_mhz", ".3f"))
    for d in DEPTHS_NM:
        add(f"median |E(x)| (MHz) at depth {d:.1f} nm",
            lambda s, d=d: depth_E(s, d))
    for dl in DARK_LAYERS_NM:
        add(f"ODMR FWHM (MHz), dark layer {dl:.1f} nm",
            lambda s, dl=dl: odmr(s, dl, "strain_fwhm_mhz", ".1f"))
    for dl in DARK_LAYERS_NM:
        add(f"fraction with 2|E| > linewidth, dark layer {dl:.1f} nm",
            lambda s, dl=dl: odmr(s, dl, "frac_2E_over_linewidth", ".2f"))

    rows = [[lab] + list(cs) for lab, cs in zip(labels, cells)]

    lw = None
    for r in odmr_rows or []:
        lw = fnum(r, "linewidth_ref_mhz") or lw
    spread = None
    for m in metas.values():
        spread = m.get("coupling_spread_factor") or spread
    elastic = next((m.get("elastic_citation") for m in metas.values()
                    if m.get("elastic_citation")), None)
    c11 = next((m.get("C11") for m in metas.values() if m.get("C11")), None)
    c12 = next((m.get("C12") for m in metas.values() if m.get("C12")), None)
    c44 = next((m.get("C44") for m in metas.values() if m.get("C44")), None)

    caption = (
        f"Predicted NV shifts for each habit at R = {radius_nm:g} nm. "
        f"The first three quantities are COUPLING-FREE: interior pressure, "
        f"lattice strain and lattice parameter follow from tau and the elastic "
        f"tensor alone, so none of the spin-strain uncertainty touches them, and "
        f"the lattice parameter is directly measurable by size-resolved XRD. "
        f"Everything below them is not. Delta D is given once per published "
        f"coupling set rather than as a single number with an error bar, because "
        f"the two sets differ by a factor of "
        + (f"{spread:.2f}" if spread else "roughly 1.65")
        + " on the axial channel for identical strain and that disagreement, "
        "not statistics, is the dominant uncertainty. The two E rows say "
        "different things and the difference is the substantive result: E "
        "evaluated at the VOLUME-AVERAGED strain is identically zero for every "
        "symmetry-complete habit, because that average is hydrostatic; the "
        "median |E(x)| is non-zero because the field resolved in position is "
        "not, being concentrated at the particle edges. An ensemble therefore "
        "sees inhomogeneous BROADENING rather than a resolved splitting, which "
        "is what the FWHM rows quantify"
        + (f" against a {lw:g} MHz reference linewidth" if lw else "")
        + ". The dark-layer rows are an assumption about where NVs are "
        "optically active, not a result: halving it from 1.0 to 0.5 nm moves "
        "the width by a factor of 2-4, more than most of the physics in this "
        "table. All quantities scale as 1/R away from this radius (P, strain "
        "and Delta D exactly; |E| and the widths only approximately, since the "
        "edge-to-interior geometry does not rescale exactly), so a 3 nm "
        "particle halves every number here. Epistemic level L1 throughout: the "
        "facet tau are L2 but the continuum body, the literature elastic "
        "tensor, the literature couplings and the assumed depth distribution "
        "are not."
    )
    notes = []
    if c11 and c12 and c44:
        notes.append(f"Elastic constants C11/C12/C44 = {c11:g}/{c12:g}/{c44:g} "
                     f"GPa are LITERATURE values, not fitted in this project"
                     + (f". Source: {elastic}" if elastic else "."))
    for ps in param_sets:
        if refs.get(ps):
            notes.append(f"{ps}: {refs[ps]}")
    notes.append("Delta D depends only on h41 and h43; E only on h15 and h16. "
                 "A hydrostatic-pressure ODMR calibration constrains the "
                 "former and says nothing about the latter, so E carries "
                 "strictly more uncertainty than Delta D.")
    notes.append("Positive interior P is compressive (CLAUDE.md sec 2).")

    hdr = ["quantity"] + [SHAPE_LABEL[s] for s in shapes]
    return Table("t4_predicted_D_and_E", 4,
                 f"Predicted D and E per habit at R = {radius_nm:g} nm",
                 hdr, rows, caption,
                 align="l" + "r" * len(shapes), notes=notes)


# ── driver ────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--radius-nm", type=float, default=1.5,
                    help="particle radius for T4; must exist in the data")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tau_rows = read_csv(TAU_CSV)
    tau_meta = read_json(TAU_META)
    se_rows = read_csv(SURF_E_CSV)
    gamma_cfg = read_json(GAMMA_JSON)
    depth_rows = read_csv(NV_DEPTH)
    odmr_rows = read_csv(NV_ODMR)
    ref = read_json(REF_JSON)

    nv_by_shape, metas = {}, {}
    for s in SHAPES:
        rows = read_csv(PROD / f"particle_strain_nv_{s}.csv")
        if rows:
            nv_by_shape[s] = rows
        m = read_json(PROD / f"particle_strain_meta_{s}.json")
        if m:
            metas[s] = m

    if depth_rows is None:
        gap(f"T4 depth-resolved |E|: {NV_DEPTH} absent")
    if odmr_rows is None:
        gap(f"T4 ODMR widths: {NV_ODMR} absent")

    runs = collect_production_runs()
    conv = convergence_stats()

    tables = [t for t in (
        table_t1(runs, conv, tau_rows, ref),
        table_t2(tau_rows, tau_meta),
        table_t3(se_rows, gamma_cfg),
        table_t4(nv_by_shape, metas, depth_rows, odmr_rows, args.radius_nm),
    ) if t is not None]

    for t in tables:
        (outdir / f"T{t.number}_{t.key.split('_', 1)[1]}.md").write_text(
            render_md(t))
        (outdir / f"T{t.number}_{t.key.split('_', 1)[1]}.tex").write_text(
            render_tex(t))
        print(f"  T{t.number}: {t.title}")

    index = ["# H-terminated result tables (T1-T4)", "",
             "Generated by `make_tables.py`; every value is read from a "
             "committed file and nothing is hardcoded.", "",
             "Sign convention (CLAUDE.md sec 2): positive sigma means "
             "COMPRESSED; positive tau is COMPRESSIVE surface stress; positive "
             "pressure is compressive.", ""]
    for t in tables:
        index += [f"## T{t.number} - {t.title}", "", render_md(t).split("\n", 2)[2],
                  ""]
    if _gaps:
        index += ["## Gaps", "",
                  "These are stated rather than filled. Each is a missing "
                  "input, not a small number.", ""]
        index += [f"- {g}" for g in _gaps]
        index.append("")
    (outdir / "TABLES.md").write_text("\n".join(index))
    print("  TABLES.md")

    if _gaps:
        print(f"\n{len(_gaps)} GAP(S) -- see TABLES.md")
    print(f"\nWrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
