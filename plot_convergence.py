#!/usr/bin/env python3
"""
plot_convergence.py - Journal-quality convergence and reference figures.

Builds the five figures that establish the production standard for the
nanodiamond surface-stress campaign, plus a CAPTIONS.md describing each one.
All styling lives in figstyle.py so the set is visually identical.

    Fig 1  plane-wave cutoff convergence      (single column, 2 panels)
    Fig 2  k-point convergence                (single column, 2 panels)
    Fig 3  vacuum convergence, sigma vs tau   (single column, twin axes)
    Fig 4  slab-thickness ladder              (single column)
    Fig 5  bulk equation of state             (single column, 2 panels)

Data provenance
---------------
Every number is read from a file. Nothing about the physics is hardcoded here,
including the adopted production settings, which are derived from what the runs
actually used rather than asserted:

    results/convergence/convergence_summary.csv      cutoff / k-point / vacuum /
                                                     thickness sweeps, written by
                                                     parse_convergence.py
    results/production/regression~kpt_C110~k_*       the (110) k-point sweep, run
                                                     after the summary CSV was
                                                     frozen and read from its run
                                                     directories
    results/reference_90_720/reference_summary.csv   bulk E(V), P(eps) series
    results/reference_90_720/bulk_fit_summary.json   fitted EOS parameters
    results/production/tau_infinity.csv              tau_infinity, written by
                                                     fit_tau_infinity.py; gates
                                                     Fig 4 (see TAU_INF_CSV)

Runs that did not reach JOB DONE or did not converge are excluded and named on
stdout. A quantity that is missing for one point becomes a NaN, which leaves a
visible gap in the line; it is never interpolated across. No smoothing or
extrapolation is applied anywhere except the explicit linear fits in Fig 4 and
the EOS fits in Fig 5, both of which are drawn as fits and labelled as such.

SIGN CONVENTION (CLAUDE.md section 2, corrected 2026-08):

    sigma > 0  ->  the cell is COMPRESSED; it wants to expand. Pressure-like.
    sigma < 0  ->  the cell is in TENSION; it wants to contract.
    QE pressure P = +(1/3) tr(sigma).

Anchored on bulk diamond at -1% strain (unambiguously compression), which gives
sigma_xx = sigma_yy = sigma_zz = P = +162.86 kbar. This docstring previously
stated the convention backwards; captions here use the corrected reading.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import figstyle
from fit_bulk_reference import ScaledPoly, linear_fit

USE_TEX = figstyle.apply()

import matplotlib.pyplot as plt  # noqa: E402

AA = figstyle.angstrom()
AA_TXT = "A"

DEFAULT_OUTDIR = Path.home() / "Documents" / "diamonds" / "convergence"
CONVERGENCE_CSV = Path("results/convergence/convergence_summary.csv")
BULK_CSV = Path("results/reference_90_720/reference_summary.csv")
BULK_FIT_JSON = Path("results/reference_90_720/bulk_fit_summary.json")

PRODUCTION_DIR = Path("results/production")

# Figure 4 reports tau_infinity, which is the property of the extrapolation
# fit owned by fit_tau_infinity.py. This module must never refit it: two
# independent implementations of the same fit can disagree silently, and the
# disagreement would look exactly like a physical result. If the CSV is absent
# the figure is skipped and said so on stdout, never recomputed here.
TAU_INF_CSV = PRODUCTION_DIR / "tau_infinity.csv"

# The (110) k-point mesh sweep was run after results/convergence/ was frozen and
# lives under results/production/ as a regression set, so it is absent from
# convergence_summary.csv and has to be read from its run directories.
KPT_EXTRA_GLOB = "regression~kpt_C110~k_*"

# tau [N/m] = sigma [kbar] * Lz [A] * 0.005; see CLAUDE.md section 2.
TAU_FACTOR = 0.005

PCT = r"\%" if USE_TEX else "%"


# ── Data loading ──────────────────────────────────────────────────────────────

def fnum(row, key):
    """Float from a CSV cell, or None when blank/unparseable."""
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_convergence(path):
    """
    Read the sweep summary, dropping runs that did not complete.

    Returns (rows, excluded). Exclusion is on the two hard criteria only:
    JOB DONE and SCF convergence. Provenance flags such as a folder name that
    omits its orientation are reported by parse_convergence.py and do not
    invalidate the number itself.
    """
    if not path.exists():
        raise SystemExit(f"missing input: {path}\nRun parse_convergence.py first.")
    rows, excluded = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            ok = (row.get("job_done", "").strip() == "yes"
                  and row.get("scf_converged", "").strip() == "yes")
            (rows if ok else excluded).append(row)
    return rows, excluded


def load_extra_kpoint_runs(root=PRODUCTION_DIR, pattern=KPT_EXTRA_GLOB):
    """
    k-point runs that live outside results/convergence/ and so are not in the CSV.

    Returns (rows, skipped). The rows are shaped exactly like csv.DictReader
    rows -- every value a string -- so the plotting code cannot tell them from
    summary rows, and `fnum` keeps working unchanged.

    Parsing goes through parse_convergence's own parse_pw_in/parse_pw_out rather
    than a second implementation, so a fix to the QE output format propagates
    here for free. A run that did not reach JOB DONE, did not converge, or is
    missing the stress tensor is skipped and named, never silently dropped.
    """
    import parse_convergence as pc

    rows, skipped = [], []
    for d in sorted(Path(root).glob(pattern)):
        if not (d / "pw.out").exists():
            skipped.append((d.name, "no pw.out"))
            continue
        pin = pc.parse_pw_in(d / "pw.in")
        pout = pc.parse_pw_out(d / "pw.out")

        if not pout["job_done"] or not pout["scf_converged"]:
            skipped.append((d.name, f"job_done={pout['job_done']} "
                                    f"scf_converged={pout['scf_converged']}"))
            continue
        sxx, syy = pout["sigma_xx_kbar"], pout["sigma_yy_kbar"]
        if sxx is None or syy is None:
            skipped.append((d.name, "no stress tensor in pw.out"))
            continue
        if pin["k1"] is None or pin["a1_angstrom"] is None:
            skipped.append((d.name, "no k-mesh or cell in pw.in"))
            continue

        # Orientation from the source slab recorded in the input, not the folder
        # name -- CLAUDE.md section 1.3: names are checked, never trusted.
        surface, trusted = pc.surface_of(pin["source_slab"], d.name)
        if surface is None:
            skipped.append((d.name, "orientation not identifiable"))
            continue
        if not trusted:
            skipped.append((d.name, "orientation only inferable from folder name"))
            continue

        rows.append({
            "run": d.name,
            "sweep": "kpoint",
            "surface": surface,
            "source_slab": pin["source_slab"] or "",
            "nat": str(pout["nat"] if pout["nat"] is not None else ""),
            "ecutwfc_ry": f"{pout['ecutwfc_ry']:.6f}" if pout["ecutwfc_ry"] else "",
            "kmesh": pin["kmesh"] or "",
            "k1": str(pin["k1"]), "k2": str(pin["k2"]), "k3": str(pin["k3"]),
            "a1_angstrom": f"{pin['a1_angstrom']:.6f}",
            "a2_angstrom": f"{pin['a2_angstrom']:.6f}",
            "lz_angstrom": f"{pin['lz_angstrom']:.6f}",
            "k_density_a1_angstrom": f"{pin['k1'] * pin['a1_angstrom']:.6f}",
            "k_density_a2_angstrom": f"{pin['k2'] * pin['a2_angstrom']:.6f}",
            "energy_ry": f"{pout['energy_ry']:.8f}" if pout["energy_ry"] else "",
            "sigma_xx_kbar": f"{sxx:.6f}",
            "sigma_yy_kbar": f"{syy:.6f}",
            "sigma_mean_kbar": f"{0.5 * (sxx + syy):.6f}",
            "anisotropy_kbar": f"{sxx - syy:.6f}",
            "scf_converged": "yes",
            "job_done": "yes",
        })
    return rows, skipped


def require_columns(rows, *names):
    """
    Fail loudly if the summary CSV predates a column a figure needs.

    parse_convergence.py has gained columns over the campaign. A summary CSV
    written before one of them existed is not corrupt and loads cleanly; the
    column is simply absent, every lookup returns None, and the figure dies
    several frames later on an empty list. Naming the missing column and the
    command that regenerates it turns a confusing IndexError into an
    instruction.
    """
    if not rows:
        return
    missing = [n for n in names if n not in rows[0]]
    if missing:
        raise SystemExit(
            f"{CONVERGENCE_CSV} is missing column(s) {', '.join(missing)}.\n"
            f"The summary predates the current parse_convergence.py. "
            f"Regenerate it with:\n"
            f"    python3 parse_convergence.py --indir results/convergence "
            f"--out <path>\n"
            f"then pass --csv <path>.")


def sweep_rows(rows, sweep, surface=None):
    out = [r for r in rows if r["sweep"] == sweep]
    if surface is not None:
        out = [r for r in out if r["surface"] == surface]
    return out


def surfaces_in(rows, sweep):
    """Surfaces actually present in a sweep, in canonical order. Never assumed."""
    present = {r["surface"] for r in sweep_rows(rows, sweep)}
    return [s for s in figstyle.SURFACES if s in present]


def series(rows, xkey, ykey):
    """
    (xs, ys) sorted by x, with a missing y as NaN so the line breaks visibly
    rather than being bridged.
    """
    pts = []
    for r in rows:
        x = fnum(r, xkey)
        if x is None:
            continue
        y = fnum(r, ykey)
        pts.append((x, float("nan") if y is None else y))
    pts.sort(key=lambda p: p[0])
    return [p[0] for p in pts], [p[1] for p in pts]


# ── Production settings, derived from the data ────────────────────────────────

def production_cutoff(rows):
    """
    The adopted ecutwfc: the one the non-cutoff sweeps were all run at.

    Derived, not asserted. If those runs disagree among themselves then there is
    no single adopted cutoff, which is an error worth stopping for.
    """
    used = Counter(fnum(r, "ecutwfc_ry") for r in rows
                   if r["sweep"] in ("kpoint", "vacuum", "thickness"))
    used.pop(None, None)
    if len(used) != 1:
        raise SystemExit(
            f"cannot derive a single production cutoff; sweeps used {dict(used)}")
    return next(iter(used))


def production_kmesh(rows, surface):
    """The k-mesh this surface's cutoff sweep ran at, with its k-density."""
    cut = sweep_rows(rows, "cutoff", surface)
    meshes = {r["kmesh"] for r in cut}
    if len(meshes) != 1:
        raise SystemExit(f"{surface}: cutoff sweep used mixed k-meshes {meshes}")
    return next(iter(meshes)), fnum(cut[0], "k_density_a1_angstrom")


def fmt_mesh(mesh):
    """'5 5 1' -> '5x5x1', typeset with a proper multiplication sign."""
    return r"$\times$".join(mesh.split())


LAYERS_RE = re.compile(r"_(\d+)L(?:_|$)")


def declared_layers(rows):
    """
    The carbon layer count shared by a set of runs, from their source slab names
    and cross-checked against the carbon atom count.

    `n_layers_geometry` in the CSV now agrees with this (parse_convergence's
    count_layers equalises layer populations, so the (100) 2x2 slab whose
    0.292 A interior buckling used to split it into 8 reports 6). This stays a
    separate derivation rather than reading that column because it checks a
    different thing: that the NAME describes the geometry, CLAUDE.md invariant
    3. Two independent routes to the layer count disagreeing is a signal worth
    keeping, not duplication worth collapsing.

    Returns the layer count, or None if the runs disagree or the check fails.
    """
    declared = set()
    for r in rows:
        m = LAYERS_RE.search(r.get("source_slab") or "")
        if not m:
            return None
        n_layers = int(m.group(1))
        n_c = fnum(r, "n_C")
        # A name that does not divide the atom count is a name that does not
        # describe the geometry; refuse it rather than print it.
        if n_c is None or n_layers == 0 or int(n_c) % n_layers != 0:
            return None
        declared.add(n_layers)
    return next(iter(declared)) if len(declared) == 1 else None


# ── Small helpers ─────────────────────────────────────────────────────────────

def finite(xs, ys):
    """Only the pairs where y is finite — for fitting, never for plotting."""
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


def rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def linspace(lo, hi, n=200):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


SIGMA_MEAN = r"$\sigma_{\mathrm{mean}}$"
TAU_MEAN = r"$\tau_{\mathrm{mean}}$"


# ── Figure 1: plane-wave cutoff ───────────────────────────────────────────────

def figure_cutoff(rows):
    surfaces = surfaces_in(rows, "cutoff")
    ecut_prod = production_cutoff(rows)

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(figstyle.SINGLE_COL, 3.9), sharex=True)

    stats = {}
    for surf in surfaces:
        rs = sweep_rows(rows, "cutoff", surf)
        xs, ys = series(rs, "ecutwfc_ry", "sigma_mean_kbar")
        # Reference is the densest cutoff actually computed, taken from the data.
        fx, fy = finite(xs, ys)
        ref_x, ref_y = fx[-1], fy[-1]
        dys = [y - ref_y for y in ys]
        ax_a.plot(xs, dys, **figstyle.series_kwargs(surf),
                  label=figstyle.surface_label(surf))

        # Panel (b) is a delta against its own densest-cutoff value, exactly as
        # (a) is, and the two panels then share a y-scale. Plotted as an absolute
        # anisotropy instead, (b) spans ~80 kbar and a 0.2 kbar drift is a line
        # width: the flatness claim is then only legible as an annotation the
        # reader has to trust. Against the same ruler as (a) it is checkable --
        # (a) visibly drifts, (b) visibly does not.
        axs, ays = series(rs, "ecutwfc_ry", "anisotropy_kbar")
        afx, afy = finite(axs, ays)
        aref_y = afy[-1] if afy else float("nan")
        adys = [y - aref_y for y in ays]
        ax_b.plot(axs, adys, **figstyle.series_kwargs(surf))

        spread = (max(afy) - min(afy)) if afy else float("nan")
        scale = max(abs(v) for v in afy) if afy else 0.0
        # Residual drift still left at the adopted cutoff: the number that
        # actually justifies adopting it.
        at_prod = dict(zip(xs, dys)).get(ecut_prod, float("nan"))
        anis_at_prod = dict(zip(axs, adys)).get(ecut_prod, float("nan"))
        stats[surf] = {
            "ref_cut": ref_x,
            "sigma_ref": ref_y,
            "residual_at_prod": abs(at_prod),
            "drift": max((d for d in dys if not math.isnan(d)), key=abs),
            "anis": afy[-1] if afy else float("nan"),
            "anis_ref": aref_y,
            "anis_drift": max((d for d in adys if not math.isnan(d)), key=abs),
            "anis_residual_at_prod": abs(anis_at_prod),
            "anis_spread": spread,
            "anis_pct": (100.0 * spread / scale) if scale else 0.0,
        }

    for ax in (ax_a, ax_b):
        ax.axvline(ecut_prod, color=figstyle.GUIDE, linestyle="--",
                   linewidth=0.7, zorder=0)
        ax.axhline(0.0, color=figstyle.GUIDE, linestyle=":", linewidth=0.6,
                   zorder=0)

    ref_cuts = {s["ref_cut"] for s in stats.values()}
    ref_txt = f"{next(iter(ref_cuts)):.0f}" if len(ref_cuts) == 1 else "densest"

    ax_a.set_ylabel(r"$\Delta\sigma_{\mathrm{mean}}$ (kbar)")
    ax_b.set_ylabel(r"$\Delta(\sigma_{xx}-\sigma_{yy})$ (kbar)")
    ax_b.set_xlabel(r"$E_{\mathrm{cut}}$ (Ry)")

    # The whole point of the restructure: one ruler for both panels. Taken as the
    # union of what the two actually need, so neither is clipped and the
    # comparison is not rigged by a hand-picked limit.
    ylo = min(min(ax_a.get_ylim()), min(ax_b.get_ylim()))
    yhi = max(max(ax_a.get_ylim()), max(ax_b.get_ylim()))
    ax_a.set_ylim(ylo, yhi)
    ax_b.set_ylim(ylo, yhi)

    for ax in (ax_a, ax_b):
        ax.annotate(f"relative to {ref_txt} Ry", xy=(0.97, 0.06),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=figstyle.FS_ANNOT)
    ax_a.annotate(f"{ecut_prod:.0f} Ry", xy=(ecut_prod, 1.02),
                  xycoords=("data", "axes fraction"), ha="center", va="bottom",
                  fontsize=figstyle.FS_ANNOT, color=figstyle.GUIDE)

    figstyle.legend(ax_a, loc="center right")
    figstyle.panel_label(ax_a, "a")
    figstyle.panel_label(ax_b, "b")
    fig.align_ylabels((ax_a, ax_b))
    fig.subplots_adjust(hspace=0.10)

    lo = min(abs(s["drift"]) for s in stats.values())
    hi = max(abs(s["drift"]) for s in stats.values())
    anis_hi = max(abs(s["anis_drift"]) for s in stats.values())
    sigma_resid = max(s["residual_at_prod"] for s in stats.values())
    anis_resid = max(s["anis_residual_at_prod"] for s in stats.values())
    caption = (
        f"Plane-wave cutoff convergence of the symmetric H-terminated diamond "
        f"slabs, both panels plotted as a deviation from the same {ref_txt} Ry "
        f"reference and on a common y-scale so the two are directly comparable. "
        f"(a) The mean in-plane stress drifts by {lo:.1f}-{hi:.1f} kbar between "
        f"60 and {ref_txt} Ry, in the same direction on every surface: a residual "
        f"Pulay contribution to the stress rather than any change in the surface. "
        f"(b) The in-plane anisotropy, on the identical ruler, never leaves "
        f"{anis_hi:.2f} kbar of its converged value, so anisotropy-derived "
        f"quantities are already converged at 60 Ry while the mean stress is not "
        f"-- the difference is visible rather than asserted. {ecut_prod:.0f} Ry "
        f"(dashed) is adopted for production, where the residual is "
        f"{sigma_resid:.2f} kbar in the mean stress and {anis_resid:.2f} kbar in "
        f"the anisotropy, both far below the surface-to-surface differences the "
        f"campaign is measuring. Positive sigma means the cell is compressed."
    )
    return fig, caption, stats


# ── Figure 2: k-points ────────────────────────────────────────────────────────

def figure_kpoints(rows):
    surfaces = surfaces_in(rows, "kpoint")
    fig, axes = plt.subplots(
        len(surfaces), 1, figsize=(figstyle.SINGLE_COL, 3.6), sharex=True)
    axes = list(axes) if len(surfaces) > 1 else [axes]

    stats = {}
    for ax, surf in zip(axes, surfaces):
        rs = sweep_rows(rows, "kpoint", surf)
        xs, ys = series(rs, "k_density_a1_angstrom", "sigma_mean_kbar")
        ax.plot(xs, ys, **figstyle.series_kwargs(surf))

        mesh, dens = production_kmesh(rows, surf)
        ax.axvline(dens, color=figstyle.GUIDE, linestyle="--", linewidth=0.7,
                   zorder=0)
        # Top of the panel: both curves fall steeply on the left and flatten
        # near the bottom, so a bottom-anchored label lands on the data.
        ax.annotate(fmt_mesh(mesh), xy=(dens, 0.94),
                    xycoords=("data", "axes fraction"),
                    ha="left", va="top", fontsize=figstyle.FS_ANNOT,
                    color=figstyle.GUIDE, xytext=(3, 0),
                    textcoords="offset points")

        fx, fy = finite(xs, ys)
        at_prod = dict(zip(xs, ys)).get(dens, float("nan"))
        stats[surf] = {
            "mesh": mesh, "density": dens,
            "sigma_prod": at_prod, "sigma_dense": fy[-1],
            "residual": abs(at_prod - fy[-1]),
            "max_density": fx[-1],
        }
        ax.set_ylabel(SIGMA_MEAN + " (kbar)")
        # Without this the first point sits on the frame and its marker is
        # clipped in half by the axis line.
        ax.margins(y=0.15)
        ax.annotate(figstyle.surface_label(surf), xy=(0.97, 0.90),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=figstyle.FS_ANNOT,
                    color=figstyle.SURFACE_STYLE[surf]["color"])

    axes[-1].set_xlabel(rf"$k$-point density $n_k\,a$ (${AA}$)")
    for letter, ax in zip("abcdef", axes):
        figstyle.panel_label(ax, letter)
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.10)

    missing = [s for s in figstyle.SURFACES if s not in surfaces]
    per_surface = " ".join(
        f"({figstyle.SURFACE_STYLE[s]['label'].strip('()')}) The production "
        f"{stats[s]['mesh'].replace(' ', 'x')} mesh (dashed, "
        f"{stats[s]['density']:.1f} {AA_TXT}) sits within "
        f"{stats[s]['residual']:.3f} kbar of the densest mesh computed "
        f"({stats[s]['max_density']:.1f} {AA_TXT})."
        for s in surfaces)
    caption = (
        f"k-point convergence of the mean in-plane stress, plotted against "
        f"k-point density $n_k a$ rather than mesh integers because the surface "
        f"cells differ in size, so equal meshes do not mean equal sampling. "
        f"{per_surface}"
        + (f" The {' and '.join(figstyle.SURFACE_STYLE[m]['label'] for m in missing)} "
           f"surface was never swept, so it is not shown and no k-point "
           f"convergence is claimed for it." if missing else "")
    )
    return fig, caption, stats


# ── Figure 3: vacuum ──────────────────────────────────────────────────────────

def figure_vacuum(rows):
    require_columns(rows, "vacuum_angstrom", "tau_mean_n_per_m", "n_C")
    surfaces = surfaces_in(rows, "vacuum")
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL, 2.7))
    ax_r = ax.twinx()

    stats = {}
    for surf in surfaces:
        rs = sweep_rows(rows, "vacuum", surf)
        xs, sig = series(rs, "vacuum_angstrom", "sigma_mean_kbar")
        _, tau = series(rs, "vacuum_angstrom", "tau_mean_n_per_m")

        kw = figstyle.series_kwargs(surf, markersize=figstyle.MS * 0.85)
        ax.plot(xs, sig, linestyle="--", **kw)
        ax_r.plot(xs, tau, linestyle="-", **kw)

        fx, fs = finite(xs, sig)
        _, ft = finite(xs, tau)
        spread = (max(ft) - min(ft)) if ft else float("nan")
        scale = max(abs(v) for v in ft) if ft else 0.0
        stats[surf] = {
            "vac_min": fx[0], "vac_max": fx[-1],
            "sigma_near": fs[0], "sigma_far": fs[-1],
            "sigma_ratio": abs(fs[-1] / fs[0]) if fs[0] else float("nan"),
            "tau": ft[0] if ft else float("nan"),
            "tau_spread": spread,
            "tau_pct": (100.0 * spread / scale) if scale else 0.0,
        }

    ax.set_xlabel(rf"vacuum thickness (${AA}$)")
    ax.set_ylabel(SIGMA_MEAN + " (kbar), dashed")
    ax_r.set_ylabel(TAU_MEAN + r" (N m$^{-1}$), solid")

    # The twin axis needs the same furniture as the primary one, and the primary
    # must stop drawing right-hand ticks or the two sets overlap.
    ax_r.tick_params(direction="in", which="both", right=True)
    ax_r.minorticks_on()
    for spine in ax_r.spines.values():
        spine.set_linewidth(figstyle.AXIS_LW)
        spine.set_color("black")
    ax.tick_params(right=False, which="both")

    handles = [plt.Line2D([], [], linestyle="none",
                          **figstyle.series_kwargs(s),
                          label=figstyle.surface_label(s)) for s in surfaces]
    handles += [
        plt.Line2D([], [], color="black", linestyle="--",
                   linewidth=figstyle.LW, label=SIGMA_MEAN),
        plt.Line2D([], [], color="black", linestyle="-",
                   linewidth=figstyle.LW, label=TAU_MEAN),
    ]
    figstyle.legend(ax, handles=handles, loc="center left", ncol=2,
                    bbox_to_anchor=(0.0, 0.55))

    worst_abs = max(s["tau_spread"] for s in stats.values())
    biggest = min(stats.values(), key=lambda s: s["sigma_ratio"])
    n_layers = declared_layers(sweep_rows(rows, "vacuum"))
    layer_txt = f"{n_layers}-layer " if n_layers else ""
    caption = (
        f"Vacuum convergence of the symmetric H-terminated {layer_txt}slabs, and "
        f"the reason surface stress is reported as tau rather than sigma. The raw "
        f"QE stress (dashed, left axis) is averaged over the whole supercell "
        f"including the vacuum, so it falls by a factor of "
        f"{1 / biggest['sigma_ratio']:.1f} as the vacuum grows from "
        f"{biggest['vac_min']:.0f} to {biggest['vac_max']:.0f} {AA_TXT} while "
        f"nothing physical changes. The vacuum-corrected tau = sigma*Lz/2 (solid, "
        f"right axis) moves by at most {worst_abs:.4f} N/m across that same range, "
        f"non-monotonically, which is numerical noise rather than drift. tau is "
        f"therefore the transferable quantity and {biggest['vac_min']:.0f} "
        f"{AA_TXT} of vacuum already suffices for it; the apparent sigma "
        f"convergence problem is an artefact of dividing by an arbitrary cell "
        f"height, not an unconverged calculation."
    )
    return fig, caption, stats


# ── Figure 4: thickness ───────────────────────────────────────────────────────

LEGACY_THICKNESS_DIR = Path("results/thickness_stress")
LEGACY_RUN_RE = re.compile(r"^C\d{3}_\d+L_stress_scf$")


def _fit_ladder(runs_dir, prefix, a0, exclude_layers):
    """Load and fit one thickness ladder with fit_tau_infinity's own code."""
    import fit_tau_infinity as fti

    omega_c = fti.bulk_volume_per_carbon(a0)
    points = fti.discover_points(runs_dir, prefix, omega_c)
    fits = {s: fti.fit_surface(s, pts, exclude_layers)
            for s, pts in points.items()}
    return {"a0": a0, "omega_c": omega_c, "points": points, "fits": fits}


def _fit_legacy_ladder(a0, exclude_layers):
    """
    Fit the superseded-a0 ladder in results/thickness_stress/.

    Those folders predate the `<prefix>~` naming that fit_tau_infinity's
    discovery regex requires, so they are exposed to it through a throwaway
    directory of symlinks that adds the prefix and changes nothing else. The
    alternative -- a second loader here -- would be a second implementation of
    the ladder invariants (area constant, atoms per layer constant, name matches
    geometry), which is exactly what must not be duplicated. Nothing is written
    into results/; the symlinks live in a temporary directory and point at the
    original read-only run folders.
    """
    import tempfile

    if not LEGACY_THICKNESS_DIR.is_dir():
        return None
    sources = sorted(d for d in LEGACY_THICKNESS_DIR.iterdir()
                     if d.is_dir() and LEGACY_RUN_RE.match(d.name)
                     and (d / "pw.out").exists())
    if not sources:
        return None

    with tempfile.TemporaryDirectory(prefix="legacy_ladder_") as tmp:
        staged = Path(tmp)
        for src in sources:
            (staged / f"legacy~{src.name}").symlink_to(src.resolve())
        return _fit_ladder(staged, "legacy", a0, exclude_layers)


def _assert_matches_published(fits):
    """
    The corrected-a0 fit computed here must equal the published one.

    This figure and results/production/tau_infinity.csv are the same fit run
    twice, once by fit_tau_infinity.py's CLI and once through its API from here.
    If they ever disagree the figure is wrong in a way no reader could see, so
    the disagreement is made fatal instead.
    """
    if not TAU_INF_CSV.exists():
        return []
    published = {}
    with TAU_INF_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            published[row["surface"]] = row

    checked = []
    for surface, fit in sorted(fits.items()):
        row = published.get(surface)
        if row is None:
            continue
        for label, got, want in (
            ("tau_xx", fit.fit_xx.tau_inf_n_per_m,
             float(row["tau_xx_inf_n_per_m"])),
            ("tau_yy", fit.fit_yy.tau_inf_n_per_m,
             float(row["tau_yy_inf_n_per_m"])),
            ("sigma_res_mean", fit.sigma_res_mean_kbar,
             float(row["sigma_res_mean_kbar"])),
        ):
            if abs(got - want) > 1e-6:
                raise SystemExit(
                    f"Figure 4 disagrees with {TAU_INF_CSV} for {surface} "
                    f"{label}: {got!r} here vs {want!r} published. Refusing to "
                    f"plot two different values of the same fitted quantity.")
        checked.append(surface)
    return checked


def thickness_convention_shift(ladder, exclude_layers):
    """
    How much tau_inf moves if t is the carbon z-extent instead of N_C*Omega/A.

    This is the size of the convention the atom-counted thickness removes, and
    it is measured rather than asserted: the same points are refitted through
    fit_tau_infinity.fit_surface with only the thickness field swapped, so the
    two numbers differ in the thickness definition and nothing else.

    Returns the largest |delta tau_inf| / |tau_inf| over the surfaces, in
    percent, or None if it cannot be computed.
    """
    import dataclasses

    import fit_tau_infinity as fti

    shifts = []
    for surface, pts in ladder["points"].items():
        geometric = [
            dataclasses.replace(
                p, thickness_angstrom=p.thickness_geometric_angstrom)
            for p in pts
        ]
        if any(p.thickness_geometric_angstrom is None for p in pts):
            continue
        try:
            alt = fti.fit_surface(surface, geometric, exclude_layers)
        except fti.TauFitError:
            continue
        ref = ladder["fits"][surface].tau_mean_n_per_m
        if ref:
            shifts.append(abs(alt.tau_mean_n_per_m - ref) / abs(ref) * 100.0)
    return max(shifts) if shifts else None


def load_thickness_ladders(exclude_layers=(6,)):
    """
    Both thickness ladders, keyed 'corrected' and 'legacy'.

    The corrected ladder is the production one; the legacy ladder is the same
    slabs rebuilt at the superseded lattice constant, kept only as the control
    that shows what the a0 correction removed. Each ladder uses its own a0 for
    Omega_bulk, since t is that ladder's own bulk-equivalent thickness.
    """
    import fit_tau_infinity as fti

    cfg = json.loads(Path(fti.DEFAULT_REFERENCE).read_text())
    bulk = cfg["bulk_reference"]
    a0_new = float(bulk["a0_fit_angstrom"])
    superseded = bulk.get("superseded") or {}
    a0_old = superseded.get("a0_fit_angstrom")

    corrected = _fit_ladder(fti.DEFAULT_RUNS_DIR, fti.DEFAULT_PREFIX,
                            a0_new, exclude_layers)
    checked = _assert_matches_published(corrected["fits"])
    if checked:
        print(f"  Figure 4: fit agrees with {TAU_INF_CSV} for "
              f"{', '.join(checked)} (tau_xx, tau_yy, sigma_res).")

    ladders = {"corrected": corrected}
    legacy = _fit_legacy_ladder(float(a0_old), exclude_layers) if a0_old else None
    if legacy:
        ladders["legacy"] = legacy
    else:
        print(f"  Figure 4: no superseded-a0 ladder under "
              f"{LEGACY_THICKNESS_DIR}; plotting the corrected ladder alone.")
    return ladders


def _series_xy(points):
    """(t, sigma_mean*Lz) for a ladder, sorted by thickness."""
    pts = sorted(points, key=lambda p: p.thickness_angstrom)
    return ([p.thickness_angstrom for p in pts],
            [0.5 * (p.sigma_xx_kbar + p.sigma_yy_kbar) * p.Lz_angstrom
             for p in pts])


def figure_thickness(rows=None, exclude_layers=(6,)):
    """
    The thickness ladder at both lattice constants, one panel per surface.

    Two things are separated here and they must not be conflated. The intercept
    of sigma*Lz against t is 2*tau_inf, the thickness-independent surface stress.
    The slope is sigma_res, a residual stress carried by the slab interior. A
    slab built at a lattice constant that is not the functional's own equilibrium
    a0 carries a spurious interior stress, and that shows up entirely in the
    slope.

    Overlaying the superseded a0 ladder on the corrected one is the whole point:
    the slope collapses while the intercept barely moves, which is what
    distinguishes "the correction removed an artefact" from "the correction
    changed the physics". Either ladder alone shows only half of that.

    x is the bulk-equivalent thickness t = N_C * Omega_bulk / A rather than the
    carbon z-extent. The z-extent depends on where one declares the slab to end
    -- a convention worth ~6% in tau_inf -- while atom-counting does not.

    Neither ladder is fitted here. Both go through fit_tau_infinity.fit_surface,
    so this figure and results/production/tau_infinity.csv cannot disagree; the
    corrected-a0 numbers are asserted against that CSV before anything is drawn.

    `rows` is unused and accepted only so the builder signature matches the other
    figures; the ladders come from their run directories, not the sweep CSV.
    """
    import fit_tau_infinity as fti

    ladders = load_thickness_ladders(exclude_layers)
    corrected = ladders["corrected"]
    surfaces = [s for s in figstyle.SURFACES if s in corrected["fits"]]
    if not surfaces:
        raise SystemExit("no thickness ladders found; cannot build Figure 4")

    fig, axes = plt.subplots(
        len(surfaces), 1, figsize=(figstyle.SINGLE_COL, 5.0), sharex=True)
    axes = list(axes) if len(surfaces) > 1 else [axes]

    def mean_line(fit):
        """Intercept and slope of the mean in-plane component, in kbar*A."""
        tau_mean = fit.tau_mean_n_per_m
        return (2.0 * tau_mean / fti.KBAR_ANGSTROM_TO_N_PER_M,
                fit.sigma_res_mean_kbar)

    stats = {}
    for ax, surf in zip(axes, surfaces):
        entry = {}
        for key, style in (("legacy", dict(filled=False, linestyle="--")),
                           ("corrected", dict(filled=True, linestyle="-"))):
            lad = ladders.get(key)
            if not lad or surf not in lad["fits"]:
                continue
            pts = sorted(lad["points"][surf], key=lambda p: p.thickness_angstrom)
            fit = lad["fits"][surf]

            xs, ys = _series_xy(pts)
            ax.plot(xs, ys, linestyle="none",
                    **figstyle.series_kwargs(surf, filled=style["filled"]))

            # Excluded points stay plotted but are struck through, so the
            # exclusion is visible rather than inferred from a caption.
            for p, x, y in zip(pts, xs, ys):
                if p.layers in set(exclude_layers):
                    ax.plot([x], [y], marker="x", linestyle="none",
                            color=figstyle.GUIDE, markersize=figstyle.MS * 1.1,
                            markeredgewidth=0.8, zorder=6)

            intercept, slope = mean_line(fit)
            xf = linspace(0.0, max(xs) * 1.05)
            ax.plot(xf, [intercept + slope * x for x in xf],
                    color=figstyle.SURFACE_STYLE[surf]["color"],
                    linestyle=style["linestyle"],
                    linewidth=figstyle.LW * 0.8, zorder=1)

            entry[key] = {
                "tau_mean": fit.tau_mean_n_per_m,
                "sigma_res": fit.sigma_res_mean_kbar,
                "intercept": intercept,
                "a0": lad["a0"],
                "layers_used": fit.layers_used,
                "layers_excluded": fit.layers_excluded,
            }
        stats[surf] = entry
        ax.set_ylabel(rf"$\sigma_{{\mathrm{{mean}}}}L_z$ (kbar ${AA}$)")
        ax.margins(y=0.28)

        # Which bottom corner is free differs per panel and is not guessable:
        # (100)'s excluded 6L point sits low on the left, while (110) and (111)
        # fall away to the right. Pick the side whose lowest plotted point is
        # highest, i.e. the corner with the most clearance, from the data.
        drawn = [(x, y) for key in entry
                 for x, y in zip(*_series_xy(ladders[key]["points"][surf]))]
        mid = 0.5 * max(x for x, _ in drawn)
        left_floor = min((y for x, y in drawn if x <= mid), default=float("inf"))
        right_floor = min((y for x, y in drawn if x > mid), default=float("inf"))
        on_left = left_floor > right_floor
        xy = (0.03, 0.05) if on_left else (0.97, 0.05)
        ha = "left" if on_left else "right"

        lines = [figstyle.surface_label(surf)]
        if "corrected" in entry:
            lines.append(rf"$\tau_\infty = {entry['corrected']['tau_mean']:+.2f}$"
                         rf" N m$^{{-1}}$")
            lines.append(rf"$\sigma_{{\mathrm{{res}}}} = "
                         rf"{entry['corrected']['sigma_res']:+.2f}$ kbar")
        if "legacy" in entry:
            lines.append(rf"superseded: {entry['legacy']['tau_mean']:+.2f}, "
                         rf"{entry['legacy']['sigma_res']:+.2f}")
        ax.annotate("\n".join(lines), xy=xy,
                    xycoords="axes fraction", ha=ha, va="bottom",
                    fontsize=figstyle.FS_ANNOT, linespacing=1.5,
                    color=figstyle.SURFACE_STYLE[surf]["color"])

    # One x-limit for the shared axis, set after every series is drawn and from
    # the widest ladder. Calling set_xlim per panel froze the shared axis on the
    # first panel's range and silently cropped the (110) 16L point off the right
    # edge -- a point that was still inside the fit, so the line appeared to
    # extrapolate past data that was simply not shown.
    t_max = max(p.thickness_angstrom
                for lad in ladders.values()
                for pts in lad["points"].values()
                for p in pts)
    axes[0].set_xlim(0.0, t_max * 1.06)

    axes[-1].set_xlabel(
        rf"bulk-equivalent thickness $t = N_\mathrm{{C}}\Omega_\mathrm{{bulk}}/A$"
        rf" (${AA}$)")

    # Labels stay short and the two lattice constants go in the caption: spelled
    # out to six decimals they make a one-row legend far wider than a 3.46 in
    # column, and a wrapped legend would have to sit inside a panel.
    handles = [
        plt.Line2D([], [], color="black", linestyle="-",
                   linewidth=figstyle.LW, marker="o", markerfacecolor="black",
                   markeredgecolor="black", markeredgewidth=figstyle.MEW,
                   markersize=figstyle.MS, label=r"corrected $a_0$"),
    ]
    if ladders.get("legacy"):
        handles.append(
            plt.Line2D([], [], color="black", linestyle="--",
                       linewidth=figstyle.LW, marker="o",
                       markerfacecolor="white", markeredgecolor="black",
                       markeredgewidth=figstyle.MEW, markersize=figstyle.MS,
                       label=r"superseded $a_0$"))
    handles.append(
        plt.Line2D([], [], color=figstyle.GUIDE, linestyle="none", marker="x",
                   markersize=figstyle.MS * 1.1, markeredgewidth=0.8,
                   label="excluded"))
    # Above the top panel, horizontal: both bottom corners of every panel are
    # now spoken for by the per-panel annotations, and the data fills the top.
    figstyle.legend(axes[0], handles=handles, ncol=len(handles),
                    loc="lower left", bbox_to_anchor=(0.0, 1.01),
                    columnspacing=1.0, handletextpad=0.4)

    for letter, ax in zip("abcdef", axes):
        figstyle.panel_label(ax, letter)
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.10)

    excl = sorted({n for e in stats.values() for k in e
                   for n in e[k]["layers_excluded"]})
    excl_txt = ", ".join(f"{n}L" for n in excl) if excl else "none"
    res_new = [e["corrected"]["sigma_res"] for e in stats.values() if "corrected" in e]
    res_old = [e["legacy"]["sigma_res"] for e in stats.values() if "legacy" in e]
    tau_shift = [
        abs(e["corrected"]["tau_mean"] - e["legacy"]["tau_mean"])
        / abs(e["legacy"]["tau_mean"]) * 100.0
        for e in stats.values()
        if "corrected" in e and "legacy" in e and e["legacy"]["tau_mean"]
    ]

    caption = (
        f"Slab-thickness ladder at both lattice constants. Plotting "
        f"sigma_mean*Lz against the bulk-equivalent thickness "
        f"t = N_C*Omega_bulk/A splits the total in-plane force per unit length "
        f"into a thickness-independent intercept, twice the surface stress "
        f"tau_inf, and a slope sigma_res carried by the slab interior. "
    )
    shift = thickness_convention_shift(corrected, exclude_layers)
    shift_old = (thickness_convention_shift(ladders["legacy"], exclude_layers)
                 if ladders.get("legacy") else None)
    if shift is not None:
        caption += (
            f"t is atom-counted rather than measured as a carbon z-extent "
            f"because the z-extent depends on where one declares the slab to "
            f"end: refitting these same points against the z-extent moves "
            f"tau_inf by up to {shift:.1f}{PCT} here"
        )
        # The convention's cost is the interior stress times the offset between
        # the two definitions, so it shrinks with sigma_res rather than being a
        # fixed property of the geometry. Quoting only the corrected ladder
        # would understate why the atom-counted definition was adopted.
        caption += (
            f", and up to {shift_old:.1f}{PCT} on the superseded ladder, where "
            f"the larger interior stress amplifies it. "
            if shift_old is not None else ". ")
    if res_old and tau_shift:
        caption += (
            f"Rebuilding the ladder at the corrected a0 = "
            f"{corrected['a0']:.6f} {AA_TXT} (filled, solid) collapses the slope "
            f"from {min(res_old):+.2f}..{max(res_old):+.2f} kbar at the "
            f"superseded a0 = {ladders['legacy']['a0']:.6f} {AA_TXT} (open, "
            f"dashed) to {min(res_new):+.2f}..{max(res_new):+.2f} kbar, while "
            f"tau_inf moves by at most {max(tau_shift):.1f}{PCT}. The interior "
            f"stress was therefore an artefact of the reference lattice "
            f"constant and not a property of any surface, and removing it left "
            f"the surface stress intact -- which is what licenses the corrected "
            f"ladder as the production result. "
        )
    caption += (
        f"The {excl_txt} point (struck through) lies outside the asymptotic "
        f"regime and is excluded from both fits but still plotted. Each surface "
        f"keeps its own y-scale because the three sit ~1000 kbar {AA_TXT} apart. "
        f"Positive sigma means the cell is compressed; tau inherits that sign, "
        f"so positive tau_inf is a compressive surface stress (minus the "
        f"continuum f). Fits are read from fit_tau_infinity.py, not "
        f"recomputed here."
    )
    return fig, caption, stats


# ── Figure 5: bulk equation of state ──────────────────────────────────────────

def figure_bulk():
    if not BULK_CSV.exists():
        raise SystemExit(f"missing input: {BULK_CSV}")

    import fit_bulk_reference as fbr

    points = fbr.load_data(BULK_CSV, strain_type=None)
    a_ref, _ = fbr.assign_epsilon_from_volume(points)
    eps = [p["epsilon"] for p in points]
    P = [p["pressure_kbar"] for p in points]

    lin = fbr.pressure_poly_fit(eps, P, a_ref, degree=1)
    quad = fbr.pressure_poly_fit(eps, P, a_ref, degree=2)
    lin_poly = ScaledPoly(eps, P, 1)
    quad_poly = ScaledPoly(eps, P, 2)

    bm3 = json.loads(BULK_FIT_JSON.read_text()) if BULK_FIT_JSON.exists() else {}
    bm3_a0 = bm3.get("bm3_a0_angstrom")
    bm3_B = bm3.get("bm3_B0_gpa")

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(figstyle.SINGLE_COL, 3.9), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]})

    xf = linspace(min(eps), max(eps))
    ax_a.plot([e * 100 for e in xf], [lin_poly.value(e) for e in xf],
              color=figstyle.GUIDE, linestyle="--", linewidth=figstyle.LW,
              label="linear")
    ax_a.plot([e * 100 for e in xf], [quad_poly.value(e) for e in xf],
              color="#0072B2", linestyle="-", linewidth=figstyle.LW,
              label="quadratic")
    ax_a.plot([e * 100 for e in eps], P, linestyle="none", marker="o",
              markerfacecolor="white", markeredgecolor="black",
              markeredgewidth=figstyle.MEW, markersize=figstyle.MS,
              label="DFT", zorder=5)
    ax_a.axhline(0.0, color="black", linewidth=0.5, linestyle=":", zorder=0)

    for e0, colour in ((lin["epsilon0"], figstyle.GUIDE),
                       (quad["epsilon0"], "#0072B2")):
        ax_a.plot([e0 * 100], [0.0], marker="v", markersize=3.2, color=colour,
                  markeredgecolor="black", markeredgewidth=0.4, zorder=6,
                  linestyle="none")

    ax_a.set_ylabel(r"$P$ (kbar)")
    figstyle.legend(ax_a, loc="upper right")
    ax_a.annotate(
        rf"$\varepsilon_0$ = {lin['epsilon0']*100:.4f}{PCT} vs "
        rf"{quad['epsilon0']*100:.4f}{PCT}",
        xy=(0.03, 0.06), xycoords="axes fraction", ha="left", va="bottom",
        fontsize=figstyle.FS_ANNOT)

    r_lin = [p - lin_poly.value(e) for e, p in zip(eps, P)]
    r_quad = [p - quad_poly.value(e) for e, p in zip(eps, P)]
    ax_b.axhline(0.0, color="black", linewidth=0.5, linestyle=":", zorder=0)
    ax_b.plot([e * 100 for e in eps], r_lin, marker="s", linestyle="--",
              color=figstyle.GUIDE, markerfacecolor=figstyle.GUIDE,
              markeredgecolor="black", markeredgewidth=figstyle.MEW,
              markersize=figstyle.MS, linewidth=figstyle.LW,
              label=f"linear ({lin['rms_residual_kbar']:.2f})")
    ax_b.plot([e * 100 for e in eps], r_quad, marker="o", linestyle="-",
              color="#0072B2", markerfacecolor="#0072B2",
              markeredgecolor="black", markeredgewidth=figstyle.MEW,
              markersize=figstyle.MS, linewidth=figstyle.LW,
              label=f"quadratic ({quad['rms_residual_kbar']:.3f})")
    ax_b.set_xlabel(rf"hydrostatic strain $\varepsilon$ ({PCT})")
    ax_b.set_ylabel("residual (kbar)")
    # The linear residual is a downward parabola filling the lower half, so the
    # legend goes top-centre where the only thing it can overlap is empty space
    # between the two rising arms.
    ax_b.margins(y=0.30)
    figstyle.legend(ax_b, loc="upper center", ncol=2)

    figstyle.panel_label(ax_a, "a")
    figstyle.panel_label(ax_b, "b")
    fig.align_ylabels((ax_a, ax_b))
    fig.subplots_adjust(hspace=0.10)

    ratio = lin["rms_residual_kbar"] / quad["rms_residual_kbar"]
    if bm3_a0 and bm3_B:
        production = (
            f" The production reference is neither of the fits drawn here but a "
            f"3rd-order Birch-Murnaghan fit to E(V) over the same series, giving "
            f"a0 = {bm3_a0:.6f} {AA_TXT} and B = {bm3_B:.2f} GPa; the quadratic "
            f"P(eps) fit shown in (a) is its cross-check, not its source."
        )
    else:
        production = ""
    caption = (
        f"Bulk diamond equation of state at 90/720 Ry, and the bias in a linear "
        f"P(eps) fit. (a) P(eps) is visibly curved over +/-1{PCT} strain, so the "
        f"straight-line fit misses the zero-pressure crossing: the two markers "
        f"on P = 0 differ by "
        f"{abs(lin['epsilon0']-quad['epsilon0'])*100:.4f}{PCT} in strain, which "
        f"is {abs(lin['a0_angstrom']-quad['a0_angstrom']):.4f} {AA_TXT} in a0. "
        f"(b) The residuals make the bias unmistakable: the linear fit leaves "
        f"{lin['rms_residual_kbar']:.2f} kbar rms with obvious systematic "
        f"curvature, {ratio:.0f} times the {quad['rms_residual_kbar']:.3f} kbar "
        f"of the quadratic.{production}"
    )
    return fig, caption, {"linear": lin, "quadratic": quad,
                          "bm3_a0": bm3_a0, "bm3_B_gpa": bm3_B}


# ── Caption plain-texting ─────────────────────────────────────────────────────

def detex(s):
    """Strip math markup so CAPTIONS.md reads as plain prose."""
    for a, b in ((r"$\sigma_{\mathrm{mean}}$", "sigma_mean"),
                 (r"$\tau_{\mathrm{mean}}$", "tau_mean"),
                 (r"$n_k a$", "n_k*a"),
                 (r"$\varepsilon_0$", "eps0"),
                 (r"\%", "%"),
                 (r"\AA", "A")):
        s = s.replace(a, b)
    return s


# ── Driver ────────────────────────────────────────────────────────────────────

TITLES = {
    "fig1_cutoff": "Figure 1 - Plane-wave cutoff convergence",
    "fig2_kpoints": "Figure 2 - k-point convergence",
    "fig3_vacuum": "Figure 3 - Vacuum convergence: sigma versus tau",
    "fig4_thickness": "Figure 4 - Slab-thickness ladder",
    "fig5_bulk_eos": "Figure 5 - Bulk equation of state",
}


def previous_caption(outdir, stem):
    """
    The caption for `stem` from an existing CAPTIONS.md, or None.

    Used only for a figure this run deliberately skipped. Its artwork stays on
    disk untouched, so its caption has to survive the rewrite verbatim instead
    of silently disappearing and leaving an unexplained pair of files behind.
    """
    path = Path(outdir) / "CAPTIONS.md"
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    heading = f"## {TITLES[stem]}"
    if heading not in lines:
        return None
    body = []
    for ln in lines[lines.index(heading) + 1:]:
        if ln.startswith("## ") or ln.startswith("Files:"):
            break
        body.append(ln)
    return "\n".join(body).strip() or None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--csv", default=str(CONVERGENCE_CSV))
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    print("Typography: "
          + ("LaTeX (text.usetex)" if USE_TEX
             else "mathtext fontset=cm (LaTeX unusable, fell back)"))

    rows, excluded = load_convergence(Path(args.csv))
    if excluded:
        print(f"\nEXCLUDED {len(excluded)} run(s) that did not complete:")
        for r in excluded:
            print(f"  {r['run']}: JOB DONE={r.get('job_done')} "
                  f"scf_converged={r.get('scf_converged')}")
    else:
        print("All runs in the summary reached JOB DONE and SCF convergence.")

    extra, skipped = load_extra_kpoint_runs()
    if skipped:
        print(f"\nEXCLUDED {len(skipped)} run(s) from "
              f"{PRODUCTION_DIR}/{KPT_EXTRA_GLOB}:")
        for name, why in skipped:
            print(f"  {name}: {why}")
    if extra:
        merged = ", ".join(sorted({r["surface"] for r in extra}))
        print(f"\nMerged {len(extra)} k-point run(s) for {merged} from "
              f"{PRODUCTION_DIR}/{KPT_EXTRA_GLOB} "
              f"(not present in {Path(args.csv).name}).")
        rows = rows + extra
    print(f"Plotting from {len(rows)} runs.\n")

    builders = [
        ("fig1_cutoff", lambda: figure_cutoff(rows)),
        ("fig2_kpoints", lambda: figure_kpoints(rows)),
        ("fig3_vacuum", lambda: figure_vacuum(rows)),
        ("fig4_thickness", lambda: figure_thickness(rows)),
        ("fig5_bulk_eos", figure_bulk),
    ]

    # Figure 4 quotes tau_infinity, which fit_tau_infinity.py owns. Without its
    # CSV the figure is skipped, not refitted here: a second implementation of
    # the same extrapolation could disagree with the first, and the disagreement
    # would be invisible in the output. See the note on TAU_INF_CSV.
    carried = {}
    if not TAU_INF_CSV.exists():
        stem = "fig4_thickness"
        builders = [b for b in builders if b[0] != stem]
        old = previous_caption(outdir, stem)
        if old:
            carried[stem] = old
        print(f"SKIPPED {stem}: {TAU_INF_CSV} is absent, so tau_infinity cannot "
              f"be read and will not be refitted here.")
        print(f"  {stem}.pdf/.png left on disk unchanged; its caption is "
              f"{'carried forward verbatim' if old else 'MISSING and cannot be carried forward'}.")
        print()

    # Pre-flight before anything is deleted. The deletion below is destructive
    # and the build is not atomic, so validating input afterwards would mean a
    # stale CSV wipes five figures and then replaces two -- worse than the state
    # it started from. Everything the figures read is checked up front instead.
    require_columns(rows,
                    "ecutwfc_ry", "sigma_mean_kbar", "anisotropy_kbar",
                    "k_density_a1_angstrom", "kmesh", "source_slab",
                    "vacuum_angstrom", "tau_mean_n_per_m", "n_C")

    # Delete before regenerating, so a figure that changes shape cannot leave a
    # stale variant sitting beside the new one. Only the stems this run actually
    # rebuilds are touched.
    for stem, _ in builders:
        for suffix in (".pdf", ".png", "_annotated.pdf"):
            path = outdir / f"{stem}{suffix}"
            if path.exists():
                path.unlink()
    (outdir / "CAPTIONS.md").unlink(missing_ok=True)

    captions = {}
    for stem, build in builders:
        fig, caption, _ = build()
        paths = figstyle.save(fig, outdir, stem, caption=caption)
        plt.close(fig)
        captions[stem] = caption
        print(f"  {stem}: " + ", ".join(p.name for p in paths))

    lines = [
        "# Figure captions",
        "",
        "Generated by `plot_convergence.py`. Every number is derived from the",
        "source CSVs and run directories rather than transcribed.",
        "",
        "Sign convention (CLAUDE.md section 2): positive sigma means the cell is",
        "COMPRESSED and wants to expand; negative sigma means it is in TENSION.",
        "All slabs are symmetric H/H terminated, 90/720 Ry unless stated.",
        "",
    ]
    for stem in TITLES:
        if stem in captions:
            body, note = detex(captions[stem]), ""
        elif stem in carried:
            body = carried[stem]
            note = ("\n\n_Not regenerated in this run: `tau_infinity.csv` was "
                    "absent. Caption and artwork are carried forward unchanged._")
        else:
            continue
        lines += [f"## {TITLES[stem]}", "", body + note, "",
                  f"Files: `{stem}.pdf`, `{stem}.png` (600 dpi), "
                  f"`{stem}_annotated.pdf`", ""]
    (outdir / "CAPTIONS.md").write_text("\n".join(lines))
    print("  CAPTIONS.md")
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()
