#!/usr/bin/env python3
"""
plot_h_results.py - the H-terminated result figures, F1-F5.

WHAT THIS SET ARGUES
--------------------
The chain, in order, and each figure is one link:

    tau per facet (F1)
      -> the habit depends on mu_H (F2, F3a)
      -> the interior stress changes sign (F3b)
      -> observables: lattice strain (F4), D and E (F5)

This replaces the earlier N-series, which was built when the argument was a
different one. Two of those figures are retired here rather than redrawn: N1
(tau per facet) is absorbed into F1(b), and N2 (gamma against |tau_aniso|) is
deleted because its premise is now known to be false. N2 asserted that the (100)
in-plane anisotropy drives the transverse NV splitting E. It does not: E is
produced by the edge-concentrated deviatoric field and survives with perfectly
isotropic facet surface stress -- zeroing the (100) anisotropy while preserving
its mean leaves 54% of E on a {100} cube (nv_local_strain.py). A figure that
asserts something the later work contradicts is worse than no figure.

SIGN CONVENTION (CLAUDE.md section 2)
-------------------------------------
Positive sigma means the cell is COMPRESSED. tau = sigma*Lz/2 inherits that, so
positive tau is COMPRESSIVE surface stress and equals minus the continuum
surface stress f. Every caption below uses that reading.

DATA
----
Every value is read from a committed file; nothing is hardcoded, and a missing
input produces a gap and a message on stdout rather than a filled value:

    results/production/tau_infinity.csv               F1b, and its axis labels
    config/surface_energies_h.json                    F2
    results/production/particle_strain_mu_h_scan.csv  F2, F3
    results/production/particle_strain_sign_change_tp.csv   F3 temperature axis
    results/production/particle_strain_size_sweep_*.csv     F4
    results/production/particle_strain_nv_*.csv       F5a
    results/production/nv_local_strain_depth.csv      F5b

EPISTEMIC LEVEL: every caption carries its own. Nothing here is above L2, and
anything touching a spin-strain coupling set or the literature elastic tensor
says so at the point of use.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import figstyle

USE_TEX = figstyle.apply()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AA = figstyle.angstrom()
# Captions are plain prose, not math. figstyle.angstrom() returns markup for
# whichever text path is active ("\AA" or "\mathrm{\AA}"), and stripping the
# backslashes out of it to reuse it in prose produced the literal string
# "mathrm{\AA}" in a committed caption. Captions use this instead.
AA_TXT = "A"
PCT = r"\%" if USE_TEX else "%"

REPO = Path(__file__).resolve().parent
PROD = REPO / "results" / "production"
DEFAULT_OUTDIR = REPO / "results" / "figures" / "h-results"

TAU_CSV = PROD / "tau_infinity.csv"
GAMMA_JSON = REPO / "config" / "surface_energies_h.json"
MU_SCAN = PROD / "particle_strain_mu_h_scan.csv"
SIGN_TP = PROD / "particle_strain_sign_change_tp.csv"
NV_LOCAL = PROD / "nv_local_strain_depth.csv"

# The habit each single-family shape corresponds to, and its surface key.
HABIT = {"octahedron": "111", "cube": "100", "rhombic-dodecahedron": "110",
         "mixture": None}
FAMILY_STYLE = {"100": "C100", "110": "C110", "111": "C111"}

# Draw and legend order, fixed for the whole set: (100), (110), (111), then the
# mixture. Iterating HABIT instead would order legends by habit name, which
# differs figure to figure from the (100)/(110)/(111) order used everywhere
# else and makes the panels harder to read against each other.
SHAPE_ORDER = ("cube", "rhombic-dodecahedron", "octahedron", "mixture")

# Residual-vacuum H2 partial pressure the temperature axis is quoted at. Stated
# rather than assumed silently: the mu_H -> T conversion is pressure dependent.
P_H2_BAR = 1.0e-9

LINEWIDTH_MHZ = 3.0

_missing = []


def note_missing(what):
    _missing.append(what)
    print(f"  GAP: {what}")


def read_csv(path):
    if not Path(path).exists():
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(row, key):
    v = (row.get(key) or "").strip()
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ───────────────────────────────────────────────── F1: surfaces and tau

def figure_f1(tau_rows):
    """
    (a) atomic renderings -- slot, see caption; (b) tau_inf dumbbells.
    """
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(figstyle.SINGLE_COL, 4.2),
        gridspec_kw={"height_ratios": [1.0, 1.15]})

    # (a) reserved slot: no structure renderer is available in this environment
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for s in ax_a.spines.values():
        s.set_linestyle((0, (4, 4)))
    ax_a.text(0.5, 0.5,
              "slot: atomic renderings of the three relaxed\n"
              "H-terminated surfaces (side + top view)\n"
              "requires VESTA/OVITO -- not available here",
              ha="center", va="center", fontsize=figstyle.FS_ANNOT,
              color=figstyle.GUIDE, transform=ax_a.transAxes)

    order = ["100", "110", "111"]
    present = [r for k in order for r in tau_rows if r["surface"][1:] == k]
    ypos = np.arange(len(present))[::-1]

    ax_b.axvline(0.0, color="black", lw=0.8, zorder=1)
    for y, r in zip(ypos, present):
        fam = r["surface"][1:]
        st = figstyle.SURFACE_STYLE[FAMILY_STYLE[fam]]
        xx = fnum(r, "tau_xx_inf_n_per_m")
        yy = fnum(r, "tau_yy_inf_n_per_m")
        mean = fnum(r, "tau_mean_n_per_m")
        ax_b.plot([xx, yy], [y, y], color=st["color"], lw=figstyle.LW * 1.6,
                  solid_capstyle="round", zorder=2)
        ax_b.plot([xx], [y], marker=st["marker"], color=st["color"],
                  markeredgecolor="black", markeredgewidth=figstyle.MEW,
                  markersize=figstyle.MS + 1, zorder=3, linestyle="none")
        ax_b.plot([yy], [y], marker=st["marker"], color="white",
                  markeredgecolor=st["color"], markeredgewidth=figstyle.MEW * 1.6,
                  markersize=figstyle.MS + 1, zorder=3, linestyle="none")
        ax_b.plot([mean], [y], marker="|", color="black",
                  markersize=figstyle.MS + 3, markeredgewidth=1.0, zorder=4,
                  linestyle="none")
        ax_b.annotate(f"{r.get('axis_x','')}", xy=(xx, y), xytext=(0, 6),
                      textcoords="offset points", ha="center",
                      fontsize=figstyle.FS_ANNOT, color=st["color"])
        ax_b.annotate(f"{r.get('axis_y','')}", xy=(yy, y), xytext=(0, -11),
                      textcoords="offset points", ha="center",
                      fontsize=figstyle.FS_ANNOT, color=st["color"])

    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([figstyle.SURFACE_STYLE[FAMILY_STYLE[r["surface"][1:]]]["label"]
                          for r in present])
    # The lower bound leaves a clear strip beneath the (111) row for the key.
    # (111)'s dumbbell collapses to a point, so its two axis labels stack
    # vertically around that row and a legend placed on the data would land on
    # them.
    ax_b.set_ylim(-1.35, len(present) - 0.35)
    ax_b.tick_params(axis="y", which="minor", left=False, right=False)
    ax_b.set_xlabel(r"$\tau_\infty$ (N m$^{-1}$), positive = compressive")

    handles = [
        plt.Line2D([], [], linestyle="none", marker="o", color="black",
                   markeredgecolor="black", label=r"$\tau_{xx}$"),
        plt.Line2D([], [], linestyle="none", marker="o", color="white",
                   markeredgecolor="black", label=r"$\tau_{yy}$"),
        plt.Line2D([], [], linestyle="none", marker="|", color="black",
                   markeredgewidth=1.0, label=r"mean"),
    ]
    # Lower left, not lower centre: centring puts the open tau_yy key directly
    # on the sigma = 0 rule, where the two read as one glyph.
    figstyle.legend(ax_b, handles=handles, loc="lower left", ncol=3,
                    fontsize=figstyle.FS_ANNOT)
    figstyle.panel_label(ax_a, "a")
    figstyle.panel_label(ax_b, "b")
    fig.subplots_adjust(hspace=0.30)

    vals = {r["surface"][1:]: (fnum(r, "tau_xx_inf_n_per_m"),
                               fnum(r, "tau_yy_inf_n_per_m")) for r in present}
    cap = (
        "The three H-terminated surfaces and their thickness-extrapolated "
        "surface stress. (a) is a reserved slot: rendering the relaxed "
        "structures needs VESTA or OVITO, neither of which is available in the "
        "environment that produced this figure, and (100)'s dimer rows cannot "
        "be drawn from the data files alone. (b) Each facet's tau_xx and "
        "tau_yy as the ends of a dumbbell with the mean marked; axis labels "
        "are the crystallographic directions recorded in tau_infinity.csv. "
        "Positive tau is COMPRESSIVE (CLAUDE.md sec 2). The asymmetry is the "
        f"finding: (100) straddles zero ({vals['100'][0]:+.2f} to "
        f"{vals['100'][1]:+.2f} N/m, opposite signs on the two in-plane axes), "
        f"(110) sits entirely positive ({vals['110'][0]:+.2f} to "
        f"{vals['110'][1]:+.2f}), and (111) collapses to a point "
        f"({vals['111'][0]:+.2f}) because C3v symmetry forces its in-plane "
        "anisotropy to vanish identically. Epistemic level L2: converged in "
        "cutoff, k-points, vacuum and thickness."
    )
    return fig, cap


# ────────────────────────────────────── F2: gamma(mu_H) and the window

def figure_f2(gamma_cfg, scan):
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL, 2.7))

    se = gamma_cfg["surface_energies"]
    mu = np.array([fnum(r, "delta_mu_h_ev") for r in scan]) if scan else None
    lo = 0.0
    hi = float(mu.max()) if mu is not None else 2.0
    grid = np.linspace(lo, hi, 200)

    for fam in ("100", "110", "111"):
        e = se[fam]
        g = e["value"] + e["dgamma_dmu_j_m2_per_ev"] * grid
        st = figstyle.SURFACE_STYLE[FAMILY_STYLE[fam]]
        ax.plot(grid, g, color=st["color"], lw=figstyle.LW,
                label=st["label"])
        zc = e.get("zero_crossing_delta_mu_ev")
        if zc is not None and lo <= zc <= hi:
            ax.plot([zc], [0.0], marker=st["marker"], color=st["color"],
                    markeredgecolor="black", markeredgewidth=figstyle.MEW,
                    markersize=figstyle.MS, linestyle="none", zorder=5)

    ax.axhline(0.0, color="black", lw=0.6, ls=":", zorder=1)
    ylo, yhi = ax.get_ylim()
    ax.axhspan(ylo, 0.0, color=figstyle.GUIDE, alpha=0.13, lw=0, zorder=0)
    # Right of the zero crossings, so the text sits in empty shading rather
    # than across the (111) line on its way down through zero.
    ax.annotate("$\\gamma < 0$: Wulff undefined", xy=(0.55, 0.06),
                xycoords="axes fraction", fontsize=figstyle.FS_ANNOT,
                color=figstyle.GUIDE, ha="left", va="bottom")
    ax.set_ylim(ylo, yhi)

    # dehydrogenation ceiling: only if the bare ladder exists
    ceiling = None
    for fam in ("100", "110", "111"):
        c = se[fam].get("dehydrogenation_ceiling_delta_mu_ev")
        if c is not None:
            ceiling = c if ceiling is None else min(ceiling, c)
    if ceiling is not None:
        ax.axvline(ceiling, color="black", lw=1.2)
        ax.axvspan(ceiling, hi, color="black", alpha=0.10, lw=0)
    else:
        note_missing("F2 dehydrogenation ceiling: no bare-facet ladder in "
                     "results/ (the ladder is on the cluster). Drawn without "
                     "it; the H-poor side is unbounded in this figure.")
        # Lower right: all three gamma curves rise to the left of it, so the
        # box lands on empty axes instead of over the (110) line.
        ax.annotate("dehydrogenation ceiling:\nNOT COMPUTED (no bare ladder)",
                    xy=(0.97, 0.18), xycoords="axes fraction", ha="right",
                    va="bottom", fontsize=figstyle.FS_ANNOT, color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="black", lw=0.6))

    ax.set_xlabel(r"$\Delta\mu_{\mathrm{H}}$ below the H-rich limit (eV)")
    ax.set_ylabel(r"$\gamma$ (J m$^{-2}$)")
    figstyle.legend(ax, loc="upper left")

    cap = (
        "Surface energy against hydrogen chemical potential, which sets the "
        "range every later figure is quoted over. gamma rises with "
        "$\\Delta\\mu_H$ at facet-dependent rates, so the stability ORDER "
        "changes and with it the equilibrium habit. The shaded band is where "
        "gamma < 0 and a Wulff construction is undefined -- shown rather than "
        "hidden, because at the H-rich limit all three H-terminated facets sit "
        "there, which is why `particle_strain.py --shape wulff` refuses. "
        + ("The dehydrogenation ceiling is NOT DRAWN: it requires a bare-facet "
           "ladder that is not in results/, so the hydrogen-poor side of this "
           "figure is unbounded and every statement made there is an "
           "extrapolation with no stop. That is the single largest limitation "
           "on this figure and on F3."
           if ceiling is None else
           "The vertical boundary is the dehydrogenation ceiling; beyond it "
           "the H-terminated surface is no longer the stable termination.")
        + " Epistemic level L1: gamma rests on a Boettger mu_C fit whose "
        "residual drift is recorded per facet in the config."
    )
    return fig, cap


# ─────────────────────────────── F3: habit and interior pressure vs mu_H

def mu_to_temperature(scan_mu, tp_rows, p_h2_bar):
    """
    Linear map from delta_mu to T at fixed p_H2, anchored on the committed
    sign-change row for that pressure. Returns (T_uncorrected, T_with_zpe) at
    the crossing, plus a callable, or None if that pressure was not tabulated.
    """
    if not tp_rows:
        return None
    match = [r for r in tp_rows
             if abs(fnum(r, "p_h2_bar") - p_h2_bar) / p_h2_bar < 1e-6]
    if not match:
        return None
    r = match[0]
    return {"delta_mu_ev": fnum(r, "delta_mu_h_ev"),
            "T_k": fnum(r, "temperature_k"),
            "T_zpe_k": fnum(r, "temperature_with_zpe_k"),
            "shift_k": fnum(r, "zpe_shift_k")}


def figure_f3(scan, tp_rows, tau_rows):
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(figstyle.SINGLE_COL, 4.0), sharex=True)

    # Read rather than asserted: the caption below names which facet is
    # tensile, and that is a claim about the data, not a fixed fact.
    tau_mean_100 = next((fnum(r, "tau_mean_n_per_m") for r in (tau_rows or [])
                         if r["surface"] == "C100"), None)

    mu = np.array([fnum(r, "delta_mu_h_ev") for r in scan])
    for fam in ("100", "110", "111"):
        y = np.array([fnum(r, f"area_fraction_{fam}") for r in scan])
        st = figstyle.SURFACE_STYLE[FAMILY_STYLE[fam]]
        ax_a.plot(mu, y, color=st["color"], lw=figstyle.LW, label=st["label"])
    ax_a.set_ylabel("Wulff area fraction")
    ax_a.set_ylim(-0.03, 1.03)
    figstyle.legend(ax_a, loc="center left")

    P = np.array([fnum(r, "pressure_gpa") for r in scan])
    ax_b.axhline(0.0, color="black", lw=0.6, ls=":")
    ax_b.plot(mu, P, color="black", lw=figstyle.LW)
    ax_b.fill_between(mu, 0, P, where=P > 0, color=figstyle.GUIDE, alpha=0.18, lw=0)
    ax_b.set_ylabel(r"interior $P$ (GPa)")
    ax_b.set_xlabel(r"$\Delta\mu_{\mathrm{H}}$ below the H-rich limit (eV)")

    cross = None
    s = np.sign(P)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx):
        i = idx[0]
        cross = mu[i] + (mu[i + 1] - mu[i]) * (0 - P[i]) / (P[i + 1] - P[i])
        for ax in (ax_a, ax_b):
            ax.axvline(cross, color="black", lw=0.8, ls="--")
        ax_b.annotate(f"sign change\n$\\Delta\\mu_H$ = {cross:.3f} eV",
                      xy=(cross, 0), xytext=(6, 12), textcoords="offset points",
                      fontsize=figstyle.FS_ANNOT, ha="left")
    else:
        note_missing("F3: interior pressure does not change sign in the "
                     "committed scan range")

    tmap = mu_to_temperature(mu, tp_rows, P_H2_BAR)
    if tmap is None:
        note_missing(f"F3 temperature axis: p_H2 = {P_H2_BAR:g} bar not in "
                     "particle_strain_sign_change_tp.csv")
    else:
        ax_t = ax_a.twiny()
        ax_t.set_xlim(ax_a.get_xlim())
        ax_t.set_xticks([tmap["delta_mu_ev"]])
        ax_t.set_xticklabels([f"{tmap['T_k']:.0f} K"])
        ax_t.tick_params(direction="in", which="both", top=True)
        ax_t.set_xlabel(f"annealing $T$ at $p_{{\\mathrm{{H_2}}}}$ = "
                        f"{P_H2_BAR:g} bar", fontsize=figstyle.FS_ANNOT)
        ax_b.annotate(
            f"with ZPE: {tmap['T_zpe_k']:.0f} K\n({tmap['shift_k']:+.0f} K)",
            xy=(cross if cross else tmap["delta_mu_ev"], P.min()),
            xytext=(6, 14), textcoords="offset points",
            fontsize=figstyle.FS_ANNOT, color=figstyle.GUIDE, ha="left")

    figstyle.panel_label(ax_a, "a")
    figstyle.panel_label(ax_b, "b")
    fig.align_ylabels((ax_a, ax_b))
    fig.subplots_adjust(hspace=0.12)

    cap = ("The headline. (a) The equilibrium habit evolves with hydrogen "
           "chemical potential: as $\\Delta\\mu_H$ grows the {100} area "
           "fraction rises at the expense of {111}. (b) Because (100) is the "
           "one facet whose net surface stress is TENSILE in the continuum "
           "convention -- f = -tau, so its tau_mean of "
           f"{tau_mean_100:+.2f} N/m is f = {-tau_mean_100:+.2f} N/m -- while "
           "(110) and (111) are compressive, that change drives the interior "
           "from tension into "
           "compression and through zero"
           + (f" at $\\Delta\\mu_H$ = {cross:.3f} eV" if cross else "")
           + (f", which at $p_{{H_2}}$ = {P_H2_BAR:g} bar (a realistic "
              f"residual vacuum) is {tmap['T_k']:.0f} K uncorrected and "
              f"{tmap['T_zpe_k']:.0f} K once zero-point energy is included. "
              f"The {abs(tmap['shift_k']):.0f} K ZPE shift is the dominant "
              "systematic on this number and is drawn rather than "
              "footnoted." if tmap else ".")
           + " At the crossing the sign of every strain-induced NV shift "
           "reverses, which is a sharper experimental signature than any "
           "magnitude here because it needs no coupling constant and no "
           "absolute tau. Epistemic level L1; the H-poor side is unbounded "
           "while the dehydrogenation ceiling is uncomputed (see F2).")
    return fig, cap


# ──────────────────────────────── F4: the coupling-free observable

def figure_f4(sweeps, a0):
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL, 2.7))
    ax.axhline(0.0, color="black", lw=0.6, ls=":")

    for shape in [s for s in SHAPE_ORDER if s in sweeps]:
        rows = sweeps[shape]
        fam = HABIT.get(shape)
        rows = [r for r in rows
                if r["spin_strain_param_set"] == "udvarhelyi2018_dft"]
        x = np.array([1.0 / fnum(r, "radius_nm") for r in rows])
        y = np.array([fnum(r, "lattice_strain_percent") for r in rows])
        o = np.argsort(x)
        if fam is None:
            kw = dict(color="black", marker="D", markerfacecolor="white",
                      markeredgecolor="black", markeredgewidth=figstyle.MEW,
                      markersize=figstyle.MS, lw=figstyle.LW)
            label = "mixture"
        else:
            st = figstyle.SURFACE_STYLE[FAMILY_STYLE[fam]]
            kw = figstyle.series_kwargs(FAMILY_STYLE[fam])
            label = st["label"]
        ax.plot(x[o], y[o], label=label, **kw)

    ax.set_xlabel(r"$1/R$ (nm$^{-1}$)")
    ax.set_ylabel(r"lattice strain (\%)" if USE_TEX
                  else r"lattice strain (%)")
    figstyle.legend(ax, loc="upper left")
    ax.annotate("expand", xy=(0.97, 0.72), xycoords="axes fraction",
                ha="right", fontsize=figstyle.FS_ANNOT, color=figstyle.GUIDE)
    ax.annotate("contract", xy=(0.97, 0.10), xycoords="axes fraction",
                ha="right", fontsize=figstyle.FS_ANNOT, color=figstyle.GUIDE)

    cap = ("The coupling-independent prediction. Linear lattice strain against "
           "inverse particle radius, one line per habit, exactly linear in "
           "1/R because every quantity in the chain is. **No spin-strain "
           "coupling constant enters this figure**, so none of the 1.65x "
           "spread between the two published coupling sets applies to it -- "
           "unlike every MHz number in F5. The sign split is the signature: a "
           "{100}-dominated particle CONTRACTS while {110}- and "
           "{111}-dominated ones EXPAND, because (100) is the one surface with "
           "net tensile f. That direction is also independent of the absolute "
           "tau scale, so a size-resolved powder-XRD series identifies the "
           "dominant facet family from the sign of the shift alone and then "
           f"tests the magnitude against the slope. a0 = {a0:.6f} {AA_TXT} "
           "from config/reference_pbe_sssp.json. Epistemic level L1.")
    return fig, cap


# ─────────────────────────── F5: D and E are different kinds of quantity

def figure_f5(nv_rows_by_shape, local_rows):
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(figstyle.SINGLE_COL, 4.0))

    shapes = [s for s in SHAPE_ORDER if s in nv_rows_by_shape]
    ypos = np.arange(len(shapes))[::-1]
    ax_a.axvline(0.0, color="black", lw=0.8)
    for y, shape in zip(ypos, shapes):
        rows = nv_rows_by_shape[shape]
        vals = [fnum(r, "delta_D_mhz") for r in rows]
        lo, hi = min(vals), max(vals)
        fam = HABIT.get(shape)
        color = ("black" if fam is None
                 else figstyle.SURFACE_STYLE[FAMILY_STYLE[fam]]["color"])
        ax_a.plot([lo, hi], [y, y], color=color, lw=figstyle.LW * 2.2,
                  solid_capstyle="round")
        for v in (lo, hi):
            ax_a.plot([v], [y], marker="|", color="black", markersize=6,
                      markeredgewidth=0.9, linestyle="none")
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels([s.replace("rhombic-dodecahedron", "rh. dodec.")
                          for s in shapes], fontsize=figstyle.FS_ANNOT)
    ax_a.set_ylim(-0.7, len(shapes) - 0.3)
    ax_a.tick_params(axis="y", which="minor", left=False, right=False)
    ax_a.set_xlabel(r"$\Delta D$ (MHz) at $R$ = 1.5 nm")
    # Without this the bars read as error bars. They are not: the span is the
    # disagreement between the two published coupling sets, not a statistical
    # uncertainty on either.
    ax_a.annotate("bar spans the two published\nspin-strain coupling sets",
                  xy=(0.03, 0.96), xycoords="axes fraction", ha="left",
                  va="top", fontsize=figstyle.FS_ANNOT,
                  color=figstyle.GUIDE)

    by_shape = {}
    for r in local_rows:
        by_shape.setdefault(r["shape"], []).append(r)
    for shape in [s for s in SHAPE_ORDER if s in by_shape]:
        rows = sorted(by_shape[shape], key=lambda r: fnum(r, "depth_nm"))
        x = [fnum(r, "depth_nm") for r in rows]
        y = [fnum(r, "E_median_mhz") for r in rows]
        fam = HABIT.get(shape)
        if fam is None:
            kw = dict(color="black", marker="D", markerfacecolor="white",
                      markeredgecolor="black", markeredgewidth=figstyle.MEW,
                      markersize=figstyle.MS, lw=figstyle.LW)
            label = "mixture"
        else:
            kw = figstyle.series_kwargs(FAMILY_STYLE[fam])
            label = figstyle.SURFACE_STYLE[FAMILY_STYLE[fam]]["label"]
        ax_b.plot(x, y, label=label, **kw)
    ax_b.axhline(LINEWIDTH_MHZ, color=figstyle.GUIDE, ls="--", lw=0.8)
    # Left-anchored: the curves cross this line on the right-hand side, and a
    # right-anchored label lands on the (100) curve at its crossing.
    ax_b.annotate(f"{LINEWIDTH_MHZ:.0f} MHz ensemble linewidth",
                  xy=(0.03, LINEWIDTH_MHZ), xycoords=("axes fraction", "data"),
                  xytext=(0, 4), textcoords="offset points", ha="left",
                  fontsize=figstyle.FS_ANNOT, color=figstyle.GUIDE)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("NV depth (nm)")
    ax_b.set_ylabel(r"median $|E|$ (MHz)")
    figstyle.legend(ax_b, loc="lower left")

    figstyle.panel_label(ax_a, "a")
    figstyle.panel_label(ax_b, "b")
    fig.subplots_adjust(hspace=0.45)

    cap = ("D and E are different kinds of quantity, and the figure is drawn "
           "to make that visible. (a) The axial shift Delta D at R = 1.5 nm, "
           "each bar spanning the two published spin-strain coupling sets -- "
           "the bar width IS the coupling uncertainty, about 1.65x, and it "
           "dominates every other error here. Delta D is a VOLUME AVERAGE: it "
           "depends only on which facets are present and on nothing about "
           "where an NV sits. (b) The transverse splitting |E| against NV "
           "depth, which has no counterpart in (a)'s framing at all. For a "
           "symmetry-complete habit the volume-averaged E is exactly zero, so "
           "(b) exists only because the field is resolved in position: E is "
           "generated by the edge-concentrated deviatoric stress and falls "
           "roughly 4x per 0.5 nm of depth as the site retreats from the "
           "edges that carry the load. It is NOT driven by the (100) "
           "anisotropy -- zeroing that leaves 54% of E on a cube. Both panels "
           "use literature C11/C12/C44 and published coupling sets, neither "
           "fitted here. Epistemic level L1.")
    return fig, cap


# ────────────────────────────────────────────────────────────── driver

TITLES = {
    "F1_surfaces_and_tau": "Figure F1 - The H-terminated surfaces and their surface stress",
    "F2_gamma_vs_mu_h": "Figure F2 - Surface energy against hydrogen chemical potential",
    "F3_habit_and_pressure": "Figure F3 - Habit and interior pressure against mu_H",
    "F4_lattice_strain": "Figure F4 - The coupling-free observable",
    "F5_D_and_E": "Figure F5 - D and E are different kinds of quantity",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)

    print("Typography: " + ("LaTeX" if USE_TEX else "mathtext cm (LaTeX unusable)"))

    tau_rows = read_csv(TAU_CSV)
    gamma_cfg = json.loads(GAMMA_JSON.read_text()) if GAMMA_JSON.exists() else None
    scan = read_csv(MU_SCAN)
    tp_rows = read_csv(SIGN_TP)
    local_rows = read_csv(NV_LOCAL)
    a0 = json.loads((REPO / "config" / "reference_pbe_sssp.json").read_text()
                    )["bulk_reference"]["a0_fit_angstrom"]

    sweeps, nv_by_shape = {}, {}
    for shape in HABIT:
        s = read_csv(PROD / f"particle_strain_size_sweep_{shape}.csv")
        if s:
            sweeps[shape] = s
        n = read_csv(PROD / f"particle_strain_nv_{shape}.csv")
        if n:
            nv_by_shape[shape] = n

    builders = []
    if tau_rows:
        builders.append(("F1_surfaces_and_tau", lambda: figure_f1(tau_rows)))
    else:
        note_missing(f"F1: {TAU_CSV} absent")
    if gamma_cfg and scan:
        builders.append(("F2_gamma_vs_mu_h", lambda: figure_f2(gamma_cfg, scan)))
    else:
        note_missing("F2: surface energies or mu_H scan absent")
    if scan:
        builders.append(("F3_habit_and_pressure",
                         lambda: figure_f3(scan, tp_rows, tau_rows)))
    else:
        note_missing(f"F3: {MU_SCAN} absent")
    if sweeps:
        builders.append(("F4_lattice_strain", lambda: figure_f4(sweeps, a0)))
    else:
        note_missing("F4: no particle_strain size sweeps")
    if nv_by_shape and local_rows:
        builders.append(("F5_D_and_E",
                         lambda: figure_f5(nv_by_shape, local_rows)))
    else:
        note_missing("F5: NV predictions or nv_local_strain depth data absent")

    captions = {}
    for stem, build in builders:
        fig, cap = build()
        paths = figstyle.save(fig, outdir, stem, caption=cap)
        plt.close(fig)
        captions[stem] = cap
        print(f"  {stem}: " + ", ".join(p.name for p in paths))

    lines = ["# H-terminated result figures (F1-F5)", "",
             "Generated by `plot_h_results.py`; every number is read from a "
             "committed file.", "",
             "Sign convention (CLAUDE.md sec 2): positive sigma means "
             "COMPRESSED; positive tau is COMPRESSIVE surface stress.", "",
             "This set replaces the earlier N-series. N1 is absorbed into "
             "F1(b); N2 was deleted because its premise -- that the (100) "
             "anisotropy drives the E channel -- is contradicted by "
             "nv_local_strain.py.", ""]
    for stem, _ in builders:
        lines += [f"## {TITLES[stem]}", "", captions[stem], "",
                  f"Files: `{stem}.pdf`, `{stem}.png` (600 dpi), "
                  f"`{stem}_annotated.pdf`", ""]
    if _missing:
        lines += ["## Gaps in this set", ""]
        for m in _missing:
            lines.append(f"- {m}")
        lines.append("")
    (outdir / "CAPTIONS.md").write_text("\n".join(lines))
    print("  CAPTIONS.md")
    if _missing:
        print(f"\n{len(_missing)} GAP(S) -- see CAPTIONS.md")
    print(f"\nWrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
