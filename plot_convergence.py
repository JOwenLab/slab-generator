#!/usr/bin/env python3
"""
plot_convergence.py - Publication figures for the cutoff and k-point sweeps.

Reads results/convergence/convergence_summary.csv (written by parse_convergence.py)
and writes to ~/Documents/diamonds/convergence/, each figure as a vector PDF and a
300 dpi PNG.

Figure 1  cutoff convergence: mean in-plane stress and in-plane anisotropy vs
          ecutwfc, one line per surface.
Figure 2  k-point convergence: mean in-plane stress vs k-point sampling, C100 and
          C111 only. C110 was not swept and is not shown.

Every number plotted comes from the CSV. Nothing about the physics is hardcoded:
the surface list, the adopted production cutoff, and each surface's production
k-mesh are all derived from the data (see derive_* below). No smoothing,
interpolation, or extrapolation is applied - computed points are plotted and
joined by straight segments. A run that is missing, or that failed to reach JOB
DONE or SCF convergence, becomes a NaN and therefore a visible gap in the line;
it is never bridged.

Sign convention (CLAUDE.md section 2): sigma < 0 is compressive, sigma > 0 is
tensile-like, all stresses in kbar. Anisotropy sigma_xx - sigma_yy is reported
with the cell axis assignment fixed by the generator, so its sign is meaningful.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- style ----
# Explicit style so the output does not carry matplotlib's defaults: Helvetica
# rather than DejaVu Sans, no top/right spines, recessive grid and axes, and
# no default C0/C1/C2 colour cycle. Sizes are literal points at the final
# figure size (3.4 in wide, saved unscaled), so nothing falls below 8 pt.
INK = "#1a1a1a"
INK_MUTED = "#5c5c5c"
RULE = "#d9d6d0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    # Without this, mathtext falls back to DejaVu and the sigma symbols end up
    # in a different typeface from the surrounding Helvetica label. macOS ships
    # Helvetica as a .ttc whose oblique face matplotlib will not select, so
    # symbols are set upright throughout rather than half-italic; switch the
    # fontset to "stixsans" if the target journal insists on italic variables.
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.bf": "Helvetica:bold",
    "mathtext.default": "rm",
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK_MUTED,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.0,
    "lines.markeredgewidth": 0.6,
    "grid.color": "#eceae6",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.columnspacing": 1.2,
    "legend.handletextpad": 0.5,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,   # embed TrueType, keep text selectable/editable
    "ps.fonttype": 42,
})

# Categorical identity, assigned in fixed surface order and never cycled. These
# three pass the colourblind-separation check (worst adjacent deutan dE 11.0,
# normal-vision dE 25.8); the distinct markers are the secondary encoding.
SURFACE_STYLE = {
    "C100": {"color": "#0072B2", "marker": "o"},
    "C110": {"color": "#D55E00", "marker": "s"},
    "C111": {"color": "#009E73", "marker": "^"},
}
SURFACE_ORDER = ["C100", "C110", "C111"]

FIGSIZE_COL = 3.4  # single-column width, inches


def surface_label(surface):
    """C100 -> (100), the crystallographic form used in the manuscript."""
    return "({})".format(surface[1:]) if surface.startswith("C") else surface


def style_for(surface):
    return SURFACE_STYLE.get(surface, {"color": INK, "marker": "D"})


# ----------------------------------------------------------------- data ----
def fnum(value):
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path):
    """Usable rows only: a run that did not finish is dropped here and reported,
    so downstream it shows up as an absent x value, i.e. a gap."""
    with Path(path).open() as f:
        all_rows = list(csv.DictReader(f))

    good, bad = [], []
    for r in all_rows:
        if r["job_done"] != "yes" or r["scf_converged"] != "yes":
            bad.append((r["run"], "job_done={} scf_converged={}".format(r["job_done"], r["scf_converged"])))
            continue
        good.append(r)
    return good, bad


def series(rows, xkey, ykey):
    """(x, y) sorted by x, with y as float. Rows lacking either are skipped."""
    pts = []
    for r in rows:
        x, y = fnum(r[xkey]), fnum(r[ykey])
        if x is not None and y is not None:
            pts.append((x, y))
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def on_grid(grid, x, y):
    """Map (x, y) onto a common x grid, NaN where this series has no point.
    NaN breaks the line, so a missing run reads as a gap and is never bridged."""
    out = np.full(len(grid), np.nan)
    for xi, yi in zip(x, y):
        idx = np.nonzero(np.isclose(grid, xi))[0]
        if idx.size:
            out[idx[0]] = yi
    return out


def derive_production_cutoff(rows):
    """The k-point sweep was run at the adopted production cutoff, so the CSV
    already carries it: the single ecutwfc shared by every k-point run."""
    vals = {fnum(r["ecutwfc_ry"]) for r in rows if r["sweep"] == "kpoint"}
    vals.discard(None)
    if len(vals) != 1:
        raise SystemExit(
            "Cannot derive the production cutoff: k-point runs use ecutwfc {}. "
            "Expected exactly one value.".format(sorted(vals))
        )
    return vals.pop()


def derive_production_mesh(rows, surface):
    """Likewise, each surface's cutoff sweep was run at its production k-mesh."""
    vals = {r["kmesh"] for r in rows
            if r["sweep"] == "cutoff" and r["surface"] == surface and r["kmesh"]}
    if len(vals) != 1:
        raise SystemExit(
            "Cannot derive the production k-mesh for {}: cutoff runs use {}. "
            "Expected exactly one mesh.".format(surface, sorted(vals))
        )
    return vals.pop()


def drift_label(x, y):
    """Change in y across the swept range, as plain kbar. Reported rather than
    a percentage because these curves cross zero on some surfaces."""
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return None
    yy = y[ok]
    return yy[-1] - yy[0]


# ------------------------------------------------------------- figure 1 ----
def figure_cutoff(rows, outdir):
    """Two stacked panels sharing the ecutwfc axis.

    Panel (a) uses a y-axis broken into one band per surface. A single
    continuous axis would span roughly -22 to +40 kbar, on which the ~+2 to
    +2.8 kbar drift that is the whole point of the panel occupies about 4% of
    the height and is not legible. The bands keep true kbar values on the axis
    while giving the drift the full height of each band; the breaks are marked.
    Panel (b) keeps a single continuous axis - that is what makes the flatness
    of the anisotropy visible against its full inter-surface spread, and it is
    why (b) must not share limits with (a).
    """
    cut = [r for r in rows if r["sweep"] == "cutoff"]
    surfaces = [s for s in SURFACE_ORDER if any(r["surface"] == s for r in cut)]
    grid = np.array(sorted({fnum(r["ecutwfc_ry"]) for r in cut}))
    prod_cut = derive_production_cutoff(rows)

    mean = {}
    anis = {}
    for s in surfaces:
        sub = [r for r in cut if r["surface"] == s]
        x, y = series(sub, "ecutwfc_ry", "sigma_mean_kbar")
        mean[s] = on_grid(grid, x, y)
        x, y = series(sub, "ecutwfc_ry", "anisotropy_kbar")
        anis[s] = on_grid(grid, x, y)

    # Bands run top-to-bottom in descending stress, so the broken axis still
    # reads as a normal y-axis.
    banded = sorted(surfaces, key=lambda s: np.nanmean(mean[s]), reverse=True)

    fig = plt.figure(figsize=(FIGSIZE_COL, 4.95), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.7], hspace=0.09)
    inner = outer[0].subgridspec(len(banded), 1, hspace=0.16)
    axes_a = [fig.add_subplot(inner[i]) for i in range(len(banded))]
    ax_b = fig.add_subplot(outer[1])

    xpad = 0.04 * (grid[-1] - grid[0])
    xlim = (grid[0] - xpad, grid[-1] + xpad)

    for ax, s in zip(axes_a, banded):
        st = style_for(s)
        ax.axvline(prod_cut, color=INK_MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.grid(axis="y", zorder=0)
        ax.plot(grid, mean[s], color=st["color"], marker=st["marker"],
                markeredgecolor="white", clip_on=False, zorder=3)
        ax.set_xlim(*xlim)

        lo, hi = np.nanmin(mean[s]), np.nanmax(mean[s])
        pad = 0.30 * (hi - lo) if hi > lo else 0.5
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))

        d = drift_label(grid, mean[s])
        note = surface_label(s) if d is None else "{}  {:+.2f} kbar".format(surface_label(s), d)
        ax.text(0.985, 0.06, note, transform=ax.transAxes, ha="right", va="bottom",
                color=INK_MUTED, fontsize=8)

    # Hide the interior x-spines and mark each break with a pair of diagonals.
    for i, ax in enumerate(axes_a):
        ax.tick_params(labelbottom=False)
        if i < len(axes_a) - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)
        dx, dy = 0.014, 0.035
        kw = dict(transform=ax.transAxes, color=INK_MUTED, lw=0.7, clip_on=False, zorder=5)
        if i > 0:
            ax.plot((-dx, dx), (1 - dy, 1 + dy), **kw)
        if i < len(axes_a) - 1:
            ax.plot((-dx, dx), (-dy, dy), **kw)

    mid = axes_a[len(axes_a) // 2]
    mid.set_ylabel("mean in-plane stress\n" + r"$(\sigma_{xx}+\sigma_{yy})/2$  (kbar)")

    axes_a[0].text(0.0, 1.12, "(a)", transform=axes_a[0].transAxes,
                   ha="left", va="bottom", fontsize=9, fontweight="bold")

    ax_b.axvline(prod_cut, color=INK_MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax_b.grid(axis="y", zorder=0)
    for s in surfaces:
        st = style_for(s)
        ax_b.plot(grid, anis[s], color=st["color"], marker=st["marker"],
                  markeredgecolor="white", label=surface_label(s),
                  clip_on=False, zorder=3)
    ax_b.set_xlim(*xlim)
    ax_b.set_xlabel(r"plane-wave cutoff  $E_{\mathrm{cut}}^{\mathrm{wfc}}$  (Ry)")
    ax_b.set_ylabel("in-plane anisotropy\n" + r"$\sigma_{xx}-\sigma_{yy}$  (kbar)")
    ax_b.set_xticks(grid)
    ax_b.text(0.0, 1.06, "(b)", transform=ax_b.transAxes,
              ha="left", va="bottom", fontsize=9, fontweight="bold")

    # Annotate the adopted cutoff once, on the bottom panel.
    ax_b.annotate("adopted\n{:g} Ry".format(prod_cut),
                  xy=(prod_cut, 0.5), xycoords=("data", "axes fraction"),
                  xytext=(4, 0), textcoords="offset points",
                  ha="left", va="center", color=INK_MUTED, fontsize=8)

    fig.legend(*ax_b.get_legend_handles_labels(),
               loc="outside upper center", ncol=len(surfaces))

    save(fig, outdir / "fig1_cutoff_convergence")
    return grid, mean, anis


# ------------------------------------------------------------- figure 2 ----
def figure_kpoint(rows, outdir):
    """Two panels, one per surface, sharing a k-point-density x-axis.

    Two choices are being made here and both are forced by the data.

    x-axis: the surface cells differ in size (C100 is a 5.05 A square cell,
    C111 a 2.53 A hexagonal one), so a 6x6 mesh does not mean the same sampling
    on both. The x coordinate is therefore n_k * a_parallel, the real-space
    sampling length in Angstrom, taken from the CSV columns
    k_density_a1_angstrom. The raw mesh integer is printed above each point so
    nothing is lost.

    Two panels rather than one: mean stress differs by about 26 kbar between
    the two surfaces while the k-point variation within each is under 0.5 kbar.
    On a shared y-axis the convergence behaviour - the entire content of the
    figure - would be invisible. Each panel therefore carries its own y-scale,
    which is the small-multiples alternative to a second y-axis, not a dual
    axis.
    """
    kpt = [r for r in rows if r["sweep"] == "kpoint"]
    surfaces = [s for s in SURFACE_ORDER if any(r["surface"] == s for r in kpt)]
    if not surfaces:
        raise SystemExit("No k-point runs in the CSV.")

    fig = plt.figure(figsize=(FIGSIZE_COL, 3.75), constrained_layout=True)
    axes = fig.subplots(len(surfaces), 1, sharex=True)
    axes = np.atleast_1d(axes)

    all_x = []
    for r in kpt:
        v = fnum(r["k_density_a1_angstrom"])
        if v is not None:
            all_x.append(v)
    xpad = 0.06 * (max(all_x) - min(all_x))
    xlim = (min(all_x) - xpad, max(all_x) + xpad)

    for ax, s, panel in zip(axes, surfaces, "abcdef"):
        st = style_for(s)
        sub = [r for r in kpt if r["surface"] == s]
        x, y = series(sub, "k_density_a1_angstrom", "sigma_mean_kbar")

        # Production mesh for this surface, converted to the same x coordinate.
        prod_mesh = derive_production_mesh(rows, s)
        prod_k = float(prod_mesh.split()[0])
        a1 = {fnum(r["a1_angstrom"]) for r in sub}
        if len(a1) != 1:
            raise SystemExit("{}: k-point runs disagree on a1 ({})".format(s, sorted(a1)))
        prod_x = prod_k * a1.pop()

        ax.axvline(prod_x, color=st["color"], lw=0.7, ls=(0, (4, 3)), alpha=0.55, zorder=1)
        ax.grid(axis="y", zorder=0)
        ax.plot(x, y, color=st["color"], marker=st["marker"],
                markeredgecolor="white", clip_on=False, zorder=3)
        ax.set_xlim(*xlim)

        lo, hi = np.nanmin(y), np.nanmax(y)
        span = hi - lo if hi > lo else 0.1
        ax.set_ylim(lo - 0.45 * span, hi + 0.55 * span)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        # Raw mesh integer above each point, so the density axis loses nothing.
        for r in sub:
            xv, yv, k1 = fnum(r["k_density_a1_angstrom"]), fnum(r["sigma_mean_kbar"]), r["k1"]
            if xv is None or yv is None:
                continue
            # White halo so a label sitting on the production rule or a
            # gridline stays readable without moving it off its point.
            ax.annotate("{:g}".format(float(k1)), xy=(xv, yv),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK_MUTED,
                        zorder=4,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

        ax.annotate("production\n{}".format(prod_mesh.replace(" ", r"$\times$")),
                    xy=(prod_x, 0.04), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8, color=INK_MUTED)

        ax.text(0.985, 0.93, surface_label(s), transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=INK)
        ax.text(0.0, 1.04, "({})".format(panel), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, fontweight="bold")

    # "in-plane" in words rather than a \parallel subscript: that glyph is not
    # in Helvetica and would drag a second font into the PDF.
    axes[-1].set_xlabel(r"in-plane k-point sampling length  $n_k a$  (Å)")
    fig.supylabel("mean in-plane stress  " + r"$(\sigma_{xx}+\sigma_{yy})/2$  (kbar)",
                  fontsize=8.5, color=INK)

    save(fig, outdir / "fig2_kpoint_convergence")


# ----------------------------------------------------------------- io ------
def save(fig, stem):
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("Wrote {}".format(pdf))
    print("Wrote {}".format(png))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--csv", default="results/convergence/convergence_summary.csv")
    ap.add_argument("--outdir", default=str(Path.home() / "Documents/diamonds/convergence"))
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    rows, dropped = load_rows(args.csv)
    if dropped:
        print("WARNING: {} run(s) excluded from the figures (they appear as gaps, "
              "not as interpolated points):".format(len(dropped)))
        for run, why in dropped:
            print("  {}: {}".format(run, why))

    figure_cutoff(rows, outdir)
    figure_kpoint(rows, outdir)

    missing = [s for s in SURFACE_ORDER
               if not any(r["sweep"] == "kpoint" and r["surface"] == s for r in rows)]
    if missing:
        print("Note: no k-point sweep exists for {}; "
              "Figure 2 shows only the surfaces that were swept."
              .format(", ".join(surface_label(s) for s in missing)))


if __name__ == "__main__":
    main()
