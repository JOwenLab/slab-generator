#!/usr/bin/env python3
"""
figstyle.py - Shared journal figure style for the nanodiamond project.

Every figure in results/figures imports this module so that typography, axis
furniture, colours and markers are identical across the whole set. Nothing here
touches data; it is presentation only.

House style
-----------
Nature / PRB conventions: Computer Modern typography, a full black axis box on
all four sides, inward ticks on all four sides with minor ticks on, no
gridlines, no figure titles, and panel labels as lowercase bold (a), (b) set
outside the axes at the top left.

Column widths are the Nature values: 88 mm single column, 180 mm double.

Typography
----------
`text.usetex` is attempted first and verified by actually rendering a figure,
not by checking whether a `latex` binary exists on PATH. On this machine
`latex` and `pdflatex` are present but `dvipng` is not, so a which-based probe
reports LaTeX as available and then matplotlib raises at the first raster
save. The fallback is mathtext with `fontset='cm'` and `font.family='serif'`,
which is also Computer Modern, so the two paths look nearly identical.

Surface identity
----------------
One colour and one marker per surface, fixed for the whole figure set, drawn
from the Okabe-Ito colour-blind-safe palette. The three hues also separate in
greyscale, and the marker shape carries the identity independently of colour so
the figures survive being printed in black and white.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


# ── Column widths (inches) ────────────────────────────────────────────────────
SINGLE_COL = 3.46      # 88 mm
DOUBLE_COL = 7.09      # 180 mm

# ── Font sizes at final printed size (pt) ─────────────────────────────────────
FS_LABEL = 8
FS_TICK = 7
FS_LEGEND = 7
FS_PANEL = 8
FS_ANNOT = 6.5

# ── Line and marker weights ───────────────────────────────────────────────────
LW = 0.9
MS = 3.5
MEW = 0.6
AXIS_LW = 0.8

# ── Surface identity: Okabe-Ito ───────────────────────────────────────────────
SURFACES = ("C100", "C110", "C111")

SURFACE_STYLE = {
    # key      colour     marker  legend label
    "C100": {"color": "#0072B2", "marker": "o", "label": "(100)"},
    "C110": {"color": "#D55E00", "marker": "s", "label": "(110)"},
    "C111": {"color": "#009E73", "marker": "^", "label": "(111)"},
}

# Neutral accents (annotations, reference lines). Deliberately not one of the
# three surface hues, so a guide line can never be misread as a surface.
GUIDE = "#4D4D4D"

_USETEX = None

# Suppress the PDF CreationDate. matplotlib stamps the wall clock into every
# PDF it writes, so two runs over identical data produce different bytes and
# the figures can never be checked for reproducibility -- `regenerate.sh
# --check` would report all 18 PDFs as changed on every invocation, which
# trains the reader to ignore it. PNGs do not carry the stamp and are already
# byte-stable. Producer is pinned for the same reason: it embeds the matplotlib
# version, which would otherwise churn the files on every upgrade.
PDF_METADATA = {"CreationDate": None, "Producer": "slab-generator"}


def _probe_usetex() -> bool:
    """
    Return True only if a figure actually renders through LaTeX.

    Checking for a `latex` executable is not sufficient: matplotlib's Agg path
    also needs dvipng, and a missing dvipng only surfaces at savefig time.
    """
    import io

    try:
        with matplotlib.rc_context({"text.usetex": True}):
            fig = plt.figure(figsize=(0.5, 0.5))
            fig.text(0.5, 0.5, r"$\sigma_{xx}$ 1")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=50)
            plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False


def usetex_active() -> bool:
    """Whether the LaTeX text path is in use (probed once, then cached)."""
    global _USETEX
    if _USETEX is None:
        _USETEX = _probe_usetex()
    return _USETEX


def angstrom() -> str:
    """Angstrom symbol that renders under whichever text path is active."""
    return r"\AA" if usetex_active() else r"\mathrm{\AA}"


def apply() -> bool:
    """
    Install the house style globally. Returns True if LaTeX typography is in use.

    Call once at the top of a figure script, before creating any figure.
    """
    use_tex = usetex_active()

    rc = {
        # Typography — Computer Modern either way
        "text.usetex": use_tex,
        "font.family": "serif",
        # Without these two the "serif" fallback is DejaVu Serif, which is not
        # Computer Modern and reads as generic matplotlib output. cmr10 is the
        # Computer Modern Roman that ships with matplotlib.
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        # cmr10 has no U+2212; rendering tick labels through mathtext avoids the
        # missing-glyph warning and the tofu box it leaves on negative ticks.
        "axes.formatter.use_mathtext": True,
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,

        # Full black box on all four sides
        "axes.linewidth": AXIS_LW,
        "axes.edgecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,

        # Ticks inward on all four sides, minor ticks on
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": AXIS_LW,
        "ytick.major.width": AXIS_LW,
        "xtick.minor.width": AXIS_LW,
        "ytick.minor.width": AXIS_LW,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.color": "black",
        "ytick.color": "black",

        # No gridlines
        "axes.grid": False,

        # Lines and markers
        "lines.linewidth": LW,
        "lines.markersize": MS,
        "lines.markeredgewidth": MEW,

        # Legend: no frame by default, never a shadow
        "legend.frameon": False,
        "legend.shadow": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderpad": 0.3,
        "legend.columnspacing": 1.0,

        # Vector output with editable text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }

    if use_tex:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"

    matplotlib.rcParams.update(rc)
    return use_tex


def series_kwargs(surface, filled=True, **override):
    """
    Plot kwargs for a surface's data series. Identical in every figure.

    `filled=False` gives an open marker, used to show a point that is plotted
    but deliberately excluded from a fit.
    """
    style = SURFACE_STYLE[surface]
    kw = {
        "color": style["color"],
        "marker": style["marker"],
        "markerfacecolor": style["color"] if filled else "white",
        "markeredgecolor": "black",
        "markeredgewidth": MEW,
        "markersize": MS,
        "linewidth": LW,
    }
    kw.update(override)
    return kw


def surface_label(surface):
    """Legend label: the Miller index alone. Termination goes in the caption."""
    return SURFACE_STYLE[surface]["label"]


def panel_label(ax, letter, dx=-0.02, dy=1.0):
    """
    Lowercase bold panel label, e.g. (a), outside the axes at the top left.

    Placed in axes coordinates just above and left of the frame so it never
    collides with tick labels.
    """
    weight = "bold"
    text = f"({letter})"
    if usetex_active():
        text = r"\textbf{(" + letter + r")}"
        weight = "normal"      # the boldness is in the LaTeX markup
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight=weight,
            ha="right", va="bottom")


def legend(ax, **kwargs):
    """House legend: frameless by default, never shadowed."""
    opts = {"frameon": False, "shadow": False, "fontsize": FS_LEGEND}
    opts.update(kwargs)
    leg = ax.legend(**opts)
    if leg is not None and opts.get("frameon"):
        leg.get_frame().set_linewidth(0.6)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_boxstyle("Square", pad=0.3)
    return leg


def tex_escape(s):
    """Escape characters that would break the LaTeX text path in a caption."""
    if not usetex_active():
        return s
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        s = s.replace(ch, "\\" + ch)
    return s


def save(fig, outdir, stem, caption=None, caption_width=None):
    """
    Write the figure as a clean vector PDF plus a 600 dpi PNG.

    If `caption` is given, additionally write <stem>_annotated.pdf with the
    caption set beneath the axes. The primary PDF never carries a baked-in
    caption — journals set captions in the manuscript, not the artwork.

    Returns the list of paths written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    for suffix, kwargs in ((".pdf", {"metadata": PDF_METADATA}),
                           (".png", {"dpi": 600})):
        path = outdir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        written.append(path)

    if caption:
        width_in = fig.get_size_inches()[0]
        # Computer Modern at 6.5 pt averages ~0.055 in per character, so ~18
        # characters per inch. Wrapping narrower than this leaves the caption
        # block visibly inset from the figure it belongs to.
        wrap_at = caption_width or max(40, int(width_in * 18))
        wrapped = "\n".join(textwrap.wrap(caption, wrap_at))
        txt = fig.text(
            0.0, -0.04, tex_escape(wrapped),
            ha="left", va="top", fontsize=FS_ANNOT,
            transform=fig.transFigure,
        )
        path = outdir / f"{stem}_annotated.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                    metadata=PDF_METADATA)
        written.append(path)
        txt.remove()

    return written
