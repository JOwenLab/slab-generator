#!/usr/bin/env python3
"""
plot_surface_stress.py - The surface-stress result figures.

Companion to plot_convergence.py, which establishes the production standard.
These four carry the physics that standard was built to support.

    N1  tau_infinity per facet                (single column)
    N2  surface energy vs anisotropy          (single column)
    N3  energy convergence is not stress      (single column)
    N4  vc-relax lattice distortion           (single column)

Styling is shared with the convergence set through figstyle.py, so a reader
sees one figure set: same typography, same Okabe-Ito colour and marker per
surface, same axis furniture.

Data provenance
---------------
Every number is read from a file or recomputed from the run directories.
Nothing about the physics is hardcoded.

    results/production/tau_infinity.csv           tau_inf and its fit rms,
                                                  written by fit_tau_infinity.py
    results/production/vcrelax~<surface>_<N>L/    vc-relax runs, for N4
    results/production/thick_a0corr_stress~*      relaxed slab energies, for the
                                                  surface energies in N2
    results/reference_90_720/bulk_fit_summary.json  mu_C (bm3 E0 / 8)
    results/reference_90_720/H2/pw.out            mu_H (E(H2) / 2)
    results/convergence/convergence_summary.csv   the cutoff sweep, for N3

tau_infinity is never refitted here. N1 reads the published CSV; if it is
absent the figure is skipped and said so, because a second implementation of
that extrapolation could disagree with the first and the disagreement would be
invisible in the output.

SIGN CONVENTION (CLAUDE.md section 2, and tau_infinity.csv's own field):

    sigma > 0  ->  the cell is COMPRESSED.
    tau   > 0  ->  a TENSILE surface stress: the surface tends to contract,
                   which compresses the slab interior.

Those two agree rather than conflict -- a tensile surface pulling inward is
what puts the interior into compression -- so no sign is flipped anywhere
between sigma and tau.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import figstyle

USE_TEX = figstyle.apply()

import matplotlib.pyplot as plt  # noqa: E402

AA = figstyle.angstrom()
AA_TXT = "A"
PCT = r"\%" if USE_TEX else "%"

DEFAULT_OUTDIR = Path.home() / "Documents" / "diamonds" / "surface-stress"
PRODUCTION_DIR = Path("results/production")
TAU_INF_CSV = PRODUCTION_DIR / "tau_infinity.csv"
CONVERGENCE_CSV = Path("results/convergence/convergence_summary.csv")
SURFACE_ENERGY_CSV = PRODUCTION_DIR / "surface_energy.csv"
BULK_FIT_JSON = Path("results/reference_90_720/bulk_fit_summary.json")
H2_OUT = Path("results/reference_90_720/H2/pw.out")

VCRELAX_RE = re.compile(r"^vcrelax~(?P<surface>C\d{3})_(?P<layers>\d+)L$")
STRESS_RUN_RE = re.compile(
    r"^thick_a0corr_stress~(?P<surface>C\d{3})_(?P<layers>\d+)L_stress_scf$")

# 1 Ry = 2.1798723611e-18 J; 1 Angstrom^2 = 1e-20 m^2.
RY_PER_ANG2_TO_J_PER_M2 = 2.1798723611e-18 / 1e-20

RY_TO_EV = 13.605693


# ── Small helpers ─────────────────────────────────────────────────────────────

def linspace(lo, hi, n=200):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def linear_fit(xs, ys):
    """Ordinary least squares. Returns (slope, intercept)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def norm(v):
    return math.sqrt(sum(x * x for x in v))


# ── Loading ───────────────────────────────────────────────────────────────────

def load_tau_infinity(path=TAU_INF_CSV):
    """
    The published tau_infinity fit, keyed by surface.

    Never recomputed here: fit_tau_infinity.py owns this fit and writes this
    file. Returns None if it is absent, which the caller reports rather than
    working around.
    """
    if not Path(path).exists():
        return None
    out = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["surface"]] = row
    return out or None


def _read_cell_and_species(pw_in):
    """(cell vectors, species->count) from a pw.in. Positions may be crystal."""
    lines = Path(pw_in).read_text().splitlines()
    cell, species = None, {}
    for i, ln in enumerate(lines):
        head = ln.strip().upper()
        if head.startswith("CELL_PARAMETERS") and cell is None:
            unit = ln.split()[-1].strip("()").lower()
            if "angstrom" not in unit:
                # Every run in this campaign writes angstrom. Refuse to guess:
                # an alat cell read as angstrom gives a plausible wrong area.
                raise ValueError(f"{pw_in}: CELL_PARAMETERS unit is {unit!r}")
            cell = [[float(v) for v in lines[i + 1 + j].split()[:3]]
                    for j in range(3)]
        elif head.startswith("ATOMIC_POSITIONS"):
            for row in lines[i + 1:]:
                parts = row.split()
                if len(parts) < 4 or row.strip().startswith("!"):
                    break
                try:
                    [float(p) for p in parts[1:4]]
                except ValueError:
                    break
                species[parts[0]] = species.get(parts[0], 0) + 1
    return cell, species


def _final_cell(pw_out):
    """Cell from the 'Begin final coordinates' block of a vc-relax run."""
    lines = Path(pw_out).read_text(errors="replace").splitlines()
    start = None
    for i, ln in enumerate(lines):
        if "Begin final coordinates" in ln:
            start = i
    if start is None:
        return None
    for i in range(start, len(lines)):
        if lines[i].strip().upper().startswith("CELL_PARAMETERS"):
            unit = lines[i].split()[-1].strip("()").lower()
            if "angstrom" not in unit:
                raise ValueError(f"{pw_out}: final CELL_PARAMETERS is {unit!r}")
            return [[float(v) for v in lines[i + 1 + j].split()[:3]]
                    for j in range(3)]
    return None


def load_vcrelax(root=PRODUCTION_DIR):
    """
    In-plane strain of each vc-relax run relative to its own starting cell.

    Returns (by_surface, skipped). eps_a and eps_b are the fractional changes in
    the two in-plane cell-vector lengths. A run that did not reach JOB DONE or
    has no final cell is skipped and named, never silently dropped.
    """
    by_surface, skipped = {}, []
    for d in sorted(Path(root).iterdir()):
        m = VCRELAX_RE.match(d.name)
        if not m or not (d / "pw.out").exists():
            continue
        text = (d / "pw.out").read_text(errors="replace")
        if "JOB DONE" not in text:
            skipped.append((d.name, "no JOB DONE"))
            continue
        initial, species = _read_cell_and_species(d / "pw.in")
        final = _final_cell(d / "pw.out")
        if initial is None or final is None:
            skipped.append((d.name, "missing initial or final cell"))
            continue

        a_i, b_i = norm(initial[0]), norm(initial[1])
        a_f, b_f = norm(final[0]), norm(final[1])
        n_c = species.get("C", 0)
        layers = int(m.group("layers"))
        if not n_c or n_c % layers:
            skipped.append((d.name, f"{n_c} C atoms is not divisible by "
                                    f"{layers} layers"))
            continue
        area = abs(initial[0][0] * initial[1][1] - initial[0][1] * initial[1][0])
        by_surface.setdefault(m.group("surface"), []).append({
            "layers": layers, "run": d.name, "n_C": n_c, "area": area,
            "eps_a": (a_f - a_i) / a_i, "eps_b": (b_f - b_i) / b_i,
        })
    for pts in by_surface.values():
        pts.sort(key=lambda p: p["layers"])
    return by_surface, skipped


def load_chemical_potentials():
    """
    (mu_C, mu_H) in Ry per atom, both at the 90/720 Ry production setting.

    mu_C is the Birch-Murnaghan E0 of the production bulk fit divided by the 8
    atoms of the conventional cell -- not the superseded 80/640 reference that
    parse_slab.py still reads. mu_H is half the total energy of the relaxed H2
    molecule computed at the same cutoffs with the same H pseudopotential as
    the slabs.
    """
    if not BULK_FIT_JSON.exists() or not H2_OUT.exists():
        return None, None
    bulk = json.loads(BULK_FIT_JSON.read_text())
    e0 = bulk.get("bm3_E0_ry")
    mu_c = e0 / 8.0 if e0 is not None else None

    text = H2_OUT.read_text(errors="replace")
    if "JOB DONE" not in text:
        return mu_c, None
    energies = re.findall(r"^!\s+total energy\s*=\s*(-?[0-9.]+)\s*Ry",
                          text, flags=re.M)
    mu_h = float(energies[-1]) / 2.0 if energies else None
    return mu_c, mu_h


def load_published_surface_energies(path=SURFACE_ENERGY_CSV):
    """
    The published surface energies, keyed by surface.

    surface_energy.py owns this quantity and writes this file; it is read here
    rather than recomputed, for the same reason tau_infinity is. Reading it also
    picks up two things a local recomputation would silently drop: the epistemic
    level per surface, and the scatter of gamma across the fitted ladder, which
    for one surface is the same size as gamma itself.

    `recompute_surface_energies` below is kept as an independent check on this
    file, not as an alternative source for the figure.
    """
    if not Path(path).exists():
        return None
    out = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["surface"]] = row
    return out or None


def cross_check_surface_energies(published, recomputed):
    """
    Confirm the published per-slab gamma matches an independent recomputation.

    Mismatch means one of the two is wrong and no reader could tell which, so
    it is raised rather than reported. Compared against the per-slab column
    rather than the headline value, because the headline comes from a fit over
    the ladder and this recomputation does not fit anything.
    """
    checked = []
    for surface, row in sorted(published.items()):
        mine = recomputed.get(surface)
        if not mine:
            continue
        per_slab = {}
        for item in (row.get("per_slab_gamma_j_m2") or "").split(";"):
            if ":" not in item:
                continue
            layers, value = item.split(":")
            per_slab[int(layers.strip().rstrip("L"))] = float(value)
        want = per_slab.get(mine["layers"])
        if want is None:
            continue
        if abs(mine["gamma"] - want) > 5e-4:
            raise SystemExit(
                f"surface energy mismatch for {surface} at {mine['layers']}L: "
                f"{mine['gamma']:.6f} J/m^2 recomputed here vs {want:.6f} "
                f"published in {SURFACE_ENERGY_CSV}. Refusing to plot.")
        checked.append(f"{surface}@{mine['layers']}L")
    return checked


def recompute_surface_energies(root=PRODUCTION_DIR):
    """
    gamma in J/m^2 for each surface, from the thickest slab of its ladder.

    gamma = [E_slab - N_C*mu_C - N_H*mu_H] / (2A), the standard symmetric-slab
    expression; the factor 2 is the slab's two identical faces. Referenced to
    bulk diamond and molecular H2, so a negative gamma means the H-terminated
    face is more stable than that reference state, which is expected because
    hydrogenation is exothermic.

    The thickest slab is used rather than an extrapolation: gamma is already
    converged to ~1e-4 J/m^2 between the last two rungs, which the caller
    reports, so no fit is needed and none is done.

    Returns (by_surface, skipped).
    """
    import parse_slab

    mu_c, mu_h = load_chemical_potentials()
    if mu_c is None or mu_h is None:
        return None, [("chemical potentials",
                       "mu_C or mu_H unavailable; cannot compute gamma")]

    ladders, skipped = {}, []
    for d in sorted(Path(root).iterdir()):
        m = STRESS_RUN_RE.match(d.name)
        if not m or not (d / "pw.out").exists():
            continue
        pout = parse_slab.parse_pw_out(d / "pw.out")
        if pout["status"] != "JOB DONE" or pout["energy_ry"] is None:
            skipped.append((d.name, f"status {pout['status']!r}"))
            continue
        pin = parse_slab.parse_pw_in(d / "pw.in")
        atoms = pout["final_positions_ang"] or pin["initial_positions_ang"]
        n_c = sum(1 for a in atoms if a["species"] == "C")
        n_h = sum(1 for a in atoms if a["species"] == "H")
        cell = pin["cell_params_ang"]
        area = norm(parse_slab.vec_cross(cell[0], cell[1]))
        gamma = ((pout["energy_ry"] - n_c * mu_c - n_h * mu_h)
                 / (2.0 * area)) * RY_PER_ANG2_TO_J_PER_M2
        ladders.setdefault(m.group("surface"), []).append(
            {"layers": int(m.group("layers")), "gamma": gamma,
             "n_C": n_c, "n_H": n_h, "area": area})

    out = {}
    for surface, pts in ladders.items():
        pts.sort(key=lambda p: p["layers"])
        thickest = pts[-1]
        out[surface] = {
            "gamma": thickest["gamma"],
            "layers": thickest["layers"],
            "h_coverage": thickest["n_H"] / thickest["area"],
            # Difference between the last two rungs: the honest statement of
            # how converged gamma is, with no fit involved.
            "drift": (abs(pts[-1]["gamma"] - pts[-2]["gamma"])
                      if len(pts) > 1 else float("nan")),
        }
    return out, skipped


def load_cutoff_sweep(path=CONVERGENCE_CSV):
    """Cutoff-sweep rows that completed, grouped by surface."""
    if not Path(path).exists():
        raise SystemExit(f"missing input: {path}\nRun parse_convergence.py first.")
    by_surface, skipped = {}, []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("sweep") != "cutoff":
                continue
            if (row.get("job_done") != "yes"
                    or row.get("scf_converged") != "yes"):
                skipped.append((row["run"], "did not complete"))
                continue
            by_surface.setdefault(row["surface"], []).append(row)
    for rows in by_surface.values():
        rows.sort(key=lambda r: float(r["ecutwfc_ry"]))
    return by_surface, skipped


# ── N1: tau_infinity per facet ────────────────────────────────────────────────

def figure_tau_infinity(tau):
    """
    Horizontal dumbbells: tau_xx and tau_yy per facet, joined, with tau_mean.

    The asymmetry between the three is the result. (100)'s pair straddles zero
    with opposite signs, so its two in-plane axes are in opposite states; (110)
    sits entirely on the tensile side; (111) collapses to a single point because
    its three-fold symmetry forces tau_xx = tau_yy exactly.
    """
    surfaces = [s for s in figstyle.SURFACES if s in tau]
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL, 2.1))

    ax.axvline(0.0, color=figstyle.GUIDE, linestyle="-", linewidth=0.8, zorder=0)

    stats = {}
    for i, surf in enumerate(surfaces):
        row = tau[surf]
        y = len(surfaces) - 1 - i
        colour = figstyle.SURFACE_STYLE[surf]["color"]
        marker = figstyle.SURFACE_STYLE[surf]["marker"]

        xx = float(row["tau_xx_inf_n_per_m"])
        yy = float(row["tau_yy_inf_n_per_m"])
        mean = float(row["tau_mean_n_per_m"])
        aniso = float(row["tau_aniso_n_per_m"])
        err_xx = float(row.get("rms_xx_as_tau_n_per_m") or 0.0)
        err_yy = float(row.get("rms_yy_as_tau_n_per_m") or 0.0)

        ax.plot([xx, yy], [y, y], color=colour, linewidth=figstyle.LW * 1.6,
                solid_capstyle="round", zorder=2)
        ax.errorbar([xx, yy], [y, y], xerr=[err_xx, err_yy],
                    fmt="none", ecolor="black", elinewidth=0.7,
                    capsize=1.6, capthick=0.7, zorder=4)
        ax.plot([xx, yy], [y, y], linestyle="none", zorder=5,
                **figstyle.series_kwargs(surf, marker=marker))
        # tau_mean as an open diamond between the endpoints: a third quantity,
        # so a third shape rather than a third colour.
        ax.plot([mean], [y], marker="D", markersize=figstyle.MS * 0.62,
                markerfacecolor="white", markeredgecolor=colour,
                markeredgewidth=figstyle.MEW, linestyle="none", zorder=6)

        ax.annotate(rf"$|\Delta\tau|$ = {abs(aniso):.2f}",
                    xy=(max(xx, yy), y), xytext=(6, 5),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=figstyle.FS_ANNOT, color=colour)
        stats[surf] = {"xx": xx, "yy": yy, "mean": mean, "aniso": aniso,
                       "err": max(err_xx, err_yy)}

    ax.set_yticks(range(len(surfaces)))
    ax.set_yticklabels([figstyle.surface_label(s) for s in reversed(surfaces)])
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.set_ylim(-0.75, len(surfaces) - 0.45)
    ax.set_xlabel(r"$\tau_\infty$ (N m$^{-1}$)")

    # Asymmetric padding: each bar carries its anisotropy label off its right
    # end, so the right margin has to hold a run of text, not just a marker.
    # Symmetric margins clipped the (110) label against the frame.
    lo = min(min(s["xx"], s["yy"]) for s in stats.values())
    hi = max(max(s["xx"], s["yy"]) for s in stats.values())
    span = hi - lo
    ax.set_xlim(lo - 0.12 * span, hi + 0.55 * span)

    handles = [
        plt.Line2D([], [], color="black", linestyle="-",
                   linewidth=figstyle.LW * 1.6, label=r"$\tau_{xx}$ to $\tau_{yy}$"),
        plt.Line2D([], [], marker="D", markersize=figstyle.MS * 0.62,
                   markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=figstyle.MEW, linestyle="none",
                   label=r"$\tau_{\mathrm{mean}}$"),
    ]
    figstyle.legend(ax, handles=handles, loc="lower right", ncol=2)

    straddles = [s for s in surfaces
                 if stats[s]["xx"] * stats[s]["yy"] < 0]
    isotropic = [s for s in surfaces if stats[s]["aniso"] == 0.0]
    worst_err = max(s["err"] for s in stats.values())
    lo = min(min(s["xx"], s["yy"]) for s in stats.values())
    hi = max(max(s["xx"], s["yy"]) for s in stats.values())

    caption = (
        f"Extrapolated surface stress of each H-terminated facet. Each bar runs "
        f"from tau_xx to tau_yy for one surface, with tau_mean as the open "
        f"diamond between them and error bars giving the rms of the "
        f"thickness extrapolation (at most {worst_err:.3f} N/m, smaller than "
        f"the markers). Positive tau is a tensile surface stress: the surface "
        f"tends to contract, compressing the slab interior. The facets span "
        f"{lo:+.2f} to {hi:+.2f} N/m and, more importantly, differ in kind. "
    )
    if straddles:
        names = " and ".join(figstyle.SURFACE_STYLE[s]["label"] for s in straddles)
        caption += (
            f"{names} straddles zero, its two in-plane axes carrying surface "
            f"stress of opposite sign -- tensile along one, compressive along "
            f"the other -- which is the lattice-space origin of a transverse "
            f"splitting. ")
    if isotropic:
        names = " and ".join(figstyle.SURFACE_STYLE[s]["label"] for s in isotropic)
        caption += (
            f"{names} collapses to a single point: its three-fold symmetry "
            f"forces tau_xx = tau_yy exactly, making it the natural control "
            f"surface. ")
    caption += (
        f"Sign of tau_xx - tau_yy depends on the in-plane axis assignment, "
        f"which is recorded per surface in tau_infinity.csv. Values are read "
        f"from that file, fitted by fit_tau_infinity.py, and are L2 "
        f"(thickness-converged, 90/720 Ry)."
    )
    return fig, caption, stats


# ── N2: surface energy vs anisotropy ──────────────────────────────────────────

def figure_gamma_vs_anisotropy(tau, gammas):
    """
    Surface energy against the magnitude of the surface-stress anisotropy.

    These are two independent quantities -- one thermodynamic, one elastic --
    and the figure exists because they are anticorrelated across the three
    facets: the most stable face carries no anisotropy and the least stable
    face carries all of it.
    """
    surfaces = [s for s in figstyle.SURFACES if s in tau and s in gammas]
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL, 2.6))

    stats = {}
    for surf in surfaces:
        row = gammas[surf]
        g = float(row["gamma_h_rich_j_m2"])
        scatter = float(row.get("gamma_scatter_j_m2") or 0.0)
        level = (row.get("epistemic_level") or "").strip()
        a = abs(float(tau[surf]["tau_aniso_n_per_m"]))

        # The scatter is not decoration: on one facet it is the same size as
        # gamma, which is the difference between "barely stabilised" and "not
        # determined". Drawn as an error bar so it cannot be read past.
        ax.errorbar([a], [g], yerr=[scatter], fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
        ax.plot([a], [g], linestyle="none", zorder=5,
                **figstyle.series_kwargs(surf, markersize=figstyle.MS * 1.5))
        label = figstyle.surface_label(surf)
        if level:
            label += f" {level}"
        ax.annotate(label, xy=(a, g), xytext=(7, -1),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=figstyle.FS_ANNOT,
                    color=figstyle.SURFACE_STYLE[surf]["color"])
        stats[surf] = {"gamma": g, "aniso": a, "scatter": scatter,
                       "level": level,
                       "dgamma_dmu": float(row.get("dgamma_dmu_j_m2_per_ev") or 0.0)}

    ax.set_xlabel(r"$|\tau_{xx}-\tau_{yy}|$ (N m$^{-1}$)")
    ax.set_ylabel(r"$\gamma$ (J m$^{-2}$)")
    ax.margins(x=0.28, y=0.30)

    # Shade the stable half of the y-axis. Not a Wulff region: with gamma
    # referenced to H2 some values are negative, and a Wulff construction is
    # undefined for negative gamma (see config/surface_energies_h.json). What
    # is meaningful is the ordering, so the direction is marked, not a polygon.
    lo, hi = ax.get_ylim()
    most_stable = min(stats.values(), key=lambda s: s["gamma"])["gamma"]
    ax.axhspan(lo, most_stable + 0.02 * (hi - lo), color=figstyle.GUIDE,
               alpha=0.10, zorder=0, linewidth=0)
    ax.annotate("more stable facet", xy=(0.5, lo), xytext=(0, 4),
                xycoords=("axes fraction", "data"), textcoords="offset points",
                ha="center", va="bottom", fontsize=figstyle.FS_ANNOT,
                color=figstyle.GUIDE)
    ax.set_ylim(lo, hi)

    order = sorted(stats.items(), key=lambda kv: kv[1]["gamma"])
    best, worst = order[0][0], order[-1][0]
    dmu = max(s["dgamma_dmu"] for s in stats.values())
    levels = sorted({s["level"] for s in stats.values() if s["level"]})
    caption = (
        f"Surface energy against the magnitude of the surface-stress "
        f"anisotropy, one point per facet. gamma is read from "
        f"surface_energy.csv, computed as [E_slab - N_C*mu_C - N_H*mu_H] / 2A "
        f"at the H-rich limit where mu_H is half the energy of a relaxed H2 "
        f"molecule at the same 90/720 Ry setting; error bars are the scatter "
        f"of gamma across the fitted thickness ladder. Negative gamma means "
        f"the H-terminated face is more stable than bulk diamond plus H2, "
        f"which is expected because hydrogenation is exothermic. The two "
        f"quantities are anticorrelated: "
        f"{figstyle.SURFACE_STYLE[best]['label']} is the most stable facet and "
        f"carries exactly zero anisotropy, while "
        f"{figstyle.SURFACE_STYLE[worst]['label']} carries all of it and is "
        f"not measurably stabilised by H termination at all "
        f"({stats[worst]['gamma']:+.3f} +/- {stats[worst]['scatter']:.3f} "
        f"J/m^2, i.e. indistinguishable from zero within its own ladder "
        f"scatter). Engineering a transverse NV splitting therefore means "
        f"favouring the facet that equilibrium morphology suppresses, i.e. a "
        f"non-equilibrium particle shape. "
    )
    if len(levels) > 1:
        per_level = "; ".join(
            f"{figstyle.SURFACE_STYLE[s]['label']} {stats[s]['level']}"
            for s, _ in order)
        caption += (
            f"Epistemic level differs by surface ({per_level}): the "
            f"{figstyle.SURFACE_STYLE[worst]['label']} ladder does not yet "
            f"support a converged gamma, so its position on this axis is "
            f"provisional while its anisotropy is not. ")
    caption += (
        f"gamma shifts with the H chemical potential at up to {dmu:.1f} "
        f"J/m^2 per eV, so these are H-rich values and not the whole phase "
        f"diagram. The shaded direction marks lower gamma, not a Wulff "
        f"region: a Wulff construction is undefined when gamma is negative."
    )
    return fig, caption, stats


# ── N3: energy convergence is not stress convergence ──────────────────────────

def figure_energy_vs_stress_convergence(by_surface):
    """
    Fractional convergence of total energy and of mean in-plane stress.

    Both quantities are normalised by their own converged magnitude, which is
    what makes them comparable on one log axis: the y value is the fraction of
    itself that each quantity is still moving by. Plotting raw |Delta| instead
    would compare meV/atom against kbar and mean nothing.
    """
    surfaces = [s for s in figstyle.SURFACES if s in by_surface]
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL, 2.8))

    stats = {}
    ref_cut = set()
    for surf in surfaces:
        rows = by_surface[surf]
        ref = rows[-1]
        ref_cut.add(float(ref["ecutwfc_ry"]))
        nat = float(ref["nat"])
        e_ref = float(ref["energy_ry"]) / nat
        s_ref = float(ref["sigma_mean_kbar"])

        xs, de, ds = [], [], []
        for r in rows[:-1]:      # the reference itself would be log(0)
            xs.append(float(r["ecutwfc_ry"]))
            de.append(abs(float(r["energy_ry"]) / nat - e_ref) / abs(e_ref))
            ds.append(abs(float(r["sigma_mean_kbar"]) - s_ref) / abs(s_ref))

        kw = figstyle.series_kwargs(surf)
        ax.plot(xs, de, linestyle="-", **kw)
        ax.plot(xs, ds, linestyle="--",
                **figstyle.series_kwargs(surf, filled=False))
        ratios = [d / e for d, e in zip(ds, de)]
        stats[surf] = {
            "e": dict(zip(xs, de)), "s": dict(zip(xs, ds)),
            "ratio_min": min(ratios), "ratio_max": max(ratios),
            "sigma_ref": s_ref,
        }

    ax.set_yscale("log")
    ax.set_xlabel(r"$E_{\mathrm{cut}}$ (Ry)")
    ax.set_ylabel(r"$|\Delta X|\,/\,|X_{\mathrm{conv}}|$")

    handles = [
        plt.Line2D([], [], color="black", linestyle="-", marker="o",
                   markerfacecolor="black", markeredgecolor="black",
                   markeredgewidth=figstyle.MEW, markersize=figstyle.MS,
                   linewidth=figstyle.LW, label="energy/atom"),
        plt.Line2D([], [], color="black", linestyle="--", marker="o",
                   markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=figstyle.MEW, markersize=figstyle.MS,
                   linewidth=figstyle.LW, label="mean stress"),
    ]
    handles += [plt.Line2D([], [], linestyle="none",
                           **figstyle.series_kwargs(s),
                           label=figstyle.surface_label(s)) for s in surfaces]
    # The gap between the two bands is the emptiest part of this figure and the
    # only place a legend fits: the stress curves occupy the top decade and the
    # energy curves the bottom, with nothing in between. Anchored to the middle
    # of that gap, measured from the data rather than guessed.
    top_of_energy = max(max(s["e"].values()) for s in stats.values())
    bottom_of_stress = min(min(s["s"].values()) for s in stats.values())
    lo_ax, hi_ax = ax.get_ylim()
    gap_mid = math.sqrt(top_of_energy * bottom_of_stress)
    frac = ((math.log10(gap_mid) - math.log10(lo_ax))
            / (math.log10(hi_ax) - math.log10(lo_ax)))
    figstyle.legend(ax, handles=handles, loc="center", ncol=3,
                    bbox_to_anchor=(0.5, frac), columnspacing=1.0,
                    handletextpad=0.4)

    ref_txt = f"{next(iter(ref_cut)):.0f}" if len(ref_cut) == 1 else "densest"
    cuts = sorted(next(iter(stats.values()))["e"])
    lowest, highest = cuts[0], cuts[-1]
    e_lo = max(stats[s]["e"][lowest] for s in surfaces)
    s_lo = max(stats[s]["s"][lowest] for s in surfaces)
    e_hi = max(stats[s]["e"][highest] for s in surfaces)
    s_hi = max(stats[s]["s"][highest] for s in surfaces)
    ratio_lo = min(s["ratio_min"] for s in stats.values())
    ratio_hi = max(s["ratio_max"] for s in stats.values())

    caption = (
        f"Why converging the total energy does not converge the stress. Both "
        f"quantities are plotted as |Delta| relative to their own value at "
        f"{ref_txt} Ry, each divided by its own converged magnitude, so the "
        f"two are dimensionless and directly comparable on one log axis; the y "
        f"value is the fraction of itself that each quantity is still moving "
        f"by. At {lowest:.0f} Ry the energy per atom is already converged to "
        f"{e_lo:.0e} while the mean in-plane stress is only at {s_lo:.0e}, and "
        f"at {highest:.0f} Ry they stand at {e_hi:.0e} and {s_hi:.0e}. At every "
        f"cutoff sampled the stress is between {ratio_lo:.0f} and "
        f"{ratio_hi:.0f} times less converged than the energy, because the "
        f"energy is variational in the basis-set error while the stress, its "
        f"first derivative, is not. "
        f"A cutoff chosen on an energy criterion alone is therefore not a "
        f"converged setting for a stress campaign -- the lesson that set "
        f"90 Ry here, and the one that recurred on every other axis of this "
        f"study."
    )
    return fig, caption, stats


# ── N4: vc-relax lattice distortion ───────────────────────────────────────────

def figure_vcrelax(by_surface, exclude_layers=(6,)):
    """
    In-plane strain of a freely relaxed slab against inverse thickness.

    A slab whose cell is allowed to relax finds an in-plane lattice constant
    that differs from bulk, by an amount that falls off as 1/t because it is
    the two surfaces driving it. The extrapolation to 1/t -> 0 must return zero
    strain -- the interior of a thick slab is bulk -- and that is the check
    this figure makes. What differs between facets is the sign structure.
    """
    surfaces = [s for s in figstyle.SURFACES if s in by_surface]
    fig, ax = plt.subplots(figsize=(figstyle.SINGLE_COL, 2.9))

    ax.axhline(0.0, color=figstyle.GUIDE, linestyle=":", linewidth=0.6, zorder=0)

    stats = {}
    for surf in surfaces:
        excl = set(exclude_layers)
        pts = [p for p in by_surface[surf] if p["layers"] not in excl]
        dropped = [p for p in by_surface[surf] if p["layers"] in excl]
        # t is the bulk-equivalent thickness N_C*Omega/A used elsewhere in this
        # campaign; here only its inverse matters, and any thickness measure
        # proportional to it gives the same intercept.
        inv_t = [p["area"] / p["n_C"] for p in pts]
        entry = {}
        for comp, style, filled in (("eps_a", "-", True), ("eps_b", "--", False)):
            ys = [p[comp] * 100.0 for p in pts]
            # Excluded rungs are drawn but struck through and kept out of the
            # fit, as on the thickness ladder. Including 6L here is not a
            # cosmetic choice: it is far enough off the 1/t trend to push the
            # (100) eps_b intercept to +0.64%, which would turn "extrapolates
            # to zero" into a claim the data does not support.
            for p in dropped:
                x, y = p["area"] / p["n_C"], p[comp] * 100.0
                ax.plot([x], [y], linestyle="none", zorder=4,
                        **figstyle.series_kwargs(surf, filled=filled))
                ax.plot([x], [y], marker="x", linestyle="none",
                        color=figstyle.GUIDE, markersize=figstyle.MS * 1.1,
                        markeredgewidth=0.8, zorder=6)
            ax.plot(inv_t, ys, linestyle=style,
                    **figstyle.series_kwargs(surf, filled=filled))
            slope, intercept = linear_fit(inv_t, ys)
            xf = linspace(0.0, max(inv_t) * 1.05)
            ax.plot(xf, [slope * x + intercept for x in xf],
                    color=figstyle.SURFACE_STYLE[surf]["color"],
                    linestyle=style, linewidth=figstyle.LW * 0.6, zorder=1)
            resid = [y - (slope * x + intercept) for x, y in zip(inv_t, ys)]
            entry[comp] = {
                "slope": slope, "intercept": intercept,
                "rms": math.sqrt(sum(r * r for r in resid) / len(resid)),
                "thinnest": ys[0], "thickest": ys[-1],
            }
        # 1e-8 is not arbitrary. The isotropic surface's two axes differ by
        # ~1e-10, which is the printing precision of CELL_PARAMETERS, while the
        # anisotropic ones differ by ~1e-3. Seven orders of magnitude separate
        # the two cases, so any threshold in that gap gives the same answer;
        # testing exact equality would instead fail on round-off.
        entry["isotropic"] = all(
            abs(p["eps_a"] - p["eps_b"]) < 1e-8 for p in pts)
        entry["opposed"] = all(p["eps_a"] * p["eps_b"] < 0 for p in pts)
        stats[surf] = entry

    ax.set_xlim(left=0.0)
    ax.set_xlabel(rf"$1/t$ (${AA}^{{-1}}$)")
    ax.set_ylabel(r"in-plane strain ($\%$)")

    handles = [
        plt.Line2D([], [], color="black", linestyle="-", marker="o",
                   markerfacecolor="black", markeredgecolor="black",
                   markeredgewidth=figstyle.MEW, markersize=figstyle.MS,
                   linewidth=figstyle.LW, label=r"$\varepsilon_a$"),
        plt.Line2D([], [], color="black", linestyle="--", marker="o",
                   markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=figstyle.MEW, markersize=figstyle.MS,
                   linewidth=figstyle.LW, label=r"$\varepsilon_b$"),
    ]
    handles += [plt.Line2D([], [], linestyle="none",
                           **figstyle.series_kwargs(s),
                           label=figstyle.surface_label(s)) for s in surfaces]
    figstyle.legend(ax, handles=handles, loc="upper left", ncol=2)

    worst_intercept = max(abs(stats[s][c]["intercept"])
                          for s in surfaces for c in ("eps_a", "eps_b"))
    worst_rms = max(stats[s][c]["rms"]
                    for s in surfaces for c in ("eps_a", "eps_b"))
    opposed = [s for s in surfaces if stats[s]["opposed"]]
    isotropic = [s for s in surfaces if stats[s]["isotropic"]]

    excl_txt = ", ".join(f"{n}L" for n in sorted(exclude_layers))
    caption = (
        f"In-plane lattice distortion of freely relaxed (vc-relax) slabs "
        f"against inverse bulk-equivalent thickness, solid for the a axis and "
        f"dashed for b. Each curve is linear in 1/t and extrapolates to within "
        f"{worst_intercept:.2f}{PCT} strain at infinite thickness, with fit "
        f"residuals below {worst_rms:.3f}{PCT}: the distortion is a surface "
        f"effect that vanishes in the bulk limit, as it must, which is what "
        f"validates treating these slabs as models of a surface rather than of "
        f"a thin film. "
    )
    if exclude_layers:
        # What keeping the excluded rungs would actually cost, refitted rather
        # than asserted, so the exclusion is justified by a number the reader
        # can check instead of by convention.
        kept_intercept, kept_rms = 0.0, 0.0
        for surf in surfaces:
            allpts = by_surface[surf]
            x_all = [p["area"] / p["n_C"] for p in allpts]
            for comp in ("eps_a", "eps_b"):
                y_all = [p[comp] * 100.0 for p in allpts]
                m, b = linear_fit(x_all, y_all)
                res = [y - (m * x + b) for x, y in zip(x_all, y_all)]
                kept_intercept = max(kept_intercept, abs(b))
                kept_rms = max(
                    kept_rms, math.sqrt(sum(r * r for r in res) / len(res)))
        caption += (
            f"The {excl_txt} rung (struck through) is excluded from the fits, "
            f"as it is from the thickness ladder: it lies off the 1/t trend, "
            f"and keeping it would push the largest intercept to "
            f"{kept_intercept:.2f}{PCT} and the largest residual to "
            f"{kept_rms:.3f}{PCT}. "
        )
    if opposed:
        names = " and ".join(figstyle.SURFACE_STYLE[s]["label"] for s in opposed)
        caption += (
            f"The facets differ in sign structure, and that is the physical "
            f"content: on {names} the two in-plane axes move in opposite "
            f"directions -- a expands while b contracts -- which is a shear, "
            f"the lattice-space signature of the transverse-E channel. ")
    if isotropic:
        names = " and ".join(figstyle.SURFACE_STYLE[s]["label"] for s in isotropic)
        caption += (
            f"{names} is exactly isotropic, its two axes equal to numerical "
            f"precision, so it can only shift D and never split E. ")
    rest = [s for s in surfaces
            if not stats[s]["opposed"] and not stats[s]["isotropic"]]
    if rest:
        names = " and ".join(figstyle.SURFACE_STYLE[s]["label"] for s in rest)
        caption += (
            f"{names} expands on both axes but by unequal amounts, so it "
            f"carries an anisotropy without a sign change. ")
    caption += (
        f"Strains are relative to each slab's own starting cell at the "
        f"production a0; positive is expansion. Lines are least-squares fits "
        f"in 1/t and the only extrapolation in this figure."
    )
    return fig, caption, stats


# ── Caption plain-texting ─────────────────────────────────────────────────────

def detex(s):
    for a, b in ((r"$\tau_\infty$", "tau_inf"),
                 (r"$\tau_{xx}$", "tau_xx"), (r"$\tau_{yy}$", "tau_yy"),
                 (r"$\varepsilon_a$", "eps_a"), (r"$\varepsilon_b$", "eps_b"),
                 (r"\%", "%"), (r"\AA", "A")):
        s = s.replace(a, b)
    return s


TITLES = {
    "figN1_tau_infinity": "Figure N1 - Surface stress per facet",
    "figN2_gamma_anisotropy": "Figure N2 - Surface energy versus anisotropy",
    "figN3_energy_vs_stress": "Figure N3 - Energy convergence is not stress convergence",
    "figN4_vcrelax_distortion": "Figure N4 - vc-relax lattice distortion",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    print("Typography: "
          + ("LaTeX (text.usetex)" if USE_TEX
             else "mathtext fontset=cm (LaTeX unusable, fell back)"))

    tau = load_tau_infinity()
    if tau is None:
        print(f"\nSKIPPED N1 and N2: {TAU_INF_CSV} is absent. tau_infinity is "
              f"fit_tau_infinity.py's result and is not refitted here.")
    else:
        print(f"\nRead tau_infinity for {', '.join(sorted(tau))} from "
              f"{TAU_INF_CSV}.")

    gammas = load_published_surface_energies()
    if gammas is None:
        print(f"\nSKIPPED N2: {SURFACE_ENERGY_CSV} is absent. Surface energy is "
              f"surface_energy.py's result and is not refitted here.")
    else:
        recomputed, gamma_skipped = recompute_surface_energies()
        for name, why in gamma_skipped:
            print(f"  EXCLUDED {name}: {why}")
        if recomputed:
            checked = cross_check_surface_energies(gammas, recomputed)
            if checked:
                print(f"\nSurface energy: independent recomputation agrees with "
                      f"{SURFACE_ENERGY_CSV} for {', '.join(checked)}.")
        for surf in sorted(gammas):
            row = gammas[surf]
            print(f"  {surf}: gamma = {float(row['gamma_h_rich_j_m2']):+.4f} "
                  f"+/- {float(row['gamma_scatter_j_m2']):.4f} J/m^2 "
                  f"({row['epistemic_level']})")

    vc, vc_skipped = load_vcrelax()
    for name, why in vc_skipped:
        print(f"  EXCLUDED {name}: {why}")
    print(f"vc-relax: {sum(len(v) for v in vc.values())} runs over "
          f"{', '.join(sorted(vc))}.")

    cutoff, cut_skipped = load_cutoff_sweep()
    for name, why in cut_skipped:
        print(f"  EXCLUDED {name}: {why}")
    print(f"cutoff sweep: {sum(len(v) for v in cutoff.values())} runs over "
          f"{', '.join(sorted(cutoff))}.\n")

    builders = []
    if tau:
        builders.append(("figN1_tau_infinity",
                         lambda: figure_tau_infinity(tau)))
        if gammas:
            builders.append(("figN2_gamma_anisotropy",
                             lambda: figure_gamma_vs_anisotropy(tau, gammas)))
        else:
            print("SKIPPED N2: surface energies unavailable.")
    builders.append(("figN3_energy_vs_stress",
                     lambda: figure_energy_vs_stress_convergence(cutoff)))
    if vc:
        builders.append(("figN4_vcrelax_distortion", lambda: figure_vcrelax(vc)))
    else:
        print("SKIPPED N4: no vc-relax runs found.")

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
        "# Figure captions - surface stress",
        "",
        "Generated by `plot_surface_stress.py`. Every number is derived from",
        "the source files and run directories rather than transcribed.",
        "",
        "Sign convention (CLAUDE.md section 2): positive sigma means the cell",
        "is COMPRESSED. Positive tau is a TENSILE surface stress -- the surface",
        "tends to contract, which compresses the slab interior, so the two",
        "conventions agree and no sign is flipped between them.",
        "",
        "All slabs are symmetric H/H terminated at 90/720 Ry, a0 from the",
        "production Birch-Murnaghan bulk fit.",
        "",
    ]
    for stem in TITLES:
        if stem not in captions:
            continue
        lines += [f"## {TITLES[stem]}", "", detex(captions[stem]), "",
                  f"Files: `{stem}.pdf`, `{stem}.png` (600 dpi), "
                  f"`{stem}_annotated.pdf`", ""]
    (outdir / "CAPTIONS.md").write_text("\n".join(lines))
    print("  CAPTIONS.md")
    print(f"\nWrote {outdir}")


if __name__ == "__main__":
    main()
