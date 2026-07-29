#!/usr/bin/env python3
"""
fit_slab_strain.py - Fit slab in-plane strain series.

Reads results/slabs/slab_stress_summary.csv and fits both the raw QE stress
sigma (kbar, cell-averaged over slab + vacuum) and the vacuum-corrected 2D
surface stress tau (N/m, already computed per-row by analyze_slab_stress.py)
against in-plane strain:

    mean_sigma_kbar(eps) = m_sigma * eps + b_sigma
    tau_mean_N_per_m(eps) = m_tau * eps + b_tau

tau is the physically intrinsic surface quantity (vacuum-thickness
independent); sigma is retained as a diagnostic cross-check. Since
tau = sigma * Lz * 0.005 pointwise with Lz fixed within a strain series, the
two fits' zero-stress crossings (zero_stress_epsilon, zero_tau_strain) must
agree to numerical precision -- this is checked explicitly and a warning is
raised on disagreement beyond --zero-strain-tolerance.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

RY_TO_EV = 13.605693122994

FIELDS = [
    "series",
    "orientation",
    "termination",
    "formula",
    "strain_mode",
    "n_points",
    "epsilon_min",
    "epsilon_max",
    "stress_slope_kbar_per_eps",
    "stress_intercept_kbar",
    "zero_stress_epsilon",
    "stress_at_zero_kbar",
    "tau_slope_n_per_m_per_eps",
    "tau_intercept_n_per_m",
    "tau_at_zero_n_per_m",
    "r2_stress_fit",
    "n_tau_points",
    "zero_tau_strain",
    "r2_tau_fit",
    "zero_strain_discrepancy",
    "fit_status",
    "warnings",
]

DEFAULT_ZERO_STRAIN_TOLERANCE = 1e-10


def ffloat(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def infer_strain_from_name(name):
    m = re.search(r"_strain_([A-Za-z]+)_([mp])(\d+)p(\d+)_scf$", name)
    if not m:
        return None, None
    mode = m.group(1)
    sign = -1.0 if m.group(2) == "m" else 1.0
    whole = m.group(3)
    frac = m.group(4)
    eps = sign * float(f"{whole}.{frac}")
    return mode, eps


def series_name(name):
    return re.sub(r"_strain_[A-Za-z]+_[mp]\d+p\d+_scf$", "", name)


def linear_fit(points):
    # points: [(x, y)]
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    den = n * sxx - sx * sx
    if abs(den) < 1e-20:
        return None
    m = (n * sxy - sx * sy) / den
    b = (sy - m * sx) / n

    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for _, y in points)
    ss_res = sum((y - (m * x + b)) ** 2 for x, y in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return m, b, r2


def build_groups(rows):
    groups = {}
    for r in rows:
        name = r.get("folder_name", "")
        if "_strain_" not in name:
            continue
        mode, eps = infer_strain_from_name(name)
        if eps is None:
            continue
        stress = ffloat(r.get("inplane_mean_stress_kbar"))
        tau = ffloat(r.get("tau_mean_n_per_m"))
        if stress is None:
            continue
        key = (series_name(name), mode)
        groups.setdefault(key, []).append((eps, stress, tau, r))
    return groups


def analyze_group(key, items, zero_strain_tolerance=DEFAULT_ZERO_STRAIN_TOLERANCE):
    series, mode = key
    items = sorted(items, key=lambda x: x[0])
    warnings = []
    stress_points = [(eps, stress) for eps, stress, _tau, _r in items]
    tau_points = [(eps, tau) for eps, _stress, tau, _r in items if tau is not None]

    fit = linear_fit(stress_points)
    if fit is None:
        return None
    m, b, r2 = fit
    eps0 = -b / m if abs(m) > 1e-20 else None

    tau_fit = linear_fit(tau_points) if len(tau_points) >= 2 else None
    if tau_fit:
        tm, tb, tr2 = tau_fit
        zero_tau_strain = -tb / tm if abs(tm) > 1e-20 else None
    else:
        tm, tb, tr2, zero_tau_strain = None, None, None, None

    discrepancy = None
    if eps0 is not None and zero_tau_strain is not None:
        discrepancy = abs(eps0 - zero_tau_strain)
        if discrepancy > zero_strain_tolerance:
            warnings.append(
                f"zero_strain_mismatch_sigma_vs_tau: "
                f"zero_sigma={eps0:.10g} zero_tau={zero_tau_strain:.10g} "
                f"diff={discrepancy:.3g} > tol={zero_strain_tolerance:.1g}")

    if len(items) < 5:
        warnings.append("fewer_than_5_points")
    if r2 is not None and r2 < 0.98:
        warnings.append("stress_fit_not_highly_linear")
    if eps0 is not None and not (min(e for e, *_ in items) <= eps0 <= max(e for e, *_ in items)):
        warnings.append("zero_stress_outside_sampled_range")

    ref = items[0][3]
    status = "good" if not warnings else "warning"

    return {
        "series": series,
        "orientation": ref.get("orientation"),
        "termination": ref.get("termination"),
        "formula": ref.get("formula"),
        "strain_mode": mode,
        "n_points": len(items),
        "epsilon_min": min(e for e, *_ in items),
        "epsilon_max": max(e for e, *_ in items),
        "stress_slope_kbar_per_eps": m,
        "stress_intercept_kbar": b,
        "zero_stress_epsilon": eps0,
        "stress_at_zero_kbar": b,
        "tau_slope_n_per_m_per_eps": tm,
        "tau_intercept_n_per_m": tb,
        "tau_at_zero_n_per_m": tb,
        "r2_stress_fit": r2,
        "n_tau_points": len(tau_points),
        "zero_tau_strain": zero_tau_strain,
        "r2_tau_fit": tr2,
        "zero_strain_discrepancy": discrepancy,
        "fit_status": status,
        "warnings": "; ".join(warnings),
    }


def write_csv(rows, path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})


def fmt(x, nd=6):
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def write_md(rows, path):
    lines = []
    lines.append("# Slab Strain Fit Summary")
    lines.append("")
    lines.append("Fits the slab's in-plane strain response from strain-series SCF calculations. "
                 "The primary result is the vacuum-corrected 2D surface stress tau (N/m); the raw "
                 "QE cell-averaged stress sigma (kbar) is retained as a diagnostic cross-check. "
                 "Both are strain-linear fits of the same underlying data, so their zero-stress "
                 "strains must agree (see Consistency Check below); tau changes units and physical "
                 "interpretation, not the equilibrium strain.")
    lines.append("")
    lines.append("## Surface stress fit (primary)")
    lines.append("")
    lines.append("```text")
    lines.append("tau_mean_N_per_m(epsilon) = m_tau * epsilon + b_tau")
    lines.append("```")
    lines.append("")
    lines.append("| Series | Orient. | Mode | Points | tau slope (N/m/strain) | tau at zero (N/m) | Zero-tau eps | R^2 | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['series']} | {r.get('orientation') or ''} | {r['strain_mode']} | "
            f"{r['n_tau_points']} | {fmt(r['tau_slope_n_per_m_per_eps'], 4)} | "
            f"{fmt(r['tau_at_zero_n_per_m'], 5)} | {fmt(r['zero_tau_strain'], 6)} | "
            f"{fmt(r['r2_tau_fit'], 5)} | {r['fit_status']} |"
        )
    lines.append("")
    lines.append("## Diagnostic QE stress fit (sigma, vacuum-dependent)")
    lines.append("")
    lines.append("```text")
    lines.append("mean_sigma_kbar(epsilon) = m_sigma * epsilon + b_sigma")
    lines.append("```")
    lines.append("")
    lines.append("| Series | Orient. | Mode | sigma slope (kbar/strain) | sigma at zero (kbar) | Zero-sigma eps | R^2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['series']} | {r.get('orientation') or ''} | {r['strain_mode']} | "
            f"{fmt(r['stress_slope_kbar_per_eps'], 2)} | {fmt(r['stress_at_zero_kbar'], 3)} | "
            f"{fmt(r['zero_stress_epsilon'], 6)} | {fmt(r['r2_stress_fit'], 5)} |"
        )
    lines.append("")
    lines.append("## Consistency check: zero-sigma vs zero-tau strain")
    lines.append("")
    mismatches = [r for r in rows if "zero_strain_mismatch_sigma_vs_tau" in (r.get("warnings") or "")]
    if mismatches:
        lines.append(f"**{len(mismatches)} series show a zero-strain mismatch beyond tolerance** "
                     "-- see their `warnings` field; this indicates an implementation error or "
                     "inconsistent cell heights (Lz) within the series, not a physical effect.")
    else:
        lines.append("All series: `zero_stress_epsilon` (sigma) and `zero_tau_strain` (tau) agree "
                     "within tolerance, as expected since tau = sigma * Lz * 0.005 pointwise with "
                     "Lz fixed within each strain series.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `tau_at_zero_n_per_m` / `stress_at_zero_kbar` is the residual in-plane surface "
                 "stress / QE stress at the relaxed generated geometry (epsilon = 0).")
    lines.append("- `zero_tau_strain` / `zero_stress_epsilon` estimate the in-plane strain that "
                 "would null the surface stress / QE stress; these are the same physical strain "
                 "expressed via two equivalent linear fits.")
    lines.append("- tau (N/m) is vacuum-thickness independent and directly comparable to surface-"
                 "stress literature; sigma (kbar) depends on the arbitrary supercell vacuum and is "
                 "retained here only as a diagnostic cross-check.")
    lines.append("- Large slopes indicate a stiff slab response to imposed in-plane strain.")
    lines.append("- This fit is a more reliable diagnostic than a single stress-SCF point, but final "
                 "surface stress should also consider ionic relaxation under strain.")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/slabs/slab_stress_summary.csv")
    ap.add_argument("--outdir", default="results/slabs")
    ap.add_argument("--zero-strain-tolerance", type=float,
                     default=DEFAULT_ZERO_STRAIN_TOLERANCE,
                     help="Max allowed |zero_stress_epsilon - zero_tau_strain| "
                          "before warning (default: 1e-10)")
    args = ap.parse_args()

    with Path(args.input).open() as f:
        rows = list(csv.DictReader(f))

    groups = build_groups(rows)
    fits = []
    for key, items in sorted(groups.items()):
        fit = analyze_group(key, items, args.zero_strain_tolerance)
        if fit:
            fits.append(fit)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "slab_strain_fit_summary.csv"
    json_path = outdir / "slab_strain_fit_summary.json"
    md_path = outdir / "slab_strain_fit_summary.md"

    write_csv(fits, csv_path)
    json_path.write_text(json.dumps(fits, indent=2, sort_keys=True))
    write_md(fits, md_path)

    print(f"Fitted strain series: {len(fits)}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
