#!/usr/bin/env python3
"""
fit_slab_strain.py - Fit slab in-plane strain series.

Reads results/slabs/slab_stress_summary.csv and fits stress-vs-strain for
biaxial slab strain series. This is the first quantitative route from surface
functionalization/orientation to lattice-pressure-like Raman relevance.

For each group, fit:

    mean_sigma_kbar(eps) = m * eps + b

and report:
    zero-stress strain eps0 = -b/m
    residual mean stress at generated zero strain
    stress slope
    tau_mean slope
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
    "fit_status",
    "warnings",
]


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


def analyze_group(key, items):
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
        tm, tb, _tr2 = tau_fit
    else:
        tm, tb = None, None

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
    lines.append("Fits in-plane slab stress response from strain-series SCF calculations.")
    lines.append("")
    lines.append("Model:")
    lines.append("")
    lines.append("```text")
    lines.append("mean_sigma_kbar(epsilon) = m * epsilon + b")
    lines.append("```")
    lines.append("")
    lines.append("| Series | Orient. | Mode | Points | Stress slope (kbar/strain) | Stress at zero (kbar) | Zero-stress eps | R^2 | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['series']} | {r.get('orientation') or ''} | {r['strain_mode']} | "
            f"{r['n_points']} | {fmt(r['stress_slope_kbar_per_eps'], 2)} | "
            f"{fmt(r['stress_at_zero_kbar'], 3)} | {fmt(r['zero_stress_epsilon'], 6)} | "
            f"{fmt(r['r2_stress_fit'], 5)} | {r['fit_status']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `stress_at_zero_kbar` is the residual in-plane mean stress at the relaxed generated geometry.")
    lines.append("- `zero_stress_epsilon` estimates the in-plane strain that would null the mean slab stress.")
    lines.append("- Large stress slopes indicate a stiff slab response to imposed in-plane strain.")
    lines.append("- This fit is a more reliable stress diagnostic than a single stress-SCF point, but final surface stress should also consider ionic relaxation under strain.")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/slabs/slab_stress_summary.csv")
    ap.add_argument("--outdir", default="results/slabs")
    args = ap.parse_args()

    with Path(args.input).open() as f:
        rows = list(csv.DictReader(f))

    groups = build_groups(rows)
    fits = []
    for key, items in sorted(groups.items()):
        fit = analyze_group(key, items)
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
