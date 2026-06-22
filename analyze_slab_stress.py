#!/usr/bin/env python3
"""
analyze_slab_stress.py - Estimate vacuum-corrected slab surface stress.

Reads results/slabs/slab_summary.csv from parse_slab.py and converts the
cell-averaged QE stress tensor into approximate 2D surface stress:

    tau_ij = sigma_ij * Lz / 2

where Lz is the slab supercell height and 2 accounts for the two slab faces.
This is a first diagnostic for surface-induced lattice pressure/strain. A
proper surface-stress fit should later use explicit in-plane strain series.
"""

import argparse
import csv
import json
import math
from pathlib import Path

KBAR_TO_GPA = 0.1
# 1 GPa * 1 Angstrom = 0.1 N/m
GPA_ANG_TO_N_PER_M = 0.1


FIELDS = [
    "folder_name",
    "orientation",
    "termination",
    "formula",
    "pseudo_consistency",
    "complete",
    "needs_attention",
    "cell_area_angstrom2",
    "cell_height_angstrom",
    "carbon_thickness_angstrom",
    "vacuum_estimate_angstrom",
    "stress_xx_kbar",
    "stress_yy_kbar",
    "stress_zz_kbar",
    "stress_xy_kbar",
    "inplane_mean_stress_kbar",
    "inplane_anisotropy_kbar",
    "tau_xx_n_per_m",
    "tau_yy_n_per_m",
    "tau_xy_n_per_m",
    "tau_mean_n_per_m",
    "tau_anisotropy_n_per_m",
    "qe_pressure_kbar",
    "effective_inplane_pressure_kbar",
    "stress_interpretation",
    "raman_relevance",
]


def ffloat(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def fmt(value, nd=6):
    if value is None:
        return ""
    return f"{value:.{nd}f}"


def analyze_row(row):
    sxx = ffloat(row.get("stress_xx_kbar"))
    syy = ffloat(row.get("stress_yy_kbar"))
    szz = ffloat(row.get("stress_zz_kbar"))
    sxy = ffloat(row.get("stress_xy_kbar"))
    lz = ffloat(row.get("cell_height_angstrom"))

    mean = anis = None
    tau_xx = tau_yy = tau_xy = tau_mean = tau_anis = None
    eff_p = None

    if sxx is not None and syy is not None:
        mean = 0.5 * (sxx + syy)
        anis = sxx - syy
        # QE stress sign is preserved. For pressure-like language, use -mean.
        eff_p = -mean

    # Convert kbar stress to vacuum-corrected 2D stress:
    # kbar * 0.1 GPa/kbar * Angstrom * 0.1 N/m/(GPa Angstrom) / 2 surfaces.
    # tau = stress_kbar * Lz * 0.005
    if lz is not None:
        factor = lz * KBAR_TO_GPA * GPA_ANG_TO_N_PER_M / 2.0
        if sxx is not None:
            tau_xx = sxx * factor
        if syy is not None:
            tau_yy = syy * factor
        if sxy is not None:
            tau_xy = sxy * factor
        if mean is not None:
            tau_mean = mean * factor
        if anis is not None:
            tau_anis = anis * factor

    interpretation = "unknown"
    if eff_p is not None:
        if eff_p > 1.0:
            interpretation = "compressive_inplane_pressure_like"
        elif eff_p < -1.0:
            interpretation = "tensile_inplane_stress_like"
        else:
            interpretation = "near_zero_inplane_mean_stress"

    relevance = "use_for_clean_H_baseline"
    if row.get("pseudo_consistency") != "ok":
        relevance = "exclude_from_comparison_pseudo_mismatch"
    elif not truthy(row.get("complete")):
        relevance = "exclude_incomplete"

    return {
        "folder_name": row.get("folder_name"),
        "orientation": row.get("orientation"),
        "termination": row.get("termination"),
        "formula": row.get("formula"),
        "pseudo_consistency": row.get("pseudo_consistency"),
        "complete": row.get("complete"),
        "needs_attention": row.get("needs_attention"),
        "cell_area_angstrom2": row.get("cell_area_angstrom2"),
        "cell_height_angstrom": row.get("cell_height_angstrom"),
        "carbon_thickness_angstrom": row.get("carbon_thickness_angstrom"),
        "vacuum_estimate_angstrom": row.get("vacuum_estimate_angstrom"),
        "stress_xx_kbar": sxx,
        "stress_yy_kbar": syy,
        "stress_zz_kbar": szz,
        "stress_xy_kbar": sxy,
        "inplane_mean_stress_kbar": mean,
        "inplane_anisotropy_kbar": anis,
        "tau_xx_n_per_m": tau_xx,
        "tau_yy_n_per_m": tau_yy,
        "tau_xy_n_per_m": tau_xy,
        "tau_mean_n_per_m": tau_mean,
        "tau_anisotropy_n_per_m": tau_anis,
        "qe_pressure_kbar": ffloat(row.get("pressure_kbar")),
        "effective_inplane_pressure_kbar": eff_p,
        "stress_interpretation": interpretation,
        "raman_relevance": relevance,
    }


def write_csv(rows, path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in FIELDS})


def write_md(rows, path):
    clean = [r for r in rows if r["raman_relevance"] == "use_for_clean_H_baseline" and r["stress_xx_kbar"] is not None and r["stress_yy_kbar"] is not None]

    lines = []
    lines.append("# Slab Stress Summary")
    lines.append("")
    lines.append("This report estimates vacuum-corrected 2D surface stress from relaxed slab QE stress tensors.")
    lines.append("")
    lines.append("Important caveat: QE slab stress is averaged over the full vacuum-containing supercell. The values below multiply by the cell height and divide by two slab faces, giving a first surface-stress diagnostic. The more rigorous next step is an explicit in-plane strain series.")
    lines.append("")
    lines.append("## Clean H-Terminated Baseline")
    lines.append("")
    lines.append("| Run | Orient. | Formula | sigma_xx (kbar) | sigma_yy (kbar) | mean sigma (kbar) | tau_mean (N/m) | effective pressure (kbar) | Interpretation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in clean:
        lines.append(
            f"| {r['folder_name']} | {r['orientation']} | {r['formula']} | "
            f"{fmt(r['stress_xx_kbar'], 3)} | {fmt(r['stress_yy_kbar'], 3)} | "
            f"{fmt(r['inplane_mean_stress_kbar'], 3)} | {fmt(r['tau_mean_n_per_m'], 4)} | "
            f"{fmt(r['effective_inplane_pressure_kbar'], 3)} | {r['stress_interpretation']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `tau_mean_n_per_m` is the approximate per-surface in-plane stress after correcting for vacuum dilution.")
    lines.append("- `effective_inplane_pressure_kbar = -mean(sigma_xx, sigma_yy)` uses QE stress sign convention to provide pressure-like language.")
    lines.append("- Large positive effective pressure indicates a compressive pressure-like surface contribution; large negative values indicate tensile stress-like behavior.")
    lines.append("- These values are best used to rank orientations and functionalizations before doing explicit strain fits.")
    lines.append("")
    lines.append("## Excluded or Flagged Rows")
    lines.append("")
    lines.append("| Run | Reason |")
    lines.append("|---|---|")
    flagged = [r for r in rows if r["raman_relevance"] != "use_for_clean_H_baseline"]
    if flagged:
        for r in flagged:
            lines.append(f"| {r['folder_name']} | {r['raman_relevance']} |")
    else:
        lines.append("| none | |")

    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append("Generate fixed-cell in-plane strain series for the clean H-terminated slabs and fit energy/stress versus strain to obtain robust surface stress tensors.")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/slabs/slab_summary.csv")
    ap.add_argument("--outdir", default="results/slabs")
    ap.add_argument("--include-flagged", action="store_true")
    args = ap.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with inpath.open() as f:
        rows = [analyze_row(r) for r in csv.DictReader(f)]

    rows.sort(key=lambda r: r["folder_name"])

    csv_path = outdir / "slab_stress_summary.csv"
    json_path = outdir / "slab_stress_summary.json"
    md_path = outdir / "slab_stress_summary.md"

    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    write_md(rows, md_path)

    clean = sum(1 for r in rows if r["raman_relevance"] == "use_for_clean_H_baseline")
    print(f"Parsed slab stress rows: {len(rows)}")
    print(f"Clean H baseline rows: {clean}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
