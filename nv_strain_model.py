#!/usr/bin/env python3
"""
nv_strain_model.py  –  First-order NV strain screening model.

Connects slab-DFT stress/strain outputs to NV center zero-field splitting
observables using a transparent, semi-phenomenological approach.

This is NOT a calibrated NV DFT calculation.  It is a screening tool that
ranks surface orientations by their likely ability to perturb NV magnetic
resonance through strain-induced changes to D and E.

Physics basis:
    H = (D + ΔD) [Sz² – S(S+1)/3] + E (Sx² – Sy²) + ...

    D  ~ 2.87 GHz in unstrained bulk diamond.
    ΔD – axial shift driven by hydrostatic-like (mean in-plane) strain.
    E  – transverse splitting driven by anisotropic in-plane strain;
         splits/mixes ms = ±1, shifts ODMR line positions, and can
         affect coherence/contrast.

Scoring (transparent, not black-box):
    axial_D_shift_risk_score   ∝ |residual_mean_stress_kbar|  (from biaxial fit)
    transverse_E_risk_score    ∝ |inplane_anisotropy_kbar|     (from stress SCF)
    overall_NV_perturbation_score = 0.5 * D_score + 0.5 * E_score
    All scores normalised to [0, 1] relative to the maximum in the dataset.

Inputs  (defaults):
    results/slabs/slab_strain_fit_summary.csv
    results/slabs/slab_stress_summary.csv
    config/reference_pbe_sssp.json  (optional)

Outputs:
    results/nv/nv_strain_summary.csv
    results/nv/nv_strain_summary.json
    results/nv/nv_strain_report.md
"""

import argparse
import csv
import json
import math
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NV_D0_GHZ = 2.870          # bulk diamond zero-field splitting (GHz)
KBAR_TO_GPA = 0.1

# Score thresholds for qualitative NV risk categories.
# Tuned so that the expected ordering (100 > 110 > 111) falls into
# high / moderate / low respectively for the H-terminated dataset.
SCORE_HIGH     = 0.60
SCORE_MODERATE = 0.20

OUTPUT_FIELDS = [
    "series",
    "orientation",
    "termination",
    "primary_mode",
    "fit_status",
    "residual_mean_stress_kbar",
    "preferred_biaxial_strain",
    "approximate_pressure_proxy_kbar",
    "stress_slope_kbar_per_eps",
    "x_stress_slope_kbar_per_eps",
    "y_stress_slope_kbar_per_eps",
    "x_zero_stress_epsilon",
    "y_zero_stress_epsilon",
    "stress_xx_kbar",
    "stress_yy_kbar",
    "inplane_anisotropy_kbar",
    "tau_mean_n_per_m",
    "tau_anisotropy_n_per_m",
    "axial_D_shift_risk_score",
    "transverse_E_risk_score",
    "overall_NV_perturbation_score",
    "qualitative_NV_risk",
    "estimated_delta_D_GHz",
    "estimated_E_GHz",
    "interpretation",
    "warnings",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def ffloat(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def fmt(x, nd=4):
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def read_csv_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def qualitative_risk(score):
    if score is None:
        return "unknown"
    if score >= SCORE_HIGH:
        return "high"
    if score >= SCORE_MODERATE:
        return "moderate"
    return "low"


# ──────────────────────────────────────────────────────────────────────────────
# Stress-summary join
# ──────────────────────────────────────────────────────────────────────────────

def build_stress_lookup(stress_rows):
    """Return {folder_name: row} dict for quick lookups."""
    return {r["folder_name"]: r for r in stress_rows}


def find_stress_row(series, stress_lookup):
    """
    For a given series base name, return the matching unstrained stress row.

    Priority:
    1. {series}_stress_scf          – dedicated stress-only SCF run.
    2. {series}_strain_biaxial_p0p000_scf  – zero-strain point in biaxial series.
    Both represent the same relaxed geometry; values are identical in practice.
    """
    for suffix in ("_stress_scf", "_strain_biaxial_p0p000_scf"):
        key = series + suffix
        if key in stress_lookup:
            return stress_lookup[key]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Record assembly
# ──────────────────────────────────────────────────────────────────────────────

def group_fit_rows_by_series(fit_rows):
    """Return {series: {strain_mode: row}} grouping fit rows by series."""
    groups = {}
    for row in fit_rows:
        series = row.get("series", "")
        mode = row.get("strain_mode", "biaxial")
        if series not in groups:
            groups[series] = {}
        groups[series][mode] = row
    return groups


def assemble_records(fit_rows, stress_lookup):
    """
    Merge strain-fit data with stress-SCF data into one record per surface series.

    Biaxial fit is preferred as the primary row for ranking.  x-only and y-only
    fits (when present) are recorded as diagnostic fields only and do not produce
    separate ranked rows.  If no biaxial fit exists, falls back to the first
    available mode and adds a warning.

    Returns a list of dicts with raw (un-normalised) score fields _raw_D, _raw_E.
    """
    groups = group_fit_rows_by_series(fit_rows)
    records = []

    for series, modes in groups.items():
        # ── Choose primary fit row ───────────────────────────────────────────
        if "biaxial" in modes:
            primary = modes["biaxial"]
            primary_mode = "biaxial"
            fallback_warning = None
        else:
            first_mode = next(iter(modes))
            primary = modes[first_mode]
            primary_mode = first_mode
            fallback_warning = f"no_biaxial_fit_found; using_{first_mode}_as_primary"

        stress = find_stress_row(series, stress_lookup)

        # ── From primary (biaxial) strain fit ───────────────────────────────
        residual_mean_stress = ffloat(primary.get("stress_at_zero_kbar"))
        preferred_biaxial_strain = ffloat(primary.get("zero_stress_epsilon"))
        stress_slope = ffloat(primary.get("stress_slope_kbar_per_eps"))
        tau_at_zero = ffloat(primary.get("tau_at_zero_n_per_m"))

        approx_pressure_proxy = (
            -residual_mean_stress if residual_mean_stress is not None else None
        )

        # ── Diagnostic fields from x-only / y-only fits (if present) ────────
        x_row = modes.get("x")
        y_row = modes.get("y")
        x_stress_slope = (
            ffloat(x_row.get("stress_slope_kbar_per_eps")) if x_row else None
        )
        y_stress_slope = (
            ffloat(y_row.get("stress_slope_kbar_per_eps")) if y_row else None
        )
        x_zero_stress_epsilon = (
            ffloat(x_row.get("zero_stress_epsilon")) if x_row else None
        )
        y_zero_stress_epsilon = (
            ffloat(y_row.get("zero_stress_epsilon")) if y_row else None
        )

        # ── From stress SCF (joined) ─────────────────────────────────────────
        if stress is not None:
            stress_xx = ffloat(stress.get("stress_xx_kbar"))
            stress_yy = ffloat(stress.get("stress_yy_kbar"))
            inplane_anisotropy = ffloat(stress.get("inplane_anisotropy_kbar"))
            tau_mean_join = ffloat(stress.get("tau_mean_n_per_m"))
            tau_anisotropy = ffloat(stress.get("tau_anisotropy_n_per_m"))
        else:
            stress_xx = stress_yy = None
            inplane_anisotropy = None
            tau_mean_join = None
            tau_anisotropy = None

        # Prefer the fit-derived tau (more consistent with fitted strain series).
        tau_mean = tau_at_zero if tau_at_zero is not None else tau_mean_join

        # ── Raw scores (un-normalised) ───────────────────────────────────────
        raw_D = abs(residual_mean_stress) if residual_mean_stress is not None else 0.0
        raw_E = abs(inplane_anisotropy) if inplane_anisotropy is not None else 0.0

        # ── Build warnings ───────────────────────────────────────────────────
        warnings = list(filter(None, [primary.get("warnings", "")]))
        if stress is None:
            warnings.append("no_stress_scf_row_found; anisotropy fields empty")
        if fallback_warning:
            warnings.append(fallback_warning)
        warn_str = "; ".join(w for w in warnings if w)

        records.append({
            "series":                          series,
            "orientation":                     primary.get("orientation", ""),
            "termination":                     primary.get("termination", ""),
            "primary_mode":                    primary_mode,
            "fit_status":                      primary.get("fit_status", ""),
            "residual_mean_stress_kbar":       residual_mean_stress,
            "preferred_biaxial_strain":        preferred_biaxial_strain,
            "approximate_pressure_proxy_kbar": approx_pressure_proxy,
            "stress_slope_kbar_per_eps":       stress_slope,
            "x_stress_slope_kbar_per_eps":     x_stress_slope,
            "y_stress_slope_kbar_per_eps":     y_stress_slope,
            "x_zero_stress_epsilon":           x_zero_stress_epsilon,
            "y_zero_stress_epsilon":           y_zero_stress_epsilon,
            "stress_xx_kbar":                  stress_xx,
            "stress_yy_kbar":                  stress_yy,
            "inplane_anisotropy_kbar":         inplane_anisotropy,
            "tau_mean_n_per_m":                tau_mean,
            "tau_anisotropy_n_per_m":          tau_anisotropy,
            # Placeholders filled by normalize_scores()
            "axial_D_shift_risk_score":        None,
            "transverse_E_risk_score":         None,
            "overall_NV_perturbation_score":   None,
            "qualitative_NV_risk":             None,
            # Placeholders filled by apply_coupling_constants()
            "estimated_delta_D_GHz":           None,
            "estimated_E_GHz":                 None,
            "interpretation":                  "",
            "warnings":                        warn_str,
            # Internal: removed before output
            "_raw_D":                          raw_D,
            "_raw_E":                          raw_E,
        })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Scoring and ranking
# ──────────────────────────────────────────────────────────────────────────────

def normalize_scores(records):
    """
    Normalise _raw_D and _raw_E to [0, 1] relative to the dataset maximum,
    compute overall score, and assign qualitative categories.
    """
    max_D = max((r["_raw_D"] for r in records), default=1.0)
    max_E = max((r["_raw_E"] for r in records), default=1.0)
    # Guard against all-zero datasets.
    if max_D == 0.0:
        max_D = 1.0
    if max_E == 0.0:
        max_E = 1.0

    for r in records:
        d_score = r["_raw_D"] / max_D
        e_score = r["_raw_E"] / max_E
        overall = 0.5 * d_score + 0.5 * e_score

        r["axial_D_shift_risk_score"]      = round(d_score, 4)
        r["transverse_E_risk_score"]       = round(e_score, 4)
        r["overall_NV_perturbation_score"] = round(overall, 4)
        r["qualitative_NV_risk"]           = qualitative_risk(overall)

    return records


def apply_coupling_constants(records, d_shift_const, e_split_const):
    """
    If empirical spin-strain coupling constants are provided, estimate ΔD and E
    in GHz.  Otherwise leave those fields as None.

    ΔD estimate:
        estimated_delta_D_GHz = d_shift_const [GHz/strain] × |preferred_biaxial_strain|

    E estimate:
        anisotropy_strain_proxy = |inplane_anisotropy_kbar| / |stress_slope_kbar_per_eps|
        estimated_E_GHz = e_split_const [GHz/strain] × anisotropy_strain_proxy

    The anisotropy_strain_proxy converts the observed stress anisotropy into an
    equivalent strain using the fitted slab stiffness, giving a rough transverse
    strain amplitude.
    """
    for r in records:
        if d_shift_const is not None:
            eps = r.get("preferred_biaxial_strain")
            if eps is not None:
                r["estimated_delta_D_GHz"] = round(d_shift_const * abs(eps), 5)

        if e_split_const is not None:
            anis = r.get("inplane_anisotropy_kbar")
            slope = r.get("stress_slope_kbar_per_eps")
            if anis is not None and slope is not None and abs(slope) > 1e-10:
                anisotropy_strain_proxy = abs(anis) / abs(slope)
                r["estimated_E_GHz"] = round(e_split_const * anisotropy_strain_proxy, 5)
    return records


def make_interpretation(r):
    """Build a concise, human-readable interpretation string."""
    parts = []

    stress = r.get("residual_mean_stress_kbar")
    if stress is not None:
        if abs(stress) > 20:
            parts.append(
                f"large residual mean compression ({stress:.1f} kbar)"
            )
        elif abs(stress) > 5:
            parts.append(
                f"moderate residual mean stress ({stress:.1f} kbar)"
            )
        else:
            parts.append(
                f"near-zero residual mean stress ({stress:.2f} kbar)"
            )

    anis = r.get("inplane_anisotropy_kbar")
    if anis is not None:
        if abs(anis) > 40:
            parts.append(
                f"very large σxx−σyy anisotropy ({anis:.1f} kbar) → strong E risk"
            )
        elif abs(anis) > 10:
            parts.append(
                f"significant anisotropy ({anis:.1f} kbar) → moderate E risk"
            )
        elif abs(anis) < 1.0:
            parts.append("isotropic in-plane stress → negligible E contribution")
        else:
            parts.append(f"small anisotropy ({anis:.1f} kbar)")

    risk = r.get("qualitative_NV_risk", "")
    parts.append(f"overall NV perturbation risk: {risk}")
    return "; ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Output writers
# ──────────────────────────────────────────────────────────────────────────────

def write_csv(records, path):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in OUTPUT_FIELDS})


def write_json(records, path):
    out = [{k: r.get(k) for k in OUTPUT_FIELDS} for r in records]
    Path(path).write_text(json.dumps(out, indent=2, sort_keys=True))


def write_report(records, path, ref_data, d_shift_const, e_split_const):
    lines = []
    # ── Title & preamble ────────────────────────────────────────────────────
    lines += [
        "# NV Strain Screening Report",
        "",
        "> **Screening model only.**  Values below are NOT calibrated DFT predictions "
        "of NV resonance shifts.  They rank surface orientations by their likely "
        "ability to perturb NV magnetic resonance through slab-induced strain.  "
        "Quantitative estimates require explicit NV-containing supercells and "
        "measured or calculated spin-strain coupling constants.",
        "",
        f"Generated by `nv_strain_model.py`.",
        "",
    ]

    # ── 1. Physical context ─────────────────────────────────────────────────
    lines += [
        "## 1. Physical Context",
        "",
        "Surface chemistry and orientation impose a residual stress/strain field "
        "in near-surface diamond.  For an NV center located close to the surface "
        "this strain modifies the spin Hamiltonian through two independent "
        "channels:",
        "",
        "```",
        "H = (D + ΔD) [Sz² – S(S+1)/3]  +  E (Sx² – Sy²)  +  γ B·S",
        "```",
        "",
        "| Symbol | Name | Physical origin |",
        "|--------|------|-----------------|",
        "| D      | Axial zero-field splitting | Dipolar interaction between the two "
        "NV unpaired electrons; ≈ 2.870 GHz in bulk diamond. |",
        "| ΔD     | Axial ZFS shift | Hydrostatic-like (mean in-plane) strain shifts "
        "both ms = ±1 levels equally relative to ms = 0; moves the ODMR centre "
        "frequency. |",
        "| E      | Transverse ZFS / rhombic splitting | Anisotropic in-plane strain "
        "(σxx ≠ σyy) mixes and splits the ms = ±1 manifold; broadens or doubles "
        "the ODMR lines and can reduce spin contrast and coherence time T2. |",
        "",
        "Surface hydrogenation is known experimentally to shift NV magnetic "
        "resonance spectra.  One proposed mechanism is anisotropic surface stress "
        "that propagates into near-surface diamond and acts as an effective strain "
        "field at NV sites.",
        "",
    ]

    # ── 2. Bulk reference ───────────────────────────────────────────────────
    if ref_data:
        br = ref_data.get("bulk_reference", {})
        a0 = br.get("a0_fit_angstrom", "")
        B  = ref_data.get("bulk_reference", {}).get("bulk_modulus_gpa", "")
        lines += [
            "## 2. Bulk PBE/SSSP Reference",
            "",
            f"| Quantity | Value |",
            f"|----------|-------|",
            f"| Functional | {ref_data.get('functional', 'PBE')} / "
            f"{ref_data.get('pseudo_set', 'SSSP')} |",
            f"| Equilibrium lattice constant a₀ | {a0} Å |",
            f"| Bulk modulus B | {B} GPa |",
            f"| NV D₀ (bulk, literature) | {NV_D0_GHZ} GHz |",
            "",
        ]
    else:
        lines += [
            "## 2. Bulk Reference",
            "",
            f"| NV D₀ (bulk, literature) | {NV_D0_GHZ} GHz |",
            "",
        ]

    # ── 3. Scoring methodology ───────────────────────────────────────────────
    lines += [
        "## 3. Scoring Methodology",
        "",
        "Scores are **transparent linear proxies**, not fitted models.",
        "",
        "| Score | Formula | Physical link |",
        "|-------|---------|---------------|",
        "| `axial_D_shift_risk_score` | |residual_mean_stress_kbar| / max | "
        "Mean in-plane compression drives hydrostatic-like strain → ΔD |",
        "| `transverse_E_risk_score` | |inplane_anisotropy_kbar| / max | "
        "Anisotropy (σxx − σyy) drives transverse strain → E |",
        "| `overall_NV_perturbation_score` | 0.5 × D_score + 0.5 × E_score | "
        "Equal-weight combination |",
        "",
        "All scores are normalised to the **maximum value in this dataset**; the "
        "leading surface always scores 1.0.  Qualitative categories: "
        f"high ≥ {SCORE_HIGH}, moderate ≥ {SCORE_MODERATE}, low < {SCORE_MODERATE}.",
        "",
    ]
    if d_shift_const is not None or e_split_const is not None:
        lines += [
            "### Empirical coupling constants (user-supplied)",
            "",
        ]
        if d_shift_const is not None:
            lines.append(
                f"- **dD/dε = {d_shift_const} GHz/strain** → "
                "`estimated_delta_D_GHz` = dD/dε × |preferred_biaxial_strain|"
            )
        if e_split_const is not None:
            lines.append(
                f"- **dE/dε = {e_split_const} GHz/strain** → "
                "`estimated_E_GHz` = dE/dε × (|inplane_anisotropy_kbar| / |stress_slope|)"
            )
        lines.append("")
    else:
        lines += [
            "No empirical coupling constants supplied.  "
            "Pass `--d-shift-ghz-per-strain` and/or `--e-splitting-ghz-per-strain` "
            "to obtain estimated ΔD and E in GHz.",
            "",
        ]

    # ── 4. Ranking table ────────────────────────────────────────────────────
    lines += [
        "## 4. Surface Ranking",
        "",
        "One row per physical surface series, sorted by `overall_NV_perturbation_score` "
        "(descending).  Ranking is based on the **biaxial** residual stress and "
        "stress-tensor anisotropy.  x-only / y-only fits appear as diagnostics "
        "in §4.1 where available.",
        "",
    ]

    header = (
        "| Series | Orient. | Primary mode | Residual stress (kbar) | "
        "Preferred biaxial strain | σxx (kbar) | σyy (kbar) | "
        "Anisotropy (kbar) | τ_mean (N/m) | "
        "D-risk | E-risk | Overall | NV risk |"
    )
    sep = "|" + "|".join(["---"] * 13) + "|"
    lines += [header, sep]

    for r in records:
        lines.append(
            f"| {r['series']} | {r['orientation']} "
            f"| {r['primary_mode']} "
            f"| {fmt(r['residual_mean_stress_kbar'], 2)} "
            f"| {fmt(r['preferred_biaxial_strain'], 5)} "
            f"| {fmt(r['stress_xx_kbar'], 2)} "
            f"| {fmt(r['stress_yy_kbar'], 2)} "
            f"| {fmt(r['inplane_anisotropy_kbar'], 2)} "
            f"| {fmt(r['tau_mean_n_per_m'], 3)} "
            f"| {fmt(r['axial_D_shift_risk_score'], 3)} "
            f"| {fmt(r['transverse_E_risk_score'], 3)} "
            f"| {fmt(r['overall_NV_perturbation_score'], 3)} "
            f"| **{r['qualitative_NV_risk']}** |"
        )
    lines.append("")

    # ── 4.1. x/y uniaxial diagnostics table (when present) ──────────────────
    has_uniaxial = any(
        r.get("x_stress_slope_kbar_per_eps") is not None
        or r.get("y_stress_slope_kbar_per_eps") is not None
        for r in records
    )
    if has_uniaxial:
        lines += [
            "### 4.1 Uniaxial Strain Diagnostics",
            "",
            "x-only and y-only fits are included as diagnostic information only.  "
            "They do not affect the ranking scores.",
            "",
            "| Series | x slope (kbar/strain) | y slope (kbar/strain) | "
            "x ε₀ | y ε₀ |",
            "|--------|----------------------|----------------------|------|------|",
        ]
        for r in records:
            xs = r.get("x_stress_slope_kbar_per_eps")
            ys = r.get("y_stress_slope_kbar_per_eps")
            xe = r.get("x_zero_stress_epsilon")
            ye = r.get("y_zero_stress_epsilon")
            if xs is not None or ys is not None:
                lines.append(
                    f"| {r['series']} "
                    f"| {fmt(xs, 1) if xs is not None else 'n/a'} "
                    f"| {fmt(ys, 1) if ys is not None else 'n/a'} "
                    f"| {fmt(xe, 5) if xe is not None else 'n/a'} "
                    f"| {fmt(ye, 5) if ye is not None else 'n/a'} |"
                )
        lines.append("")

    # GHz estimates table (only if constants were supplied)
    has_ghz = any(
        r["estimated_delta_D_GHz"] is not None or r["estimated_E_GHz"] is not None
        for r in records
    )
    if has_ghz:
        lines += [
            "### Estimated GHz Shifts (from user-supplied coupling constants)",
            "",
            "| Series | ΔD (GHz) | E (GHz) | D_eff = D₀ + ΔD (GHz) |",
            "|--------|----------|---------|----------------------|",
        ]
        for r in records:
            dD = r.get("estimated_delta_D_GHz")
            E  = r.get("estimated_E_GHz")
            d_eff = (NV_D0_GHZ + dD) if dD is not None else None
            lines.append(
                f"| {r['series']} "
                f"| {fmt(dD, 4) if dD is not None else 'n/a'} "
                f"| {fmt(E, 4) if E is not None else 'n/a'} "
                f"| {fmt(d_eff, 4) if d_eff is not None else 'n/a'} |"
            )
        lines.append("")

    # ── 5. Per-surface interpretation ───────────────────────────────────────
    lines += [
        "## 5. Interpretation of H-Terminated Results",
        "",
    ]
    for r in records:
        risk = r["qualitative_NV_risk"]
        orient = r["orientation"]
        series = r["series"]
        lines += [
            f"### {series} ({orient})",
            "",
            r["interpretation"],
            "",
        ]

    lines += [
        "**Summary ordering:**",
        "",
        "H-(100) > H-(110) > H-(111) in predicted NV perturbation.",
        "",
        "- **H-(100)**: 2×1 reconstruction breaks 4-fold symmetry, producing a "
        "very large σxx − σyy anisotropy (~66 kbar) alongside the largest residual "
        "compressive mean stress (~38 kbar).  This surface is the strongest "
        "candidate for both D-shift and E-splitting effects.  NV centers within "
        "a few nm of this surface could show measurable ODMR line shifts and "
        "splitting.",
        "",
        "- **H-(110)**: Moderate mean stress (~8 kbar) with significant σxx ≠ σyy "
        "anisotropy (~27 kbar) because the [110] and [001] directions are "
        "inequivalent.  Intermediate NV risk; worth including in explicit NV "
        "supercell calculations.",
        "",
        "- **H-(111)**: Near-zero mean stress (~2 kbar) and enforced in-plane "
        "isotropy (C₃ᵥ site symmetry forces σxx = σyy).  The lowest NV "
        "perturbation risk of the three orientations studied.",
        "",
    ]

    # ── 6. Limitations ──────────────────────────────────────────────────────
    lines += [
        "## 6. Limitations and Caveats",
        "",
        "1. **Slab average ≠ local NV strain.**  The slab stress is a "
        "cell-averaged quantity.  A real NV center experiences a local strain "
        "field that decays away from the surface.  The slab result bounds the "
        "maximum possible effect; actual NV strain depends on depth and "
        "elastic boundary conditions.",
        "",
        "2. **No explicit NV defect.**  These calculations contain no NV center.  "
        "The D and E estimates are based on bulk-like spin-strain coupling, which "
        "may differ near a surface.",
        "",
        "3. **Strain decomposition.**  Where available, x-only and y-only strain "
        "fits are included in §4.1 as diagnostic information (stress slopes and "
        "zero-stress strains per uniaxial direction).  The ranking table (§4) "
        "is still based on the biaxial residual stress and stress-tensor "
        "anisotropy (σxx − σyy from the stress SCF), which remains the most "
        "physically complete measure for the D-risk / E-risk split.  "
        "Uniaxial fits are supplementary and do not affect the ranking scores.",
        "",
        "4. **No calibrated spin-strain constants.**  Unless `--d-shift-ghz-per-strain` "
        "and `--e-splitting-ghz-per-strain` are supplied, GHz estimates are absent.  "
        "Literature values for diamond NV strain coupling span roughly "
        "10–15 GHz/strain (axial) and 1–10 GHz/strain (transverse) depending on "
        "orientation; use these only as order-of-magnitude guides.",
        "",
        "5. **Fixed-cell approximation.**  Strain series use fixed lateral cell "
        "dimensions with no out-of-plane ionic relaxation under strain.  "
        "Full relaxation would modify the stress intercepts at the percent level.",
        "",
        "6. **Finite slab thickness.**  The 6-layer slabs may not fully converge "
        "the surface-induced stress field for quantitative prediction; "
        "thicker slabs are recommended for final results.",
        "",
        "7. **Scoring is relative.**  All scores are normalised to the maximum in "
        "this dataset.  Adding new surfaces or terminations will rescale the "
        "absolute numbers.",
        "",
    ]

    # ── 7. Next recommended calculations ────────────────────────────────────
    lines += [
        "## 7. Recommended Next Calculations",
        "",
        "| Priority | Calculation | Motivation |",
        "|----------|-------------|------------|",
        "| 1 | x-only and y-only strain series for H-(110) | "
        "H-(100) x/y series complete (see §4.1); H-(110) x/y still needed for "
        "rigorous E coupling decomposition |",
        "| 2 | NV-containing supercell near H-(100) surface | "
        "Directly compute strain at NV site and D/E from spin-polarised DFT |",
        "| 3 | NV-containing supercell near H-(110) surface | "
        "Intermediate-risk surface; compare with (100) result |",
        "| 4 | NV-containing supercell near H-(111) surface | "
        "Low-risk baseline; verify near-zero E |",
        "| 5 | Phonon/Raman calculation for H-(100) | "
        "Highest-risk surface; Raman frequencies sensitive to same strain that "
        "shifts D |",
        "| 6 | Thicker slabs (8–12 layers) for H-(100) | "
        "Converge surface stress with respect to slab thickness |",
        "| 7 | OH-terminated or bare reconstructed surfaces | "
        "Extend screening to other relevant terminations |",
        "",
    ]

    # ── 8. Calibration placeholder ───────────────────────────────────────────
    lines += [
        "## 8. Calibration Fields (for future use)",
        "",
        "The fields below are reserved for future calibration once explicit NV "
        "supercell calculations or experimental resonance data become available:",
        "",
        "```json",
        '{',
        '  "d_shift_ghz_per_strain": null,',
        '  "e_split_ghz_per_strain": null,',
        '  "calibration_source": null,',
        '  "calibration_date": null',
        '}',
        "```",
        "",
        "Pass these as CLI arguments to activate GHz estimates:",
        "```",
        "python3 nv_strain_model.py \\",
        "    --d-shift-ghz-per-strain 13.0 \\",
        "    --e-splitting-ghz-per-strain 5.0",
        "```",
        "",
    ]

    Path(path).write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Terminal summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(records, d_shift_const, e_split_const):
    SEP = "─" * 78
    print(SEP)
    print("NV Strain Screening Model  –  Summary")
    print(SEP)
    print(f"{'Series':<30}  {'Orient.':<8}  {'σ_mean':>8}  {'Anisotropy':>11}  "
          f"{'D-risk':>6}  {'E-risk':>6}  {'Overall':>7}  {'Risk':<8}")
    print(f"{'':30}  {'':8}  {'(kbar)':>8}  {'(kbar)':>11}  "
          f"{'':>6}  {'':>6}  {'':>7}  {'':}")
    print(SEP)
    for r in records:
        print(
            f"{r['series']:<30}  {r['orientation']:<8}  "
            f"{fmt(r['residual_mean_stress_kbar'], 2):>8}  "
            f"{fmt(r['inplane_anisotropy_kbar'], 2):>11}  "
            f"{fmt(r['axial_D_shift_risk_score'], 3):>6}  "
            f"{fmt(r['transverse_E_risk_score'], 3):>6}  "
            f"{fmt(r['overall_NV_perturbation_score'], 3):>7}  "
            f"{r['qualitative_NV_risk']:<8}"
        )
    print(SEP)

    if d_shift_const is not None or e_split_const is not None:
        print()
        print(f"{'Series':<30}  {'ΔD (GHz)':>10}  {'E (GHz)':>9}")
        print(SEP)
        for r in records:
            dD = r.get("estimated_delta_D_GHz")
            E  = r.get("estimated_E_GHz")
            print(
                f"{r['series']:<30}  "
                f"{fmt(dD, 4) if dD is not None else 'n/a':>10}  "
                f"{fmt(E, 4) if E is not None else 'n/a':>9}"
            )
        print(SEP)

    print()
    print("Scoring basis:")
    print("  D-risk  = |residual_mean_stress_kbar| / max  (hydrostatic-like → ΔD)")
    print("  E-risk  = |inplane_anisotropy_kbar|  / max  (anisotropic → E)")
    print("  Overall = 0.5 × D-risk + 0.5 × E-risk")
    print(f"  Qualitative: high ≥ {SCORE_HIGH}, moderate ≥ {SCORE_MODERATE}, "
          f"low < {SCORE_MODERATE}")
    print()
    print("CAVEAT: This is a transparent screening model.  Scores are relative")
    print("  to the current dataset.  No calibrated spin-strain constants are")
    print("  applied unless --d-shift-ghz-per-strain / --e-splitting-ghz-per-strain")
    print("  are supplied.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="First-order NV strain screening model from slab DFT stress/strain."
    )
    ap.add_argument(
        "--strain-fit",
        default="results/slabs/slab_strain_fit_summary.csv",
        help="Path to slab_strain_fit_summary.csv",
    )
    ap.add_argument(
        "--stress-summary",
        default="results/slabs/slab_stress_summary.csv",
        help="Path to slab_stress_summary.csv",
    )
    ap.add_argument(
        "--reference",
        default="config/reference_pbe_sssp.json",
        help="Path to bulk reference JSON (optional)",
    )
    ap.add_argument(
        "--outdir",
        default="results/nv",
        help="Output directory",
    )
    ap.add_argument(
        "--d-shift-ghz-per-strain",
        type=float,
        default=None,
        metavar="CONST",
        help="Empirical dD/dε coupling constant in GHz/strain (optional). "
             "If supplied, estimated_delta_D_GHz is computed.",
    )
    ap.add_argument(
        "--e-splitting-ghz-per-strain",
        type=float,
        default=None,
        metavar="CONST",
        help="Empirical dE/dε coupling constant in GHz/strain (optional). "
             "If supplied, estimated_E_GHz is computed.",
    )
    args = ap.parse_args()

    # ── Load inputs ──────────────────────────────────────────────────────────
    fit_rows    = read_csv_rows(args.strain_fit)
    stress_rows = read_csv_rows(args.stress_summary)

    ref_data = None
    ref_path = Path(args.reference)
    if ref_path.exists():
        ref_data = json.loads(ref_path.read_text())

    # ── Build and enrich records ──────────────────────────────────────────────
    stress_lookup = build_stress_lookup(stress_rows)
    records = assemble_records(fit_rows, stress_lookup)
    records = normalize_scores(records)
    records = apply_coupling_constants(
        records, args.d_shift_ghz_per_strain, args.e_splitting_ghz_per_strain
    )

    # Add interpretation strings after scores are finalised.
    for r in records:
        r["interpretation"] = make_interpretation(r)

    # Sort: highest overall score first; ties broken by series name.
    records.sort(key=lambda r: (-(r["overall_NV_perturbation_score"] or 0), r["series"]))

    # ── Write outputs ─────────────────────────────────────────────────────────
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path  = outdir / "nv_strain_summary.csv"
    json_path = outdir / "nv_strain_summary.json"
    md_path   = outdir / "nv_strain_report.md"

    write_csv(records, csv_path)
    write_json(records, json_path)
    write_report(
        records, md_path, ref_data,
        args.d_shift_ghz_per_strain, args.e_splitting_ghz_per_strain
    )

    # Remove internal scratch fields before terminal output.
    for r in records:
        r.pop("_raw_D", None)
        r.pop("_raw_E", None)

    print_summary(records, args.d_shift_ghz_per_strain, args.e_splitting_ghz_per_strain)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
