#!/usr/bin/env python3
"""
fit_bulk_reference.py — Fit hydrostatic bulk diamond reference EOS.

Reads the parsed reference_summary.csv, filters to complete SCF hydrostatic
points, and estimates PBE/SSSP equilibrium parameters.

PRIMARY fit: 3rd-order Birch-Murnaghan on E(V) → V0, a0, B0, B0'.
CROSS-CHECK: quadratic P(ε) → ε0, a0, B.
Both are reported, and disagreement between them is flagged rather than
silently resolved in favour of one.

Why not a linear P(ε) fit
-------------------------
P(ε) over ±1 % hydrostatic strain is strongly curved: on the 90/720 Ry series
the local slopes run −14520, −13682, −12876, −12118 kbar/strain. A straight
line through five such points is biased. On that series:

    linear    P(ε):  ε0 = +0.001945   rms = 3.35 kbar
    quadratic P(ε):  ε0 = +0.001661   rms = 0.06 kbar
    cubic     P(ε):  ε0 = +0.001663   rms = 0.00 kbar

The linear fit's ε0 is wrong by ~2.8e-4 strain, which is ~1e-3 Å in a0 and
~3.6 kbar of spurious pressure — comparable to the surface-stress signals this
project is trying to measure. The same bias affects a bulk modulus taken from
a linear P(V) slope (445.1 GPa) versus the Birch-Murnaghan B0 (433.7 GPa).

The pre-2026-08 versions of this script used the linear P(ε) fit and an E(V)
*quadratic* minimum; both are retained below under `legacy_*` field names so
the size of the bias stays visible in the output rather than being erased.

Birch-Murnaghan implementation note
-----------------------------------
The 3rd-order BM energy is *exactly* a cubic polynomial in x = V^(-2/3), so
the fit is a linear least-squares problem — no iteration, no scipy. V0, B0 and
B0' are then recovered analytically from the polynomial derivatives at the
minimum (see `birch_murnaghan_fit`). The recovery is round-trip tested against
synthetic BM3 data in tests/test_fit_bulk_reference.py; that test exists
because an earlier draft of the B0' recovery dropped a dx/dV factor and
returned B0' ≈ 222 for diamond while V0 and B0 stayed correct to 5 digits.

Writes:
  bulk_fit_summary.csv
  bulk_fit_summary.json
  bulk_fit_report.md
  (optional) bulk_energy_vs_epsilon.png
  (optional) bulk_pressure_vs_epsilon.png
  (optional) bulk_energy_vs_volume.png
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import elastic_reference

# ── Physical constants ────────────────────────────────────────────────────────
RY_TO_EV    = 13.605693122994
KBAR_TO_GPA = 0.1
EV_A3_TO_GPA = 160.21766208
# 1 Ry/Å³ expressed in GPa — used to convert the Birch-Murnaghan B0, which comes
# out in Ry/Å³ because energies are Ry and volumes Å³.
RY_A3_TO_GPA = RY_TO_EV * EV_A3_TO_GPA

# ── Pure-Python least-squares helpers ─────────────────────────────────────────

def _linspace(lo, hi, n):
    """Pure-Python linspace."""
    if n < 2:
        return [lo]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _solve_3x3(mat, rhs):
    """Gaussian elimination with partial pivoting for Ax = b, A is 3×3."""
    M = [list(mat[i]) + [float(rhs[i])] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-14:
            raise ValueError("Singular matrix in least-squares system")
        for row in range(col + 1, 3):
            f = M[row][col] / M[col][col]
            M[row] = [M[row][j] - f * M[col][j] for j in range(4)]
    x = [0.0] * 3
    for i in range(2, -1, -1):
        x[i] = M[i][3] - sum(M[i][j] * x[j] for j in range(i + 1, 3))
        x[i] /= M[i][i]
    return x


def quadratic_fit(xs, ys):
    """
    Fit y = A x^2 + B x + C via normal equations.
    Returns (A, B, C).
    """
    n = len(xs)
    s = [sum(x ** k for x in xs) for k in range(5)]
    sy = [sum(y * (x ** k) for x, y in zip(xs, ys)) for k in range(3)]
    mat = [
        [s[4], s[3], s[2]],
        [s[3], s[2], s[1]],
        [s[2], s[1], n],
    ]
    return _solve_3x3(mat, [sy[2], sy[1], sy[0]])


def linear_fit(xs, ys):
    """
    Simple linear regression y = m x + b.
    Returns (m, b).
    """
    n = len(xs)
    sx  = sum(xs)
    sy  = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sx2 = sum(x * x for x in xs)
    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-14:
        raise ValueError("Degenerate linear fit (are all x values equal?)")
    m = (n * sxy - sx * sy) / denom
    b = (sy - m * sx) / n
    return m, b


def _solve_n(mat, rhs):
    """Gaussian elimination with partial pivoting for Ax = b, A is n×n."""
    n = len(rhs)
    M = [list(mat[i]) + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-300:
            raise ValueError("Singular matrix in least-squares system")
        for row in range(col + 1, n):
            f = M[row][col] / M[col][col]
            M[row] = [M[row][j] - f * M[col][j] for j in range(n + 1)]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] /= M[i][i]
    return x


class ScaledPoly:
    """
    Least-squares polynomial in a centred, unit-scaled variable.

    Fits y ≈ y_c + p(t) with t = (x - x_c)/s. The centring matters: the raw
    Birch-Murnaghan variable is x = V^(-2/3) ≈ 0.077 Å⁻², while y = E ≈ -147 Ry.
    Solving the normal equations in raw x means recovering a cubic coefficient
    of order 1e-2 from sums of order 1e2, and the cubic term — the only term
    carrying B0' — is lost to cancellation. In the centred variable the design
    matrix entries are O(1).

    Derivatives are returned with respect to the *raw* x via the chain rule,
    since t is affine in x: dⁿy/dxⁿ = p⁽ⁿ⁾(t) / sⁿ.
    """

    def __init__(self, xs, ys, degree):
        if len(xs) < degree + 1:
            raise ValueError(
                f"degree-{degree} fit needs ≥{degree + 1} points; got {len(xs)}"
            )
        self.degree = degree
        self.x_c = sum(xs) / len(xs)
        span = max(xs) - min(xs)
        self.s = (span / 2.0) if span > 0 else 1.0
        self.y_c = sum(ys) / len(ys)
        t = [(x - self.x_c) / self.s for x in xs]
        z = [y - self.y_c for y in ys]
        n = degree + 1
        mat = [[sum(tt ** (i + j) for tt in t) for j in range(n)] for i in range(n)]
        rhs = [sum(zz * tt ** i for tt, zz in zip(t, z)) for i in range(n)]
        self.coeffs = _solve_n(mat, rhs)          # ascending, in t
        self.residuals = [y - self.value(x) for x, y in zip(xs, ys)]
        self.rms = math.sqrt(
            sum(r * r for r in self.residuals) / len(self.residuals)
        )

    def _t(self, x):
        return (x - self.x_c) / self.s

    def _dp(self, t, order):
        """order-th derivative of the polynomial with respect to t."""
        total = 0.0
        for i, c in enumerate(self.coeffs):
            if i < order:
                continue
            f = 1
            for k in range(order):
                f *= (i - k)
            total += c * f * t ** (i - order)
        return total

    def value(self, x):
        return self.y_c + self._dp(self._t(x), 0)

    def deriv(self, x, order):
        """d^order y / dx^order at raw x."""
        if order == 0:
            return self.value(x)
        return self._dp(self._t(x), order) / self.s ** order

    def stationary_point(self, x_guess):
        """Newton solve of dy/dx = 0, returned in raw x."""
        x = float(x_guess)
        for _ in range(200):
            d1, d2 = self.deriv(x, 1), self.deriv(x, 2)
            if abs(d2) < 1e-300:
                break
            step = d1 / d2
            x -= step
            if abs(step) < 1e-15 * max(1.0, abs(x)):
                break
        return x

    def real_root_near(self, x_guess):
        """Newton solve of y = 0, returned in raw x."""
        x = float(x_guess)
        for _ in range(200):
            f, fp = self.value(x), self.deriv(x, 1)
            if abs(fp) < 1e-300:
                break
            step = f / fp
            x -= step
            if abs(step) < 1e-16 * max(1.0, abs(x)):
                break
        return x


def birch_murnaghan_fit(V, E, order=3):
    """
    Fit a Birch-Murnaghan EOS to E(V) and return V0, a0, B0, B0'.

    The 3rd-order BM energy

        E(V) = E0 + (9 V0 B0 / 16) [ (u-1)³ B0' + (u-1)² (6 - 4u) ],
        u = (V0/V)^(2/3)

    is, with x = V^(-2/3) and u = V0^(2/3) x, a cubic polynomial in x. So the
    fit is ordinary linear least squares in x, and V0/B0/B0' are recovered from
    the polynomial's derivatives at its stationary point x0 = V0^(-2/3):

        B0  = (4/9) · E_xx · V0^(-7/3)
        B0' = 4 + (2/3) · V0^(-2/3) · E_xxx / E_xx

    order=2 fixes B0' = 4 exactly (the cubic term vanishes), giving a
    3-parameter fit useful as a stability check when points are scarce.

    Returns a dict; a0 assumes V is the volume of a *cubic* cell so a = V^(1/3).
    """
    if order not in (2, 3):
        raise ValueError(f"Birch-Murnaghan order must be 2 or 3; got {order}")
    x = [v ** (-2.0 / 3.0) for v in V]
    poly = ScaledPoly(x, E, order)

    x0 = poly.stationary_point(sum(x) / len(x))
    if x0 <= 0:
        raise ValueError(f"Birch-Murnaghan fit gave non-physical x0={x0:.6g}")
    V0 = x0 ** -1.5

    E_xx = poly.deriv(x0, 2)
    if E_xx <= 0:
        raise ValueError(
            f"Birch-Murnaghan fit gave non-positive curvature E_xx={E_xx:.6g}; "
            "E(V) has no minimum in this range"
        )
    E_xxx = poly.deriv(x0, 3) if order == 3 else 0.0

    B0_ry_a3 = (4.0 / 9.0) * E_xx * V0 ** (-7.0 / 3.0)
    B0_prime = 4.0 + (2.0 / 3.0) * V0 ** (-2.0 / 3.0) * E_xxx / E_xx

    return {
        "order":          order,
        "V0_angstrom3":   V0,
        "a0_angstrom":    V0 ** (1.0 / 3.0),
        "B0_gpa":         B0_ry_a3 * RY_A3_TO_GPA,
        "B0_prime":       B0_prime,
        "E0_ry":          poly.value(x0),
        "rms_residual_ry": poly.rms,
        "V0_in_sampled_range": min(V) <= V0 <= max(V),
    }


def pressure_poly_fit(eps, P, a_ref, degree=2):
    """
    Fit P(ε) to a polynomial and locate the zero-pressure strain.

    Bulk modulus at ε0 uses V = V_ref (1+ε)³, so dV/V = 3 dε and

        B = -V dP/dV = -(1/3) dP/dε |_{ε0}

    Returns a dict with ε0, a0 = a_ref (1 + ε0), B in GPa, and the fit rms.
    """
    poly = ScaledPoly(eps, P, degree)
    eps0 = poly.real_root_near(0.0)
    dP_deps = poly.deriv(eps0, 1)
    B_kbar = -(1.0 / 3.0) * dP_deps
    return {
        "degree":            degree,
        "epsilon0":          eps0,
        "a0_angstrom":       a_ref * (1.0 + eps0),
        "B_gpa":             B_kbar * KBAR_TO_GPA,
        "dP_depsilon_kbar":  dP_deps,
        "rms_residual_kbar": poly.rms,
        "epsilon0_in_sampled_range": min(eps) <= eps0 <= max(eps),
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(csv_path, strain_type="hydrostatic"):
    """
    Read reference_summary.csv and return qualifying rows sorted by volume.

    Filters: calculation_type=scf, complete=True, and energy_ry /
    pressure_kbar / volume_angstrom3 all parseable. `strain_type` filters the
    strain_type column; pass None to accept any (needed for series whose run
    folders are named `<prefix>~eps_<value>`, which parse_reference.py records
    with the folder name in strain_type and a blank epsilon).

    `epsilon` is NOT required here and is not trusted if present — it is
    recomputed from the cell volume by `assign_epsilon_from_volume`.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("calculation_type", "").strip() != "scf":
                continue
            if strain_type is not None:
                if row.get("strain_type", "").strip() != strain_type:
                    continue
            if row.get("complete", "").strip().lower() != "true":
                continue
            try:
                energy_ry     = float(row["energy_ry"])
                pressure_kbar = float(row["pressure_kbar"])
                volume        = float(row["volume_angstrom3"])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                epsilon_declared = float(row["epsilon"])
            except (KeyError, ValueError, TypeError):
                epsilon_declared = None
            point = {
                "epsilon_declared": epsilon_declared,
                "energy_ry":        energy_ry,
                "pressure_kbar":    pressure_kbar,
                "volume_angstrom3": volume,
                "folder_name":      row.get("folder_name", ""),
            }
            for col in ("a_angstrom", "a_from_volume_angstrom"):
                try:
                    v = row.get(col, "").strip()
                    point[col] = float(v) if v else None
                except (ValueError, TypeError):
                    point[col] = None
            # Provenance: parse_reference.py takes these from pw.out, which
            # CLAUDE.md §1.7 makes ground truth for what was actually run.
            for col in ("ecutwfc", "ecutrho", "kpoints", "pseudo_C"):
                point[col] = (row.get(col) or "").strip() or None
            rows.append(point)
    return sorted(rows, key=lambda r: r["volume_angstrom3"])


def assign_epsilon_from_volume(points, tol=1e-4):
    """
    Recompute each point's hydrostatic strain from its cell volume.

    Per CLAUDE.md §1: read the geometry, do not trust the label. The strain
    recorded in the CSV comes from the run folder name; the volume comes from
    the QE output. Here ε is defined from the volume,

        ε = (V / V_ref)^(1/3) - 1

    and the declared value, where present, is only cross-checked against it.

    V_ref is the point declared as ε=0 if there is one, else the median-volume
    point. The choice does not affect the fitted a0: a0 = a_ref (1 + ε0) with
    a_ref = V_ref^(1/3), and shifting V_ref rescales ε0 to compensate exactly.
    It only changes the ε0 that gets *reported*, which is why the reference
    volume is recorded alongside it.

    `tol` is in strain. Declared and volume-derived strains differ by ~1.4e-5
    on the existing series simply because `alat` was rounded to six decimals in
    the generated pw.in (6.673247 bohr for nominal ε = -0.010); that is 5e-5 Å
    in a and does not warrant a warning. A genuinely mislabelled run would be
    off by ≥1e-3.

    Mutates points (adds "epsilon"); returns (a_ref, warnings).
    """
    warnings = []
    ref = None
    for p in points:
        if p["epsilon_declared"] is not None and abs(p["epsilon_declared"]) < 1e-12:
            ref = p
            break
    if ref is None:
        ref = points[len(points) // 2]
        warnings.append(
            f"No point declares ε=0; using median-volume point "
            f"'{ref['folder_name']}' (V={ref['volume_angstrom3']:.4f} Å³) as the "
            "strain reference. Fitted a0 is unaffected by this choice; the "
            "reported ε0 is relative to that volume."
        )

    V_ref = ref["volume_angstrom3"]
    a_ref = V_ref ** (1.0 / 3.0)

    for p in points:
        p["epsilon"] = (p["volume_angstrom3"] / V_ref) ** (1.0 / 3.0) - 1.0
        dec = p["epsilon_declared"]
        if dec is not None and abs(dec - p["epsilon"]) > tol:
            warnings.append(
                f"{p['folder_name']}: declared ε={dec:+.6f} disagrees with "
                f"ε={p['epsilon']:+.6f} computed from the cell volume "
                f"(|Δ|={abs(dec - p['epsilon']):.2e} > {tol:.0e}). Using the "
                "volume-derived value; check the run folder naming."
            )

    points.sort(key=lambda r: r["epsilon"])
    return a_ref, warnings


def collect_run_settings(points):
    """
    Return (settings_dict, warnings) describing the QE settings the fitted
    points were actually run with, taken from pw.out via reference_summary.csv.

    A series fitted across mixed cutoffs or k-meshes is not a valid EOS, so
    any variation is reported rather than averaged or silently taking the
    first value.
    """
    warnings = []
    settings = {}
    for col, label in (("ecutwfc", "ecutwfc"), ("ecutrho", "ecutrho"),
                       ("kpoints", "kpoints_bulk"), ("pseudo_C", "pseudo_C")):
        seen = sorted({p.get(col) for p in points if p.get(col) is not None})
        if not seen:
            warnings.append(f"{col} not recorded in the input summary")
            continue
        if len(seen) > 1:
            warnings.append(
                f"MIXED {col} across the fitted series: {', '.join(seen)}. "
                "An EOS fitted across different settings is not meaningful."
            )
        settings[label] = seen[0] if len(seen) == 1 else list(seen)
    for key in ("ecutwfc", "ecutrho"):
        if isinstance(settings.get(key), str):
            try:
                settings[key] = float(settings[key])
            except ValueError:
                pass
    return settings, warnings


# ── Analysis ───────────────────────────────────────────────────────────────────

def run_analysis(points, a_ref=None, prior_warnings=None):
    """
    Run all fits and assemble result dict.
    Returns (results_dict, warnings_list).
    """
    warnings = list(prior_warnings or [])
    n = len(points)

    if n < 4:
        raise ValueError(
            f"At least 4 qualifying points required for a 3rd-order "
            f"Birch-Murnaghan fit; found {n}"
        )
    if n < 5:
        warnings.append(
            f"Only {n} sampled points; a 3rd-order Birch-Murnaghan fit has 4 "
            "parameters, so this leaves 0 residual degrees of freedom"
        )

    if a_ref is None:
        a_ref, extra = assign_epsilon_from_volume(points)
        warnings.extend(extra)

    eps = [p["epsilon"]          for p in points]
    E   = [p["energy_ry"]        for p in points]
    P   = [p["pressure_kbar"]    for p in points]
    V   = [p["volume_angstrom3"] for p in points]

    # ── 1. E(ε) quadratic fit ─────────────────────────────────────────────
    A_e, B_e, C_e = quadratic_fit(eps, E)
    if A_e <= 0:
        warnings.append(
            f"E(ε) fit: leading coefficient A={A_e:.4e} ≤ 0; "
            "parabola opens downward — fit may be unreliable"
        )
    eps0_energy = -B_e / (2.0 * A_e) if abs(A_e) > 1e-20 else 0.0
    E0_ry = A_e * eps0_energy ** 2 + B_e * eps0_energy + C_e
    a0_energy = a_ref * (1.0 + eps0_energy)

    if not (min(eps) <= eps0_energy <= max(eps)):
        warnings.append(
            f"E(ε) minimum ε₀={eps0_energy:+.5f} lies outside the sampled range "
            f"[{min(eps):.4f}, {max(eps):.4f}]; consider extending the strain sweep"
        )

    # ── 2. P(ε) linear fit ────────────────────────────────────────────────
    m_P, b_P = linear_fit(eps, P)
    eps0_pressure = -b_P / m_P if abs(m_P) > 1e-10 else 0.0
    a0_pressure = a_ref * (1.0 + eps0_pressure)

    if not (min(eps) <= eps0_pressure <= max(eps)):
        warnings.append(
            f"P(ε) zero crossing ε₀={eps0_pressure:+.5f} lies outside sampled range"
        )

    # ── 3. E(V) quadratic fit ─────────────────────────────────────────────
    A_v, B_v, C_v = quadratic_fit(V, E)
    if A_v <= 0:
        warnings.append(
            f"E(V) fit: leading coefficient A={A_v:.4e} ≤ 0; fit may be unreliable"
        )
    V0_ev = -B_v / (2.0 * A_v) if abs(A_v) > 1e-30 else sum(V) / n
    a0_ev = V0_ev ** (1.0 / 3.0)

    # ── 4. P(V) linear fit → legacy bulk modulus ──────────────────────────
    m_PV, b_PV = linear_fit(V, P)
    # B = −V₀ dP/dV   (kbar/Å³ × Å³ → kbar → GPa)
    B_gpa_legacy = -V0_ev * m_PV * KBAR_TO_GPA

    # ── 5. PRIMARY: 3rd-order Birch-Murnaghan on E(V) ─────────────────────
    bm3 = birch_murnaghan_fit(V, E, order=3)
    # 2nd-order BM (B0' fixed at 4) as a stability check on the extra parameter.
    # This is a diagnostic, so its failure must not abort the primary fit.
    try:
        bm2 = birch_murnaghan_fit(V, E, order=2)
    except ValueError as exc:
        bm2 = None
        warnings.append(f"BM2 stability check unavailable: {exc}")

    # ── 6. CROSS-CHECK: quadratic P(ε) ────────────────────────────────────
    pq = pressure_poly_fit(eps, P, a_ref, degree=2)
    pl = pressure_poly_fit(eps, P, a_ref, degree=1)   # legacy, for the bias table

    # ── Agreement between primary and cross-check ─────────────────────────
    A0_TOL_ANG  = 0.0005     # 5e-4 Å ≈ 1.4e-4 strain ≈ 1.8 kbar in this material
    B_TOL_GPA   = 5.0
    BP_LO, BP_HI = 2.0, 8.0  # physical range for B0' in a tetrahedral solid
    B_LO, B_HI  = 200.0, 900.0

    delta_a0_methods = abs(bm3["a0_angstrom"] - pq["a0_angstrom"])
    delta_B_methods  = abs(bm3["B0_gpa"] - pq["B_gpa"])

    if delta_a0_methods > A0_TOL_ANG:
        warnings.append(
            f"PRIMARY/CROSS-CHECK DISAGREEMENT in a0: Birch-Murnaghan E(V) gives "
            f"{bm3['a0_angstrom']:.6f} Å, quadratic P(ε) gives "
            f"{pq['a0_angstrom']:.6f} Å, |Δ| = {delta_a0_methods:.6f} Å > "
            f"{A0_TOL_ANG:.4f} Å. Not resolved automatically — inspect before use."
        )
    if delta_B_methods > B_TOL_GPA:
        warnings.append(
            f"PRIMARY/CROSS-CHECK DISAGREEMENT in B: Birch-Murnaghan gives "
            f"{bm3['B0_gpa']:.2f} GPa, quadratic P(ε) gives {pq['B_gpa']:.2f} GPa, "
            f"|Δ| = {delta_B_methods:.2f} GPa > {B_TOL_GPA:.1f} GPa. "
            "Not resolved automatically — inspect before use."
        )
    if not (BP_LO <= bm3["B0_prime"] <= BP_HI):
        warnings.append(
            f"Birch-Murnaghan B0' = {bm3['B0_prime']:.3f} lies outside the "
            f"physical range [{BP_LO}, {BP_HI}]. B0' is the least-constrained "
            "parameter of the fit; treat V0/B0 with suspicion too until the "
            "strain window is widened."
        )
    if not (B_LO <= bm3["B0_gpa"] <= B_HI):
        warnings.append(
            f"Bulk modulus {bm3['B0_gpa']:.1f} GPa outside broad sanity range "
            f"[{B_LO:.0f}, {B_HI:.0f}] GPa"
        )
    if not bm3["V0_in_sampled_range"]:
        warnings.append(
            f"Birch-Murnaghan V0 = {bm3['V0_angstrom3']:.4f} Å³ lies outside the "
            f"sampled volume range [{min(V):.4f}, {max(V):.4f}] Å³; extrapolated"
        )
    if not pq["epsilon0_in_sampled_range"]:
        warnings.append(
            f"Quadratic P(ε) zero crossing ε₀={pq['epsilon0']:+.6f} lies outside "
            f"the sampled range [{min(eps):.4f}, {max(eps):.4f}]"
        )

    consistency_status = (
        "consistent"
        if delta_a0_methods <= A0_TOL_ANG and delta_B_methods <= B_TOL_GPA
        else "inconsistent"
    )

    results = {
        # Sampling metadata
        "n_points":              n,
        "epsilon_min_sampled":   min(eps),
        "epsilon_max_sampled":   max(eps),
        "a_ref_angstrom":        a_ref,

        # ── PRIMARY: 3rd-order Birch-Murnaghan on E(V) ────────────────────
        "fit_method":                 "birch_murnaghan_3rd_order_EV",
        "a0_fit_angstrom":            bm3["a0_angstrom"],
        "bulk_modulus_gpa":           bm3["B0_gpa"],
        "bm3_V0_angstrom3":           bm3["V0_angstrom3"],
        "bm3_a0_angstrom":            bm3["a0_angstrom"],
        "bm3_B0_gpa":                 bm3["B0_gpa"],
        "bm3_B0_prime":               bm3["B0_prime"],
        "bm3_E0_ry":                  bm3["E0_ry"],
        "bm3_E0_ev":                  bm3["E0_ry"] * RY_TO_EV,
        "bm3_rms_residual_ry":        bm3["rms_residual_ry"],

        # 2nd-order BM (B0' ≡ 4) — stability check on the 4th parameter
        "bm2_a0_angstrom":            bm2["a0_angstrom"] if bm2 else None,
        "bm2_B0_gpa":                 bm2["B0_gpa"] if bm2 else None,
        "bm2_rms_residual_ry":        bm2["rms_residual_ry"] if bm2 else None,

        # ── CROSS-CHECK: quadratic P(ε) ───────────────────────────────────
        "pquad_epsilon0":             pq["epsilon0"],
        "pquad_a0_angstrom":          pq["a0_angstrom"],
        "pquad_B_gpa":                pq["B_gpa"],
        "pquad_dP_depsilon_kbar":     pq["dP_depsilon_kbar"],
        "pquad_rms_residual_kbar":    pq["rms_residual_kbar"],

        # ── Method agreement ──────────────────────────────────────────────
        "delta_a0_methods_angstrom":  delta_a0_methods,
        "delta_B_methods_gpa":        delta_B_methods,
        "consistency_status":         consistency_status,

        # ── LEGACY (biased) fits, retained so the bias stays visible ──────
        "legacy_epsilon0_pressure_linear":   eps0_pressure,
        "legacy_a0_pressure_linear_angstrom": a0_pressure,
        "legacy_pressure_linear_slope_kbar_per_eps": m_P,
        "legacy_pressure_linear_intercept_kbar":     b_P,
        "legacy_pressure_linear_rms_kbar":   pl["rms_residual_kbar"],
        "legacy_V0_energy_volume_quadratic_angstrom3": V0_ev,
        "legacy_a0_energy_volume_quadratic_angstrom":  a0_ev,
        "legacy_bulk_modulus_pv_linear_gpa": B_gpa_legacy,
        "legacy_epsilon0_energy_quadratic":  eps0_energy,
        "legacy_a0_energy_quadratic_angstrom": a0_energy,
        "legacy_E0_energy_quadratic_ry":     E0_ry,

        "warnings":            "; ".join(warnings) if warnings else "",

        # Internal (not exported to CSV/JSON)
        "_a_ref":    a_ref,
        "_eps":      eps,
        "_E":        E,
        "_P":        P,
        "_V":        V,
        "_points":   points,
        "_bm3":      bm3,
        "_bm2":      bm2,
        "_pq":       pq,
        "_pl":       pl,
        "_m_PV":     m_PV,
        "_b_PV":     b_PV,
        "_energy_fit":  (A_e, B_e, C_e),
        "_volume_fit":  (A_v, B_v, C_v),
    }
    return results, warnings


# ── Output writers ─────────────────────────────────────────────────────────────

FIT_FIELDS = [
    "n_points", "epsilon_min_sampled", "epsilon_max_sampled", "a_ref_angstrom",
    # primary
    "fit_method", "a0_fit_angstrom", "bulk_modulus_gpa",
    "bm3_V0_angstrom3", "bm3_a0_angstrom", "bm3_B0_gpa", "bm3_B0_prime",
    "bm3_E0_ry", "bm3_E0_ev", "bm3_rms_residual_ry",
    # stability check
    "bm2_a0_angstrom", "bm2_B0_gpa", "bm2_rms_residual_ry",
    # cross-check
    "pquad_epsilon0", "pquad_a0_angstrom", "pquad_B_gpa",
    "pquad_dP_depsilon_kbar", "pquad_rms_residual_kbar",
    # agreement
    "delta_a0_methods_angstrom", "delta_B_methods_gpa", "consistency_status",
    # legacy / bias record
    "legacy_epsilon0_pressure_linear", "legacy_a0_pressure_linear_angstrom",
    "legacy_pressure_linear_slope_kbar_per_eps",
    "legacy_pressure_linear_intercept_kbar", "legacy_pressure_linear_rms_kbar",
    "legacy_V0_energy_volume_quadratic_angstrom3",
    "legacy_a0_energy_volume_quadratic_angstrom",
    "legacy_bulk_modulus_pv_linear_gpa",
    "legacy_epsilon0_energy_quadratic", "legacy_a0_energy_quadratic_angstrom",
    "legacy_E0_energy_quadratic_ry",
    "warnings",
]


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow({k: results.get(k, "") for k in FIT_FIELDS})


def write_json(results, path):
    out = {k: results[k] for k in FIT_FIELDS if k in results}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _f(v, fmt=".6f"):
    if v is None:
        return "—"
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def write_markdown(results, warnings, path):
    points = results["_points"]
    bm3    = results["_bm3"]
    bm2    = results["_bm2"]
    pq     = results["_pq"]
    pl     = results["_pl"]
    a_ref  = results["_a_ref"]
    a0     = results["a0_fit_angstrom"]
    B_gpa  = results["bulk_modulus_gpa"]

    lines = [
        "# Bulk Diamond Reference — Equation-of-State Fit",
        "",
        "## Project Context",
        "",
        "This report summarises the equation-of-state analysis of the PBE/SSSP",
        "bulk diamond reference series computed with Quantum ESPRESSO `pw.x`.",
        "A five-point hydrostatic strain series (ε = −0.010 … +0.010) was fitted",
        "to extract the equilibrium lattice constant and bulk modulus.  These",
        "values define the zero-strain baseline for downstream slab",
        "surface-energy, surface-stress, Raman-shift, and NV-centre analyses.",
        "",
        "**Method.** The primary fit is a 3rd-order Birch-Murnaghan EOS on E(V).",
        "A quadratic P(ε) fit is reported as an independent cross-check.  Both",
        "are shown; disagreement is flagged rather than silently resolved.",
        "A *linear* P(ε) fit — used by this script before 2026-08 — is biased,",
        "because P(ε) is strongly curved over ±1 %; it is retained below under",
        "'Superseded fits' so the size of that bias stays visible.",
        "",
        "**ε is computed from the cell volume**, ε = (V/V_ref)^(1/3) − 1, not",
        "read from the run folder name (CLAUDE.md §1: read the geometry).  The",
        f"reference is V_ref = {a_ref**3:.4f} Å³, a_ref = {a_ref:.5f} Å.  The",
        "fitted a₀ does not depend on that choice; only the reported ε₀ does.",
        "",
        "## Input Data",
        "",
        "| folder | ε (from V) | a (Å) | V (Å³) | E (Ry) | P (kbar) |",
        "|--------|-----------|-------|--------|--------|---------|",
    ]
    for p in points:
        a_val = p.get("a_from_volume_angstrom") or p.get("a_angstrom")
        lines.append(
            f"| {p['folder_name']} "
            f"| {p['epsilon']:+.6f} "
            f"| {_f(a_val, '.5f')} "
            f"| {p['volume_angstrom3']:.4f} "
            f"| {p['energy_ry']:.8f} "
            f"| {p['pressure_kbar']:.2f} |"
        )

    lines += [
        "",
        "## 1. PRIMARY — 3rd-order Birch-Murnaghan on E(V)",
        "",
        "BM3 is exactly a cubic polynomial in x = V^(−2/3), so this is a linear",
        "least-squares fit; V₀, B₀ and B₀′ follow analytically from the",
        "polynomial's derivatives at its minimum.",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| V₀ (Å³) | {bm3['V0_angstrom3']:.5f} |",
        f"| **a₀ (Å)** | **{bm3['a0_angstrom']:.6f}** |",
        f"| **B₀ (GPa)** | **{bm3['B0_gpa']:.2f}** |",
        f"| B₀′ | {bm3['B0_prime']:.3f} |",
        f"| E₀ (Ry) | {bm3['E0_ry']:.8f} |",
        f"| fit residual rms (Ry) | {bm3['rms_residual_ry']:.3e} |",
        f"| V₀ inside sampled range | {bm3['V0_in_sampled_range']} |",
        "",
        "**Stability of the 4th parameter.** With 5 points a BM3 fit has one",
        "residual degree of freedom, so B₀′ is the least-constrained quantity.",
        "Refitting with B₀′ fixed at 4 (2nd-order BM, 3 parameters) gives:",
        "",
        "| | a₀ (Å) | B₀ (GPa) | rms (Ry) |",
        "|---|--------|----------|----------|",
        f"| BM3 (B₀′ free = {bm3['B0_prime']:.3f}) | {bm3['a0_angstrom']:.6f} | "
        f"{bm3['B0_gpa']:.2f} | {bm3['rms_residual_ry']:.3e} |",
    ]
    if bm2 is None:
        lines.append("| BM2 (B₀′ ≡ 4) | unavailable — see Warnings | | |")
    else:
        lines += [
            f"| BM2 (B₀′ ≡ 4) | {bm2['a0_angstrom']:.6f} | {bm2['B0_gpa']:.2f} | "
            f"{bm2['rms_residual_ry']:.3e} |",
            f"| difference | {abs(bm3['a0_angstrom']-bm2['a0_angstrom']):.6f} | "
            f"{abs(bm3['B0_gpa']-bm2['B0_gpa']):.2f} | — |",
        ]
    lines += [
        "",
        "## 2. CROSS-CHECK — quadratic P(ε)",
        "",
        "B from the cross-check uses V = V_ref(1+ε)³, so B = −(1/3) dP/dε|₍ε₀₎.",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| ε₀ | {pq['epsilon0']:+.6f} |",
        f"| a₀ (Å) | {pq['a0_angstrom']:.6f} |",
        f"| B (GPa) | {pq['B_gpa']:.2f} |",
        f"| dP/dε at ε₀ (kbar) | {pq['dP_depsilon_kbar']:.1f} |",
        f"| fit residual rms (kbar) | {pq['rms_residual_kbar']:.4f} |",
        "",
        "## 3. Method agreement",
        "",
        "| Quantity | BM3 E(V) | quadratic P(ε) | |Δ| |",
        "|----------|----------|----------------|-----|",
        f"| a₀ (Å) | {bm3['a0_angstrom']:.6f} | {pq['a0_angstrom']:.6f} | "
        f"{results['delta_a0_methods_angstrom']:.6f} |",
        f"| B (GPa) | {bm3['B0_gpa']:.2f} | {pq['B_gpa']:.2f} | "
        f"{results['delta_B_methods_gpa']:.2f} |",
        "",
        f"**Status: {results['consistency_status'].upper()}**",
        "",
        "## 4. Superseded fits (biased — recorded, not used)",
        "",
        "| Fit | ε₀ | a₀ (Å) | B (GPa) | rms |",
        "|-----|----|--------|---------|-----|",
        f"| P(ε) **linear** | {results['legacy_epsilon0_pressure_linear']:+.6f} | "
        f"{results['legacy_a0_pressure_linear_angstrom']:.6f} | — | "
        f"{pl['rms_residual_kbar']:.3f} kbar |",
        f"| P(ε) quadratic (cross-check above) | {pq['epsilon0']:+.6f} | "
        f"{pq['a0_angstrom']:.6f} | {pq['B_gpa']:.2f} | "
        f"{pq['rms_residual_kbar']:.4f} kbar |",
        f"| E(V) **quadratic** minimum | — | "
        f"{results['legacy_a0_energy_volume_quadratic_angstrom']:.6f} | "
        f"{results['legacy_bulk_modulus_pv_linear_gpa']:.2f} (from linear P(V)) | — |",
        f"| E(ε) quadratic minimum | "
        f"{results['legacy_epsilon0_energy_quadratic']:+.6f} | "
        f"{results['legacy_a0_energy_quadratic_angstrom']:.6f} | — | — |",
        "",
        f"The linear P(ε) rms is {pl['rms_residual_kbar']:.2f} kbar against "
        f"{pq['rms_residual_kbar']:.3f} kbar for the quadratic — a factor of "
        f"{pl['rms_residual_kbar']/max(pq['rms_residual_kbar'], 1e-12):.0f}. "
        "That residual structure is the curvature the linear fit cannot",
        "represent, and it biases ε₀ by "
        f"{abs(results['legacy_epsilon0_pressure_linear'] - pq['epsilon0']):.6f} "
        "in strain.",
        "",
        "## Interpretation",
        "",
        f"The recommended **PBE/SSSP bulk reference lattice constant** is",
        f"**a₀ = {a0:.6f} Å** (3rd-order Birch-Murnaghan on E(V), "
        f"V₀ = {bm3['V0_angstrom3']:.4f} Å³), with the quadratic P(ε)",
        f"cross-check giving {pq['a0_angstrom']:.6f} Å.",
        "",
        f"The bulk modulus is **B₀ = {B_gpa:.2f} GPa** with B₀′ = "
        f"{bm3['B0_prime']:.2f}; the cross-check gives {pq['B_gpa']:.2f} GPa.",
        "",
        f"a_ref = {a_ref:.5f} Å is smaller than the fitted a₀, i.e. the input "
        f"lattice constant sits on the compressed side of the PBE/SSSP "
        f"equilibrium; removing that residual would need ~"
        f"{(a0/a_ref - 1)*100:.3f}% isotropic expansion.",
    ]

    if 400 <= B_gpa <= 470:
        lines.append("")
        lines.append(
            "B₀ is in the expected range for PBE diamond (~430–445 GPa "
            "depending on EOS form and strain window)."
        )
    elif 350 <= B_gpa <= 550:
        lines.append("")
        lines.append(
            "B₀ is physically reasonable for diamond (PBE bulk modulus ~430 GPa)."
        )
    else:
        lines.append("")
        lines.append(
            "B₀ deviates from the typical PBE diamond range (~430 GPa); "
            "consider extending the strain range or verifying the input data."
        )

    lines += [
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("None.")
    lines.append("")

    path.write_text("\n".join(lines))


# ── Optional matplotlib plots ──────────────────────────────────────────────────

def try_plots(results, outdir):
    """
    Generate PNG plots if matplotlib is available.
    Returns list of written Paths, or None if matplotlib is absent.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    eps   = results["_eps"]
    E     = results["_E"]
    P     = results["_P"]
    V     = results["_V"]
    bm3   = results["_bm3"]
    A_e, B_e, C_e = results["_energy_fit"]
    A_v, B_v, C_v = results["_volume_fit"]
    m_P, b_P      = (results["legacy_pressure_linear_slope_kbar_per_eps"],
                     results["legacy_pressure_linear_intercept_kbar"])
    eps0_e = results["legacy_epsilon0_energy_quadratic"]
    eps0_p = results["pquad_epsilon0"]
    V0     = bm3["V0_angstrom3"]

    # Re-fit the two curved models so the plotted lines are the ones actually used
    pquad = ScaledPoly(eps, P, 2)
    bm_poly = ScaledPoly([v ** (-2.0 / 3.0) for v in V], E, 3)

    pad_eps = (max(eps) - min(eps)) * 0.15
    pad_V   = (max(V)   - min(V))   * 0.15
    eps_fine = _linspace(min(eps) - pad_eps, max(eps) + pad_eps, 200)
    V_fine   = _linspace(min(V)   - pad_V,   max(V)   + pad_V,   200)

    written = []

    # 1. E vs ε
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps, E, "o", color="steelblue", zorder=3, label="DFT points")
    ax.plot(eps_fine,
            [A_e * e**2 + B_e * e + C_e for e in eps_fine],
            "-", color="steelblue", label="Quadratic fit")
    ax.axvline(eps0_e, color="crimson", ls="--", lw=0.9,
               label=f"ε₀ = {eps0_e:+.4f}")
    ax.set_xlabel("Hydrostatic strain ε")
    ax.set_ylabel("Total energy (Ry)")
    ax.set_title("E vs ε — PBE/SSSP bulk diamond")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "bulk_energy_vs_epsilon.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 2. P vs ε
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps, P, "s", color="darkorange", zorder=3, label="DFT points")
    ax.plot(eps_fine,
            [pquad.value(e) for e in eps_fine],
            "-", color="darkorange", label="Quadratic fit (used)")
    ax.plot(eps_fine,
            [m_P * e + b_P for e in eps_fine],
            "--", color="gray", lw=1.0, label="Linear fit (superseded)")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(eps0_p, color="crimson", ls="--", lw=0.9,
               label=f"ε₀ = {eps0_p:+.4f}")
    ax.axvline(results["legacy_epsilon0_pressure_linear"], color="gray",
               ls="--", lw=0.8,
               label=f"ε₀ linear = {results['legacy_epsilon0_pressure_linear']:+.4f}")
    ax.set_xlabel("Hydrostatic strain ε")
    ax.set_ylabel("Pressure (kbar)")
    ax.set_title("P vs ε — PBE/SSSP bulk diamond")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "bulk_pressure_vs_epsilon.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 3. E vs V
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(V, E, "^", color="mediumseagreen", zorder=3, label="DFT points")
    ax.plot(V_fine,
            [bm_poly.value(v ** (-2.0 / 3.0)) for v in V_fine],
            "-", color="mediumseagreen", label="Birch-Murnaghan 3rd order (used)")
    ax.plot(V_fine,
            [A_v * v**2 + B_v * v + C_v for v in V_fine],
            "--", color="gray", lw=1.0, label="Quadratic fit (superseded)")
    ax.axvline(V0, color="crimson", ls="--", lw=0.9,
               label=f"V₀ = {V0:.3f} Å³")
    ax.set_xlabel("Volume (Å³)")
    ax.set_ylabel("Total energy (Ry)")
    ax.set_title("E vs V — PBE/SSSP bulk diamond")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "bulk_energy_vs_volume.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    return written


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit hydrostatic bulk diamond reference EOS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 fit_bulk_reference.py
  python3 fit_bulk_reference.py --input results/reference_diamond/reference_summary.csv
  python3 fit_bulk_reference.py --outdir results/reference_diamond
  python3 fit_bulk_reference.py --no-plots
""",
    )
    parser.add_argument(
        "--input",
        default="results/reference_diamond/reference_summary.csv",
        metavar="CSV",
        help="Path to reference_summary.csv",
    )
    parser.add_argument(
        "--outdir",
        default="results/reference_diamond",
        metavar="DIR",
        help="Output directory for fit results (default: results/reference_diamond)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib plot generation",
    )
    parser.add_argument(
        "--strain-type", default="hydrostatic", metavar="NAME",
        help="Value of the strain_type column to select (default: hydrostatic). "
             "Pass 'any' to disable the filter — needed for series whose folders "
             "are named '<prefix>~eps_<value>', which parse_reference.py records "
             "with the folder name in strain_type.",
    )
    parser.add_argument(
        "--update-config", metavar="JSON",
        help="Write the fitted a0/bulk modulus into this reference config "
             "(e.g. config/reference_pbe_sssp.json). Elastic tensor (C11/C12/C44) "
             "and citation are left untouched. Not run by default.",
    )
    parser.add_argument(
        "--supersede-reason", metavar="TEXT",
        help="With --update-config, archive the config's existing "
             "bulk_reference values under bulk_reference.superseded with this "
             "reason attached, instead of overwriting them silently.",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    outdir   = Path(args.outdir)

    if not csv_path.exists():
        print(f"Error: input file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    strain_type = None if args.strain_type.lower() == "any" else args.strain_type

    print(f"Loading data from: {csv_path}")
    points = load_data(csv_path, strain_type=strain_type)

    if not points:
        print(
            "Error: no qualifying rows found (need calculation_type=scf, "
            f"strain_type={args.strain_type}, complete=True, with numeric "
            "E/P/V). Try --strain-type any.",
            file=sys.stderr,
        )
        sys.exit(1)

    a_ref, load_warnings = assign_epsilon_from_volume(points)
    run_settings, settings_warnings = collect_run_settings(points)
    load_warnings.extend(settings_warnings)

    print(f"Loaded {len(points)} qualifying points (ε recomputed from volume, "
          f"a_ref = {a_ref:.6f} Å):")
    for p in points:
        print(
            f"  ε={p['epsilon']:+.6f}  "
            f"E={p['energy_ry']:.8f} Ry  "
            f"P={p['pressure_kbar']:8.2f} kbar  "
            f"V={p['volume_angstrom3']:.4f} Å³"
        )

    print("\nRunning fits...")
    results, warnings = run_analysis(points, a_ref=a_ref,
                                     prior_warnings=load_warnings)

    print(f"\n{'─'*66}")
    print("EQUILIBRIUM ESTIMATES")
    print(f"{'─'*66}")
    print(f"  PRIMARY  Birch-Murnaghan 3rd order on E(V)")
    print(f"           V₀  = {results['bm3_V0_angstrom3']:.5f} Å³")
    print(f"           a₀  = {results['bm3_a0_angstrom']:.6f} Å")
    print(f"           B₀  = {results['bm3_B0_gpa']:.2f} GPa")
    print(f"           B₀' = {results['bm3_B0_prime']:.3f}")
    print(f"           rms = {results['bm3_rms_residual_ry']:.3e} Ry")
    print(f"  CHECK    quadratic P(ε)")
    print(f"           ε₀  = {results['pquad_epsilon0']:+.6f}")
    print(f"           a₀  = {results['pquad_a0_angstrom']:.6f} Å")
    print(f"           B   = {results['pquad_B_gpa']:.2f} GPa")
    print(f"           rms = {results['pquad_rms_residual_kbar']:.4f} kbar")
    print(f"  AGREEMENT  Δa₀ = {results['delta_a0_methods_angstrom']:.6f} Å   "
          f"ΔB = {results['delta_B_methods_gpa']:.2f} GPa   "
          f"→ {results['consistency_status'].upper()}")
    print(f"  SUPERSEDED (biased, for reference)")
    print(f"           P(ε) linear:      a₀ = "
          f"{results['legacy_a0_pressure_linear_angstrom']:.6f} Å  "
          f"(rms {results['legacy_pressure_linear_rms_kbar']:.2f} kbar)")
    print(f"           E(V) quadratic:   a₀ = "
          f"{results['legacy_a0_energy_volume_quadratic_angstrom']:.6f} Å")
    print(f"           P(V) linear B:    B  = "
          f"{results['legacy_bulk_modulus_pv_linear_gpa']:.2f} GPa")

    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    ! {w}")

    csv_out  = outdir / "bulk_fit_summary.csv"
    json_out = outdir / "bulk_fit_summary.json"
    md_out   = outdir / "bulk_fit_report.md"

    write_csv(results, csv_out)
    write_json(results, json_out)
    write_markdown(results, warnings, md_out)

    print(f"\nOutput files written:")
    print(f"  CSV:      {csv_out}")
    print(f"  JSON:     {json_out}")
    print(f"  Markdown: {md_out}")

    if not args.no_plots:
        plot_paths = try_plots(results, outdir)
        if plot_paths is None:
            msg = "matplotlib not available; plots skipped"
            print(f"  ! {msg}")
            warnings.append(msg)
            results["warnings"] = "; ".join(warnings)
            write_csv(results, csv_out)
            write_json(results, json_out)
            write_markdown(results, warnings, md_out)
        else:
            for pp in plot_paths:
                print(f"  Plot:     {pp}")
    else:
        print("  (plots skipped — --no-plots)")

    if args.update_config:
        elastic_reference.update_bulk_reference(
            args.update_config,
            a0_angstrom=results["a0_fit_angstrom"],
            bulk_modulus_gpa=results["bulk_modulus_gpa"],
            source_summary_json=str(json_out),
            extra_fields={
                "fit_method": results["fit_method"],
                "fit_method_description": (
                    "3rd-order Birch-Murnaghan fit to E(V) over the hydrostatic "
                    "series; B0 = V d2E/dV2 at the fitted V0. Cross-checked "
                    "against a quadratic P(epsilon) fit."
                ),
                "bm3_B0_prime": results["bm3_B0_prime"],
                "bm3_V0_angstrom3": results["bm3_V0_angstrom3"],
                "bm3_rms_residual_ry": results["bm3_rms_residual_ry"],
                "crosscheck_method": "pressure_epsilon_quadratic",
                "crosscheck_a0_angstrom": results["pquad_a0_angstrom"],
                "crosscheck_bulk_modulus_gpa": results["pquad_B_gpa"],
                "delta_a0_methods_angstrom": results["delta_a0_methods_angstrom"],
                "delta_B_methods_gpa": results["delta_B_methods_gpa"],
                "fit_status": results["consistency_status"],
                "source_summary_csv": str(csv_out),
                "source_report": str(md_out),
                "fitted_with_qe_settings": run_settings,
            },
            supersede_reason=args.supersede_reason,
            # Outputs of the pre-2026-08 fit that this method does not produce.
            # Left in place they would sit next to the new numbers and read as
            # current; they survive in bulk_reference.superseded.
            stale_keys=[
                "a0_energy_fit_angstrom",
                "a0_pressure_fit_angstrom",
                "epsilon0_energy_fit",
                "epsilon0_pressure_fit",
            ],
        )
        print(f"  Updated config: {args.update_config} (a0, bulk_modulus_gpa and "
              "fit provenance; elastic_tensor untouched)")
        if args.supersede_reason:
            print("  Previous bulk_reference values archived under "
                  "bulk_reference.superseded")

        # The config carries a top-level qe_settings block describing the
        # reference calculation. It is maintained by hand, so it can drift away
        # from the series actually fitted — which is exactly the kind of silent
        # provenance error CLAUDE.md §1.7 warns about. Compare and complain.
        try:
            cfg_now = json.loads(Path(args.update_config).read_text())
        except (OSError, json.JSONDecodeError):
            cfg_now = None
        if cfg_now:
            declared = cfg_now.get("qe_settings", {})
            for key in ("ecutwfc", "ecutrho"):
                dv, fv = declared.get(key), run_settings.get(key)
                if dv is not None and fv is not None and float(dv) != float(fv):
                    print(
                        f"  ! WARNING: config qe_settings.{key} = {dv} but the "
                        f"fitted series was run at {fv} (from pw.out). The "
                        f"top-level qe_settings block is stale — fix it, or the "
                        f"config will misreport how its own a0 was obtained.",
                        file=sys.stderr,
                    )

    print()


if __name__ == "__main__":
    main()
