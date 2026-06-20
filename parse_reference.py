#!/usr/bin/env python3
"""
parse_reference.py — Parse QE bulk diamond reference results.

Scans a results directory for completed Quantum ESPRESSO pw.x runs,
extracts energies, stresses, geometry, and metadata, computes derived
quantities, and writes machine-readable summaries for downstream
energy-volume fitting, pressure-strain analysis, Raman calibration, etc.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

# ── Physical constants ──────────────────────────────────────────────────────
RY_TO_EV = 13.605693122994       # 1 Ry in eV  (NIST 2018 CODATA)
BOHR_TO_ANG = 0.529177210903     # 1 bohr in Å  (NIST 2018 CODATA)
KBAR_TO_GPA = 0.1                # 1 kbar in GPa

# QE internal conversion: 1 Ry/bohr³ = 147105 kbar  (matches QE uakbar)
RY_BOHR3_TO_KBAR = 147105.0

# ── Hydrostatic quality thresholds (kbar) ───────────────────────────────────
_ANISO_WARN = 0.5
_ANISO_BAD = 2.0
_SHEAR_WARN = 0.5
_SHEAR_BAD = 2.0

# ── Regex helpers ────────────────────────────────────────────────────────────
_NUM = r"[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?"  # matches floats in scientific / fixed notation


# ── Directory discovery ──────────────────────────────────────────────────────

def find_result_dirs(root: Path) -> list:
    """Return all directories under root that contain a pw.out file."""
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if "pw.out" in files:
            found.append(Path(dirpath))
    return found


# ── Cell-parameter parsing (shared between pw.in and pw.out) ─────────────────

def _parse_cell_parameters_text(text: str, last: bool = True):
    """
    Locate CELL_PARAMETERS block(s) and return the lattice matrix in Å.

    Handles units: angstrom (default), bohr.  Returns None for alat units
    (requires separate alat value).  If last=True, returns the final block
    (vc-relax final geometry); otherwise the first.
    """
    pattern = re.compile(
        r"CELL_PARAMETERS\s*[\({]?\s*(angstrom|bohr|alat)?\s*[\)}]?\s*\n"
        r"\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*\n"
        r"\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*\n"
        r"\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    m = matches[-1] if last else matches[0]
    unit = (m.group(1) or "angstrom").lower().strip()
    if unit == "alat":
        return None  # caller must handle with alat value
    scale = BOHR_TO_ANG if unit == "bohr" else 1.0
    rows = []
    for i in range(3):
        base = 2 + i * 3
        rows.append([float(m.group(base + j)) * scale for j in range(3)])
    return rows  # 3×3 list of lists, in Å


def _crystal_axes_from_pw_out(text: str, alat_bohr: float):
    """
    Parse the 'crystal axes' block from pw.out and return the lattice matrix
    in Å.  Used as fallback when no CELL_PARAMETERS appears in pw.out.
    """
    m = re.search(
        r"crystal axes:.*?\n"
        r"\s*a\(1\) = \(\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*\).*?\n"
        r"\s*a\(2\) = \(\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*\).*?\n"
        r"\s*a\(3\) = \(\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*\)",
        text,
    )
    if not m:
        return None
    alat_ang = alat_bohr * BOHR_TO_ANG
    vals = [float(m.group(i + 1)) for i in range(9)]
    return [
        [vals[0] * alat_ang, vals[1] * alat_ang, vals[2] * alat_ang],
        [vals[3] * alat_ang, vals[4] * alat_ang, vals[5] * alat_ang],
        [vals[6] * alat_ang, vals[7] * alat_ang, vals[8] * alat_ang],
    ]


# ── File parsers ─────────────────────────────────────────────────────────────

def parse_pw_in(path: Path) -> dict:
    """Extract cell, species, and control parameters from pw.in."""
    data = {}
    try:
        text = path.read_text()
    except OSError:
        return data

    def _first(pattern, conv=str):
        m = re.search(pattern, text, re.IGNORECASE)
        return conv(m.group(1)) if m else None

    data["nat"] = _first(r"\bnat\s*=\s*(\d+)", int)
    data["ntyp"] = _first(r"\bntyp\s*=\s*(\d+)", int)
    data["ecutwfc"] = _first(r"\becutwfc\s*=\s*(" + _NUM + r")", float)
    data["ecutrho"] = _first(r"\becutrho\s*=\s*(" + _NUM + r")", float)
    data["calculation"] = _first(r"calculation\s*=\s*'([^']+)'")
    data["pseudo_dir"] = _first(r"pseudo_dir\s*=\s*'([^']+)'")

    # First pseudopotential file from ATOMIC_SPECIES block
    m = re.search(
        r"ATOMIC_SPECIES.*?\n\s*\S+\s+\S+\s+(\S+\.UPF)",
        text, re.IGNORECASE | re.DOTALL,
    )
    data["pseudo_C"] = m.group(1) if m else None

    # K_POINTS automatic  →  "N1 N2 N3"
    m = re.search(
        r"K_POINTS\s+automatic\s*\n\s*(\d+)\s+(\d+)\s+(\d+)",
        text, re.IGNORECASE,
    )
    data["kpoints"] = f"{m.group(1)} {m.group(2)} {m.group(3)}" if m else None

    # CELL_PARAMETERS
    cp = _parse_cell_parameters_text(text, last=False)
    data["cell_params_ang"] = cp  # may be None

    return data


def parse_pw_out(path: Path) -> dict:
    """Parse pw.out and return a flat dict of all extracted quantities."""
    d = {
        "status": "UNKNOWN",
        "job_done": False,
        "scf_converged": False,
        "energy_ry": None,
        "nat": None,
        "ntyp": None,
        "pressure_kbar": None,
        "stress_xx_ry_bohr3": None, "stress_yy_ry_bohr3": None, "stress_zz_ry_bohr3": None,
        "stress_xy_ry_bohr3": None, "stress_xz_ry_bohr3": None, "stress_yz_ry_bohr3": None,
        "stress_xx_kbar": None, "stress_yy_kbar": None, "stress_zz_kbar": None,
        "stress_xy_kbar": None, "stress_xz_kbar": None, "stress_yz_kbar": None,
        "cell_params_ang": None,
        "alat_bohr": None,
        "calculation_type": None,
        "scf_iterations": None,
        "ionic_steps": None,
        "wall_time": None,
        "warnings": [],
        "errors": [],
    }

    try:
        text = path.read_text()
    except OSError as e:
        d["errors"].append(f"Cannot read pw.out: {e}")
        d["status"] = "UNREADABLE"
        return d

    # ── Job completion ────────────────────────────────────────────────────
    if re.search(r"JOB DONE", text):
        d["job_done"] = True
        d["status"] = "JOB DONE"

    # ── Errors ────────────────────────────────────────────────────────────
    error_patterns = [
        r"%%%% Error[^\n]*",
        r"Error in routine[^\n]*",
        r"Segmentation fault[^\n]*",
        r"forrtl:[^\n]*",
        r"floating[- ]point exception[^\n]*",
    ]
    for pat in error_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            d["errors"].append(m.group(0).strip()[:200])
            if d["status"] == "UNKNOWN":
                d["status"] = "ERROR"

    # ── Warnings ─────────────────────────────────────────────────────────
    warn_patterns = [
        r"Warning:[^\n]*",
        r"using ibrav=0 with symmetry is DISCOURAGED[^\n]*",
        r"SCF correction compared to forces is large[^\n]*",
        r"This is a supercell, fractional translations are disabled[^\n]*",
    ]
    seen_warns = set()
    for pat in warn_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            w = m.group(0).strip()[:150]
            if w not in seen_warns:
                d["warnings"].append(w)
                seen_warns.add(w)

    # ── Calculation type ─────────────────────────────────────────────────
    m = re.search(r"calculation\s*=\s*'([^']+)'", text)
    if m:
        d["calculation_type"] = m.group(1).strip()
    elif re.search(r"Program PWSCF", text):
        d["calculation_type"] = "scf"

    # ── Atom / type counts ────────────────────────────────────────────────
    m = re.search(r"number of atoms/cell\s*=\s*(\d+)", text)
    if m:
        d["nat"] = int(m.group(1))
    m = re.search(r"number of atomic types\s*=\s*(\d+)", text)
    if m:
        d["ntyp"] = int(m.group(1))

    # ── Total energy (last converged value) ───────────────────────────────
    energies = re.findall(r"!\s+total energy\s*=\s*(" + _NUM + r")\s*Ry", text)
    if energies:
        d["energy_ry"] = float(energies[-1])

    # ── SCF convergence ───────────────────────────────────────────────────
    m = re.search(r"convergence has been achieved in\s+(\d+)\s+iteration", text)
    if m:
        d["scf_converged"] = True
        d["scf_iterations"] = int(m.group(1))

    # ── Ionic / BFGS steps ────────────────────────────────────────────────
    bfgs_steps = re.findall(r"number of bfgs steps\s*=\s*(\d+)", text, re.IGNORECASE)
    if bfgs_steps:
        d["ionic_steps"] = int(bfgs_steps[-1])
    else:
        count = len(re.findall(r"BFGS Geometry Optimization", text, re.IGNORECASE))
        if count:
            d["ionic_steps"] = count

    # ── alat ─────────────────────────────────────────────────────────────
    m = re.search(r"lattice parameter \(alat\)\s*=\s*(" + _NUM + r")\s*a\.u\.", text)
    if m:
        d["alat_bohr"] = float(m.group(1))

    # ── Stress tensor ─────────────────────────────────────────────────────
    # Header: "total   stress  (Ry/bohr**3)                   (kbar)     P=   21.55"
    stress_hdr = re.search(
        r"total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(" + _NUM + r")",
        text,
    )
    if stress_hdr:
        d["pressure_kbar"] = float(stress_hdr.group(1))
        # Each of the 3 rows: s_i1  s_i2  s_i3  (kbar_i1  kbar_i2  kbar_i3)
        block_start = stress_hdr.end()
        block = text[block_start: block_start + 600]
        row_re = re.compile(
            r"\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")"
            r"\s+(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")"
        )
        rows = row_re.findall(block)
        if len(rows) >= 3:
            r0, r1, r2 = rows[0], rows[1], rows[2]
            d["stress_xx_ry_bohr3"] = float(r0[0])
            d["stress_xy_ry_bohr3"] = float(r0[1])
            d["stress_xz_ry_bohr3"] = float(r0[2])
            d["stress_xx_kbar"]     = float(r0[3])
            d["stress_xy_kbar"]     = float(r0[4])
            d["stress_xz_kbar"]     = float(r0[5])

            d["stress_yy_ry_bohr3"] = float(r1[1])
            d["stress_yz_ry_bohr3"] = float(r1[2])
            d["stress_yy_kbar"]     = float(r1[4])
            d["stress_yz_kbar"]     = float(r1[5])

            d["stress_zz_ry_bohr3"] = float(r2[2])
            d["stress_zz_kbar"]     = float(r2[5])

    # ── Final cell (vc-relax writes CELL_PARAMETERS to output) ───────────
    cp = _parse_cell_parameters_text(text, last=True)
    if cp is not None:
        d["cell_params_ang"] = cp
    elif d["alat_bohr"] is not None:
        # Fallback: crystal axes block × alat
        cp = _crystal_axes_from_pw_out(text, d["alat_bohr"])
        d["cell_params_ang"] = cp  # may still be None

    # ── Wall time ─────────────────────────────────────────────────────────
    m = re.search(r"PWSCF\s*:\s*" + _NUM + r"s?\s*CPU\s+(" + _NUM + r")s?\s*WALL", text)
    if m:
        d["wall_time"] = float(m.group(1))

    return d


def parse_meta_json(path: Path) -> dict:
    """Load meta.json and normalize to internal field names."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    mapping = {
        "calculation":            "calculation_type",
        "strain_type":            "strain_type",
        "epsilon":                "epsilon",
        "lattice_constant_angstrom": "lattice_constant_angstrom",
        "functional":             "functional",
        "pseudopotential":        "pseudo_C",
        "pseudo_dir":             "pseudo_dir",
        "ecutwfc_Ry":             "ecutwfc",
        "ecutrho_Ry":             "ecutrho",
        "kpoints":                "kpoints",
        "cell_type":              "cell_type",
        "generated_at":           "generated_at",
    }
    meta = {}
    for src, dst in mapping.items():
        if src in raw:
            val = raw[src]
            if dst == "kpoints" and isinstance(val, list):
                val = " ".join(str(v) for v in val)
            meta[dst] = val
    return meta


def _infer_from_path(run_path: Path, root: Path) -> dict:
    """Infer strain_type and epsilon from directory structure."""
    info = {}
    try:
        parts = run_path.relative_to(root).parts
        if parts:
            info["strain_type"] = parts[0]
        if len(parts) >= 2:
            m = re.match(r"eps_([+-]?\d+\.\d+)", parts[1])
            if m:
                info["epsilon"] = float(m.group(1))
    except ValueError:
        pass
    return info


# ── Geometry / derived quantities ────────────────────────────────────────────

def _det3(m: list) -> float:
    """Determinant of a 3×3 list-of-lists."""
    a, b, c = m[0], m[1], m[2]
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _vec_norm(v: list) -> float:
    return math.sqrt(sum(x * x for x in v))


def geometry_quantities(cell_ang, nat) -> dict:
    """Compute volume, per-atom volume, a_angstrom, a_from_volume."""
    g = {}
    if cell_ang is None:
        return g
    vol = abs(_det3(cell_ang))
    g["volume_angstrom3"] = vol
    if nat and nat > 0:
        g["volume_per_atom_angstrom3"] = vol / nat
    # Cubic-equivalent lattice parameter from volume
    g["a_from_volume_angstrom"] = vol ** (1.0 / 3.0)
    # a_angstrom from cell norms — only assign if all three norms agree (cubic)
    norms = [_vec_norm(row) for row in cell_ang]
    if max(norms) - min(norms) < 0.01:
        g["a_angstrom"] = sum(norms) / 3.0
    return g


def _hydrostatic_quality(anisotropy, max_shear) -> str:
    if anisotropy is None or max_shear is None:
        return "unknown"
    if anisotropy > _ANISO_BAD or max_shear > _SHEAR_BAD:
        return "bad"
    if anisotropy > _ANISO_WARN or max_shear > _SHEAR_WARN:
        return "warning"
    return "good"


# ── Row builder ──────────────────────────────────────────────────────────────

def build_row(run_path: Path, root: Path) -> dict:
    """Parse one result directory into a unified data dictionary."""
    out_data  = parse_pw_out(run_path / "pw.out")
    in_data   = parse_pw_in(run_path / "pw.in") if (run_path / "pw.in").exists() else {}
    meta      = parse_meta_json(run_path / "meta.json") if (run_path / "meta.json").exists() else {}
    path_info = _infer_from_path(run_path, root)

    def first(*args):
        """Return the first non-None value."""
        for a in args:
            if a is not None:
                return a
        return None

    row = {}

    # Identifiers
    row["run_path"]    = str(run_path)
    row["folder_name"] = run_path.name

    # Strain / experiment metadata (meta > path_info)
    row["strain_type"] = first(meta.get("strain_type"), path_info.get("strain_type"))
    row["epsilon"]     = first(meta.get("epsilon"), path_info.get("epsilon"))
    row["cell_type"]   = meta.get("cell_type")
    row["functional"]  = meta.get("functional")

    # Calculation type
    row["calculation_type"] = first(
        meta.get("calculation_type"),
        out_data.get("calculation_type"),
        in_data.get("calculation"),
    )

    # Pseudopotential / cutoffs (meta > pw.in)
    row["pseudo_C"]   = first(meta.get("pseudo_C"),  in_data.get("pseudo_C"))
    row["pseudo_dir"] = first(meta.get("pseudo_dir"), in_data.get("pseudo_dir"))
    row["ecutwfc"]    = first(meta.get("ecutwfc"),    in_data.get("ecutwfc"))
    row["ecutrho"]    = first(meta.get("ecutrho"),    in_data.get("ecutrho"))
    row["kpoints"]    = first(meta.get("kpoints"),    in_data.get("kpoints"))

    # Status / completeness
    row["status"]  = out_data["status"]
    row["complete"] = bool(
        out_data["job_done"]
        and out_data["scf_converged"]
        and out_data["energy_ry"] is not None
    )

    # Atom counts
    nat  = first(out_data.get("nat"),  in_data.get("nat"))
    ntyp = first(out_data.get("ntyp"), in_data.get("ntyp"))
    row["nat"]  = nat
    row["ntyp"] = ntyp

    # Energy
    e_ry = out_data["energy_ry"]
    row["energy_ry"] = e_ry
    row["energy_ev"] = e_ry * RY_TO_EV if e_ry is not None else None
    if e_ry is not None and nat:
        row["energy_per_atom_ry"] = e_ry / nat
        row["energy_per_atom_ev"] = e_ry * RY_TO_EV / nat
    else:
        row["energy_per_atom_ry"] = None
        row["energy_per_atom_ev"] = None

    # Pressure / stress
    row["pressure_kbar"] = out_data["pressure_kbar"]
    p_kbar = out_data["pressure_kbar"]
    row["pressure_gpa"] = p_kbar * KBAR_TO_GPA if p_kbar is not None else None

    for comp in ("xx", "yy", "zz", "xy", "xz", "yz"):
        row[f"stress_{comp}_ry_bohr3"] = out_data.get(f"stress_{comp}_ry_bohr3")
        row[f"stress_{comp}_kbar"]     = out_data.get(f"stress_{comp}_kbar")

    # Derived stress quantities
    normals = [row.get(f"stress_{c}_kbar") for c in ("xx", "yy", "zz")]
    shears  = [row.get(f"stress_{c}_kbar") for c in ("xy", "xz", "yz")]

    if all(v is not None for v in normals):
        row["mean_normal_stress_kbar"]  = sum(normals) / 3.0
        row["stress_anisotropy_kbar"]   = max(normals) - min(normals)
    else:
        row["mean_normal_stress_kbar"] = None
        row["stress_anisotropy_kbar"]  = None

    if all(v is not None for v in shears):
        row["max_abs_shear_stress_kbar"] = max(abs(v) for v in shears)
    else:
        row["max_abs_shear_stress_kbar"] = None

    row["hydrostatic_quality"] = _hydrostatic_quality(
        row["stress_anisotropy_kbar"], row["max_abs_shear_stress_kbar"]
    )

    # Geometry — prefer final cell from pw.out (vc-relax), fall back to pw.in
    cell = first(out_data.get("cell_params_ang"), in_data.get("cell_params_ang"))
    geo = geometry_quantities(cell, nat)
    row["volume_angstrom3"]        = geo.get("volume_angstrom3")
    row["volume_per_atom_angstrom3"] = geo.get("volume_per_atom_angstrom3")
    row["a_angstrom"]              = geo.get("a_angstrom")
    row["a_from_volume_angstrom"]  = geo.get("a_from_volume_angstrom")

    # SCF / ionic / timing
    row["scf_iterations"] = out_data.get("scf_iterations")
    row["ionic_steps"]    = out_data.get("ionic_steps")
    row["wall_time"]      = out_data.get("wall_time")

    # Warnings / errors as semicolon-separated strings
    row["warnings"] = "; ".join(out_data.get("warnings", []))
    row["errors"]   = "; ".join(out_data.get("errors", []))

    # needs_attention flag
    reasons = []
    if not row["complete"]:
        reasons.append("incomplete")
    if not out_data["job_done"]:
        reasons.append("job_not_done")
    if row["energy_ry"] is None:
        reasons.append("missing_energy")
    if row["pressure_kbar"] is None:
        reasons.append("missing_stress")
    if row["hydrostatic_quality"] in ("warning", "bad"):
        reasons.append(f"hydrostatic_{row['hydrostatic_quality']}")
    if row["errors"]:
        reasons.append("errors_in_output")
    if not (run_path / "meta.json").exists():
        reasons.append("no_meta_json")

    row["needs_attention"]   = bool(reasons)
    row["attention_reasons"] = "; ".join(reasons)

    return row


# ── Sorting ──────────────────────────────────────────────────────────────────

def sort_rows(rows: list) -> list:
    """Sort: hydrostatic by epsilon, relax after, then everything else."""
    def _key(r):
        st  = r.get("strain_type") or "zzz"
        eps = r.get("epsilon") if r.get("epsilon") is not None else 999.0
        return (st, eps, r.get("folder_name") or "")
    return sorted(rows, key=_key)


# ── Output writers ────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "run_path", "folder_name", "calculation_type", "strain_type", "epsilon",
    "cell_type", "functional", "pseudo_C", "pseudo_dir", "ecutwfc", "ecutrho",
    "kpoints", "status", "complete", "needs_attention",
    "energy_ry", "energy_ev", "energy_per_atom_ry", "energy_per_atom_ev",
    "pressure_kbar", "pressure_gpa",
    "stress_xx_kbar", "stress_yy_kbar", "stress_zz_kbar",
    "stress_xy_kbar", "stress_xz_kbar", "stress_yz_kbar",
    "mean_normal_stress_kbar", "stress_anisotropy_kbar",
    "max_abs_shear_stress_kbar", "hydrostatic_quality",
    "volume_angstrom3", "volume_per_atom_angstrom3",
    "a_angstrom", "a_from_volume_angstrom",
    "nat", "ntyp", "scf_iterations", "ionic_steps", "wall_time",
    "warnings", "errors",
]


def write_csv(rows: list, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})


def write_json(rows: list, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, default=str)


def _f(v, fmt=".6f") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def write_markdown(rows: list, path: Path) -> None:
    completed   = [r for r in rows if r.get("complete")]
    incomplete  = [r for r in rows if not r.get("complete")]
    needs_attn  = [r for r in rows if r.get("needs_attention")]

    lines = [
        "# Bulk Diamond Reference — Calculation Summary",
        "",
        f"**Total runs found:** {len(rows)}  ",
        f"**Completed:** {len(completed)}  ",
        f"**Incomplete / failed:** {len(incomplete)}  ",
        f"**Needs attention:** {len(needs_attn)}  ",
        "",
    ]

    if completed:
        lines += [
            "## Completed Calculations",
            "",
            "| folder | strain | ε | E (Ry) | P (kbar) | P (GPa) | V (Å³) | a_vol (Å) | quality |",
            "|--------|--------|---|--------|----------|---------|--------|-----------|---------|",
        ]
        for r in completed:
            lines.append(
                f"| {r['folder_name']} "
                f"| {r.get('strain_type') or '—'} "
                f"| {_f(r.get('epsilon'), '.4f')} "
                f"| {_f(r.get('energy_ry'), '.8f')} "
                f"| {_f(r.get('pressure_kbar'), '.2f')} "
                f"| {_f(r.get('pressure_gpa'), '.4f')} "
                f"| {_f(r.get('volume_angstrom3'), '.4f')} "
                f"| {_f(r.get('a_from_volume_angstrom'), '.5f')} "
                f"| {r.get('hydrostatic_quality') or '—'} |"
            )
        lines.append("")

    if incomplete:
        lines += ["## Incomplete / Failed Calculations", ""]
        for r in incomplete:
            lines.append(
                f"- **{r['folder_name']}** "
                f"status: `{r.get('status', '?')}` "
                f"— errors: {r.get('errors') or 'none'}"
            )
        lines.append("")

    if needs_attn:
        lines += ["## Needs Attention", ""]
        for r in needs_attn:
            lines.append(
                f"- **{r['folder_name']}** — {r.get('attention_reasons', '')}"
            )
        lines.append("")

    # Stress detail table (completed only)
    if completed:
        lines += [
            "## Stress Detail (completed runs)",
            "",
            "| folder | σ_xx (kbar) | σ_yy (kbar) | σ_zz (kbar) "
            "| σ_xy (kbar) | σ_xz (kbar) | σ_yz (kbar) "
            "| shear_max (kbar) | anisotropy (kbar) |",
            "|--------|-------------|-------------|-------------|"
            "-------------|-------------|-------------|"
            "------------------|-------------------|",
        ]
        for r in completed:
            lines.append(
                f"| {r['folder_name']} "
                f"| {_f(r.get('stress_xx_kbar'), '.2f')} "
                f"| {_f(r.get('stress_yy_kbar'), '.2f')} "
                f"| {_f(r.get('stress_zz_kbar'), '.2f')} "
                f"| {_f(r.get('stress_xy_kbar'), '.3f')} "
                f"| {_f(r.get('stress_xz_kbar'), '.3f')} "
                f"| {_f(r.get('stress_yz_kbar'), '.3f')} "
                f"| {_f(r.get('max_abs_shear_stress_kbar'), '.3f')} "
                f"| {_f(r.get('stress_anisotropy_kbar'), '.3f')} |"
            )
        lines.append("")

    # Energy/volume summary
    if completed:
        lines += [
            "## Energy per Atom (completed runs)",
            "",
            "| folder | E/atom (Ry) | E/atom (eV) | V/atom (Å³) |",
            "|--------|-------------|-------------|-------------|",
        ]
        for r in completed:
            lines.append(
                f"| {r['folder_name']} "
                f"| {_f(r.get('energy_per_atom_ry'), '.8f')} "
                f"| {_f(r.get('energy_per_atom_ev'), '.6f')} "
                f"| {_f(r.get('volume_per_atom_angstrom3'), '.5f')} |"
            )
        lines.append("")

    path.write_text("\n".join(lines))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse Quantum ESPRESSO bulk diamond reference results and produce "
            "CSV, JSON, and Markdown summaries."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 parse_reference.py
  python3 parse_reference.py --root results/reference_diamond
  python3 parse_reference.py --csv-out my.csv --json-out my.json --md-out my.md
  python3 parse_reference.py --strict          # exit 1 if any run incomplete
""",
    )
    parser.add_argument(
        "--root",
        default="results/reference_diamond",
        metavar="DIR",
        help="Root directory to scan recursively for pw.out files "
             "(default: results/reference_diamond)",
    )
    parser.add_argument(
        "--csv-out", metavar="FILE",
        help="CSV output path (default: <root>/reference_summary.csv)",
    )
    parser.add_argument(
        "--json-out", metavar="FILE",
        help="JSON output path (default: <root>/reference_summary.json)",
    )
    parser.add_argument(
        "--md-out", metavar="FILE",
        help="Markdown output path (default: <root>/reference_summary.md)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 if any calculation is incomplete or needs attention",
    )
    parser.add_argument(
        "--include-incomplete", action="store_true",
        help="(Future) include directories without pw.out, marked as missing",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: root directory '{root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    csv_out  = Path(args.csv_out)  if args.csv_out  else root / "reference_summary.csv"
    json_out = Path(args.json_out) if args.json_out else root / "reference_summary.json"
    md_out   = Path(args.md_out)   if args.md_out   else root / "reference_summary.md"

    print(f"Scanning: {root.resolve()}")
    run_dirs = find_result_dirs(root)

    if not run_dirs:
        print("No result directories found (no pw.out files).")
        sys.exit(0)

    plural = "y" if len(run_dirs) == 1 else "ies"
    print(f"Found {len(run_dirs)} result director{plural}.")

    rows = sort_rows([build_row(d, root) for d in run_dirs])

    n_complete    = sum(1 for r in rows if r.get("complete"))
    n_incomplete  = len(rows) - n_complete
    n_attn        = sum(1 for r in rows if r.get("needs_attention"))

    write_csv(rows, csv_out)
    write_json(rows, json_out)
    write_markdown(rows, md_out)

    print(f"\nResults:")
    print(f"  Completed:          {n_complete}/{len(rows)}")
    print(f"  Incomplete/failed:  {n_incomplete}/{len(rows)}")
    print(f"  Needs attention:    {n_attn}/{len(rows)}")
    print(f"\nOutput files written:")
    print(f"  CSV:      {csv_out}")
    print(f"  JSON:     {json_out}")
    print(f"  Markdown: {md_out}")

    if args.strict and (n_incomplete > 0 or n_attn > 0):
        print(f"\n[--strict] {n_incomplete} incomplete, {n_attn} needing attention. Exit 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
