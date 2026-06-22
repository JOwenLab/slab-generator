#!/usr/bin/env python3
"""
parse_slab.py - Parse completed Quantum ESPRESSO diamond slab calculations.

Scans results/slabs by default, extracts energies, convergence, species,
pseudopotentials, cell geometry, approximate slab/vacuum metrics, and writes
CSV/JSON/Markdown summaries for surface-energy and surface-stress workflows.
"""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

RY_TO_EV = 13.605693122994
BOHR_TO_ANG = 0.529177210903
_NUM = r"[-+]?\d+\.?\d*(?:[Ee][+-]?\d+)?"


def read_text(path):
    try:
        return path.read_text()
    except OSError:
        return ""


def find_result_dirs(root):
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if "pw.out" in files:
            found.append(Path(dirpath))
    return sorted(found)


def parse_cell_parameters(text, last=True):
    pat = re.compile(
        r"CELL_PARAMETERS\s*[\({]?\s*(angstrom|bohr|alat)?\s*[\)}]?\s*\n"
        r"\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*\n"
        r"\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")\s*\n"
        r"\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")",
        re.IGNORECASE,
    )
    matches = list(pat.finditer(text))
    if not matches:
        return None
    m = matches[-1] if last else matches[0]
    unit = (m.group(1) or "angstrom").lower()
    if unit == "alat":
        return None
    scale = BOHR_TO_ANG if unit == "bohr" else 1.0
    vals = [float(m.group(i)) * scale for i in range(2, 11)]
    return [vals[0:3], vals[3:6], vals[6:9]]


def parse_atomic_positions(text, last=True):
    pat = re.compile(
        r"ATOMIC_POSITIONS\s*[\({]?\s*(angstrom|bohr|crystal|alat)?\s*[\)}]?\s*\n"
        r"((?:\s*[A-Za-z][A-Za-z0-9_+-]*\s+" + _NUM + r"\s+" + _NUM + r"\s+" + _NUM + r"(?:\s+[01]\s+[01]\s+[01])?\s*\n)+)",
        re.IGNORECASE,
    )
    matches = list(pat.finditer(text))
    if not matches:
        return []
    m = matches[-1] if last else matches[0]
    unit = (m.group(1) or "angstrom").lower()
    if unit not in ("angstrom", "bohr"):
        return []
    scale = BOHR_TO_ANG if unit == "bohr" else 1.0
    atoms = []
    for line in m.group(2).splitlines():
        parts = line.split()
        if len(parts) >= 4:
            atoms.append({
                "species": parts[0],
                "x": float(parts[1]) * scale,
                "y": float(parts[2]) * scale,
                "z": float(parts[3]) * scale,
                "fixed": len(parts) >= 7 and parts[-3:] == ["0", "0", "0"],
            })
    return atoms


def parse_pw_in(path):
    text = read_text(path)
    d = {
        "nat": None, "ntyp": None, "calculation_type": None, "pseudo_dir": None,
        "ecutwfc": None, "ecutrho": None, "occupations": None, "kpoints": None,
        "species": [], "pseudopotentials": {}, "cell_params_ang": None,
        "initial_positions_ang": [],
    }

    def first(pattern, conv=str):
        m = re.search(pattern, text, re.IGNORECASE)
        return conv(m.group(1)) if m else None

    d["nat"] = first(r"\bnat\s*=\s*(\d+)", int)
    d["ntyp"] = first(r"\bntyp\s*=\s*(\d+)", int)
    d["calculation_type"] = first(r"calculation\s*=\s*'([^']+)'")
    d["pseudo_dir"] = first(r"pseudo_dir\s*=\s*'([^']+)'")
    d["ecutwfc"] = first(r"\becutwfc\s*=\s*(" + _NUM + r")", float)
    d["ecutrho"] = first(r"\becutrho\s*=\s*(" + _NUM + r")", float)
    d["occupations"] = first(r"occupations\s*=\s*'([^']+)'")

    m = re.search(r"K_POINTS\s+automatic\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", text, re.IGNORECASE)
    if m:
        d["kpoints"] = f"{m.group(1)} {m.group(2)} {m.group(3)}"

    m = re.search(r"ATOMIC_SPECIES\s*\n(.*?)(?:CELL_PARAMETERS|ATOMIC_POSITIONS|K_POINTS|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        for line in m.group(1).splitlines():
            parts = line.split()
            if len(parts) >= 3:
                sym, pseudo = parts[0], parts[2]
                d["species"].append(sym)
                d["pseudopotentials"][sym] = pseudo

    d["cell_params_ang"] = parse_cell_parameters(text, last=False)
    d["initial_positions_ang"] = parse_atomic_positions(text, last=False)
    return d


def parse_pw_out(path):
    text = read_text(path)
    d = {
        "status": "UNKNOWN", "complete": False, "energy_ry": None,
        "pressure_kbar": None, "stress": {}, "final_force_ry_bohr": None,
        "scf_iterations": None, "ionic_steps": None, "wall_time": None,
        "final_cell_ang": None, "final_positions_ang": [], "warnings": [], "errors": [],
    }

    if "JOB DONE" in text:
        d["status"] = "JOB DONE"
        d["complete"] = True

    energies = re.findall(r"!\s+total energy\s*=\s*(" + _NUM + r")\s*Ry", text)
    if energies:
        d["energy_ry"] = float(energies[-1])

    forces = re.findall(r"Total force\s*=\s*(" + _NUM + r")", text)
    if forces:
        d["final_force_ry_bohr"] = float(forces[-1])

    bfgs = re.findall(r"number of bfgs steps\s*=\s*(\d+)", text, re.IGNORECASE)
    if bfgs:
        d["ionic_steps"] = int(bfgs[-1])
    else:
        m = re.search(r"bfgs converged in\s+(\d+)\s+scf cycles and\s+(\d+)\s+bfgs steps", text, re.IGNORECASE)
        if m:
            d["scf_iterations"] = int(m.group(1))
            d["ionic_steps"] = int(m.group(2))

    scf = re.findall(r"convergence has been achieved in\s+(\d+)\s+iteration", text, re.IGNORECASE)
    if scf:
        d["scf_iterations"] = int(scf[-1])

    m = re.search(r"PWSCF\s*:\s*" + _NUM + r"s?\s*CPU\s+(" + _NUM + r")s?\s*WALL", text)
    if m:
        d["wall_time"] = float(m.group(1))

    stress_hdr = re.search(
        r"total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(" + _NUM + r")",
        text,
    )
    if stress_hdr:
        d["pressure_kbar"] = float(stress_hdr.group(1))
        block = text[stress_hdr.end():stress_hdr.end() + 700]
        rows = re.findall(
            r"\s*(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")"
            r"\s+(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + r")",
            block,
        )
        if len(rows) >= 3:
            r0, r1, r2 = rows[0], rows[1], rows[2]
            d["stress"] = {
                "xx": float(r0[3]), "xy": float(r0[4]), "xz": float(r0[5]),
                "yy": float(r1[4]), "yz": float(r1[5]), "zz": float(r2[5]),
            }

    for pat in [r"Warning:[^\n]*", r"SCF correction compared to forces is large[^\n]*", r"negative rho[^\n]*"]:
        for m in re.finditer(pat, text, re.IGNORECASE):
            msg = m.group(0).strip()[:180]
            if msg not in d["warnings"]:
                d["warnings"].append(msg)

    for pat in [r"%%%% Error[^\n]*", r"Error in routine[^\n]*", r"convergence NOT achieved[^\n]*"]:
        for m in re.finditer(pat, text, re.IGNORECASE):
            msg = m.group(0).strip()[:180]
            if msg not in d["errors"]:
                d["errors"].append(msg)
                d["status"] = "ERROR"

    d["final_cell_ang"] = parse_cell_parameters(text, last=True)
    d["final_positions_ang"] = parse_atomic_positions(text, last=True)
    return d


def load_reference_config(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_bulk_energy_per_c(outdir):
    path = Path(outdir).parent / "reference_diamond" / "bulk_fit_summary.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    e0 = data.get("E0_energy_fit_ry")
    if e0 is not None:
        return float(e0) / 8.0
    return None


def vec_cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


def infer_name_fields(name):
    out = {"orientation": None, "termination": None, "layers": None, "is_sssp_rerun": False}
    m = re.match(r"C(\d{3})", name)
    if m:
        out["orientation"] = f"({m.group(1)})"
    m = re.search(r"_(\d+)L(?:_|$)", name)
    if m:
        out["layers"] = int(m.group(1))
    if "_H" in name:
        out["termination"] = "H"
    elif "_F" in name:
        out["termination"] = "F"
    elif "_O" in name:
        out["termination"] = "O"
    elif "bare" in name:
        out["termination"] = "bare"
    out["is_sssp_rerun"] = name.endswith("_SSSP")
    return out


def formula_from_counts(counts):
    pieces = []
    for sym in ["C", "H"]:
        n = counts.get(sym, 0)
        if n:
            pieces.append(sym if n == 1 else f"{sym}{n}")
    for sym in sorted(k for k in counts if k not in ("C", "H")):
        n = counts[sym]
        pieces.append(sym if n == 1 else f"{sym}{n}")
    return "".join(pieces)


def build_row(run_dir, root, ref_cfg, bulk_mu_c_ry):
    pw_in = parse_pw_in(run_dir / "pw.in")
    pw_out = parse_pw_out(run_dir / "pw.out")
    name_info = infer_name_fields(run_dir.name)

    positions = pw_out["final_positions_ang"] or pw_in["initial_positions_ang"]
    counts = {}
    for atom in positions:
        counts[atom["species"]] = counts.get(atom["species"], 0) + 1

    n_c = counts.get("C", 0)
    n_h = counts.get("H", 0)
    n_other = sum(n for s, n in counts.items() if s not in ("C", "H"))
    nat = pw_in["nat"] or sum(counts.values()) or None

    cell = pw_out["final_cell_ang"] or pw_in["cell_params_ang"]
    area = height = None
    if cell:
        area = vec_norm(vec_cross(cell[0], cell[1]))
        height = vec_norm(cell[2])

    carbon_z = [a["z"] for a in positions if a["species"] == "C"]
    zmin = min(carbon_z) if carbon_z else None
    zmax = max(carbon_z) if carbon_z else None
    thickness = (zmax - zmin) if zmin is not None else None
    vacuum = (height - thickness) if height is not None and thickness is not None else None

    e_ry = pw_out["energy_ry"]
    e_ev = e_ry * RY_TO_EV if e_ry is not None else None
    epa_ry = e_ry / nat if e_ry is not None and nat else None
    epa_ev = e_ev / nat if e_ev is not None and nat else None
    epc_ry = e_ry / n_c if e_ry is not None and n_c else None

    carbon_sub = None
    if e_ry is not None and bulk_mu_c_ry is not None and n_c:
        carbon_sub = e_ry - n_c * bulk_mu_c_ry

    ref_bulk = ref_cfg.get("bulk_reference", {})
    ref_pseudos = ref_cfg.get("pseudopotentials", {})
    pseudos = pw_in["pseudopotentials"]
    expected_c = ref_pseudos.get("C")
    pseudo_notes = []
    if expected_c and pseudos.get("C") != expected_c:
        pseudo_notes.append("C_pseudo_mismatch")
    if n_h and pseudos.get("H") != "H_ONCV_PBE-1.0.oncvpsp.upf":
        pseudo_notes.append("H_pseudo_not_project_SSSP")
    if pw_in["ecutwfc"] is not None and pw_in["ecutwfc"] < 80:
        pseudo_notes.append("ecutwfc_below_reference")
    if pw_in["ecutrho"] is not None and pw_in["ecutrho"] < 640:
        pseudo_notes.append("ecutrho_below_reference")

    surface_status = "not_computed_missing_H_chemical_potential" if n_h else "not_computed"
    needs_attention = bool(pw_out["errors"] or pseudo_notes or not pw_out["complete"])

    return {
        "run_path": str(run_dir),
        "folder_name": run_dir.name,
        "status": pw_out["status"],
        "complete": pw_out["complete"],
        "needs_attention": needs_attention,
        "orientation": name_info["orientation"],
        "termination": name_info["termination"],
        "layers": name_info["layers"],
        "calculation_type": pw_in["calculation_type"],
        "formula": formula_from_counts(counts),
        "nat": nat,
        "ntyp": pw_in["ntyp"],
        "n_C": n_c,
        "n_H": n_h,
        "n_other": n_other,
        "species": " ".join(pw_in["species"]),
        "pseudopotentials": "; ".join(f"{k}:{v}" for k, v in sorted(pseudos.items())),
        "pseudo_consistency": "ok" if not pseudo_notes else "; ".join(pseudo_notes),
        "functional_reference": ref_cfg.get("functional"),
        "pseudo_set_reference": ref_cfg.get("pseudo_set"),
        "a0_reference_angstrom": ref_bulk.get("a0_fit_angstrom"),
        "bulk_modulus_reference_gpa": ref_bulk.get("bulk_modulus_gpa"),
        "energy_ry": e_ry,
        "energy_ev": e_ev,
        "energy_per_atom_ry": epa_ry,
        "energy_per_atom_ev": epa_ev,
        "energy_per_C_ry": epc_ry,
        "bulk_reference_energy_per_C_ry": bulk_mu_c_ry,
        "carbon_bulk_subtracted_energy_ry": carbon_sub,
        "surface_energy_status": surface_status,
        "surface_energy_ev_angstrom2": None,
        "surface_energy_j_m2": None,
        "cell_area_angstrom2": area,
        "cell_height_angstrom": height,
        "carbon_zmin_angstrom": zmin,
        "carbon_zmax_angstrom": zmax,
        "carbon_thickness_angstrom": thickness,
        "vacuum_estimate_angstrom": vacuum,
        "pressure_kbar": pw_out["pressure_kbar"],
        "stress_xx_kbar": pw_out["stress"].get("xx"),
        "stress_yy_kbar": pw_out["stress"].get("yy"),
        "stress_zz_kbar": pw_out["stress"].get("zz"),
        "stress_xy_kbar": pw_out["stress"].get("xy"),
        "stress_xz_kbar": pw_out["stress"].get("xz"),
        "stress_yz_kbar": pw_out["stress"].get("yz"),
        "final_force_ry_bohr": pw_out["final_force_ry_bohr"],
        "scf_iterations": pw_out["scf_iterations"],
        "ionic_steps": pw_out["ionic_steps"],
        "wall_time": pw_out["wall_time"],
        "warnings": "; ".join(pw_out["warnings"]),
        "errors": "; ".join(pw_out["errors"]),
    }


FIELDNAMES = [
    "run_path", "folder_name", "status", "complete", "needs_attention",
    "orientation", "termination", "layers", "calculation_type", "formula",
    "nat", "ntyp", "n_C", "n_H", "n_other", "species", "pseudopotentials",
    "pseudo_consistency", "functional_reference", "pseudo_set_reference",
    "a0_reference_angstrom", "bulk_modulus_reference_gpa",
    "energy_ry", "energy_ev", "energy_per_atom_ry", "energy_per_atom_ev",
    "energy_per_C_ry", "bulk_reference_energy_per_C_ry",
    "carbon_bulk_subtracted_energy_ry", "surface_energy_status",
    "surface_energy_ev_angstrom2", "surface_energy_j_m2",
    "cell_area_angstrom2", "cell_height_angstrom", "carbon_zmin_angstrom",
    "carbon_zmax_angstrom", "carbon_thickness_angstrom",
    "vacuum_estimate_angstrom", "pressure_kbar", "stress_xx_kbar",
    "stress_yy_kbar", "stress_zz_kbar", "stress_xy_kbar", "stress_xz_kbar",
    "stress_yz_kbar", "final_force_ry_bohr", "scf_iterations",
    "ionic_steps", "wall_time", "warnings", "errors",
]


def write_outputs(rows, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "slab_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in FIELDNAMES})

    json_path = outdir / "slab_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True))

    md_path = outdir / "slab_summary.md"
    completed = sum(1 for r in rows if r["complete"])
    attention = sum(1 for r in rows if r["needs_attention"])

    lines = []
    lines.append("# Slab Summary")
    lines.append("")
    lines.append("Parsed completed and available diamond slab Quantum ESPRESSO calculations.")
    lines.append("")
    lines.append(f"- Slab calculations found: {len(rows)}")
    lines.append(f"- Completed: {completed}")
    lines.append(f"- Needs attention: {attention}")
    lines.append("")
    lines.append("## Slab Runs")
    lines.append("")
    lines.append("| Run | Formula | Orient. | Term. | Energy (Ry) | Area (A^2) | Force (Ry/Bohr) | Pseudos | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        lines.append(
            f"| {r['folder_name']} | {r['formula']} | {r['orientation'] or ''} | "
            f"{r['termination'] or ''} | {fmt(r['energy_ry'], 8)} | "
            f"{fmt(r['cell_area_angstrom2'], 3)} | {fmt(r['final_force_ry_bohr'], 6)} | "
            f"{r['pseudo_consistency']} | {r['status']} |"
        )

    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- H-terminated surface energies are not final yet because the hydrogen chemical potential is not defined.")
    lines.append("- `carbon_bulk_subtracted_energy_ry` subtracts only the fitted bulk carbon reference and should be treated as a preliminary excess-energy diagnostic.")
    lines.append("- Rows with pseudo/cutoff mismatches should not be used for direct orientation comparisons.")
    lines.append("")
    lines.append("## Next Analysis Step")
    lines.append("")
    lines.append("Add a consistent hydrogen chemical-potential reference, then compute area-normalized surface energies using:")
    lines.append("")
    lines.append("```text")
    lines.append("gamma = (E_slab - N_C * mu_C_bulk - N_H * mu_H) / (2A)")
    lines.append("```")
    lines.append("")

    md_path.write_text("\n".join(lines))
    return csv_path, json_path, md_path


def fmt(value, ndigits):
    if value is None:
        return ""
    return f"{value:.{ndigits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/slabs")
    ap.add_argument("--outdir", default="results/slabs")
    ap.add_argument("--include-incomplete", action="store_true")
    ap.add_argument("--reference-config", default="config/reference_pbe_sssp.json")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    ref_cfg = load_reference_config(args.reference_config)
    bulk_mu_c_ry = load_bulk_energy_per_c(outdir)

    rows = []
    for run_dir in find_result_dirs(root):
        row = build_row(run_dir, root, ref_cfg, bulk_mu_c_ry)
        if row["complete"] or args.include_incomplete:
            rows.append(row)

    rows.sort(key=lambda r: r["folder_name"])
    csv_path, json_path, md_path = write_outputs(rows, outdir)

    print(f"Parsed slab calculations: {len(rows)}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
