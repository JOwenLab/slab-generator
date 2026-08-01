#!/usr/bin/env python3
"""
parse_convergence.py - Summarise the cutoff and k-point convergence sweeps.

Walks results/convergence/<run>/, where each <run> holds the pw.in and pw.out of
a stress SCF (tstress, tprnfor, no ionic relaxation), and writes
results/convergence/convergence_summary.csv.

Ground-truth rules (CLAUDE.md sections 1 and 2):

- pw.out is authoritative for what the code actually did: QE version, nat,
  cutoffs, pseudopotential files opened, energy, stress, convergence. pw.in
  supplies the *requested* parameters (cutoffs, k-mesh, cell). Where both are
  available the two are compared and a mismatch is flagged, not averaged away.
- The surface is read from the source-slab provenance comment in pw.in, never
  from the folder name. The folder name is then checked against it and any
  disagreement is reported (invariant 3: names describe geometry).
- Runs that did not reach JOB DONE or did not reach SCF convergence stay in the
  CSV with the failure flagged, and are listed in a warning block on stdout.
  They are never silently dropped.

Derived quantities use the project conventions in CLAUDE.md section 2, with
sigma in kbar and Lz in Angstrom:

    sigma_mean = (sigma_xx + sigma_yy) / 2          [kbar]
    anisotropy = sigma_xx - sigma_yy                [kbar]
    tau_mean   = sigma_mean * Lz * 0.005            [N/m]

The 0.005 is 0.1 GPa/kbar * 0.1 (N/m)/(GPa*Angstrom) / 2 surfaces; the factor of
2 is because the slab has two surfaces. sigma < 0 is compressive, sigma > 0 is
tensile-like, and QE pressure P = -(1/3)tr(sigma).

Anisotropy sign depends on how the in-plane cell axes were assigned, so the
lattice vector lengths a1 and a2 are carried in the CSV alongside it.
"""

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path

KBAR_TO_GPA = 0.1
# 1 GPa * 1 Angstrom = 0.1 N/m
GPA_ANG_TO_N_PER_M = 0.1
# tau = sigma_kbar * Lz_angstrom * TAU_FACTOR, over two surfaces.
TAU_FACTOR = KBAR_TO_GPA * GPA_ANG_TO_N_PER_M / 2.0

ORIENTATIONS = ("C100", "C110", "C111")

# Which sweep a run belongs to, from the leading token of its folder name.
# Anything unrecognised is labelled "other" rather than guessed at.
SWEEP_PREFIXES = {
    "cutoff": "cutoff",
    "kpt": "kpoint",
    "vac": "vacuum",
}

FIELDS = [
    "run",
    "sweep",
    "surface",
    "source_slab",
    "git_commit",
    "qe_version",
    "nat",
    "ecutwfc_ry",
    "ecutrho_ry",
    "ecutwfc_requested_ry",
    "ecutrho_requested_ry",
    "kmesh",
    "k1",
    "k2",
    "k3",
    "a1_angstrom",
    "a2_angstrom",
    "lz_angstrom",
    "k_density_a1_angstrom",
    "k_density_a2_angstrom",
    "energy_ry",
    "sigma_xx_kbar",
    "sigma_yy_kbar",
    "sigma_zz_kbar",
    "pressure_kbar",
    "sigma_mean_kbar",
    "anisotropy_kbar",
    "tau_mean_n_per_m",
    "scf_converged",
    "job_done",
    "pseudo_C",
    "pseudo_H",
    "pseudo_consistency",
    "name_consistency",
    "cutoff_consistency",
    "notes",
]


def repo_git_commit(repo):
    """HEAD of the repo, with -dirty appended if the tree has uncommitted work."""
    # stdout=PIPE rather than capture_output: the cluster login node is py3.6.
    def git(*cmd):
        return subprocess.run(
            ["git", "-C", str(repo), *cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, check=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    try:
        if git("status", "--porcelain"):
            commit += "-dirty"
    except subprocess.CalledProcessError:
        pass
    return commit


def fmt(value, nd=6):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return value


def parse_pw_in(path):
    """Requested parameters and the cell, from the QE input."""
    out = {
        "source_slab": None,
        "ecutwfc_requested_ry": None,
        "ecutrho_requested_ry": None,
        "kmesh": None,
        "k1": None, "k2": None, "k3": None,
        "a1_angstrom": None, "a2_angstrom": None, "lz_angstrom": None,
        "pseudo_in": {},
    }
    if not path.exists():
        return out

    lines = path.read_text().splitlines()

    for line in lines:
        m = re.search(r"generated from\s+(\S+)", line)
        if m and out["source_slab"] is None:
            out["source_slab"] = m.group(1).rstrip("/").split("/")[-1]
        m = re.match(r"\s*ecutwfc\s*=\s*([0-9.eEdD+-]+)", line)
        if m:
            out["ecutwfc_requested_ry"] = float(m.group(1).replace("d", "e").replace("D", "e"))
        m = re.match(r"\s*ecutrho\s*=\s*([0-9.eEdD+-]+)", line)
        if m:
            out["ecutrho_requested_ry"] = float(m.group(1).replace("d", "e").replace("D", "e"))

    for i, line in enumerate(lines):
        if line.strip().upper().startswith("K_POINTS"):
            if "automatic" not in line.lower():
                break
            parts = lines[i + 1].split()
            if len(parts) >= 3:
                k = [int(p) for p in parts[:3]]
                out["kmesh"] = " ".join(str(v) for v in k)
                out["k1"], out["k2"], out["k3"] = k
            break

    for i, line in enumerate(lines):
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            unit = line.split()[-1].lower() if len(line.split()) > 1 else "alat"
            if "angstrom" not in unit:
                # Every run in this campaign writes angstrom. Refuse to guess.
                raise ValueError(f"{path}: CELL_PARAMETERS unit is {unit!r}, expected angstrom")
            vecs = []
            for row in lines[i + 1:i + 4]:
                vals = [float(v) for v in row.split()[:3]]
                if len(vals) != 3:
                    raise ValueError(f"{path}: malformed CELL_PARAMETERS row {row!r}")
                vecs.append(vals)
            out["a1_angstrom"] = math.sqrt(sum(v * v for v in vecs[0]))
            out["a2_angstrom"] = math.sqrt(sum(v * v for v in vecs[1]))
            # Lz is the z-component of the third cell vector.
            out["lz_angstrom"] = vecs[2][2]
            break

    for i, line in enumerate(lines):
        if line.strip().upper().startswith("ATOMIC_SPECIES"):
            for row in lines[i + 1:]:
                parts = row.split()
                if len(parts) != 3 or row.strip().startswith("!"):
                    break
                out["pseudo_in"][parts[0]] = parts[2]
            break

    return out


def parse_pw_out(path):
    """What QE actually did. Ground truth for pseudopotentials and stress."""
    out = {
        "qe_version": None,
        "nat": None,
        "ecutwfc_ry": None,
        "ecutrho_ry": None,
        "energy_ry": None,
        "sigma_xx_kbar": None,
        "sigma_yy_kbar": None,
        "sigma_zz_kbar": None,
        "pressure_kbar": None,
        "scf_converged": False,
        "job_done": False,
        "pseudo_out": {},
    }
    if not path.exists():
        return out

    lines = path.read_text(errors="replace").splitlines()

    for i, line in enumerate(lines):
        m = re.search(r"Program PWSCF v\.(\S+)", line)
        if m and out["qe_version"] is None:
            out["qe_version"] = m.group(1)
        m = re.search(r"number of atoms/cell\s*=\s*(\d+)", line)
        if m:
            out["nat"] = int(m.group(1))
        m = re.search(r"kinetic-energy cutoff\s*=\s*([0-9.]+)\s*Ry", line)
        if m:
            out["ecutwfc_ry"] = float(m.group(1))
        m = re.search(r"charge density cutoff\s*=\s*([0-9.]+)\s*Ry", line)
        if m:
            out["ecutrho_ry"] = float(m.group(1))
        # The final SCF energy is the one flagged with "!".
        m = re.match(r"!\s+total energy\s*=\s*(-?[0-9.]+)\s*Ry", line)
        if m:
            out["energy_ry"] = float(m.group(1))
        if "convergence has been achieved" in line:
            out["scf_converged"] = True
        if "JOB DONE" in line:
            out["job_done"] = True

        # PseudoPot. # 1 for C  read from file:
        #      ./C.pbe-n-kjpaw_psl.1.0.0.UPF
        m = re.search(r"PseudoPot\. #\s*\d+\s+for\s+(\S+)\s+read from file", line)
        if m and i + 1 < len(lines):
            out["pseudo_out"][m.group(1)] = lines[i + 1].strip().lstrip("./")

        # total   stress  (Ry/bohr**3)   (kbar)     P=      -13.55
        m = re.search(r"total\s+stress.*P=\s*(-?[0-9.]+)", line)
        if m:
            out["pressure_kbar"] = float(m.group(1))
            diag = []
            for j, row in enumerate(lines[i + 1:i + 4]):
                vals = row.split()
                if len(vals) < 6:
                    diag = []
                    break
                diag.append(float(vals[3 + j]))
            if len(diag) == 3:
                out["sigma_xx_kbar"], out["sigma_yy_kbar"], out["sigma_zz_kbar"] = diag

    return out


def surface_of(source_slab, run_name):
    """Orientation from the source slab; run_name only as a fallback."""
    for text, trusted in ((source_slab, True), (run_name, False)):
        if not text:
            continue
        for orient in ORIENTATIONS:
            if orient in text:
                return orient, trusted
    return None, False


def analyze_run(rundir, git_commit):
    pin = parse_pw_in(rundir / "pw.in")
    pout = parse_pw_out(rundir / "pw.out")

    run = rundir.name
    sweep = SWEEP_PREFIXES.get(run.split("_")[0], "other")

    surface, from_source = surface_of(pin["source_slab"], run)
    notes = []

    if surface is None:
        notes.append("surface_unidentified")
    if not from_source and surface is not None:
        notes.append("surface_taken_from_folder_name_no_provenance_comment")

    # Invariant 3: the folder name must describe the geometry it contains.
    if surface is None:
        name_consistency = "unknown"
    elif surface in run:
        name_consistency = "ok"
    else:
        name_consistency = f"folder_name_omits_{surface}"

    # Invariant 7: pw.out is ground truth for the pseudopotential actually opened.
    pseudo_out = pout["pseudo_out"]
    pseudo_in = pin["pseudo_in"]
    if not pseudo_out:
        pseudo_consistency = "no_pseudo_in_pw_out"
    elif not pseudo_in:
        pseudo_consistency = "no_pseudo_in_pw_in"
    elif all(pseudo_in.get(sp) == fn for sp, fn in pseudo_out.items()):
        pseudo_consistency = "ok"
    else:
        pseudo_consistency = "pw_in_pw_out_mismatch"

    req = pin["ecutwfc_requested_ry"]
    got = pout["ecutwfc_ry"]
    if req is None or got is None:
        cutoff_consistency = "unknown"
    elif abs(req - got) < 1e-6:
        cutoff_consistency = "ok"
    else:
        cutoff_consistency = f"requested_{req:g}_used_{got:g}"

    # Cross-check the sweep coordinate encoded in the folder name against the
    # value the calculation actually used.
    token = run.split("~")[-1] if "~" in run else ""
    m = re.match(r"ecut_([0-9.]+)$", token)
    if m and got is not None and abs(float(m.group(1)) - got) > 1e-6:
        notes.append(f"folder_says_ecut_{m.group(1)}_pw_out_says_{got:g}")
    m = re.match(r"k_(\d+)$", token)
    if m and pin["k1"] is not None and int(m.group(1)) != pin["k1"]:
        notes.append(f"folder_says_k_{m.group(1)}_pw_in_says_{pin['k1']}")

    if not pout["job_done"]:
        notes.append("no_JOB_DONE")
    if not pout["scf_converged"]:
        notes.append("scf_not_converged")
    if pout["sigma_xx_kbar"] is None:
        notes.append("no_stress_tensor")

    sxx = pout["sigma_xx_kbar"]
    syy = pout["sigma_yy_kbar"]
    lz = pin["lz_angstrom"]

    sigma_mean = anisotropy = tau_mean = None
    if sxx is not None and syy is not None:
        sigma_mean = 0.5 * (sxx + syy)
        anisotropy = sxx - syy
        if lz is not None:
            tau_mean = sigma_mean * lz * TAU_FACTOR

    def kdens(k, a):
        return k * a if (k is not None and a is not None) else None

    return {
        "run": run,
        "sweep": sweep,
        "surface": surface,
        "source_slab": pin["source_slab"],
        "git_commit": git_commit,
        "qe_version": pout["qe_version"],
        "nat": pout["nat"],
        "ecutwfc_ry": pout["ecutwfc_ry"],
        "ecutrho_ry": pout["ecutrho_ry"],
        "ecutwfc_requested_ry": pin["ecutwfc_requested_ry"],
        "ecutrho_requested_ry": pin["ecutrho_requested_ry"],
        "kmesh": pin["kmesh"],
        "k1": pin["k1"],
        "k2": pin["k2"],
        "k3": pin["k3"],
        "a1_angstrom": pin["a1_angstrom"],
        "a2_angstrom": pin["a2_angstrom"],
        "lz_angstrom": lz,
        "k_density_a1_angstrom": kdens(pin["k1"], pin["a1_angstrom"]),
        "k_density_a2_angstrom": kdens(pin["k2"], pin["a2_angstrom"]),
        "energy_ry": pout["energy_ry"],
        "sigma_xx_kbar": sxx,
        "sigma_yy_kbar": syy,
        "sigma_zz_kbar": pout["sigma_zz_kbar"],
        "pressure_kbar": pout["pressure_kbar"],
        "sigma_mean_kbar": sigma_mean,
        "anisotropy_kbar": anisotropy,
        "tau_mean_n_per_m": tau_mean,
        "scf_converged": "yes" if pout["scf_converged"] else "no",
        "job_done": "yes" if pout["job_done"] else "no",
        "pseudo_C": pseudo_out.get("C", ""),
        "pseudo_H": pseudo_out.get("H", ""),
        "pseudo_consistency": pseudo_consistency,
        "name_consistency": name_consistency,
        "cutoff_consistency": cutoff_consistency,
        "notes": ";".join(notes),
    }


def sort_key(row):
    sweep_order = {"cutoff": 0, "kpoint": 1, "vacuum": 2}.get(row["sweep"], 3)
    return (
        sweep_order,
        row["surface"] or "zzz",
        row["ecutwfc_ry"] if row["sweep"] == "cutoff" and row["ecutwfc_ry"] is not None else 0.0,
        row["k1"] if row["k1"] is not None else 0,
        row["run"],
    )


def write_csv(rows, path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in FIELDS})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--indir", default="results/convergence")
    ap.add_argument("--out", default=None,
                    help="output CSV (default <indir>/convergence_summary.csv)")
    args = ap.parse_args()

    indir = Path(args.indir)
    outpath = Path(args.out) if args.out else indir / "convergence_summary.csv"
    repo = Path(__file__).resolve().parent
    git_commit = repo_git_commit(repo)

    rundirs = sorted(d for d in indir.iterdir() if d.is_dir() and (d / "pw.out").exists())
    if not rundirs:
        raise SystemExit(f"No runs with a pw.out under {indir}")

    rows = [analyze_run(d, git_commit) for d in rundirs]
    rows.sort(key=sort_key)

    write_csv(rows, outpath)

    print(f"Parsed {len(rows)} convergence runs from {indir}")
    for sweep in ("cutoff", "kpoint", "vacuum", "other"):
        group = [r for r in rows if r["sweep"] == sweep]
        if not group:
            continue
        surfaces = sorted({r["surface"] or "?" for r in group})
        print(f"  {sweep}: {len(group)} runs, surfaces {', '.join(surfaces)}")
    print(f"Wrote {outpath}")
    print(f"git commit stamped on every row: {git_commit}")

    incomplete = [r for r in rows if r["job_done"] != "yes" or r["scf_converged"] != "yes"]
    if incomplete:
        print()
        print(f"WARNING: {len(incomplete)} run(s) did not reach JOB DONE or did not converge.")
        print("They are kept in the CSV and must not be treated as converged results:")
        for r in incomplete:
            print(f"  {r['run']}: JOB DONE={r['job_done']} scf_converged={r['scf_converged']}")

    flagged = [r for r in rows
               if r["notes"] and not set(r["notes"].split(";")) <= {"no_JOB_DONE", "scf_not_converged"}]
    flagged += [r for r in rows if r["pseudo_consistency"] != "ok"
                or r["name_consistency"] != "ok" or r["cutoff_consistency"] != "ok"]
    seen, unique = set(), []
    for r in flagged:
        if r["run"] not in seen:
            seen.add(r["run"])
            unique.append(r)
    if unique:
        print()
        print(f"WARNING: {len(unique)} run(s) with provenance or naming flags:")
        for r in unique:
            detail = [r["notes"]] if r["notes"] else []
            for key in ("pseudo_consistency", "name_consistency", "cutoff_consistency"):
                if r[key] != "ok":
                    detail.append(f"{key}={r[key]}")
            print(f"  {r['run']}: {'; '.join(d for d in detail if d)}")


if __name__ == "__main__":
    main()
