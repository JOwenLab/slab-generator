#!/usr/bin/env python3
"""
parse_convergence.py - Summarise the cutoff, k-point, vacuum and thickness sweeps.

Walks one or more run directories, where each <run> holds the pw.in and pw.out of
a stress SCF (tstress, tprnfor, no ionic relaxation), and writes a summary CSV.

    results/convergence/      -> cutoff, k-point and vacuum sweeps
    results/thickness_stress/ -> layer-thickness ladder (6L..16L per surface)

Both are parsed by the same code path; the sweep a run belongs to is derived
from its folder name (see `sweep_of`) and anything unrecognised is labelled
"other" rather than guessed at.

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
    sigma_lz   = sigma_mean * Lz                    [kbar*Angstrom]

The 0.005 is 0.1 GPa/kbar * 0.1 (N/m)/(GPa*Angstrom) / 2 surfaces; the factor of
2 is because the slab has two surfaces.

SIGN CONVENTION (CLAUDE.md section 2, corrected 2026-08):

    sigma > 0  ->  the cell is COMPRESSED; it wants to expand. Pressure-like.
    sigma < 0  ->  the cell is in TENSION; it wants to contract.
    QE pressure P = +(1/3) tr(sigma).

Anchored on bulk diamond at -1% strain (unambiguously compression), which gives
sigma_xx = sigma_yy = sigma_zz = P = +162.86 kbar. This docstring previously
stated both of these backwards; the numbers were never affected, only the words.

Anisotropy sign depends on how the in-plane cell axes were assigned, so the
lattice vector lengths a1 and a2 are carried in the CSV alongside it.

Geometry is read from the coordinates, never inferred from the folder name
(invariant 1/3). The carbon slab thickness, layer count and vacuum gap are all
computed from ATOMIC_POSITIONS and cross-checked against the name; a
disagreement is reported, not silently accepted.
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

# Thickness-ladder runs are named C<orientation>_<N>L_stress_scf and carry no
# sweep prefix, so they are matched separately.
THICKNESS_RE = re.compile(r"^C\d{3}_(\d+)L(?:_stress_scf)?$")

# Measured carbon z-gaps in this dataset, which is what any layer-counting rule
# has to survive:
#
#   (100)  0.076 A   buckling within a 2x1 dimer row   -> SAME layer
#   (100)  0.292 A   buckling within a 2x2 interior layer -> SAME layer
#   (100)  0.816 A   adjacent layers                   -> distinct
#   (110)  0.000 A   two atoms per layer, degenerate   -> SAME layer
#   (110)  1.235 A   adjacent layers                   -> distinct
#   (111)  0.488 A   the two halves of a (111) bilayer -> DISTINCT layers
#   (111)  1.554 A   between bilayers                  -> distinct
#
# There is no single distance that separates those: (100)'s 0.292 A intra-layer
# buckling is LARGER than (111)'s 0.488 A inter-layer split is small, and the
# two windows overlap. Any fixed tolerance therefore has to be wrong for one of
# them. The former 0.25 A value was wrong for (100): it split each buckled
# interior layer in two and reported the 6-layer (100) slab as 8 layers. That
# never reached tau_inf, because fit_tau_infinity.py keys exclusion on the
# directory name, but it is a live trap for any exclusion logic written against
# this field. count_layers below still clusters on a distance, but no longer
# relies on it being right: it repairs the clustering with the requirement that
# every layer hold the same number of atoms, which is what the old rule lacked.
# 0.25 A is kept because it is below the smallest genuine interlayer gap in the
# set ((111), 0.488 A); being too small is now the harmless direction.
LAYER_CLUSTER_TOL_ANGSTROM = 0.25


def sweep_of(run_name):
    """Sweep label for a run folder. Never guesses: unknown -> 'other'."""
    if THICKNESS_RE.match(run_name):
        return "thickness"
    return SWEEP_PREFIXES.get(run_name.split("_")[0], "other")


def declared_layer_count(run_name):
    """Layer count asserted by the folder name, for cross-checking only."""
    m = THICKNESS_RE.match(run_name)
    return int(m.group(1)) if m else None

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
    "sigma_lz_kbar_angstrom",
    "n_C",
    "n_H",
    "slab_thickness_angstrom",
    "atom_extent_angstrom",
    "vacuum_angstrom",
    "n_layers_geometry",
    "n_layers_declared",
    "layer_count_consistency",
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
        "z_by_species": {},
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

    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped.startswith("ATOMIC_POSITIONS"):
            if "ANGSTROM" not in stripped:
                # Every run in this campaign writes angstrom. Refuse to guess:
                # a crystal/bohr block read as angstrom would give a plausible
                # but wrong slab thickness.
                raise ValueError(
                    f"{path}: ATOMIC_POSITIONS unit in {line.strip()!r} is not "
                    "angstrom; refusing to guess"
                )
            for row in lines[i + 1:]:
                parts = row.split()
                if len(parts) < 4 or row.strip().startswith("!"):
                    break
                try:
                    z = float(parts[3])
                except ValueError:
                    break
                out["z_by_species"].setdefault(parts[0], []).append(z)
            break

    return out


def count_layers(zs, tol=LAYER_CLUSTER_TOL_ANGSTROM):
    """
    Number of atomic layers among these z coordinates.

    Clustering on a distance alone is not enough, and cannot be made enough:
    (100)'s 0.292 A intra-layer buckling is wider than (111)'s 0.488 A
    inter-layer split is narrow, so no single threshold classifies both (see
    the measured gaps above).

    What disambiguates them is not a distance but a counting invariant. A slab
    layer is a set of symmetry-equivalent sites, so every layer holds the same
    number of atoms, and therefore the layer count must divide the atom count.
    Clusters of unequal population are a proof that the clustering split
    something that is one layer.

    So: cluster on `tol`, then, while the cluster populations are unequal,
    merge the adjacent pair separated by the smallest gap. On the (100) 2x2
    slab that turns populations 4,4,2,2,2,2,4,4 into 4,4,4,4,4,4 by closing the
    two 0.292 A buckling splits, and reports 6 layers rather than 8. Structures
    the old rule already got right are untouched, because their clusters are
    equal-population from the start.

    Known limit, deliberately not papered over: the repair sees a split only
    when it makes populations UNEQUAL. If a tolerance were small enough to
    split every layer identically, the populations would stay equal and the
    count would come back a whole multiple too large. That case is not
    decidable from z alone -- uniform intra-layer buckling and a genuine
    bilayer like (111)'s produce the same alternating gap sequence -- so it is
    left to the name-vs-geometry check in `analyze_run`, which compares this
    count against the layer count the folder name declares. `tol` must
    therefore stay below the smallest genuine interlayer gap in the set
    ((111), 0.488 A); 0.25 A is verified against every structure in results/.

    Returns None for empty input.
    """
    n = len(zs)
    if not n:
        return None

    ordered = sorted(zs)
    clusters = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        if cur - prev > tol:
            clusters.append([])
        clusters[-1].append(cur)

    while len({len(c) for c in clusters}) > 1:
        # Smallest separation first: the narrowest split is the likeliest to be
        # buckling within one layer rather than a boundary between two.
        i = min(range(len(clusters) - 1),
                key=lambda j: clusters[j + 1][0] - clusters[j][-1])
        clusters[i:i + 2] = [clusters[i] + clusters[i + 1]]

    return len(clusters)


def slab_geometry(z_by_species, lz):
    """
    Carbon slab thickness, total atomic extent, vacuum gap and layer count,
    all from coordinates.

    `slab_thickness` is the carbon z-extent, max(z_C) - min(z_C). That is the
    elastic body whose interior stress the thickness fit is about; the
    terminating H sits outside it. The choice matters: a constant offset in the
    thickness definition shifts the intercept of the sigma*Lz vs t fit, and so
    shifts tau_inf. The slope (the residual interior stress) is unaffected by
    it. Anything quoting tau_inf must state which convention produced it.

    `vacuum` uses the full atomic extent including H, since it is the H atoms
    on opposing faces that would interact across the periodic boundary.
    """
    out = {
        "n_C": None, "n_H": None,
        "slab_thickness_angstrom": None,
        "atom_extent_angstrom": None,
        "vacuum_angstrom": None,
        "n_layers_geometry": None,
    }
    if not z_by_species:
        return out

    z_c = z_by_species.get("C", [])
    z_h = z_by_species.get("H", [])
    all_z = [z for zs in z_by_species.values() for z in zs]

    out["n_C"] = len(z_c) or None
    out["n_H"] = len(z_h) or None
    if z_c:
        out["slab_thickness_angstrom"] = max(z_c) - min(z_c)
        out["n_layers_geometry"] = count_layers(z_c)
    if all_z:
        out["atom_extent_angstrom"] = max(all_z) - min(all_z)
        if lz is not None:
            out["vacuum_angstrom"] = lz - out["atom_extent_angstrom"]
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
    sweep = sweep_of(run)

    surface, from_source = surface_of(pin["source_slab"], run)
    notes = []

    geom = slab_geometry(pin["z_by_species"], pin["lz_angstrom"])

    # Invariant 3 again, for the layer count: the name says NL, the coordinates
    # must agree. This is exactly the class of error that cost this project 45
    # calculations, so it is checked rather than assumed.
    n_declared = declared_layer_count(run)
    n_geom = geom["n_layers_geometry"]
    if n_declared is None:
        layer_count_consistency = "not_applicable"
    elif n_geom is None:
        layer_count_consistency = "no_coordinates"
    elif n_declared == n_geom:
        layer_count_consistency = "ok"
    else:
        layer_count_consistency = f"name_says_{n_declared}L_geometry_says_{n_geom}L"
        notes.append(layer_count_consistency)

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

    sigma_mean = anisotropy = tau_mean = sigma_lz = None
    if sxx is not None and syy is not None:
        sigma_mean = 0.5 * (sxx + syy)
        anisotropy = sxx - syy
        if lz is not None:
            tau_mean = sigma_mean * lz * TAU_FACTOR
            sigma_lz = sigma_mean * lz

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
        "sigma_lz_kbar_angstrom": sigma_lz,
        "n_C": geom["n_C"],
        "n_H": geom["n_H"],
        "slab_thickness_angstrom": geom["slab_thickness_angstrom"],
        "atom_extent_angstrom": geom["atom_extent_angstrom"],
        "vacuum_angstrom": geom["vacuum_angstrom"],
        "n_layers_geometry": n_geom,
        "n_layers_declared": n_declared,
        "layer_count_consistency": layer_count_consistency,
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
    sweep_order = {"cutoff": 0, "kpoint": 1, "vacuum": 2, "thickness": 3}.get(
        row["sweep"], 4)
    return (
        sweep_order,
        row["surface"] or "zzz",
        row["ecutwfc_ry"] if row["sweep"] == "cutoff" and row["ecutwfc_ry"] is not None else 0.0,
        row["k1"] if row["k1"] is not None else 0,
        row["n_layers_geometry"] or 0,
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
    ap.add_argument("--indir", nargs="+", default=["results/convergence"],
                    help="one or more run directories to scan "
                         "(e.g. results/convergence results/thickness_stress)")
    ap.add_argument("--out", default=None,
                    help="output CSV (default <first indir>/convergence_summary.csv)")
    args = ap.parse_args()

    indirs = [Path(d) for d in args.indir]
    outpath = Path(args.out) if args.out else indirs[0] / "convergence_summary.csv"
    repo = Path(__file__).resolve().parent
    git_commit = repo_git_commit(repo)

    rundirs = []
    for indir in indirs:
        if not indir.is_dir():
            raise SystemExit(f"Not a directory: {indir}")
        found = sorted(d for d in indir.iterdir()
                       if d.is_dir() and (d / "pw.out").exists())
        if not found:
            raise SystemExit(f"No runs with a pw.out under {indir}")
        rundirs.extend(found)

    rows = [analyze_run(d, git_commit) for d in rundirs]
    rows.sort(key=sort_key)

    write_csv(rows, outpath)

    print(f"Parsed {len(rows)} runs from {', '.join(str(d) for d in indirs)}")
    for sweep in ("cutoff", "kpoint", "vacuum", "thickness", "other"):
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
                or r["name_consistency"] != "ok" or r["cutoff_consistency"] != "ok"
                or r["layer_count_consistency"] not in ("ok", "not_applicable")]
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
            for key in ("pseudo_consistency", "name_consistency",
                        "cutoff_consistency", "layer_count_consistency"):
                if r[key] not in ("ok", "not_applicable"):
                    detail.append(f"{key}={r[key]}")
            print(f"  {r['run']}: {'; '.join(d for d in detail if d)}")


if __name__ == "__main__":
    main()
