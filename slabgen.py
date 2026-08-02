"""
slabgen.py — generate periodic supercell coordinates for terminated diamond
slabs from the motif library.

Usage
-----
  python slabgen.py LIST
  python slabgen.py MOTIF --layers 10 --vacuum 15 --a0 3.567 \
         [--symmetric | --bottom {bare,H}] [--format poscar|xyz|cif] \
         [--repeat N1 N2] [--out FILE]

Rule-based motifs are re-generated at the requested thickness with the same
validated decorators that built the library (registry-safe by construction).
'pending-literature' motifs refuse to generate until their coordinates are
transcribed in the retrieval pass.

Face/thickness constraints enforced:
  * (111): n_layers must be EVEN so both faces are single-dangling-bond.
  * (100) symmetric dimer motifs: n_layers must be EVEN so the dimer rows
    on both faces run parallel (dangling-bond axis alternates per layer).
"""
import argparse
import sys
import numpy as np
import yaml
import geometry as G
from build_library import dimer_cell_100

LIB_PATH = "motifs.yaml"


# --------------------------------------------------------------- decorators
def _decorate(slab, motif, top=True, bottom=False):
    o = motif["orientation"]
    term = motif["termination"]
    name = motif["_name"]
    if name.endswith("_bare") or term == "none":
        if "2x1_bare" in name:
            G.dimerize_100(slab, G.DIMER_BARE, top=top, bottom=bottom)
        return slab
    if o == "100" and "2x1" in name:
        G.dimerize_100(slab, G.DIMER_SAT, top=top, bottom=bottom)
        sp = "H" if term == "H" else "F"
        G.add_monovalent(slab, sp, G.BOND[("C", sp)], top=top, bottom=bottom)
        return slab
    if "O_ether" in name:
        lim = 2 if o == "100" else 1
        G.add_bridge_oxygen(slab, top=top, bottom=bottom, per_atom_limit=lim)
        return slab
    if "O_ketone" in name:
        G.add_ketone_oxygen(slab, top=top, bottom=bottom)
        return slab
    if term in ("H", "F"):
        G.add_monovalent(slab, term, G.BOND[("C", term)],
                         top=top, bottom=bottom)
        return slab
    raise ValueError(f"no decorator route for {name}")


def _region_cartesian(entry, a0):
    """Stored dimensionless region -> Cartesian at the requested a0."""
    scale = a0 / entry["source"]["a0_used"]
    U = np.array(entry["cell_A"][0]) * scale
    W = np.array(entry["cell_A"][1]) * scale
    pos, el = [], []
    for at in entry["region_atoms"]:
        xy = at["f"][0] * U + at["f"][1] * W
        pos.append([xy[0], xy[1], at["z_a0"] * a0])
        el.append(at["el"])
    return np.array(pos), el, U, W


def splice_literature(name, entry, n_layers, a0, bottom="bare"):
    """Replace the top `region_layers` layers of an SDB bulk slab with the
    stored literature region. Registry/handedness resolved by symmetry
    search; the splice is verified by coordination analysis."""
    nreg = entry["region_layers"]
    if n_layers % 2 or n_layers < nreg + 4:
        raise ValueError(f"(111) literature splice: use an EVEN layer "
                         f"count >= {nreg + 4}")
    rpos, rel, U, W = _region_cartesian(entry, a0)

    shift = G.detect_sdb_shift_111(a0)
    full = G.bulk_slab("111", n_layers, a0, z_shift_layers=shift)
    # supercell congruent with the literature cell: (1 x m) where the
    # m-fold cell area matches |U x W|
    area_ratio = abs(U[0] * W[1] - U[1] * W[0]) / abs(
        full.A1[0] * full.A2[1] - full.A1[1] * full.A2[0])
    m = int(round(area_ratio))
    s = G.repeat_inplane(full, 1, m)

    # delete top nreg layers
    z = np.round(s.pos[:, 2], 3)
    levels = np.unique(z)[::-1]
    keep_levels = levels[nreg:]
    keep = np.isin(z, keep_levels)
    bulk_pos = s.pos[keep]
    bulk_el = [e for e, k in zip(s.el, keep) if k]
    bulk_top_z = bulk_pos[:, 2].max()
    bulk_top_xy = bulk_pos[np.abs(bulk_pos[:, 2] - bulk_top_z) < 0.05, :2]

    # bottom sublayer of the region (atoms carrying the open seam bonds)
    rz = rpos[:, 2]
    zb = rz.min()
    bot_idx = np.where(np.abs(rz - zb) < 0.15)[0]
    vbond = a0 * np.sqrt(3) / 4.0
    z_off = (bulk_top_z + vbond) - rpos[bot_idx, 2].mean()

    # symmetry/registry search: S in {1, mx, my, mxy}, t from matching
    # region bottom atom 0 onto each bulk top site
    A1m, A2m = s.A1, s.A2
    best = None
    for sx in (1, -1):
        for sy in (1, -1):
            Sp = rpos.copy()
            Sp[:, 0] *= sx
            Sp[:, 1] *= sy
            Sp[:, 2] += z_off
            for target in bulk_top_xy:
                t = np.array([target[0] - Sp[bot_idx[0], 0],
                              target[1] - Sp[bot_idx[0], 1], 0.0])
                cand_pos = np.vstack([bulk_pos, Sp + t])
                cand = G.Slab(A1m, A2m, cand_pos,
                              bulk_el + list(rel), a0, "111",
                              ["bulk"] * len(bulk_el)
                              + ["region"] * len(rel))
                cand.wrap()
                bonds, _ = G.neighbors(cand)
                co = np.array([len(b) for b in bonds])
                # seam atoms = region bottom + bulk top: all must be 4-fold
                seamr = [len(bulk_el) + int(i) for i in bot_idx]
                seamb = [i for i in range(len(bulk_el))
                         if abs(cand.pos[i, 2] - bulk_top_z) < 0.05]
                ok = all(co[i] == 4 for i in seamr + seamb)
                if ok:
                    imgs = G._images(cand)
                    for i in seamr:
                        for j in range(cand.n):
                            if j == i:
                                continue
                            for im in imgs:
                                d = np.linalg.norm(cand.pos[j] + im
                                                   - cand.pos[i])
                                if d < 1.30:
                                    ok = False
                if ok:
                    best = cand
                    break
            if best is not None:
                break
        if best is not None:
            break
    if best is None:
        raise RuntimeError(f"{name}: could not register literature region "
                           f"onto bulk (no symmetry/translation candidate "
                           f"passed seam verification)")
    if bottom == "H":
        G.add_monovalent(best, "H", G.BOND[("C", "H")],
                         top=False, bottom=True)
    return best


def generate(name, n_layers, a0=G.A0_DEFAULT, symmetric=False,
             bottom="bare", repeat=(1, 1)):
    lib = yaml.safe_load(open(LIB_PATH))
    if name not in lib:
        raise KeyError(f"unknown motif {name}; run with LIST to see options")
    motif = dict(lib[name])
    motif["_name"] = name
    if motif.get("status") == "pending-literature":
        raise NotImplementedError(
            f"{name} awaits transcription of published coordinates "
            f"(literature retrieval pass). Note: {motif['notes']}")
    if motif.get("status") == "published-model-coordinates":
        if symmetric:
            nreg = motif["region_layers"]
            n_src = n_layers + nreg + 4
            if n_src % 2:
                n_src += 1
            src = splice_literature(name, motif, n_src, a0, bottom="bare")
            # literature region thickness + margin (mean (111) layer
            # spacing ~ a0*sqrt(3)/6)
            depth = nreg * (a0 * np.sqrt(3) / 6.0) * 1.2 + 1.5
            s = G.invert_symmetrize(src, n_layers, recon_depth=depth)
        else:
            s = splice_literature(name, motif, n_layers, a0, bottom)
        if repeat != (1, 1):
            s = G.repeat_inplane(s, *repeat)
        return s, motif

    o = motif["orientation"]
    if motif.get("face") == "TDB":
        if symmetric:
            n_src = n_layers + 7
            if n_src % 2 == 0:
                n_src += 1          # seiwatz_slab requires ODD layer counts
            src = G.seiwatz_slab(motif["tdb_variant"], n_src, a0, "bare")
            s = G.invert_symmetrize(src, n_layers, recon_depth=4.0)
        else:
            if n_layers % 2 == 0:
                raise ValueError("TDB-face (111) asymmetric: use an ODD "
                                 "layer count (TDB top + SDB bottom)")
            s = G.seiwatz_slab(motif["tdb_variant"], n_layers, a0, bottom)
        if repeat != (1, 1):
            s = G.repeat_inplane(s, *repeat)
        return s, motif
    if o == "111" and n_layers % 2:
        raise ValueError("(111): use an EVEN layer count so both faces are "
                         "single-dangling-bond faces")
    cellm = tuple(motif["surface_cell_multiple"])
    if o == "100" and cellm != (1, 1):
        cellm = dimer_cell_100()
        if symmetric and n_layers % 2 == 1:
            raise ValueError("(100) symmetric dimer slabs: use an EVEN layer "
                             "count so dimer rows on both faces are "
                             "parallel (dangling-bond axis alternates per "
                             "layer)")

    shift = G.detect_sdb_shift_111(a0) if o == "111" else 0
    s = G.bulk_slab(o, n_layers, a0, z_shift_layers=shift)
    if cellm != (1, 1):
        s = G.repeat_inplane(s, *cellm)

    _decorate(s, motif, top=True, bottom=symmetric)
    if not symmetric and bottom == "H":
        G.add_monovalent(s, "H", G.BOND[("C", "H")], top=False, bottom=True)
    if repeat != (1, 1):
        s = G.repeat_inplane(s, *repeat)
    return s, motif


# ---------------------------------------------------------------- exporters
def _cell3d(slab, vacuum):
    zmin, zmax = slab.pos[:, 2].min(), slab.pos[:, 2].max()
    span = zmax - zmin
    C = np.array([0.0, 0.0, span + vacuum])
    pos = slab.pos.copy()
    pos[:, 2] += (vacuum / 2.0 - zmin)
    return np.array([slab.A1, slab.A2, C]), pos


def to_poscar(slab, vacuum, comment):
    cell, pos = _cell3d(slab, vacuum)
    pos = pos.copy()
    pos[abs(pos) < 1e-8] = 0.0            # avoid printing -0.00000000
    species = sorted(set(slab.el), key=lambda e: ("C", "O", "H", "F").index(e))
    lines = [comment, "1.0"]
    for v in cell:
        lines.append(f"  {v[0]:18.12f} {v[1]:18.12f} {v[2]:18.12f}")
    lines.append("  " + "  ".join(species))
    lines.append("  " + "  ".join(str(slab.el.count(e)) for e in species))
    lines.append("Cartesian")
    for e in species:
        for i in range(slab.n):
            if slab.el[i] == e:
                p = pos[i]
                lines.append(f"  {p[0]:18.12f} {p[1]:18.12f} {p[2]:18.12f}")
    return "\n".join(lines) + "\n"


def to_xyz(slab, vacuum, comment):
    cell, pos = _cell3d(slab, vacuum)
    flat = " ".join(f"{x:.10f}" for v in cell for x in v)
    lines = [str(slab.n),
             f'Lattice="{flat}" Properties=species:S:1:pos:R:3 '
             f'Comment="{comment}"']
    for i in range(slab.n):
        p = pos[i]
        lines.append(f"{slab.el[i]:2s} {p[0]:16.10f} {p[1]:16.10f} "
                     f"{p[2]:16.10f}")
    return "\n".join(lines) + "\n"


def to_cif(slab, vacuum, comment):
    cell, pos = _cell3d(slab, vacuum)
    a, b = np.linalg.norm(cell[0]), np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])
    gamma = np.degrees(np.arccos(np.dot(cell[0], cell[1]) / (a * b)))
    inv = np.linalg.inv(cell.T)
    lines = [f"# {comment}", "data_slab",
             f"_cell_length_a {a:.6f}", f"_cell_length_b {b:.6f}",
             f"_cell_length_c {c:.6f}",
             "_cell_angle_alpha 90.0", "_cell_angle_beta 90.0",
             f"_cell_angle_gamma {gamma:.4f}",
             "loop_", "_atom_site_type_symbol",
             "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    for i in range(slab.n):
        f = inv @ pos[i]
        for k in (0, 1):                  # wrap in-plane fractions to [0,1)
            f[k] %= 1.0
            if f[k] > 1.0 - 5e-9:         # snap print-precision residue
                f[k] = 0.0
        f += 0.0                          # normalize -0.0 -> 0.0
        lines.append(f"{slab.el[i]} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f}")
    return "\n".join(lines) + "\n"


QE_PSEUDO = {"C": ("12.011", "C.pbe-n-kjpaw_psl.1.0.0.UPF"),
             "H": ("1.008",  "H_ONCV_PBE-1.0.oncvpsp.upf"),
             "O": ("15.999", "O.pbe-n-kjpaw_psl.1.0.0.UPF"),
             "F": ("18.998", "F.pbe-n-kjpaw_psl.1.0.0.UPF")}


def to_qe(slab, vacuum, comment, fix_bottom_layers=2, symmetric=False,
          relax_mode="ions"):
    """Quantum ESPRESSO pw.x input for a slab relaxation.

    relax_mode='ions'  : fixed-cell ionic relax (original behavior)
    relax_mode='cell2d': vc-relax with cell_dofree='2Dxy' — relaxes
        in-plane a/b vectors only; vacuum axis c is fixed.
        Positions written in fractional (crystal) coords so if_pos
        constraints strain affinely with the cell.
        conv_thr tightened to 1.0d-9 for reliable stress.

    Dipole correction:
        symmetric slab  → omitted (no net dipole)
        asymmetric slab → active (tefield/dipfield in &CONTROL,
                          edir/emaxpos/eopreg/eamp in &SYSTEM)
    """
    cell, pos = _cell3d(slab, vacuum)
    pos = pos.copy()
    pos[abs(pos) < 1e-8] = 0.0

    species = sorted(set(slab.el), key=lambda e: ("C", "O", "H", "F").index(e))

    cell2d = (relax_mode == "cell2d")

    # --- freezing setup ---
    if symmetric:
        fix_bottom_layers = 0
    elif cell2d and fix_bottom_layers > 0:
        print("WARNING: --qe-fix-bottom > 0 with --relax-mode cell2d and no "
              "--symmetric. Bottom-freezing during cell relaxation is "
              "discouraged (spurious stress). Proceeding with fractional "
              "freeze (self-consistent but physically questionable).",
              file=sys.stderr)

    if cell2d:
        # fractional coordinates: compute inv once
        inv_cell = np.linalg.inv(cell.T)
        frac_pos = (inv_cell @ pos.T).T
        # z threshold in fractional units for frozen region
        cz_frac = sorted({round(float(f), 5)
                          for e, f in zip(slab.el, frac_pos[:, 2])
                          if e == "C"})
        zfix_frac = cz_frac[min(fix_bottom_layers, len(cz_frac)) - 1] + 1e-4 \
            if fix_bottom_layers > 0 else -1e9
    else:
        cz = sorted({round(float(z), 3) for e, z in zip(slab.el, pos[:, 2])
                     if e == "C"})
        zfix = cz[min(fix_bottom_layers, len(cz)) - 1] + 0.05 \
            if fix_bottom_layers > 0 else -1e9

    nk1 = max(1, round(24.0 / np.linalg.norm(cell[0])))
    nk2 = max(1, round(24.0 / np.linalg.norm(cell[1])))

    # --- header comment ---
    if cell2d:
        dip_state = "OFF (symmetric)" if symmetric else "ON (asymmetric)"
        L = [f"! {comment}",
             f"! relax_mode=cell2d  cell_dofree=2Dxy  dipole={dip_state}",
             "! in-plane lattice free; vacuum axis fixed",
             "! STRESS IS SENSITIVE TO ecutwfc (Pulay stress): confirm",
             "! in-plane stress is converged w.r.t. cutoff separately",
             "! from your energy/force convergence tests.",
             "! Generated by diamond_slabs slabgen; verify pseudos, cutoffs",
             "! and k-mesh before production use."]
    else:
        L = [f"! {comment}",
             "! Generated by diamond_slabs slabgen; verify pseudos, cutoffs and",
             "! k-mesh against your own convergence tests before production use."]

    # --- &CONTROL ---
    calc = "vc-relax" if cell2d else "relax"
    ctrl = ["&CONTROL",
            f"  calculation   = '{calc}'",
            "  prefix        = 'slab'",
            "  pseudo_dir    = './'",
            "  outdir        = './tmp'",
            "  forc_conv_thr = 1.0d-4",
            "  etot_conv_thr = 1.0d-5"]
    if not symmetric:
        ctrl += ["  tefield       = .true.",
                 "  dipfield      = .true."]
    ctrl.append("/")
    L += ctrl

    # --- &SYSTEM ---
    sys_block = ["&SYSTEM",
                 "  ibrav    = 0",
                 f"  nat      = {slab.n}",
                 f"  ntyp     = {len(species)}",
                 "  ecutwfc  = 60.0",
                 "  ecutrho  = 480.0",
                 "  occupations = 'smearing'",
                 "  smearing    = 'mv'",
                 "  degauss     = 0.01"]
    if not symmetric:
        sys_block += ["  edir     = 3",
                      "  emaxpos  = 0.95",
                      "  eopreg   = 0.05",
                      "  eamp     = 0.0"]
    sys_block.append("/")
    L += sys_block

    # --- &ELECTRONS ---
    conv = "1.0d-9" if cell2d else "1.0d-8"
    L += ["&ELECTRONS",
          f"  conv_thr    = {conv}",
          "  mixing_beta = 0.3",
          "/"]

    # --- &IONS ---
    L += ["&IONS",
          "  ion_dynamics = 'bfgs'",
          "/"]

    # --- &CELL (cell2d only) ---
    if cell2d:
        L += ["&CELL",
              "  cell_dynamics  = 'bfgs'",
              "  cell_dofree    = '2Dxy'",
              "  press_conv_thr = 0.1",
              "/"]

    # --- ATOMIC_SPECIES ---
    L.append("ATOMIC_SPECIES")
    for e in species:
        m, p = QE_PSEUDO[e]
        L.append(f"  {e:2s} {m:>7s}  {p}")

    # --- CELL_PARAMETERS ---
    L.append("CELL_PARAMETERS angstrom")
    for v in cell:
        v = v + 0.0
        v[np.abs(v) < 1e-10] = 0.0
        L.append(f"  {v[0]:14.8f} {v[1]:14.8f} {v[2]:14.8f}")

    # --- ATOMIC_POSITIONS ---
    nfix = 0
    if cell2d:
        L.append("ATOMIC_POSITIONS crystal")
        for i in range(slab.n):
            f = frac_pos[i].copy()
            f[np.abs(f) < 1e-10] = 0.0
            if f[2] < zfix_frac:
                L.append(f"  {slab.el[i]:2s} {f[0]:14.8f} {f[1]:14.8f} "
                         f"{f[2]:14.8f}  0 0 0")
                nfix += 1
            else:
                L.append(f"  {slab.el[i]:2s} {f[0]:14.8f} {f[1]:14.8f} "
                         f"{f[2]:14.8f}")
    else:
        L.append("ATOMIC_POSITIONS angstrom")
        for i in range(slab.n):
            p = pos[i]
            if p[2] < zfix:
                L.append(f"  {slab.el[i]:2s} {p[0]:14.8f} {p[1]:14.8f} "
                         f"{p[2]:14.8f}  0 0 0")
                nfix += 1
            else:
                L.append(f"  {slab.el[i]:2s} {p[0]:14.8f} {p[1]:14.8f} "
                         f"{p[2]:14.8f}")

    L.append("K_POINTS automatic")
    L.append(f"  {nk1} {nk2} 1  0 0 0")
    L.append(f"! {nfix} atoms frozen (bottom region) via if_pos flags")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- CLI
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "LIST":
        lib = yaml.safe_load(open(LIB_PATH))
        for k, v in lib.items():
            print(f"{k:28s} {v.get('status','')}")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("motif")
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--vacuum", type=float, default=15.0)
    ap.add_argument("--a0", type=float, default=G.A0_DEFAULT)
    ap.add_argument("--symmetric", action="store_true", default=True,
                    help="(default) both faces identical: dipole-free, "
                         "equal surface stress")
    ap.add_argument("--asymmetric", action="store_true",
                    help="EXPLICIT opt-out: single reconstructed top face "
                         "with --bottom {bare,H}; QE dipole correction is "
                         "enabled automatically. Not for surface-stress "
                         "production runs.")
    ap.add_argument("--bottom", choices=["bare", "H"], default="bare",
                    help="bottom termination (only with --asymmetric)")
    ap.add_argument("--repeat", type=int, nargs=2, default=(1, 1))
    ap.add_argument("--format", choices=["poscar", "xyz", "cif", "qe"],
                    default="poscar")
    ap.add_argument("--qe-fix-bottom", type=int, default=2,
                    help="QE format: freeze this many bottom C layers "
                         "(asymmetric slabs; ignored for --symmetric)")
    ap.add_argument("--relax-mode", choices=["ions", "cell2d"], default="ions",
                    help="QE format only: 'ions' = fixed-cell relax (default); "
                         "'cell2d' = vc-relax with in-plane cell freedom (2Dxy)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.relax_mode == "cell2d" and a.format != "qe":
        sys.exit("ERROR: --relax-mode cell2d is only valid with --format qe")
    if a.asymmetric:
        a.symmetric = False
        print("WARNING: --asymmetric requested: faces differ "
              f"(top=motif, bottom={a.bottom}). QE dipole correction will "
              "be enabled; do NOT use for surface-stress production runs.",
              file=sys.stderr)

    slab, motif = generate(a.motif, a.layers, a.a0, a.symmetric,
                           a.bottom, tuple(a.repeat))
    comment = (f"{a.motif} | {a.layers} layers | a0={a.a0} | "
               f"{'symmetric' if a.symmetric else 'bottom=' + a.bottom} | "
               f"ideal-constructed, requires DFT relaxation")
    if a.format == "qe":
        def fn(slab, vacuum, comment):
            return to_qe(slab, vacuum, comment,
                         fix_bottom_layers=a.qe_fix_bottom,
                         symmetric=a.symmetric,
                         relax_mode=a.relax_mode)
    else:
        fn = {"poscar": to_poscar, "xyz": to_xyz, "cif": to_cif}[a.format]
    text = fn(slab, a.vacuum, comment)
    out = a.out or (f"{a.motif}_{a.layers}L."
                    + {"poscar": "vasp", "xyz": "xyz", "cif": "cif",
                       "qe": "pw.in"}[a.format])
    open(out, "w").write(text)
    counts = {e: slab.el.count(e) for e in sorted(set(slab.el))}
    print(f"wrote {out}  ({counts}, "
          f"z-span {np.ptp(slab.pos[:,2]):.2f} A + {a.vacuum} A vacuum)")


if __name__ == "__main__":
    main()
