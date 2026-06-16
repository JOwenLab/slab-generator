"""
build_library.py — construct the ideal motif library and write motifs.yaml.

Each entry stores:
  * metadata (orientation, cell multiple, termination, coverage, status,
    provenance, stoichiometry, symmetry/dipole notes)
  * the explicit surface-region coordinates (adsorbates + top C layers),
    DIMENSIONLESS: in-plane as fractions of the surface cell, z in units
    of a0. The library is therefore invariant to the choice of a0.

Literature-bound motifs (Pandey, Kern-Hafner graphitic, methoxyacetone,
Seiwatz-canted O) are written as 'pending-literature' stubs.
"""
import numpy as np
import yaml
import geometry as G

A0 = G.A0_DEFAULT
N_BUILD_LAYERS = 8          # layers used while constructing motifs
REGION_LAYERS = 4           # how many C layers belong to the stored region


# ----------------------------------------------------------------- helpers
def dimer_cell_100():
    """Probe which surface vector the (100) dangling lobes lie along and
    return the supercell multiple that doubles the cell in that direction."""
    s = G.bulk_slab("100", N_BUILD_LAYERS, A0)
    bonds, vecs = G.neighbors(s)
    i = G.surface_atoms(s, True)[0]
    d = G.dangling_directions(s, i, bonds, vecs)[0]
    h = np.array([d[0], d[1]])
    a1 = s.A1[:2] / np.linalg.norm(s.A1[:2])
    along_a1 = abs(np.dot(h, a1)) > abs(np.linalg.norm(h)) * 0.7
    return (2, 1) if along_a1 else (1, 2)


def build(orientation, cell, decorate, n_layers=N_BUILD_LAYERS):
    shift = G.detect_sdb_shift_111(A0) if orientation == "111" else 0
    s = G.bulk_slab(orientation, n_layers, A0, z_shift_layers=shift)
    if cell != (1, 1):
        s = G.repeat_inplane(s, *cell)
    return decorate(s)


def region_record(slab: G.Slab):
    """Extract adsorbates + top REGION_LAYERS C layers, dimensionless."""
    zc = sorted({round(z, 3) for z, e in zip(slab.pos[:, 2], slab.el)
                 if e == "C"}, reverse=True)
    zmin = zc[REGION_LAYERS - 1] - 0.05
    f = slab.frac_inplane()
    atoms = []
    for i in range(slab.n):
        if slab.el[i] != "C" or slab.pos[i, 2] > zmin:
            atoms.append({
                "el": slab.el[i],
                "f": [round(float(f[i, 0]) % 1.0, 6),
                      round(float(f[i, 1]) % 1.0, 6)],
                "z_a0": round(float(slab.pos[i, 2]) / slab.a0, 6),
            })
    # registry of the first bulk layer below the region (splice anchor)
    anchor_z = zc[REGION_LAYERS]
    anchor = [[round(float(f[i, 0]) % 1.0, 6), round(float(f[i, 1]) % 1.0, 6)]
              for i in range(slab.n)
              if slab.el[i] == "C" and abs(slab.pos[i, 2] - anchor_z) < 0.05]
    return atoms, sorted(anchor), round(float(anchor_z) / slab.a0, 6)


def bond_table(slab: G.Slab, rcut=1.9):
    """All hetero/homo bonds below rcut, for validation."""
    imgs = G._images(slab)
    out = []
    for i in range(slab.n):
        d = slab.pos[None, :, :] + imgs[:, None, :] - slab.pos[i]
        dist = np.linalg.norm(d, axis=2)
        for im in range(len(imgs)):
            for j in range(slab.n):
                if dist[im, j] < 1e-6 or dist[im, j] > rcut or j < i:
                    continue
                out.append((slab.el[i], slab.el[j], dist[im, j]))
    return out


def validate(name, slab, expect):
    """expect: list of (elA, elB, target, tol, min_count). Returns report."""
    tab = bond_table(slab)
    rep, ok = [], True
    for (a, b, tgt, tol, mc) in expect:
        ds = [d for (x, y, d) in tab
              if {x, y} == {a, b} and abs(d - tgt) < tol]
        bad = [d for (x, y, d) in tab
               if {x, y} == {a, b} and abs(d - tgt) >= tol
               and d < tgt + 3 * tol]
        if len(ds) < mc:
            ok = False
            rep.append(f"  FAIL {a}-{b}: found {len(ds)} within "
                       f"{tgt}+/-{tol} (need >= {mc}); near-misses: "
                       f"{[round(x,3) for x in bad[:4]]}")
        else:
            rng = (min(ds), max(ds))
            rep.append(f"  ok   {a}-{b}: {len(ds)} bonds, "
                       f"{rng[0]:.3f}-{rng[1]:.3f} A (target {tgt})")
    # universal: no absurdly short contacts
    short = [(x, y, d) for (x, y, d) in tab if d < 1.05]
    if short:
        ok = False
        rep.append(f"  FAIL short contacts: {short[:4]}")
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    for line in rep:
        print(line)
    return ok, rep


# ----------------------------------------------------------------- motifs
CC = np.sqrt(3) / 4 * A0     # 1.5445 bulk bond

LIB = {}
REPORTS = {}


def register(name, slab, meta, expect):
    ok, rep = validate(name, slab, expect)
    atoms, anchor, anchor_z = region_record(slab)
    entry = dict(meta)
    entry.update({
        "status_validation": "pass" if ok else "FAIL",
        "region_atoms": atoms,
        "bulk_anchor_registry": anchor,
        "bulk_anchor_z_a0": anchor_z,
        "region_layers": REGION_LAYERS,
    })
    LIB[name] = entry
    REPORTS[name] = (ok, rep)
    return slab


def meta(orientation, cell, term, cov, motif, notes, dipole, plane_group,
         sp2_atoms="none"):
    return {
        "orientation": orientation,
        "surface_cell_multiple": list(cell),
        "termination": term,
        "coverage_ML": cov,
        "motif": motif,
        "status": "ideal-constructed",
        "source": {"type": "geometric construction",
                   "bond_lengths": "C-H 1.09, C-F 1.36, C-O(ether) 1.43, "
                                   "C=O 1.21, bare dimer 1.37, "
                                   "saturated dimer 1.60 (Angstrom)",
                   "doi": None, "functional": None, "a0_used": None},
        "requires_dft_relaxation": True,
        "expected_dipole_z": dipole,
        "plane_group_approx": plane_group,
        "sp2_carbons": sp2_atoms,
        "notes": notes,
    }


def main():
    # ---------------------------------------------------------- (100) 2x1
    dc = dimer_cell_100()
    s = build("100", dc, lambda s: G.dimerize_100(s, G.DIMER_BARE))
    register("C100_2x1_bare", s,
             meta("100", dc, "none", 0.0, "pi-bonded symmetric dimer",
                  "Bare reconstructed surface; dimer C are sp2-like.",
                  "none (nonpolar termination)", "pmm", "dimer pair"),
             [("C", "C", G.DIMER_BARE, 0.06, 1),
              ("C", "C", CC, 0.12, 6)])

    s = build("100", dc, lambda s: G.add_monovalent(
        G.dimerize_100(s, G.DIMER_SAT), "H", G.BOND[("C", "H")]))
    register("C100_2x1_H", s,
             meta("100", dc, "H", 1.0, "monohydride on symmetric dimer",
                  "Canonical C(100)-(2x1):H.", "small, H+ outward (NEA)",
                  "pmm"),
             [("C", "C", G.DIMER_SAT, 0.06, 1),
              ("C", "H", 1.09, 0.03, 2),
              ("C", "C", CC, 0.12, 6)])

    s = build("100", dc, lambda s: G.add_monovalent(
        G.dimerize_100(s, G.DIMER_SAT), "F", G.BOND[("C", "F")]))
    register("C100_2x1_F", s,
             meta("100", dc, "F", 1.0, "monofluoride on symmetric dimer",
                  "Analog of monohydride; >1 ML difluoride excluded "
                  "(steric).", "F- outward (PEA)", "pmm"),
             [("C", "C", G.DIMER_SAT, 0.06, 1),
              ("C", "F", 1.36, 0.03, 2),
              ("C", "C", CC, 0.12, 6)])

    # ---------------------------------------------------------- (100) 1x1 O
    s = build("100", (1, 1), lambda s: G.add_bridge_oxygen(
        s, per_atom_limit=2))
    register("C100_1x1_O_ether", s,
             meta("100", (1, 1), "O", 1.0, "C-O-C bridge (ether), "
                  "reconstruction lifted",
                  "Classic ether model (cf. Sque/Jones/Briddon PRB 73, "
                  "085313 (2006) for relaxed coords). Ideal geometry is "
                  "strained (COC angle wide); DFT relaxation pulls surface "
                  "C together.", "moderate, O- outward", "p4m-like"),
             [("C", "O", 1.43, 0.05, 2),
              ("C", "C", CC, 0.12, 3)])

    s = build("100", (1, 1), lambda s: G.add_ketone_oxygen(s))
    register("C100_1x1_O_ketone", s,
             meta("100", (1, 1), "O", 1.0, "on-top C=O (ketone), "
                  "reconstruction lifted",
                  "Surface C is sp2; expect distinct electron affinity vs "
                  "ether (~+3.9 vs ~+2.6 eV) as validation observable.",
                  "large, C=O dipole", "p4m-like", "all surface C"),
             [("C", "O", 1.21, 0.04, 1),
              ("C", "C", CC, 0.12, 3)])

    # ---------------------------------------------------------- (111) 1x1
    s = build("111", (1, 1), lambda s: s)
    register("C111_1x1_bare", s,
             meta("111", (1, 1), "none", 0.0, "bulk-truncated SDB face "
                  "(reference only)",
                  "Real bare (111) reconstructs to Pandey 2x1; this entry "
                  "is the unreconstructed reference.", "none", "p3m1"),
             [("C", "C", CC, 0.12, 3)])

    s = build("111", (1, 1), lambda s: G.add_monovalent(
        s, "H", G.BOND[("C", "H")]))
    register("C111_1x1_H", s,
             meta("111", (1, 1), "H", 1.0, "on-top C-H, SDB face",
                  "Canonical NEA surface; full coverage implies "
                  "de-reconstruction from Pandey 2x1.",
                  "small, H+ outward (NEA)", "p3m1"),
             [("C", "H", 1.09, 0.03, 1),
              ("C", "C", CC, 0.12, 3)])

    s = build("111", (1, 1), lambda s: G.add_monovalent(
        s, "F", G.BOND[("C", "F")]))
    register("C111_1x1_F", s,
             meta("111", (1, 1), "F", 1.0, "on-top C-F, SDB face",
                  "Sen et al., J. Mater. Res. 24, 2461 (2009): stable over "
                  "wider phase space than H.", "F- outward (PEA)", "p3m1"),
             [("C", "F", 1.36, 0.03, 1),
              ("C", "C", CC, 0.12, 3)])

    s = build("111", (1, 1), lambda s: G.add_ketone_oxygen(s))
    register("C111_1x1_O_ketone", s,
             meta("111", (1, 1), "O", 1.0, "on-top C=O, SDB face",
                  "On-top O most favorable on 1x1 (Petrini & Larsson, JPCC "
                  "112, 3018 (2008)). NOT the global ground state at 1 ML "
                  "(see Seiwatz-canted, pending literature).",
                  "large, C=O dipole", "p3m1", "all surface C"),
             [("C", "O", 1.21, 0.04, 1),
              ("C", "C", CC, 0.12, 3)])

    # ---------------------------------------------------------- (110) 1x1
    s = build("110", (1, 1), lambda s: s)
    register("C110_1x1_bare", s,
             meta("110", (1, 1), "none", 0.0, "bulk-truncated zigzag "
                  "chains; relaxation only",
                  "No reconstruction observed by LEED.", "none", "pmg-like"),
             [("C", "C", CC, 0.12, 4)])

    s = build("110", (1, 1), lambda s: G.add_monovalent(
        s, "H", G.BOND[("C", "H")]))
    register("C110_1x1_H", s,
             meta("110", (1, 1), "H", 1.0, "one H per chain C, along "
                  "dangling-bond direction",
                  "NEA surface.", "small, H+ outward", "pmg-like"),
             [("C", "H", 1.09, 0.03, 2),
              ("C", "C", CC, 0.12, 4)])

    s = build("110", (1, 1), lambda s: G.add_monovalent(
        s, "F", G.BOND[("C", "F")]))
    register("C110_1x1_F", s,
             meta("110", (1, 1), "F", 1.0, "one F per chain C",
                  "PEA, no in-gap states / surface spins (NV-relevant). "
                  "Cui et al., Diam. Relat. Mater. (2022).",
                  "F- outward", "pmg-like"),
             [("C", "F", 1.36, 0.03, 2),
              ("C", "C", CC, 0.12, 4)])

    # (110) ether: geometry decides which lobe pairs can be bridged;
    # built in a 1x2 cell to allow alternating bridges if needed.
    s = build("110", (1, 2), lambda s: G.add_bridge_oxygen(
        s, per_atom_limit=1))
    register("C110_O_ether", s,
             meta("110", (1, 2), "O", "all dangling bonds saturated",
                  "ether bridge between facing chain lobes",
                  "Full saturation of (110) dangling bonds by INTER-CHAIN C-O-C "
                  "bridges (facing lobes of adjacent zigzag chains, sep. "
                  "2.96 A). Ideal placement is stretched (C-O ~1.52 A); "
                  "DFT relaxation pulls surface C together. On-top ketone "
                  "excluded on pristine (110) (needs 2 dangling bonds/C).",
                  "moderate", "tbd"),
             [("C", "O", 1.56, 0.06, 4),
              ("C", "C", CC, 0.12, 6)])

    # ------------------------------------------------- literature stubs
    # ---- TDB-face (111) Seiwatz family (constructed to literature params)
    for name, variant, term, cov, motif_d, sp2, note in [
        ("C111_3db_2x1_seiwatz", "bare", "none (carbon reconstruction)", 0.0,
         "Seiwatz single zig-zag chains on the TDB (glide-plane, 3db) face",
         "chain atoms (2 per 2x1)",
         "Constructed to Kern, Hafner & Kresse, Surf. Sci. 366, 464 (1996) "
         "(DFT-LDA): chain bond 1.45 A, buckling 0.07 A, angles 119.4 deg; "
         "Erel -1.957 eV/atom. Subsurface bucklings (0.04/0.17/0.07/0.02 A) "
         "NOT pre-applied; backbonds carry ~0.15 A construction strain -- "
         "requires DFT relaxation. Asymmetric slabs only (SDB bottom), "
         "ODD layer counts."),
        ("C111_3db_2x1_seiwatz_H", "H", "H", 1.0,
         "monohydrogenated Seiwatz chains (TDB face)",
         "none",
         "Constructed to Kern-Hafner II: chain 1.525 A, C-H 1.106 A "
         "(built 1.09 A standard), buckling removed. Requires DFT "
         "relaxation. Asymmetric slabs only, ODD layer counts."),
        ("C111_3db_2x1_O_ketone", "ketone", "O", 1.0,
         "Sei-1ML-ketone: Seiwatz chains broken into C2 pairs, each C sp2 "
         "with on-top C=O (alpha-diketone-like O=C-C=O units; TDB face)",
         "dimer carbons (2 per 2x1)",
         "Topology determined from Yang, Gao & Li, Diam. Relat. Mater. 159, "
         "112823 (2025), Fig. 6c + text (DOI 10.1016/j.diamond.2025.112823; "
         "PBE/VASP, 520 eV): C=O 1.20 A; chains break because a chain C "
         "cannot carry 2 chain bonds + backbond + C=O (pentavalent). "
         "Dimer C-C built at 1.52 A with along-chain (Peierls-like) "
         "dimerization to isolate pairs. Published: Ef -2.23 eV/unit cell, "
         "metallic (bands cross E_F), EA +2.54 eV; NOTE paper finds in-gap/"
         "spin-polarized character for ketone-type surfaces -- the "
         "NV-friendly O/(111) structures are Pan-0.5ML-ether and "
         "Cha-0.5ML-ether (0.5 ML, outside full-coverage scope). "
         "Requires DFT relaxation."),
    ]:
        s = G.seiwatz_slab(variant, 11, A0, bottom="bare")
        m = meta("111", (1, 2), term, cov, motif_d, note,
                 "small (asymmetric construction)" if variant == "ketone"
                 else "none", "pm (chains)", sp2)
        m["face"] = "TDB"
        m["tdb_variant"] = variant
        m["layer_parity"] = "odd"
        m["source_params"] = (
            "Kern-Hafner II (1996)" if variant != "ketone"
            else "Yang, Gao & Li, DRM 159, 112823 (2025), Fig. 6c")
        m["requires_dft_relaxation"] = True
        expect = {
            "bare": [("C", "C", 1.45, 0.05, 2),      # Seiwatz chain bonds
                     ("C", "C", 1.5445, 0.25, 8)],   # lattice incl. strained
            "H": [("C", "C", 1.525, 0.05, 2),
                  ("C", "H", 1.09, 0.03, 2),
                  ("C", "C", 1.5445, 0.25, 8)],
            "ketone": [("C", "C", 1.52, 0.05, 1),    # isolated dimer
                       ("C", "O", 1.20, 0.03, 2),
                       ("C", "C", 1.5445, 0.25, 8)],
        }[variant]
        register(name, s, m, expect)

    for name, note in {
        "C100_O_methoxyacetone": "Mixed carbonyl+epoxide 1 ML ground state. "
            "Source: Lu, Fan, Chen, Mei, Ma & Hu, Carbon 159, 9-15 (2020), "
            "DOI 10.1016/j.carbon.2019.12.003 (CITATION CORRECTED from "
            "earlier 'Carbon 162, 510-518'). Methods: CALYPSO + VASP/PBE/"
            "PAW, 600 eV, slab >=8 C layers, O + top 4 layers relaxed, "
            "rest fixed. Known metrics: C=O 1.18 A, ether/epoxide C-O "
            "1.45 A, Ef -0.63 eV/(1x1) vs ketone, nearest O-O 5 A, gap "
            "5.27 eV (GW), EA +2.54 eV, phonon-stable; Raman 1100 + 550 "
            "cm-1. SI checked: figure captions only, NO coordinates. "
            "Fig. 1f is a single projection with an adatom-bearing, "
            "likely non-stoichiometric top layer -- bond graph NOT safely "
            "inferable; request coordinates from authors "
            "(huxj@zjut.edu.cn).",
        "C111_O_seiwatz_canted": "Seiwatz-chain 1 ML canted O ground state "
            "(Sei-1ML-canted, surface formation energy -2.64 eV/cell). "
            "Source: Yang, Gao & Li, Diam. Relat. Mater. 159, 112823 "
            "(2025), DOI 10.1016/j.diamond.2025.112823. SI checked: "
            "figure captions only, NO coordinates ('data available on "
            "request' -- email gaon@jlu.edu.cn / hdli@jlu.edu.cn). "
            "Fig. 6d analysis: polycyclic canted-ribbon structure derived "
            "from Sei-0.5ML-canted (dissociated surface-C ribbon) + extra "
            "O forming C-O single bonds 1.41-1.51 A; C=O 1.20-1.25 A; "
            "occluded projections -- bond graph NOT safely inferable. "
            "Published: Ef -2.64 eV/unit cell (ground state), gap 1.79 eV "
            "with intruding O-derived surface levels (paper deems it "
            "NV-unfriendly despite stability). "
            "SUBSTRATE NOTE: Seiwatz single chains live on the TDB "
            "(glide-plane, 3db) face -- generator currently builds the "
            "SDB face only; TDB support required before splicing. "
            "Bare-Seiwatz reference parameters (Kern, Hafner & Kresse, "
            "Surf. Sci. 366, 464 (1996), DFT-LDA): chain bond 1.45 A, "
            "chain buckling 0.07 A, chain angles 119.4 deg, subsurface "
            "bucklings 0.04/0.17/0.07/0.02 A, Erel -1.957 eV/atom; "
            "monohydrogenated Seiwatz: chain 1.525 A, C-H 1.106 A, "
            "buckling removed. Also wanted from SI: Sei-1ML-ketone.",
    }.items():
        LIB[name] = {"status": "pending-literature", "notes": note}

    with open("motifs.yaml", "w") as fh:
        yaml.safe_dump(LIB, fh, sort_keys=False, width=78)
    n_ok = sum(1 for k, (ok, _) in REPORTS.items() if ok)
    print(f"\n{n_ok}/{len(REPORTS)} constructed motifs passed validation; "
          f"{len(LIB) - len(REPORTS)} literature stubs written.")


if __name__ == "__main__":
    main()
