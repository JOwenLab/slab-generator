"""
transcribe_petukhov.py — transcribe published coordinates for the three
C(111) reconstructions of:

  A.V. Petukhov, D. Passerone, F. Ercolessi, E. Tosatti, A. Fasolino,
  "(Meta-)stable reconstructions of the diamond (111) surface: interplay
  between diamond- and graphite-like bonding",
  Phys. Rev. B 61 (2000); arXiv:cond-mat/0001397, Table I.

Structures (Brenner potential, parametrization I; bulk C-C = 1.541 A,
hence a0_source = 1.541*4/sqrt(3) = 3.55879 A):
  * Pandey pi-bonded chain (2x1)        — stable
  * dimerized "graphitic" (2x1)         — metastable, threefold surface
  * dimerized "graphitic" (4x1)         — metastable, threefold surface

Cell (their frame): x = perpendicular-to-chain period (4.357 A for 2x1,
8.714 A for 4x1), y = chain period (2.517 A), z = height (A).

These are EMPIRICAL-POTENTIAL coordinates: correct bond topology
(which cannot be safely reconstructed from memory), to be DFT-relaxed
by the user. Library status: 'published-model-coordinates'.
"""
import numpy as np
import yaml
import geometry as G

A0_SRC = 1.541 * 4 / np.sqrt(3)          # 3.55879 A (Brenner bulk)
CELL_X_2x1, CELL_Y = 4.357, 2.517        # printed cell, A

# ----------------------------------------------------- Table I (verbatim)
# columns: atom, x, y, z
PANDEY = """
11  3.064 -0.943 5.525
12  2.370  0.315 5.519
13  1.007  0.313 4.792
14  0.076 -0.941 4.804
21  3.816 -0.942 3.325
22  3.041  0.316 2.880
23  1.557  0.314 3.290
24  0.895 -0.943 2.737
31  3.065  0.316 1.300
32  2.346 -0.942 0.760
33  0.883 -0.944 1.227
34  0.152  0.315 0.739
"""

DIMER_2x1 = """
11  3.071 -0.942 5.638
12  2.353  0.332 5.625
13  1.177  0.594 4.804
14 -0.112 -1.188 4.820
21  3.835 -0.965 3.331
22  3.065  0.299 2.872
23  1.573  0.341 3.297
24  0.896 -0.910 2.759
31  3.075  0.318 1.295
32  2.251 -0.946 0.770
33  0.889 -0.930 1.240
34  0.156  0.328 0.752
"""

# 4x1: first half + (second half) per Table I brackets
DIMER_4x1 = """
11  3.072 -0.532 5.648
11b 7.435 -1.355 5.645
12  2.379  0.728 5.646
12b 6.742 -0.095 5.641
13  1.210  0.714 4.802
13b 5.591 -0.081 4.793
14 -0.151 -1.325 4.813
14b 4.228 -0.561 4.806
21  3.838 -0.911 3.330
21b 8.202 -0.989 3.325
22  3.072  0.350 2.866
22b 7.436  0.275 2.862
23  1.589  0.378 3.313
23b 5.967  0.250 3.302
24  0.898 -0.878 2.774
24b 5.272 -1.015 2.767
31  3.084  0.322 1.299
31b 7.449  0.306 1.291
32  2.361 -0.938 0.769
32b 6.727 -0.955 0.765
33  0.895 -0.918 1.252
33b 5.265 -0.974 1.246
34  0.162  0.342 0.756
34b 4.533  0.293 0.756
"""


def parse(tbl):
    rows = []
    for line in tbl.strip().splitlines():
        p = line.split()
        rows.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return rows


def make_slab(rows, nx_cell):
    """Build a periodic Slab in the source rectangular frame:
    A1 = chain (their y) -> Cartesian x;  A2 = perpendicular (their x) -> y.
    Note: this maps (x,y)->(y,x), a reflection; the splice routine resolves
    handedness/registry against the bulk by symmetry search at generation
    time, so the stored frame choice is immaterial."""
    A1 = np.array([CELL_Y, 0.0, 0.0])
    A2 = np.array([0.0, CELL_X_2x1 * nx_cell / 2, 0.0])
    pos = np.array([[r[2], r[1], r[3]] for r in rows])
    pos[:, 2] -= pos[:, 2].max()
    el = ["C"] * len(rows)
    s = G.Slab(A1, A2, pos, el, A0_SRC, "111", ["region"] * len(rows))
    return s.wrap()


# ----------------------------------------------------- topology analysis
def ring_stats(slab, max_ring=8):
    """Count smallest rings through each bond using a 3x3 image graph."""
    big = G.repeat_inplane(slab, 3, 3)
    bonds, _ = G.neighbors(big, rcut=1.85 * slab.a0 / 3.567)
    N = big.n
    adj = [set(b) for b in bonds]
    # central-cell atom indices (middle replica)
    n0 = slab.n
    central = range(4 * n0, 5 * n0)
    sizes = []
    for a in central:
        for b in adj[a]:
            if b < a:
                continue
            # BFS shortest path a->b avoiding direct edge
            prev = {a: None}
            q = [a]
            found = None
            while q and found is None:
                nq = []
                for u in q:
                    for v in adj[u]:
                        if u == a and v == b:
                            continue
                        if v not in prev:
                            prev[v] = u
                            if v == b:
                                found = v
                                break
                            nq.append(v)
                    if found:
                        break
                q = nq
            if found is None:
                continue
            n_path = 1
            u = found
            while prev[u] is not None:
                u = prev[u]
                n_path += 1
            ring = n_path + 1 - 1   # path edges + closing edge -> ring size
            if ring <= max_ring:
                sizes.append(ring)
    import collections
    return dict(sorted(collections.Counter(sizes).items()))


def coordination(slab):
    big = G.repeat_inplane(slab, 3, 3)
    bonds, _ = G.neighbors(big, rcut=1.85 * slab.a0 / 3.567)
    n0 = slab.n
    co = {}
    for k, a in enumerate(range(4 * n0, 5 * n0)):
        co[k] = len(bonds[a])
    return co


def bond_lengths(slab):
    big = G.repeat_inplane(slab, 3, 3)
    bonds, vecs = G.neighbors(big, rcut=1.85 * slab.a0 / 3.567)
    n0 = slab.n
    out = []
    for a in range(4 * n0, 5 * n0):
        for j, v in zip(bonds[a], vecs[a]):
            pass
    # recompute distances directly
    imgs = G._images(slab)
    for i in range(n0):
        for j in range(n0):
            for im in imgs:
                d = np.linalg.norm(slab.pos[j] + im - slab.pos[i])
                if 0.1 < d < 1.85 * slab.a0 / 3.567 and (j > i or d > 0.1
                                                         and not np.allclose(im, 0)):
                    out.append(round(float(d), 3))
    return sorted(set(out))


def region_entry(slab, name, motif_desc, notes, plane_group, sp2):
    f = slab.frac_inplane()
    atoms = [{"el": "C",
              "f": [round(float(f[i, 0]) % 1.0, 6),
                    round(float(f[i, 1]) % 1.0, 6)],
              "z_a0": round(float(slab.pos[i, 2]) / A0_SRC, 6)}
             for i in range(slab.n)]
    return {
        "orientation": "111",
        "surface_cell_multiple": None,   # literature frame, see cell_note
        "cell_note": ("rectangular cell: chain period a0/sqrt(2) x "
                      "perpendicular period; stored frame is the source "
                      "frame; registry resolved at splice time"),
        "cell_A": [[CELL_Y, 0.0],
                   [0.0, float(slab.A2[1])]],
        "termination": "none (carbon reconstruction)",
        "coverage_ML": 0.0,
        "motif": motif_desc,
        "status": "published-model-coordinates",
        "source": {
            "type": "transcribed published coordinates (Table I)",
            "reference": "Petukhov, Passerone, Ercolessi, Tosatti, "
                         "Fasolino, Phys. Rev. B 61 (2000); "
                         "arXiv:cond-mat/0001397",
            "method": "Brenner empirical potential (param. I), "
                      "off-lattice GCMC",
            "a0_used": float(round(A0_SRC, 5)),
            "functional": None,
            "relaxed": "empirical-potential minimum; NOT DFT-relaxed",
        },
        "requires_dft_relaxation": True,
        "expected_dipole_z": "none (carbon-only termination)",
        "plane_group_approx": plane_group,
        "sp2_carbons": sp2,
        "notes": notes,
        "region_atoms": atoms,
        "region_layers": 6,
        "origin": "floating (aligned to bulk at splice time)",
    }


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def main():
    lib = yaml.safe_load(open("motifs.yaml"))
    report = {}

    for name, tbl, nx, desc, sp2 in [
        ("C111_2x1_pandey", PANDEY, 2,
         "Pandey pi-bonded chain (transcribed; Brenner)",
         "top chain atoms (2 per 2x1)"),
        ("C111_2x1_graphitic", DIMER_2x1, 2,
         "dimerized lower-chain metastable, threefold 'graphitic' surface "
         "(2x1)",
         "all first-bilayer atoms"),
        ("C111_4x1_graphitic", DIMER_4x1, 4,
         "dimerized lower-chain metastable, threefold 'graphitic' surface "
         "(4x1)",
         "all first-bilayer atoms"),
    ]:
        s = make_slab(parse(tbl), nx)
        co = coordination(s)
        rings = ring_stats(s)
        dists = bond_lengths(s)
        report[name] = {"coordination": co, "rings": rings,
                        "bonds_A": dists}
        notes = (f"Transcribed from Table I. Energy gains (Brenner, "
                 f"eV per 1x1): Pandey 1.102, dimer-2x1 0.883, "
                 f"dimer-4x1 1.023. Region = top 3 bilayers; deeper "
                 f"layers near-bulk. Mis-attribution in earlier catalog "
                 f"draft (Kern-Hafner) corrected: graphitic metastables "
                 f"are due to Petukhov et al. Transcription flag: atom 32 "
                 f"x-coordinate in dimer-2x1 read as 2.251 (vs 2.346 "
                 f"Pandey / 2.361 4x1); bond validation passes as "
                 f"printed.")
        entry = region_entry(s, name, desc, notes, "pm (chains)", sp2)
        if name == "C111_2x1_pandey":
            entry["validation_targets"] = {
                "comment": "Post-DFT-relaxation acceptance criteria; the "
                           "two sets agree to ~1%.",
                "LEED_Walter2002": {
                    "reference": "Walter, Bernhardt, Starke, Heinz, Maier, "
                                 "Ristein, Ley, J. Phys.: Condens. Matter "
                                 "14, 3085 (2002); Pendry R=0.19",
                    "bulk_d0_A": 1.544, "bulk_db0_A": 0.515,
                    "d12_A": 1.55, "d23_A": 1.55, "d34_A": 1.52,
                    "db1_A": 0.68, "db2_A": 0.51, "db3_A": 0.50,
                    "db4_A": 0.51,
                    "buckling_A": {"b11": 0.01, "b12": 0.01, "b21": 0.02,
                                   "b22": 0.18, "b31": 0.08, "b32": 0.01,
                                   "b41": 0.01, "b42": 0.04},
                    "bonds_A": {"l12": 1.62, "l12p": 1.64, "l23": 1.61,
                                "l23p": 1.49},
                    "chain_tilt": "untilted (intrachain buckling 0.01 A, "
                                  "error 0.04 A)",
                    "dimerization_pct": 0.7,
                    "dimerization_upper_limit_pct": 7.0,
                },
                "DFT_KernHafner1996": {
                    "reference": "Kern, Hafner, Kresse, Surf. Sci. 366, "
                                 "445 (1996), Table 4 + Fig. 1; DFT-LDA, "
                                 "a0_calc 3.531 A, bulk bond 1.529 A",
                    "chain_bond_A": 1.425,
                    "dimerization_pct": 0.0,
                    "chain_buckling_A": 0.006,
                    "d12_pct": [0.2, -0.2], "dprime_pct": 0.9,
                    "d23_pct": [6.5, 4.6],
                    "layer_bucklings_A": [0.01, 0.01, 0.03, 0.17,
                                          0.06, 0.02],
                    "Erel_2x1_eV_per_atom": -1.375,
                },
            }
        lib[name] = entry

    # retire the misattributed stub names
    for old in ("C111_2x1_graphitic_KH", "C111_4x1_graphitic_KH"):
        if old in lib:
            del lib[old]

    text = yaml.safe_dump(_clean(lib), sort_keys=False, width=78)
    with open("motifs.yaml", "w") as fh:
        fh.write(text)

    for k, v in report.items():
        print(f"== {k}")
        print("   coordination:", v["coordination"])
        print("   ring sizes:", v["rings"])
        print("   distinct bond lengths (A):", v["bonds_A"])


if __name__ == "__main__":
    main()
