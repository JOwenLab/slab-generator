"""
geometry.py — core engine for diamond slab construction.

Conventions
-----------
* a0: conventional cubic lattice constant of diamond (default 3.567 Angstrom,
  experimental). All internal construction is done at a caller-supplied a0;
  the motif LIBRARY stores coordinates in dimensionless units (in-plane:
  fractions of the surface cell; z: units of a0) so the library is invariant
  to the choice of a0.
* Surface normal is always +z. Slabs are built downward from z ~ 0.
* "Layer" = cluster of atoms with the same z (within tolerance) in the ideal
  bulk-truncated slab.

Orientation-specific surface cells (Cartesian, units of a0)
-----------------------------------------------------------
(100): A1 = (1/2, 1/2, 0), A2 = (1/2,-1/2, 0)        square,    a0/sqrt(2)
(110): A1 = (1/2,-1/2, 0), A2 = (0, 0, 1)            rectangle, a0/sqrt(2) x a0
(111): A1 = (1/2,-1/2, 0), A2 = (1/2, 0,-1/2)        hexagonal (60 deg), a0/sqrt(2)
       rotated so that the surface normal [111] -> +z.

All bulk generation is numeric (replicate conventional cell, rotate, slice);
no hand-derived registry tables, which avoids transcription errors.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# ---------------------------------------------------------------- constants
A0_DEFAULT = 3.567  # Angstrom, experimental

BOND = {           # ideal construction targets, Angstrom
    ("C", "C"): None,        # bulk: sqrt(3)/4 * a0, computed at runtime
    ("C", "H"): 1.09,
    ("C", "F"): 1.36,
    ("C", "O_ether"): 1.43,
    ("C", "O_ketone"): 1.21,
}
DIMER_BARE = 1.37        # pi-bonded bare (100) dimer
DIMER_SAT = 1.60         # mono-hydride / mono-fluoride (100) dimer
CHAIN_PANDEY = 1.44      # sp2 chain bond target for Pandey construction

TET_ANGLE_COS = -1.0 / 3.0

# conventional diamond cell, fractional
_DIAMOND_BASIS = np.array([
    [0.00, 0.00, 0.00], [0.00, 0.50, 0.50],
    [0.50, 0.00, 0.50], [0.50, 0.50, 0.00],
    [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
])


# ---------------------------------------------------------------- structure
@dataclass
class Slab:
    """Periodic in A1, A2 (Cartesian, Angstrom); finite in z."""
    A1: np.ndarray
    A2: np.ndarray
    pos: np.ndarray            # (N,3) Cartesian, Angstrom
    el: list                   # element symbols
    a0: float
    orientation: str
    tags: list = field(default_factory=list)   # per-atom annotations

    def copy(self):
        return Slab(self.A1.copy(), self.A2.copy(), self.pos.copy(),
                    list(self.el), self.a0, self.orientation, list(self.tags))

    @property
    def n(self):
        return len(self.el)

    def frac_inplane(self):
        """In-plane fractional coordinates wrt A1, A2 (z passes through)."""
        M = np.array([self.A1[:2], self.A2[:2]]).T  # 2x2
        f = np.linalg.solve(M, self.pos[:, :2].T).T
        return f

    def wrap(self):
        f = self.frac_inplane()
        f -= np.floor(f + 1e-9)
        self.pos[:, :2] = (f @ np.array([self.A1[:2], self.A2[:2]]))
        return self


# ------------------------------------------------------- orientation frames
def _frame(orientation: str, a0: float):
    """Return (A1, A2, rotation) for the orientation: surface cell vectors in
    the rotated frame (normal -> +z) and the rotation matrix R (row-vector
    convention: v_rot = v_cubic @ R.T)."""
    o = orientation
    if o == "100":
        n = np.array([0.0, 0.0, 1.0])           # use (001), equivalent
        a1c = a0 * np.array([0.5, 0.5, 0.0])
        a2c = a0 * np.array([0.5, -0.5, 0.0])
    elif o == "110":
        n = np.array([1.0, 1.0, 0.0])
        a1c = a0 * np.array([0.5, -0.5, 0.0])   # [1-10]
        a2c = a0 * np.array([0.0, 0.0, 1.0])    # [001]
    elif o == "111":
        n = np.array([1.0, 1.0, 1.0])
        a1c = a0 * np.array([0.5, -0.5, 0.0])
        a2c = a0 * np.array([0.5, 0.0, -0.5])
    else:
        raise ValueError(orientation)
    nz = n / np.linalg.norm(n)
    x = a1c / np.linalg.norm(a1c)
    y = np.cross(nz, x)
    R = np.array([x, y, nz])                    # rows = new basis in cubic
    A1 = R @ a1c
    A2 = R @ a2c
    assert abs(A1[2]) < 1e-9 and abs(A2[2]) < 1e-9
    return A1, A2, R


def bulk_slab(orientation: str, n_layers: int, a0: float = A0_DEFAULT,
              z_shift_layers: int = 0) -> Slab:
    """Bulk-truncated slab with n_layers atomic layers, top layer near z=0.

    z_shift_layers: slide the slicing window down by this many layers
    (used on (111) to select the single-dangling-bond face).
    """
    A1, A2, R = _frame(orientation, a0)
    # generate a generous block of bulk
    reach = int(np.ceil(n_layers / 2)) + 6
    cells = range(-reach, reach + 1)
    pts, base = [], _DIAMOND_BASIS * a0
    for i in cells:
        for j in cells:
            for k in cells:
                t = a0 * np.array([i, j, k])
                pts.append(base + t)
    pts = np.vstack(pts) @ R.T

    # wrap into one surface cell (in-plane) and deduplicate
    M = np.array([A1[:2], A2[:2]]).T
    f = np.linalg.solve(M, pts[:, :2].T).T
    f -= np.floor(f + 1e-7)
    f[f > 1 - 1e-7] = 0.0
    pts[:, :2] = f @ np.array([A1[:2], A2[:2]])
    key = np.round(np.column_stack([f, pts[:, 2]]), 4)
    _, idx = np.unique(key, axis=0, return_index=True)
    pts = pts[idx]

    # cluster into layers by z
    z = np.round(pts[:, 2], 4)
    levels = np.unique(z)[::-1]                  # descending
    levels = levels[4:]                          # trim incomplete block edge
    levels = levels[z_shift_layers:z_shift_layers + n_layers]
    keep = np.isin(z, levels)
    pts = pts[keep]
    pts[:, 2] -= pts[:, 2].max()                 # top layer at z = 0
    order = np.argsort(-pts[:, 2])
    pts = pts[order]
    s = Slab(A1, A2, pts, ["C"] * len(pts), a0, orientation,
             ["bulk"] * len(pts))
    return s.wrap()


# ------------------------------------------------------------- connectivity
def _images(slab: Slab):
    out = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            out.append(i * np.r_[slab.A1, ] + j * np.r_[slab.A2, ])
    return np.array(out)


def neighbors(slab: Slab, rcut: float = None):
    """List of bonded pairs and per-atom unit bond vectors (min. image)."""
    if rcut is None:
        rcut = 1.15 * np.sqrt(3) / 4 * slab.a0   # ~1.78 A for C-C
    imgs = _images(slab)
    N = slab.n
    bonds = [[] for _ in range(N)]
    vecs = [[] for _ in range(N)]
    for i in range(N):
        d = slab.pos[None, :, :] + imgs[:, None, :] - slab.pos[i]
        dist = np.linalg.norm(d, axis=2)
        dist[:, i][np.abs(dist[:, i]) < 1e-6] = 1e9   # exclude self image 0
        hits = np.argwhere(dist < rcut)
        for (im, j) in hits:
            bonds[i].append(j)
            vecs[i].append(d[im, j] / dist[im, j])
    return bonds, vecs


def dangling_directions(slab: Slab, i: int, bonds, vecs):
    """Unit vector(s) of missing tetrahedral bonds for atom i (C only)."""
    b = np.array(vecs[i])
    nb = len(b)
    if nb >= 4:
        return []
    if nb == 3:
        d = -b.sum(axis=0)
        return [d / np.linalg.norm(d)]
    if nb == 2:
        u = b.sum(axis=0)
        u /= np.linalg.norm(u)
        w = np.cross(b[0], b[1])
        w /= np.linalg.norm(w)
        ca, sa = 1 / np.sqrt(3), np.sqrt(2.0 / 3.0)
        return [-ca * u + sa * w, -ca * u - sa * w]
    raise RuntimeError(f"atom {i}: {nb} bonds — cannot infer danglings")


def surface_atoms(slab: Slab, top=True, tol=0.25):
    z = slab.pos[:, 2]
    zref = z.max() if top else z.min()
    return [i for i in range(slab.n)
            if abs(z[i] - zref) < tol and slab.el[i] == "C"]


# ------------------------------------------------------------- decorators
def add_monovalent(slab: Slab, species: str, length: float, top=True,
                   bottom=False):
    """Saturate every dangling bond on the chosen face(s) with one atom."""
    bonds, vecs = neighbors(slab)
    faces = ([True] if top else []) + ([False] if bottom else [])
    newp, newe = [], []
    for face in faces:
        for i in surface_atoms(slab, top=face):
            for d in dangling_directions(slab, i, bonds, vecs):
                if (d[2] > 0) == face:           # outward only
                    newp.append(slab.pos[i] + length * d)
                    newe.append(species)
    if newp:
        slab.pos = np.vstack([slab.pos, np.array(newp)])
        slab.el += newe
        slab.tags += ["adsorbate"] * len(newe)
    return slab


def dimerize_100(slab: Slab, target: float, top=True, bottom=False):
    """Form (2x1) dimers on a (100) face. The slab cell must already be the
    2x1 cell (two surface atoms per face). Surface atoms are displaced toward
    each other along the line joining the facing dangling lobes."""
    bonds, vecs = neighbors(slab)
    faces = ([True] if top else []) + ([False] if bottom else [])
    for face in faces:
        surf = surface_atoms(slab, top=face)
        assert len(surf) == 2, "dimerize_100 expects a 2x1 cell (2 atoms/face)"
        i, j = surf
        # minimum-image vector i -> j
        best, bv = None, None
        for img in _images(slab):
            v = slab.pos[j] + img - slab.pos[i]
            dd = np.linalg.norm(v)
            if best is None or dd < best:
                best, bv = dd, v
        u = bv / best
        # verify the lobes actually face each other
        di = dangling_directions(slab, i, bonds, vecs)
        dj = dangling_directions(slab, j, bonds, vecs)
        ok_i = max(np.dot(d, u) for d in di)
        ok_j = max(np.dot(d, -u) for d in dj)
        assert ok_i > 0.4 and ok_j > 0.4, "lobes do not face: wrong pairing"
        shift = (best - target) / 2.0
        slab.pos[i] += shift * u
        slab.pos[j] -= shift * u
    return slab


def add_bridge_oxygen(slab: Slab, top=True, bottom=False,
                      per_atom_limit=None):
    """Place ether O atoms bridging pairs of surface C whose dangling lobes
    face each other. per_atom_limit caps how many O a single C may receive
    (2 on (100) full-ML ether; 1 on (110))."""
    bonds, vecs = neighbors(slab)
    L = BOND[("C", "O_ether")]
    faces = ([True] if top else []) + ([False] if bottom else [])
    newp = []
    for face in faces:
        surf = surface_atoms(slab, top=face)
        lobes = {i: [d for d in dangling_directions(slab, i, bonds, vecs)
                     if (d[2] > 0) == face] for i in surf}
        used = {i: 0 for i in surf}
        cap = per_atom_limit or 99
        pairs = []
        for ii, i in enumerate(surf):
            for j in surf[ii:]:                  # include i==j across image
                for img in _images(slab):
                    v = slab.pos[j] + img - slab.pos[i]
                    dd = np.linalg.norm(v)
                    if dd < 0.1 or dd > 3.3:
                        continue
                    u = v / dd
                    fi = max((np.dot(d, u) for d in lobes[i]), default=-1)
                    fj = max((np.dot(d, -u) for d in lobes[j]), default=-1)
                    if fi > 0.3 and fj > 0.3:
                        pairs.append((dd, i, j, slab.pos[i] + v / 2))
        pairs.sort(key=lambda t: t[0])
        seen_mid = []
        for dd, i, j, mid in pairs:
            if used[i] >= cap or used[j] >= cap:
                continue
            if any(np.linalg.norm(mid - m) < 0.5 for m in seen_mid):
                continue
            half = dd / 2
            if L <= half:                        # stretched-ideal fallback:
                Leff = float(np.hypot(half, 0.50))   # keeps O clear of trench
            else:
                Leff = L
            h = np.sqrt(Leff * Leff - half * half)
            o = mid + np.array([0, 0, h if face else -h])
            newp.append(o)
            seen_mid.append(mid)
            used[i] += 1
            used[j] += 1
    if newp:
        slab.pos = np.vstack([slab.pos, np.array(newp)])
        slab.el += ["O"] * len(newp)
        slab.tags += ["adsorbate"] * len(newp)
    return slab


def add_ketone_oxygen(slab: Slab, top=True, bottom=False):
    """On-top C=O on every surface C (1x1 ketone). Placed along +/-z."""
    L = BOND[("C", "O_ketone")]
    faces = ([True] if top else []) + ([False] if bottom else [])
    newp = []
    for face in faces:
        for i in surface_atoms(slab, top=face):
            newp.append(slab.pos[i] + np.array([0, 0, L if face else -L]))
    slab.pos = np.vstack([slab.pos, np.array(newp)])
    slab.el += ["O"] * len(newp)
    slab.tags += ["adsorbate"] * len(newp)
    return slab


# ------------------------------------------------------------ supercell etc
def repeat_inplane(slab: Slab, n1: int, n2: int) -> Slab:
    pos, el, tg = [], [], []
    for i in range(n1):
        for j in range(n2):
            t = i * slab.A1 + j * slab.A2
            pos.append(slab.pos + t)
            el += slab.el
            tg += slab.tags
    return Slab(n1 * slab.A1, n2 * slab.A2, np.vstack(pos), el,
                slab.a0, slab.orientation, tg)


def detect_sdb_shift_111(a0: float) -> int:
    """Return the layer shift (0 or 1) so that the TOP face of a (111) slab
    is the single-dangling-bond face."""
    for shift in (0, 1):
        s = bulk_slab("111", 8, a0, z_shift_layers=shift)
        bonds, vecs = neighbors(s)
        top = surface_atoms(s, top=True)
        miss = [4 - len(bonds[i]) for i in top]
        if all(m == 1 for m in miss):
            return shift
    raise RuntimeError("could not find SDB face")


# ----------------------------------------------------------- TDB (111) face
def detect_tdb_shift_111(a0: float, n_layers: int = 9):
    """Return (shift, ok_parity) so the TOP face of a (111) slab is the
    triple-dangling-bond (glide-plane) face: top atoms keep only their
    vertical backbond (3 missing bonds). Bottom face should be SDB
    (1 missing). Returns shift or raises with a parity hint."""
    for shift in (0, 1):
        s = bulk_slab("111", n_layers, a0, z_shift_layers=shift)
        bonds, _ = neighbors(s)
        top = surface_atoms(s, top=True)
        bot = surface_atoms(s, top=False)
        if (all(4 - len(bonds[i]) == 3 for i in top)
                and all(4 - len(bonds[i]) == 1 for i in bot)):
            return shift
    raise ValueError(f"(111) TDB-top/SDB-bottom not achievable with "
                     f"n_layers={n_layers}; try changing layer parity")


def seiwatz_slab(variant: str, n_layers: int, a0: float = A0_DEFAULT,
                 bottom: str = "bare"):
    """Build a TDB-face (111) 2x1 slab with the Seiwatz family on top.

    variant:
      'bare'   - Seiwatz single zig-zag chains (Kern-Hafner II: bond 1.45 A,
                 buckling 0.07 A, angles 119.4 deg)
      'H'      - monohydrogenated Seiwatz chains (KH-II: chain 1.525 A,
                 C-H 1.106 A, buckling removed)
      'ketone' - Sei-1ML-ketone (Yang, Gao & Li, DRM 159, 112823 (2025)):
                 chains broken into C2 pairs, each C sp2 with on-top C=O
                 (1.20 A); the alpha-diketone-like O=C-C=O motif of Fig. 6c.
    Asymmetric slabs only; bottom face is SDB (bare or H)."""
    shift = detect_tdb_shift_111(a0, n_layers)
    s = bulk_slab("111", n_layers, a0, z_shift_layers=shift)
    s = repeat_inplane(s, 1, 2)            # 2x1: chain along A1
    bonds, _ = neighbors(s)
    top = [i for i in surface_atoms(s, top=True)]
    if len(top) != 2:
        raise RuntimeError("expected 2 TDB top atoms per 2x1 cell")
    a, b = top
    # minimal-image vector a->b
    imgs = _images(s)
    best = None
    for im in imgs:
        v = s.pos[b] + im - s.pos[a]
        if best is None or np.linalg.norm(v[:2]) < np.linalg.norm(best[:2]):
            best = v
    v = best
    ey = np.array([0.0, 1.0, 0.0]) if abs(v[1]) > abs(v[0]) else \
        np.array([1.0, 0.0, 0.0])
    # transverse separation now and target
    t_now = abs(np.dot(v, ey))
    dx = np.linalg.norm(v[:2] - np.dot(v[:2], ey[:2]) * ey[:2])
    chain_bond = {"bare": 1.45, "H": 1.525, "ketone": 1.52}[variant]
    half_period = None
    if variant in ("bare", "H"):
        # zig-zag chain: bond connects atoms half a chain-period apart
        half_period = np.linalg.norm(s.A1) / 2.0
        t_target = np.sqrt(max(chain_bond**2 - half_period**2, 0.04))
    else:
        # isolated dimer (Sei-1ML-ketone): the pair must dimerize ALONG the
        # chain direction as well, otherwise the half-period offset makes
        # the bond repeat on both sides (pentavalent C). Choose transverse
        # separation t and near-side x-offset so the short bond is
        # chain_bond and the far-side contact exceeds the bonding cutoff.
        t_target = 1.30
        dx_near = np.sqrt(chain_bond**2 - t_target**2)
    move = (t_now - t_target) / 2.0
    sgn = np.sign(np.dot(v, ey))
    s.pos[a] += sgn * move * ey
    s.pos[b] -= sgn * move * ey
    if variant == "ketone":
        ex = np.array([1.0, 0.0, 0.0]) if ey[1] == 1.0 else \
            np.array([0.0, 1.0, 0.0])
        dgn = np.sign(np.dot(v, ex))
        dmove = (dx - dx_near) / 2.0
        s.pos[a] += dgn * dmove * ex
        s.pos[b] -= dgn * dmove * ex
    if variant == "bare":                  # chain buckling 0.07 A
        s.pos[a][2] += 0.035
        s.pos[b][2] -= 0.035
    s.wrap()
    if variant == "H":
        add_monovalent(s, "H", BOND[("C", "H")], top=True, bottom=False)
    if variant == "ketone":
        bonds, vecs = neighbors(s)
        for i in (a, b):
            units = [u / np.linalg.norm(u) for u in vecs[i]]
            lobe = -np.sum(units, axis=0)
            lobe /= np.linalg.norm(lobe)
            s.pos = np.vstack([s.pos, s.pos[i] + 1.20 * lobe])
            s.el.append("O")
            s.tags.append("adsorbate")
    if bottom == "H":
        add_monovalent(s, "H", BOND[("C", "H")], top=False, bottom=True)
    return s


# ------------------------------------------------- inversion symmetrization
def _z_layer_centers(slab: Slab, species="C", tol=0.4):
    """Mean z of each atomic layer of `species`, bottom to top."""
    zs = sorted(float(z) for z, e in zip(slab.pos[:, 2], slab.el)
                if e == species)
    layers = [[zs[0]]]
    for z in zs[1:]:
        (layers.append([z]) if z - layers[-1][-1] > tol
         else layers[-1].append(z))
    return [float(np.mean(L)) for L in layers]


def invert_symmetrize(slab: Slab, n_layers_target: int,
                      recon_depth: float, guard: float = 1.0) -> Slab:
    """Return an inversion-symmetric slab whose both faces carry the input
    slab's TOP-face structure (reconstruction and/or adsorbates).

    Every C-C bond midpoint in the ideal-constructed bulk region is an exact
    inversion center of the diamond lattice, so reflecting the kept upper
    part through such a midpoint reproduces the identical surface on the
    bottom: equal surface stress on both faces and zero net dipole by
    symmetry.  The input's own bottom face is discarded.

    Parameters
    ----------
    n_layers_target : desired total number of C layers (the center bond is
        chosen to match it; vertical bonds give even totals, in-plane bonds
        odd totals on (111)).
    recon_depth : thickness (Angstrom) of the top region that may deviate
        from ideal bulk positions; center candidates are kept below it.
    """
    pos, el = slab.pos, slab.el
    zmax_c = max(z for z, e in zip(pos[:, 2], el) if e == "C")
    zmin_c = min(z for z, e in zip(pos[:, 2], el) if e == "C")
    ceiling = zmax_c - recon_depth
    floor = zmin_c + guard
    if ceiling - floor < 0.5:
        raise ValueError("invert_symmetrize: source slab too thin for the "
                         "requested reconstruction depth; add bulk layers")

    # --- candidate inversion centers: bulk C-C bond midpoints -------------
    imgs = _images(slab)
    rcut = 1.15 * np.sqrt(3) / 4 * slab.a0
    cands = []
    for i in range(slab.n):
        if el[i] != "C" or not (floor <= pos[i, 2] <= ceiling):
            continue
        d = pos[None, :, :] + imgs[:, None, :] - pos[i]
        dist = np.linalg.norm(d, axis=2)
        dist[:, i][np.abs(dist[:, i]) < 1e-6] = 1e9
        for (im, j) in np.argwhere(dist < rcut):
            if el[j] != "C":
                continue
            c = pos[i] + 0.5 * d[im, j]
            if floor <= c[2] <= ceiling:
                cands.append(c)
    if not cands:
        raise ValueError("invert_symmetrize: no bulk bond-midpoint centers "
                         "found in the allowed window")

    layer_z = _z_layer_centers(slab)

    def predicted_total(cz):
        above = sum(1 for L in layer_z if L > cz + 0.05)
        equatorial = any(abs(L - cz) <= 0.05 for L in layer_z)
        return 2 * above + (1 if equatorial else 0)

    best = min(cands, key=lambda c: (abs(predicted_total(c[2])
                                         - n_layers_target),
                                     abs(c[2] - 0.5 * (zmax_c + zmin_c))))
    achieved = predicted_total(best[2])
    if achieved != n_layers_target:
        opts = sorted({predicted_total(c[2]) for c in cands})
        raise ValueError(f"invert_symmetrize: cannot realize "
                         f"{n_layers_target} layers with an exact inversion "
                         f"center; achievable counts here: {opts}")

    # --- build: keep upper part, add its inversion image ------------------
    keep = [k for k in range(slab.n) if pos[k, 2] >= best[2] - 1e-4]
    new_pos = [pos[k].copy() for k in keep]
    new_el = [el[k] for k in keep]
    new_tags = [slab.tags[k] if k < len(slab.tags) else "" for k in keep]
    for k in keep:
        new_pos.append(2.0 * best - pos[k])
        new_el.append(el[k])
        new_tags.append(slab.tags[k] if k < len(slab.tags) else "")

    out = Slab(slab.A1.copy(), slab.A2.copy(), np.array(new_pos), new_el,
               slab.a0, slab.orientation, new_tags)

    # --- dedupe equatorial duplicates (periodic min-image) ----------------
    imgs2 = _images(out)
    drop = set()
    for i in range(out.n):
        if i in drop:
            continue
        d = out.pos[None, i + 1:, :] + imgs2[:, None, :] - out.pos[i]
        dist = np.linalg.norm(d, axis=2)
        for (im, j) in np.argwhere(dist < 0.10):
            drop.add(i + 1 + j)
    if drop:
        keep2 = [i for i in range(out.n) if i not in drop]
        out = Slab(out.A1, out.A2, out.pos[keep2],
                   [out.el[i] for i in keep2], out.a0, out.orientation,
                   [out.tags[i] for i in keep2])

    # --- verify exact inversion symmetry about the constructed center -----
    ctr = best
    imgs3 = _images(out)
    for i in range(out.n):
        target = 2.0 * ctr - out.pos[i]
        d = out.pos[None, :, :] + imgs3[:, None, :] - target
        dist = np.linalg.norm(d, axis=2)
        hits = np.argwhere(dist < 0.02)
        if not any(out.el[j] == out.el[i] for (_, j) in hits):
            raise RuntimeError(f"invert_symmetrize: verification failed at "
                               f"atom {i} ({out.el[i]}); no inversion "
                               f"partner found")
    # pairwise sanity: no unphysical close contacts survived the merge
    for i in range(out.n):
        d = out.pos[None, :, :] + imgs3[:, None, :] - out.pos[i]
        dist = np.linalg.norm(d, axis=2)
        dist[:, i][np.abs(dist[:, i]) < 1e-6] = 1e9
        if dist.min() < 0.8:
            raise RuntimeError("invert_symmetrize: unphysical close contact "
                               "after merge; check recon_depth")
    out.wrap()
    return out
