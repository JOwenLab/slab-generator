#!/usr/bin/env python3
"""Quantitative NV spin-strain predictions for terminated diamond slabs.

Implements the complete symmetry-allowed ground-state spin-strain Hamiltonian
of Udvarhelyi, Shkolnikov, Gali, Burkard & Palyi, Phys. Rev. B 98, 075201
(2018) [arXiv:1712.02684], Eq. (3), with their DFT coupling parameters
(Table I).  For a slab in mechanical equilibrium the in-plane strain is taken
from the strain-series fits (zero_stress_epsilon in
results/slabs/slab_strain_fit_summary.csv); the out-of-plane response
(eps_zz, and shear where symmetry allows) is obtained by solving sigma_i3 = 0
with the anisotropic cubic stiffness tensor rotated into the slab frame.

The strain tensor is then rotated into the local frame of each of the four
NV orientations and the full 3x3 spin Hamiltonian is diagonalized exactly.
Outputs per orientation: Delta D (axial ZFS shift), E_eff (transverse ZFS),
and the zero-field transition frequencies f+- = (D0 + Delta D) +- E_eff.

Sign convention (paper Appendix B): NEGATIVE strain = compression.

Physics notes
-------------
* The paper's DFT axial couplings are ~1.7x smaller than the experimental
  values of Barson et al., Nano Lett. 17, 1496 (2017) (a1: -2.66 vs
  -4.4 MHz/GPa).  The parameter set is swappable; DFT is the default and the
  Barson-scaled set is provided for sensitivity checks.
* h25/h26 (the H_eps1 terms) mix ms=0 with ms=+-1 and contribute only at
  second order ~ (h eps)^2 / D0 at zero field; they are included exactly via
  numerical diagonalization.

Slab frames follow geometry._frame() exactly:
  (100): z=[001], x=[110]/sqrt2, y=[-110]/sqrt2
  (110): z=[110]/sqrt2, x=[1-10]/sqrt2, y=[00-1]
  (111): z=[111]/sqrt3, x=[1-10]/sqrt2, y = z ^ x
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

import elastic_reference

# ----------------------------------------------------------------- constants
D0_MHZ = 2870.0  # zero-field splitting of the unperturbed NV, MHz

# Spin-1 operators (units of hbar), basis |+1>, |0>, |-1>
_SQ2 = 1.0 / math.sqrt(2.0)
SX = np.array([[0, _SQ2, 0], [_SQ2, 0, _SQ2], [0, _SQ2, 0]], dtype=complex)
SY = np.array([[0, -1j * _SQ2, 0], [1j * _SQ2, 0, -1j * _SQ2],
               [0, 1j * _SQ2, 0]], dtype=complex)
SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)


def _anti(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b + b @ a


@dataclass(frozen=True)
class SpinStrainParams:
    """Spin-strain couplings in MHz per unit strain (NV frame)."""
    name: str
    h41: float
    h43: float
    h25: float
    h26: float
    h15: float
    h16: float
    reference: str = ""


# Udvarhelyi et al., PRB 98, 075201 (2018), Table I (DFT, PBE, 512-atom cell)
UDVARHELYI_DFT = SpinStrainParams(
    name="udvarhelyi2018_dft",
    h41=-6420.0, h43=2300.0, h25=-2600.0, h26=-2830.0,
    h15=5700.0, h16=19660.0,
    reference="Udvarhelyi et al., PRB 98, 075201 (2018), Table I",
)

# Same tensor rescaled on the axial channel to match Barson et al. (2017)
# experimental a1 = -4.4 MHz/GPa (vs DFT -2.66): factor 4.4/2.66 = 1.654 on
# h41, h43.  Transverse channel scaled by b_exp/b_dft = 2.3/1.94 = 1.186.
# Crude, but brackets the plausible range for sensitivity analysis.
BARSON_SCALED = SpinStrainParams(
    name="barson2017_scaled",
    h41=-6420.0 * 1.654, h43=2300.0 * 1.654,
    h25=-2600.0 * 1.186, h26=-2830.0 * 1.186,
    h15=5700.0 * 1.186, h16=19660.0 * 1.186,
    reference="Udvarhelyi tensor rescaled to Barson et al., "
              "Nano Lett. 17, 1496 (2017) axial/transverse magnitudes",
)

PARAM_SETS = {p.name: p for p in (UDVARHELYI_DFT, BARSON_SCALED)}
PARAM_SETS["dft"] = UDVARHELYI_DFT
PARAM_SETS["barson"] = BARSON_SCALED


DEFAULT_REFERENCE_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "reference_pbe_sssp.json")

_ELASTIC_CACHE: dict = {}


def config_elastic_constants(config_path: str = DEFAULT_REFERENCE_CONFIG):
    """
    C11/C12/C44 as recorded in the reference config, cached per path.

    This is the ONLY place the cubic stiffness enters this module. It used to
    be duplicated as literals in `ElasticConstants`'s field defaults, which is
    the same failure shape as the a0 bug: the config is the thing that gets
    updated (the planned DFT elastic-tensor campaign will replace these
    literature values), and a stale second copy would keep feeding the old
    numbers into every NV shift via particle_strain, silently and without
    changing any code.

    A missing or malformed config is a hard error rather than a fallback to
    literals, because a silent fallback is precisely the failure being removed.
    """
    key = os.path.abspath(config_path)
    if key not in _ELASTIC_CACHE:
        ref = elastic_reference.load_elastic_reference(config_path)
        _ELASTIC_CACHE[key] = (ref.tensor.C11, ref.tensor.C12, ref.tensor.C44)
    return _ELASTIC_CACHE[key]


def _default_C11() -> float:
    return config_elastic_constants()[0]


def _default_C12() -> float:
    return config_elastic_constants()[1]


def _default_C44() -> float:
    return config_elastic_constants()[2]


@dataclass(frozen=True)
class ElasticConstants:
    """
    Cubic stiffness of diamond, GPa.

    Defaults are READ FROM config/reference_pbe_sssp.json, which is the single
    source of truth (`elastic_tensor.source_type: literature` — these are not
    project-derived DFT values; see that file's notes). They are not written
    here. tests/test_elastic_constants_single_source.py fails if a copy ever
    reappears in this file.
    """
    C11: float = field(default_factory=_default_C11)
    C12: float = field(default_factory=_default_C12)
    C44: float = field(default_factory=_default_C44)

    def tensor(self) -> np.ndarray:
        """Full C_ijkl in the cubic frame."""
        C = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                C[i, i, j, j] = self.C12
            C[i, i, i, i] = self.C11
        for i in range(3):
            for j in range(3):
                if i != j:
                    C[i, j, i, j] = self.C44
                    C[i, j, j, i] = self.C44
        return C


# ------------------------------------------------------------------- frames
def nv_frames() -> list[tuple[str, np.ndarray]]:
    """The four NV orientation frames (rows = e_x, e_y, e_z in cubic coords).

    Base frame per the paper: e_z=(1,1,1)/sqrt3, e_y=(1,-1,0)/sqrt2,
    e_x = e_y ^ e_z.  The other three are generated by the C2 rotations about
    the cubic axes, which are symmetries of the diamond lattice and map the
    NV sublattice orientations onto each other while preserving the internal
    C3v frame convention (e_x in a reflection plane).
    """
    ez = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    ey = np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    ex = np.cross(ey, ez)
    base = np.array([ex, ey, ez])
    ops = {
        "[111]": np.eye(3),
        "[1-1-1]": np.diag([1.0, -1.0, -1.0]),
        "[-11-1]": np.diag([-1.0, 1.0, -1.0]),
        "[-1-11]": np.diag([-1.0, -1.0, 1.0]),
    }
    return [(label, base @ op.T) for label, op in ops.items()]


def slab_frame(face: str) -> np.ndarray:
    """Slab frame (rows = x, y, z in cubic coords); mirrors geometry._frame."""
    if face in ("100", "(100)"):
        n = np.array([0.0, 0.0, 1.0])
        a1 = np.array([0.5, 0.5, 0.0])
    elif face in ("110", "(110)"):
        n = np.array([1.0, 1.0, 0.0])
        a1 = np.array([0.5, -0.5, 0.0])
    elif face in ("111", "(111)"):
        n = np.array([1.0, 1.0, 1.0])
        a1 = np.array([0.5, -0.5, 0.0])
    else:
        raise ValueError(f"unknown face {face!r}")
    z = n / np.linalg.norm(n)
    x = a1 / np.linalg.norm(a1)
    y = np.cross(z, x)
    return np.array([x, y, z])


# -------------------------------------------------------------- elasticity
def slab_equilibrium_strain(face: str, exx: float, eyy: float,
                            exy: float = 0.0,
                            elastic: ElasticConstants = ElasticConstants()
                            ) -> np.ndarray:
    """Full 3x3 strain tensor in the CUBIC frame for a slab with imposed
    in-plane strain (slab frame) and traction-free surfaces (sigma_i3 = 0).

    Solves the 3x3 linear system for (eps_xz, eps_yz, eps_zz) using the
    anisotropic cubic stiffness rotated into the slab frame — valid for any
    face, unlike the isotropic-Poisson shortcut.
    """
    R = slab_frame(face)
    C = elastic.tensor()
    # rotate stiffness into the slab frame: C'_mnop = R_mi R_nj R_ok R_pl C
    Cp = np.einsum("mi,nj,ok,pl,ijkl->mnop", R, R, R, R, C)

    eps_known = np.array([[exx, exy, 0.0], [exy, eyy, 0.0], [0.0, 0.0, 0.0]])
    # sigma_i3 = Cp[i,2,k,l] eps_kl ; unknowns u = (e13, e23, e33)
    # coefficient of e13 in sigma_i3: Cp[i,2,0,2]+Cp[i,2,2,0]  (symmetric)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(3):
        A[i, 0] = Cp[i, 2, 0, 2] + Cp[i, 2, 2, 0]
        A[i, 1] = Cp[i, 2, 1, 2] + Cp[i, 2, 2, 1]
        A[i, 2] = Cp[i, 2, 2, 2]
        b[i] = -np.einsum("kl,kl->", Cp[i, 2], eps_known)
    e13, e23, e33 = np.linalg.solve(A, b)
    eps_slab = eps_known.copy()
    eps_slab[0, 2] = eps_slab[2, 0] = e13
    eps_slab[1, 2] = eps_slab[2, 1] = e23
    eps_slab[2, 2] = e33
    # rotate strain back to the cubic frame: eps_cubic = R^T eps_slab R
    return R.T @ eps_slab @ R


# ------------------------------------------------------------- hamiltonian
def spin_strain_hamiltonian(eps_nv: np.ndarray,
                            p: SpinStrainParams) -> np.ndarray:
    """H_eps / h in MHz, Eq. (3) of PRB 98, 075201, strain in the NV frame."""
    exx, eyy, ezz = eps_nv[0, 0], eps_nv[1, 1], eps_nv[2, 2]
    exy, exz, eyz = eps_nv[0, 1], eps_nv[0, 2], eps_nv[1, 2]
    H0 = (p.h41 * (exx + eyy) + p.h43 * ezz) * (SZ @ SZ)
    H1 = (0.5 * (p.h26 * exz - 0.5 * p.h25 * (exx - eyy)) * _anti(SX, SZ)
          + 0.5 * (p.h26 * eyz + p.h25 * exy) * _anti(SY, SZ))
    H2 = (0.5 * (p.h16 * exz - 0.5 * p.h15 * (exx - eyy))
          * (SY @ SY - SX @ SX)
          + 0.5 * (p.h16 * eyz + p.h15 * exy) * _anti(SX, SY))
    return H0 + H1 + H2


def nv_observables(eps_cubic: np.ndarray, frame: np.ndarray,
                   p: SpinStrainParams = UDVARHELYI_DFT) -> dict:
    """Exact zero-field observables for one NV orientation.

    Returns dict with delta_D_mhz, E_mhz, f_plus_mhz, f_minus_mhz.
    """
    eps_nv = frame @ eps_cubic @ frame.T
    H = D0_MHZ * (SZ @ SZ) + spin_strain_hamiltonian(eps_nv, p)
    evals, evecs = np.linalg.eigh(H)
    # identify the ms=0-like state by overlap with |0> = (0,1,0)
    w0 = np.abs(evecs[1, :]) ** 2
    i0 = int(np.argmax(w0))
    e0 = evals[i0]
    others = sorted(evals[j] for j in range(3) if j != i0)
    f_minus, f_plus = others[0] - e0, others[1] - e0
    return {
        "delta_D_mhz": 0.5 * (f_plus + f_minus) - D0_MHZ,
        "E_mhz": 0.5 * (f_plus - f_minus),
        "f_plus_mhz": f_plus,
        "f_minus_mhz": f_minus,
    }


def predict_for_slab(face: str, exx: float, eyy: float, exy: float = 0.0,
                     params: SpinStrainParams = UDVARHELYI_DFT,
                     elastic: ElasticConstants = ElasticConstants()
                     ) -> list[dict]:
    """Per-orientation predictions for a slab at mechanical equilibrium."""
    eps_cubic = slab_equilibrium_strain(face, exx, eyy, exy, elastic)
    out = []
    for label, frame in nv_frames():
        row = {"nv_axis": label,
               "eps_zz_slab": float((slab_frame(face) @ eps_cubic
                                     @ slab_frame(face).T)[2, 2])}
        row.update(nv_observables(eps_cubic, frame, params))
        out.append(row)
    return out


# --------------------------------------------------------------------- CLI
def resolve_elastic_context(args) -> tuple[ElasticConstants, dict]:
    """Resolve which C11/C12/C44 to use and build a provenance dict for
    output/reporting. Precedence: CLI override > reference config > explicit
    legacy fallback (never a silent default)."""
    override_vals = [args.C11, args.C12, args.C44]
    override_used = any(v is not None for v in override_vals)
    if override_used and not all(v is not None for v in override_vals):
        raise SystemExit("--C11, --C12, and --C44 must all be supplied together")

    ref = None
    ref_error = None
    try:
        ref = elastic_reference.load_elastic_reference(
            args.reference_config, warning_threshold_pct=args.warning_threshold_pct)
    except elastic_reference.ElasticReferenceError as exc:
        ref_error = exc

    if override_used:
        elastic_reference.validate_cubic_tensor(
            args.C11, args.C12, args.C44, "--C11/--C12/--C44")
        elastic = ElasticConstants(args.C11, args.C12, args.C44)
        b_tensor = (args.C11 + 2 * args.C12) / 3.0
        provenance = {
            "elastic_source_type": "cli_override",
            "elastic_citation": "user-supplied via --C11/--C12/--C44",
            "elastic_config_path": ref.config_path if ref else "",
            "cli_override_used": True,
            "a0_angstrom": ref.bulk.a0_angstrom if ref else None,
            "bulk_modulus_fit_gpa": ref.bulk.bulk_modulus_gpa if ref else None,
            "bulk_modulus_tensor_gpa": b_tensor,
            "bulk_modulus_discrepancy_gpa": None,
            "bulk_modulus_discrepancy_pct": None,
            "warning_threshold_pct": args.warning_threshold_pct,
            "consistent": None,
        }
        if ref:
            diff = b_tensor - ref.bulk.bulk_modulus_gpa
            provenance["bulk_modulus_discrepancy_gpa"] = diff
            provenance["bulk_modulus_discrepancy_pct"] = 100.0 * diff / ref.bulk.bulk_modulus_gpa
            provenance["consistent"] = abs(provenance["bulk_modulus_discrepancy_pct"]) <= args.warning_threshold_pct
        return elastic, provenance

    if args.elastic_source == "legacy":
        print("WARNING: --elastic-source legacy bypasses the validated reference "
              "config; using hardcoded literature elastic constants "
              "(C11=1076, C12=125, C44=576 GPa). This is not project-derived "
              "DFT data.", file=sys.stderr)
        elastic = ElasticConstants()
        provenance = {
            "elastic_source_type": "legacy_hardcoded",
            "elastic_citation": "hardcoded literature defaults (pre-config)",
            "elastic_config_path": "",
            "cli_override_used": False,
            "a0_angstrom": None,
            "bulk_modulus_fit_gpa": None,
            "bulk_modulus_tensor_gpa": (elastic.C11 + 2 * elastic.C12) / 3.0,
            "bulk_modulus_discrepancy_gpa": None,
            "bulk_modulus_discrepancy_pct": None,
            "warning_threshold_pct": args.warning_threshold_pct,
            "consistent": None,
        }
        return elastic, provenance

    if ref is None:
        raise SystemExit(
            f"Could not load elastic reference from {args.reference_config}: {ref_error}\n"
            "Pass --reference-config to point at a valid config, or pass "
            "--elastic-source legacy to explicitly use hardcoded literature constants.")

    elastic = ElasticConstants(ref.tensor.C11, ref.tensor.C12, ref.tensor.C44)
    provenance = {
        "elastic_source_type": ref.tensor.source_type,
        "elastic_citation": ref.tensor.citation,
        "elastic_config_path": ref.config_path,
        "cli_override_used": False,
        "a0_angstrom": ref.bulk.a0_angstrom,
        "bulk_modulus_fit_gpa": ref.bulk.bulk_modulus_gpa,
        "bulk_modulus_tensor_gpa": ref.b_tensor_gpa,
        "bulk_modulus_discrepancy_gpa": ref.b_discrepancy_gpa,
        "bulk_modulus_discrepancy_pct": ref.b_discrepancy_pct,
        "warning_threshold_pct": ref.warning_threshold_pct,
        "consistent": ref.consistent,
    }
    return elastic, provenance


def _format_elastic_reference_section(elastic: ElasticConstants, prov: dict) -> list[str]:
    lines = ["## Elastic reference", ""]

    if prov["a0_angstrom"] is not None:
        lines += [
            "**Lattice constant:**",
            f"  {prov['a0_angstrom']:.6f} Å",
            "  source: project PBE/SSSP bulk fit",
            "",
        ]

    if prov["bulk_modulus_fit_gpa"] is not None:
        lines += [
            "**Hydrostatic bulk modulus:**",
            f"  {prov['bulk_modulus_fit_gpa']:.1f} GPa",
            "  source: project PBE/SSSP bulk fit",
            "",
        ]

    lines += [
        "**Cubic stiffness tensor:**",
        f"  C11 = {elastic.C11:.1f} GPa",
        f"  C12 = {elastic.C12:.1f} GPa",
        f"  C44 = {elastic.C44:.1f} GPa",
        f"  source: {prov['elastic_source_type']}"
        + (f" ({prov['elastic_citation']})" if prov["elastic_citation"] else ""),
        "",
        "**Tensor-implied bulk modulus:**",
        f"  {prov['bulk_modulus_tensor_gpa']:.1f} GPa",
        "",
    ]

    if prov["bulk_modulus_discrepancy_gpa"] is not None:
        lines += [
            "**Difference from project hydrostatic fit:**",
            f"  {prov['bulk_modulus_discrepancy_gpa']:+.1f} GPa "
            f"({prov['bulk_modulus_discrepancy_pct']:+.1f}%)",
            "",
        ]

    mixed = prov["elastic_source_type"] != "project_dft_fit"
    status = ("mixed-source elastic reference; full DFT Cij not yet computed" if mixed
               else "fully project-derived elastic reference")
    lines += [
        "**Status:**",
        f"  {status}",
        f"  CLI override used: {prov['cli_override_used']}",
        f"  config: {prov['elastic_config_path'] or '(none — legacy fallback)'}",
        "",
        "See `nv_spin_strain.py` for the authoritative exact NV physics model; "
        "`nv_strain_model.py` is a screening tool only.",
        "",
    ]
    return lines


def _run_from_fits(csv_path: str, params: SpinStrainParams, out_dir: str,
                   elastic: ElasticConstants, elastic_provenance: dict) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as fh:
        for rec in csv.DictReader(fh):
            mode = rec["strain_mode"]
            eps0 = float(rec["zero_stress_epsilon"])
            if mode == "biaxial":
                exx = eyy = eps0
            elif mode == "x":
                exx, eyy = eps0, 0.0
            elif mode == "y":
                exx, eyy = 0.0, eps0
            else:
                continue
            face = rec["orientation"].strip("()")
            for pred in predict_for_slab(face, exx, eyy, params=params, elastic=elastic):
                rows.append({
                    "series": rec["series"], "face": rec["orientation"],
                    "termination": rec["termination"], "strain_mode": mode,
                    "eps_inplane": eps0, "fit_status": rec["fit_status"],
                    "param_set": params.name, **pred,
                    "elastic_source_type": elastic_provenance["elastic_source_type"],
                    "elastic_C11_gpa": elastic.C11,
                    "elastic_C12_gpa": elastic.C12,
                    "elastic_C44_gpa": elastic.C44,
                })
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "nv_predictions.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-fits", metavar="CSV",
                    default="results/slabs/slab_strain_fit_summary.csv")
    ap.add_argument("--params", choices=sorted(PARAM_SETS), default="dft")
    ap.add_argument("--out-dir", default="results/nv")
    ap.add_argument("--reference-config", default="config/reference_pbe_sssp.json",
                    help="Authoritative bulk/elastic reference config "
                         "(see elastic_reference.py).")
    ap.add_argument("--elastic-source", choices=["config", "legacy"], default="config",
                    help="'config' (default) loads C11/C12/C44 from --reference-config; "
                         "'legacy' explicitly bypasses it for hardcoded literature values.")
    ap.add_argument("--C11", type=float, default=None, metavar="GPA")
    ap.add_argument("--C12", type=float, default=None, metavar="GPA")
    ap.add_argument("--C44", type=float, default=None, metavar="GPA")
    ap.add_argument("--warning-threshold-pct", type=float,
                    default=elastic_reference.DEFAULT_WARNING_THRESHOLD_PCT,
                    help="Warn (not fail) if the tensor-implied bulk modulus "
                         "differs from the project hydrostatic fit by more "
                         "than this percent.")
    args = ap.parse_args()

    params = PARAM_SETS[args.params]
    elastic, provenance = resolve_elastic_context(args)

    if not os.path.exists(args.from_fits):
        raise SystemExit(
            f"No strain-fit summary at {args.from_fits}.\n"
            "The original 6L results were archived (see "
            "results/archive_asymmetric_6L/README.md); run the symmetric "
            "batch in batches/differential_6L/ and update_slab_analysis.py "
            "first, or point --from-fits at an existing summary.")
    rows = _run_from_fits(args.from_fits, params, args.out_dir, elastic, provenance)
    if not rows:
        raise SystemExit(f"{args.from_fits} contained no usable fit rows.")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, "nv_predictions_meta.json")
    with open(meta_path, "w") as fh:
        json.dump({
            "param_set": params.name,
            "param_set_reference": params.reference,
            **provenance,
        }, fh, indent=2)

    report_path = os.path.join(out_dir, "nv_predictions_report.md")
    with open(report_path, "w") as fh:
        fh.write("\n".join(
            ["# NV Spin-Strain Predictions", "",
             f"Spin-strain parameters: {params.name} ({params.reference})", ""]
            + _format_elastic_reference_section(elastic, provenance)
        ) + "\n")

    hdr = (f"{'series':26s} {'mode':8s} {'NV axis':9s} "
           f"{'dD (MHz)':>10s} {'E (MHz)':>9s} {'f+ (MHz)':>10s} "
           f"{'f- (MHz)':>10s}")
    print(f"# spin-strain parameters: {params.name} ({params.reference})")
    print(f"# elastic reference: {provenance['elastic_source_type']} "
          f"(C11={elastic.C11:.1f}, C12={elastic.C12:.1f}, C44={elastic.C44:.1f} GPa)")
    print(hdr)
    for r in rows:
        print(f"{r['series']:26s} {r['strain_mode']:8s} {r['nv_axis']:9s} "
              f"{r['delta_D_mhz']:10.2f} {r['E_mhz']:9.2f} "
              f"{r['f_plus_mhz']:10.1f} {r['f_minus_mhz']:10.1f}")
    print(f"\nwrote {os.path.join(args.out_dir, 'nv_predictions.csv')}")
    print(f"wrote {meta_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
