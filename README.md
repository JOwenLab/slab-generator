# slab-generator

**Diamond surface slab construction for periodic DFT, built for NV-center surface science.**

`slab-generator` builds symmetric, dipole-free diamond slab models across the (111), (110), and (100) orientations with a curated library of surface terminations and reconstructions, and emits ready-to-run Quantum ESPRESSO inputs. It was developed in the Owen Group (Columbia University, Department of Chemistry) to support a systematic first-principles screen of surface terminations for nanodiamond-hosted nitrogen-vacancy (NV) quantum sensors — identifying surface chemistries that stabilize the NV⁻ charge state while minimizing surface stress and strain.

Development is active; interfaces may change ahead of the accompanying publication.

> **Working on this repo?** Read [`CLAUDE.md`](CLAUDE.md) first. It carries the
> project's invariants, sign conventions, and epistemic-level rules, and it is
> the authority where this README and it disagree.

## What it does

- **Slab construction** — symmetric slabs (matching top/bottom terminations) at user-specified thickness, with the DFT bulk lattice constant as the strain reference to avoid Pulay-stress artifacts.
- **Termination library** — 19 validated surface motifs across (111), (110), and (100): hydrogen, fluorine, oxygen (ether and ketone), hydroxyl, and clean reconstructed surfaces, including reconstructions transcribed from published coordinates. Two additional literature motifs are included as placeholder stubs pending author-provided coordinates.
- **Quantum ESPRESSO input generation** — `pw.x` inputs with sensible defaults for surface work (`tstress=.true.`, Methfessel-Paxton smearing, thickness-scaled k-grids), including constrained in-plane variable-cell relaxation (`calculation='vc-relax'`, `cell_dofree='2Dxy'`) for surface-stress extraction.
- **Structural pre-flight gate** — `preflight.py` refuses to let a structure reach the queue if it contradicts its own name or the project's invariants (see below).
- **Validation** — the library build pipeline runs 16 structural and chemical validation checks (stoichiometry, symmetry, bond-length sanity, termination coverage) on every motif.

## Installation

```bash
git clone https://github.com/JOwenLab/slab-generator.git
cd slab-generator
pip install -r requirements.txt
```

Verified against **Python 3.9**. Runtime dependencies are `numpy` and, for the
figure scripts only, `matplotlib`; `pytest` runs the suite. The analysis and
fitting modules are deliberately pure-Python standard library beyond numpy so
they run unchanged on the cluster login node.

**Pseudopotentials are not distributed** with this repository. Inputs are
written for SSSP-style UPF files (PBE); download them from the
[SSSP library](https://www.materialscloud.org/discover/sssp). The production
runs use `C.pbe-n-kjpaw_psl.1.0.0.UPF` and `H_ONCV_PBE-1.0.oncvpsp.upf`;
archived runs additionally reference `H.pbe-kjpaw_psl.1.0.0.UPF`,
`C_PBE_TM_2pj.UPF`, and `H_HSCV_PBE-1.0.UPF`.

## Reproducing every derived artifact

Every CSV, JSON, and report under `results/` is regenerated from committed
inputs by one script:

```bash
./regenerate.sh            # rebuild everything
./regenerate.sh --check    # rebuild, then fail if any tracked file changed
```

This exists because several artifacts previously required flags recorded
nowhere — the NV coupling constants, the μ_H scan range, `--write-config`.
`regenerate.sh` is the record. It runs no DFT: every `pw.out` under `results/`
is an input to it.

## Workflow for a new slab

```bash
# 1. generate. --a0 defaults to the fitted value in config/reference_pbe_sssp.json;
#    pass it explicitly only to override.
python3 slabgen.py <motif> --layers 6 --format qe --out runs/<name>/pw.in

# 2. GATE IT before it reaches the queue (CLAUDE.md section 7)
python3 preflight.py runs/<name> || { echo "REFUSING TO SUBMIT"; exit 1; }

# 3. run, parse, analyse
python3 run_queue.py --dry-run --only <name> --pseudo-source <pseudo_source>
python3 run_queue.py --run     --only <name> --pseudo-source <pseudo_source>
python3 parse_slab.py

# 4. stress SCF on the relaxed geometry
python3 make_slab_stress_scf.py
python3 run_queue.py --run --only <name>_stress_scf --pseudo-source runs/<name>
python3 update_slab_analysis.py

# 5. strain series
python3 make_slab_strain_series.py --only <name>_stress_scf
python3 run_queue.py --run --only <name>_strain_biaxial --pseudo-source runs/<name> --max-jobs 5
python3 update_slab_analysis.py
```

Rebuild and validate the full motif library:

```bash
python3 build_library.py     # see CLAUDE.md section 6 before running this
```

## PBE/SSSP bulk diamond reference

The project uses a 5-point hydrostatic QE series as the canonical bulk diamond
reference (PBE, SSSP 1.3.0 Precision, 8-atom conventional cell). The production
series is at 90/720 Ry with an 8×8×8 k-mesh; an earlier 80/640 Ry series is
retained for comparison.

**Authoritative project config:** `config/reference_pbe_sssp.json`

**Do not quote a0 or B here or anywhere else in prose — read them from that
config.** This section previously stated `a₀ ≈ 3.5736 Å, B ≈ 445 GPa` and told
readers to use those values; both were superseded when the fit was corrected
from a biased linear P(ε) to a 3rd-order Birch-Murnaghan on E(V), and this text
was not updated. The config carries the current values under `bulk_reference`
and the retired ones, with the reason they moved, under
`bulk_reference.superseded`.

- Raw parsed data: `results/reference_90_720/reference_summary.{csv,json,md}`
- Fitted parameters: `results/reference_90_720/bulk_fit_summary.{csv,json}` and `bulk_fit_report.md`

For slab surface-energy and stress calculations, subtract strain-matched bulk
references (same lateral strain as the slab) rather than the unstrained
equilibrium reference alone.

## Surface stress: sigma (diagnostic) vs tau (primary)

The slab stress/strain pipeline is:

    QE stress (sigma, kbar)
          ↓
    vacuum correction (analyze_slab_stress.py)
          ↓
    surface stress (tau, N/m)
          ↓
    thickness extrapolation (fit_tau_infinity.py) -> tau_inf
          ↓
    particle model (particle_strain.py) / exact NV physics (nv_spin_strain.py)

Quantum ESPRESSO reports `sigma_ij` averaged over the **entire periodic
supercell** (slab + vacuum), so its magnitude depends on the arbitrary vacuum
thickness chosen when building the cell. `analyze_slab_stress.py` converts this
into the vacuum-independent 2D surface stress

    tau_ij = sigma_ij * Lz * 0.005   (Lz = cell height, Å; result in N/m)

(0.005 = 0.1 GPa/kbar × 0.1 N/m per GPa·Å ÷ 2 surfaces). `tau` is the physically
intrinsic surface quantity and is directly comparable across cells with
different vacuum padding, or against surface-stress literature values — `sigma`
is not.

**Sign convention (CLAUDE.md section 2).** Positive `sigma` means the cell is
COMPRESSED. `tau` inherits that sign, so **positive `tau` is compressive**, and
equals *minus* the continuum surface stress `f` (for which positive is tensile).
Verified against free 2D `vc-relax`: positive `tau` expanded the released cell
on 6 of 6 axes. Consumers wanting `f` must negate — `particle_strain.load_taus`
does this once, at the boundary.

`fit_slab_strain.py` fits both `tau_mean(eps)` (primary) and `mean_sigma(eps)`
(diagnostic) independently and checks that their zero-stress strains agree.
Because `tau = sigma * Lz * 0.005` pointwise with `Lz` fixed within a strain
series, this is a consistency check on the arithmetic, not a new physical
prediction: **switching to tau changes units and interpretation, not the
equilibrium strain.**

## Elastic reference and the exact NV pipeline

`config/reference_pbe_sssp.json` also carries an `elastic_tensor` block (C11,
C12, C44, cubic symmetry, GPa) alongside `bulk_reference`. These two blocks have
different provenance and must not be confused:

- **`bulk_reference`** (a0, B): fitted from this project's hydrostatic PBE/SSSP
  series. A hydrostatic fit determines only `B = (C11 + 2*C12)/3`; it cannot
  separate C11, C12, and C44.
- **`elastic_tensor`** (C11, C12, C44): currently **literature-sourced**, not
  derived from this project's DFT. `elastic_tensor.source_type` and
  `elastic_tensor.citation` record this explicitly.

`elastic_reference.py` is the single validated loader for both blocks. It checks
required fields, units, positivity, and cubic mechanical stability, and reports
the tensor-implied bulk modulus against the project's fitted B.

`nv_spin_strain.py` (the exact NV spin-strain Hamiltonian; `nv_strain_model.py`
is a screening tool only) reads C11/C12/C44 through that loader — including its
`ElasticConstants` dataclass defaults, so no second copy of those constants
exists in the codebase. Every run writes `nv_predictions_meta.json` and an
"Elastic reference" section recording which source was used.

A future DFT elastic-tensor campaign would populate `elastic_tensor` with
`source_type: "project_dft_fit"` through the same config and loader;
`nv_spin_strain.py` would not need to change.

## Physics notes

- **Symmetric slabs** (identical top and bottom terminations) are required for clean surface-stress measurements; asymmetric slabs introduce dipole artifacts and complicate interpretation of the stress tensor. `preflight.py` enforces this: an asymmetric structure without a dipole correction fails the gate.
- **Surface stress** is extracted from the in-plane stress of a 2D-relaxed slab as *f*<sub>αβ</sub> = ½ · *L*<sub>z</sub> · σ<sub>αβ</sub> (1 kbar·Å ≈ 0.01 N/m), referenced to the DFT equilibrium lattice constant at the same cutoffs.
- **Every result carries an epistemic level** (L0 exploratory → L3 publication) per CLAUDE.md section 4. Do not promote a result a level without the corresponding work existing.

## Development

This package was developed iteratively with **Claude (Anthropic)** as an active
co-developer: implementation briefs are handed to Claude Code for well-scoped
features, and the accompanying DFT campaign pipeline uses the Anthropic API for
automated triage of failed HPC jobs. The repository history reflects that
workflow.

Run the test suite with `python3 -m pytest tests/ -q`.

## Citation

A publication describing the termination library and the surface-stress
screening campaign is in preparation. Until then, please cite this repository:

> Owen Group, Columbia University. *slab-generator: diamond surface slab construction for periodic DFT.* https://github.com/JOwenLab/slab-generator (2026).

## License

MIT — see [LICENSE](LICENSE).

## Contact

Jonathan S. Owen — jso2115@columbia.edu
Department of Chemistry, Columbia University
