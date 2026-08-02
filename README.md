# slab-generator
scripts for generating diamond slabs

## PBE/SSSP bulk diamond reference

The project uses a 5-point hydrostatic QE series as the canonical bulk diamond reference
(PBE functional, SSSP 1.3.0 Precision pseudopotential set, 80 Ry / 640 Ry cutoffs,
8×8×8 k-mesh, 8-atom conventional cell).

**Raw parsed reference data:** `results/reference_diamond/reference_summary.{csv,json,md}`

**Fitted reference parameters:** `results/reference_diamond/bulk_fit_summary.{csv,json}` and
`results/reference_diamond/bulk_fit_report.md`

**Authoritative project config:** `config/reference_pbe_sssp.json`

The fitted equilibrium lattice constant is **a₀ ≈ 3.5736 Å** (from E–V fit) with a bulk
modulus of **B ≈ 445 GPa**. Use this value as the default PBE/SSSP diamond lattice constant
when generating slabs or setting up strain series.

For slab surface-energy and stress calculations, subtract strain-matched bulk references
(same lateral strain as the slab) rather than the unstrained equilibrium reference alone.

## Surface stress: sigma (diagnostic) vs tau (primary)

The slab stress/strain pipeline is:

    QE stress (sigma, kbar)
          ↓
    vacuum correction (analyze_slab_stress.py)
          ↓
    surface stress (tau, N/m)
          ↓
    strain fitting (fit_slab_strain.py)
          ↓
    NV screening (nv_strain_model.py) / exact NV physics (nv_spin_strain.py)

Quantum ESPRESSO reports `sigma_ij` averaged over the **entire periodic supercell**
(slab + vacuum), so its magnitude depends on the arbitrary vacuum thickness chosen when
building the cell. `analyze_slab_stress.py` converts this into the vacuum-independent 2D
surface stress

    tau_ij = sigma_ij * Lz * 0.005   (Lz = cell height, Å; result in N/m)

(0.005 = 0.1 GPa/kbar × 0.1 N/m per GPa·Å ÷ 2 surfaces). `tau` is the physically intrinsic
surface quantity and is directly comparable across cells with different vacuum padding, or
against surface-stress literature values — `sigma` is not.

`fit_slab_strain.py` fits **both** `tau_mean(eps)` (primary — `slab_strain_fit_summary.md`'s
"Surface stress fit" table, columns `tau_slope_n_per_m_per_eps`, `tau_intercept_n_per_m`,
`zero_tau_strain`, `r2_tau_fit`) and `mean_sigma(eps)` (diagnostic — "Diagnostic QE stress
fit" table, unchanged legacy columns `stress_slope_kbar_per_eps`, `zero_stress_epsilon`,
etc.) independently, and explicitly checks that their zero-stress strains agree
(`zero_strain_discrepancy`, warns above `--zero-strain-tolerance`, default 1e-10). Because
`tau = sigma * Lz * 0.005` pointwise with `Lz` fixed within a strain series, this is a
consistency check on the arithmetic/data, not a new physical prediction: **switching to tau
changes units and physical interpretation, not the equilibrium (zero-stress) strain.**
`nv_spin_strain.py` consumes that equilibrium strain directly and is unaffected by which of
the two (mathematically equivalent) fits produced it.

`nv_strain_model.py`'s screening risk scores remain sigma-based (see its report's Scoring
Methodology section for why: cell height differs by orientation in this dataset, so
switching the score basis to tau would rescale each orientation by a different factor,
which could reorder rankings rather than just changing units) — `tau_mean_n_per_m` is still
reported there for absolute, vacuum-independent comparison.

## Elastic reference and the exact NV pipeline

`config/reference_pbe_sssp.json` also carries an `elastic_tensor` block (C11, C12, C44,
cubic symmetry, GPa) alongside `bulk_reference`. These two blocks have different
provenance and must not be confused:

- **`bulk_reference`** (a0, B): fitted from this project's five-point hydrostatic PBE/SSSP
  series. A hydrostatic fit determines only `B = (C11 + 2*C12) / 3`; it cannot separate
  C11, C12, and C44.
- **`elastic_tensor`** (C11, C12, C44): currently **literature-sourced**, not derived from
  this project's DFT. `elastic_tensor.source_type` and `elastic_tensor.citation` record this
  explicitly.

`elastic_reference.py` is the single validated loader for both blocks. It checks required
fields/units/positivity, cubic mechanical stability (`C11 - C12 > 0`, `C11 + 2*C12 > 0`,
`C44 > 0`), and reports the tensor-implied bulk modulus against the project's fitted B
(warns, does not fail, above a configurable percent threshold).

`nv_spin_strain.py` (the exact NV spin-strain Hamiltonian; `nv_strain_model.py` is a
screening tool only) loads C11/C12/C44 through this interface by default
(`--reference-config config/reference_pbe_sssp.json`). Precedence: explicit
`--C11`/`--C12`/`--C44` CLI overrides > reference config > `--elastic-source legacy`
(hardcoded literature fallback, requires an explicit flag and emits a warning). Every run
writes `nv_predictions_meta.json` and an "Elastic reference" section in
`nv_predictions_report.md` recording which source was used, so no output can be mistaken
for a project-derived DFT tensor when it isn't one.

`fit_bulk_reference.py --update-config config/reference_pbe_sssp.json` writes the newly
fitted a0/B into that config (explicit flag; not run by default) without touching
`elastic_tensor`.

A future DFT elastic-tensor campaign (independent hydrostatic/orthorhombic/monoclinic
strain modes to separately determine C11, C12, C44 — see `PLAN_nv_strain_campaign.md`)
would populate `elastic_tensor` with `source_type: "project_dft_fit"` through the same
config and loader; `nv_spin_strain.py` would not need to change.

**General Workflow for the Current Version:

python3 slabgen.py <motif> --layers 6 --a0 3.573641 --format qe --out runs/<name>/pw.in

python3 run_queue.py --dry-run --only <name> --pseudo-source <pseudo_source>
python3 run_queue.py --run --only <name> --pseudo-source <pseudo_source>

python3 parse_slab.py

python3 make_slab_stress_scf.py
python3 run_queue.py --run --only <name>_stress_scf --pseudo-source runs/<name>

python3 update_slab_analysis.py

python3 make_slab_strain_series.py --only <name>_stress_scf
python3 run_queue.py --run --only <name>_strain_biaxial --pseudo-source runs/<name> --max-jobs 5

python3 update_slab_analysis.py
