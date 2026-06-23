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

General Workflow for the Current Version:
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
