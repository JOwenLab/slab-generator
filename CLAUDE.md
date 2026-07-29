# CLAUDE.md — slab-generator

Nanodiamond surface-stress / Raman / NV project. Quantum ESPRESSO (PBE) DFT on
functionalized diamond slabs, to explain an anomalous experimental Raman feature
and post-treatment NV/ODMR changes.

You are assisting a research pipeline where **wrong results look exactly like
right results**. Every number here is a float that will print without complaint.
The only defenses are conventions, assertions, and provenance. Treat them as
load-bearing.

---

## 0. The one thing to internalize

This project has already lost 45 completed DFT calculations to a silent geometry
error: slabs intended as symmetric H/H were generated with H on the top face and
a **bare bottom face**. Nothing crashed. SCF converged. BFGS converged. `JOB DONE`
was reached. The structures were simply not the ones the science claimed.

Consequences that were invisible in the output: dangling bonds on the bare face,
a net slab dipole with no dipole correction applied, and a stress tensor mixing
H-terminated and bare-surface contributions. The NV rankings derived from it were
uninterpretable as H-termination results.

**Therefore: never infer a structure's properties from its filename, its intent,
or the script that generated it. Read the structure.**

---

## 1. Non-negotiable invariants

Violating any of these invalidates results silently. Assert them, don't assume them.

1. **Symmetric slabs are symmetric.** For any `*_sym` structure: H (or other
   adsorbate) count on the bottom face equals the count on the top face, and
   inversion symmetry holds to numerical tolerance. Verify by coordinate, not by name.
2. **No undercoordinated surface carbon** on a nominally fully terminated face.
3. **Names describe geometry.** A folder named `C100_2x1_H_6L_sym` must contain a
   (100) slab, 2x1 reconstruction, H termination, 6 carbon layers, symmetric.
   If a script derives new names by string manipulation, it must preserve every
   token — `_sym` has been silently stripped before, regenerating stress inputs
   from archived asymmetric geometries.
4. **Asymmetric slabs require dipole correction.** If a structure is deliberately
   asymmetric, `assume_isolated`/dipole correction must be active. No exceptions.
5. **No NaN/inf** in any coordinate reaching a QE input.
6. **Vacuum is present and adequate**; no atom overlaps its periodic image.
7. **Pseudopotentials are what the output says they are.** `pw.out` is ground
   truth for the pseudo actually opened, not the input declaration.
8. **Archived asymmetric data never mixes with corrected results.**
   `results/archive_asymmetric_6L/` is an asymmetric-control dataset only. It is
   never combined into H/H statistics, plots, or fits.

---

## 2. Units and sign conventions

Most silent errors in this pipeline are unit or sign errors. These are the
project's conventions; state them explicitly whenever reporting a number.

**Energies.** QE reports Ry. 1 Ry = 13.605693 eV. Total energies across different
slabs are **not comparable** — cells differ in size and stoichiometry. Compare
only surface energies (per area, with chemical potentials) or energy differences
within a fixed cell.

**Stress.** QE reports kbar. 1 GPa = 10 kbar. Sign convention in this project:

- `sigma < 0` (negative mean in-plane stress) = **compressive**, pressure-like
- `sigma > 0` (positive mean in-plane stress) = **tensile-like**
- QE pressure `P = -(1/3)tr(sigma)`, so positive `P` corresponds to a compressed cell

Sanity anchor from the bulk reference: -1.0% strain (compression) gives
P = +162.53 kbar. If a sign chain ever contradicts that, the chain is wrong.

**Surface stress.** The 2D diagnostic is

    tau_ij = sigma_ij * L_z / 2        [N/m]

with `sigma` in kbar and `L_z` in Angstrom, using 1 kbar*Angstrom = 0.01 N/m:

    tau[N/m] = 0.01 * sigma[kbar] * L_z[Angstrom] / 2

The factor of 2 is because the slab has **two** surfaces. Verified against the
(111) baseline: sigma = 3.19 kbar, L_z ~ 14.8 Angstrom gives tau = 0.236 N/m,
matching the recorded value. Any refactor of `analyze_slab_stress.py` must
reproduce this. An inverted or dropped factor of 2 yields plausible numbers.

**Anisotropy.** `sigma_xx - sigma_yy`, in kbar. Sign is meaningful (it identifies
which in-plane axis carries more stress) and depends on how the surface cell axes
were assigned. Report the axis convention alongside the number.

**Strain.** Dimensionless; report as a fraction and as a percent to avoid the
100x error. Fit slopes are kbar per unit strain, not per percent.

---

## 3. What each layer of the pipeline can and cannot claim

The physical chain being modeled:

    surface orientation + functionalization
      -> reconstruction and bond relaxation
      -> surface stress and anisotropic lattice strain
      -> Raman shifts and/or new surface vibrational modes
      -> NV zero-field-splitting shifts (axial D, transverse E)

Each arrow is a separate inferential step with its own validity conditions.
Do not collapse them in reasoning or in prose.

**Bulk reference (mature).** Five-point hydrostatic EOS, PBE/SSSP-C. Fitted
a0 = 3.573641 Angstrom, B = 445.116 GPa. This is the most trustworthy artifact in
the project and the authoritative source for lattice constant. Live in
`config/reference_pbe_sssp.json`; read it rather than hardcoding.

**Slab stress (screening).** QE stress on a relaxed slab is the stress of the
**whole slab**, including the elastic response of the carbon interior. It is not
an isolated surface-stress tensor. The `tau` diagnostic is a screening proxy.

**Strain response (screening, clamped-ion).** Current strain series apply
homogeneous strain to cell and coordinates, then run a single-point SCF. Ions are
**not re-relaxed**. This is a largely clamped-ion response and will differ from
the relaxed-ion response, potentially substantially for (100) dimers, H
relaxation, and (110) buckling. Never describe clamped-ion results as "the"
surface stress response.

**NV D/E (proxy only).** `nv_strain_model.py` uses residual mean stress as a
proxy for axial D-shift risk and in-plane anisotropy as a proxy for transverse E.
The GHz numbers rest on provisional coupling constants (dD/deps ~ 13 GHz/strain,
dE/deps ~ 5 GHz/strain), cell-averaged rather than local strain, no explicit NV
defect, no depth model, and 6-layer slabs. **The defensible output is the channel
separation and the ranking, not the GHz values.** Any elastic constants in use
(C11/C12/C44) are literature values, not fitted from this DFT setup — say so
every time they enter a result.

**Raman (not yet attempted).** No phonon calculation exists. Nothing in this
repo currently distinguishes a local functional-group mode from a strain-shifted
bulk diamond mode. Do not speculate about the experimental feature's assignment
as though the pipeline supports it.

---

## 4. Epistemic ladder — required for every reported result

Label every quantitative statement at one of these levels. Never promote a
result a level without the corresponding work existing.

- **L0 exploratory** — single run, unconverged, possibly buggy. Internal only.
- **L1 screening** — reproducible, physically sensible, but unconverged in
  thickness / vacuum / cutoff / k-points, or resting on a proxy model. *All
  current slab, strain, and NV results are L1.*
- **L2 converged** — convergence demonstrated in all relevant parameters, with
  the convergence data retained.
- **L3 publication** — L2 plus relaxed-ion response, strain-matched bulk
  subtraction, and (for NV/Raman) a direct calculation rather than a proxy.

Current honest summary of the corrected symmetric H/H 6L baseline, all L1:

- H/H-(100): strongest in-plane anisotropy (~ +56.8 kbar) -> leading transverse E candidate
- H/H-(110): largest mean in-plane stress (~ +36.8 kbar) -> leading axial D candidate
- H/H-(111): isotropic, low stress -> control surface

Biaxial fits have R^2 > 0.999 with zero crossings inside the sampled +/-1% window.
High R^2 means the linear fit is good; it says nothing about whether the
underlying calculation is converged. Do not cite R^2 as evidence of physical
correctness.

**Known not demonstrated:** thickness convergence (6L only), vacuum convergence
(8 Angstrom used, 18-20 Angstrom targeted), cutoff consistency (bulk 80/640 Ry vs
some slabs 60/480), k-point convergence, relaxed-ion response, strain-matched
bulk subtraction, full elastic tensor, layer-resolved strain vs depth, particle
size/shape model, phonons, explicit NV defect, direct D/E tensor, surface
energies, experimental calibration.

---

## 5. Pipeline map

    bulkgen.py            -> runs/reference_diamond/*/pw.in
    parse_reference.py    -> reference_summary.{csv,json,md}
    fit_bulk_reference.py -> bulk_fit_summary + config/reference_pbe_sssp.json

    geometry.py + motifs.yaml -> slabgen.py -> runs/<slab>/pw.in
    parse_slab.py             -> slab_summary.{csv,json,md}

    make_slab_stress_scf.py   -> runs/<slab>_stress_scf/    (scf, tstress, tprnfor)
    analyze_slab_stress.py    -> slab_stress_summary.*
    make_slab_strain_series.py-> runs/<slab>_strain_<mode>_<eps>_scf/
    fit_slab_strain.py        -> slab_strain_fit_summary.*
    nv_strain_model.py        -> results/nv/nv_strain_*

    run_queue.py              -> execution (currently local pw.x)
    update_slab_analysis.py   -> parse + stress + strain-fit refresh

`runs/` is scratch and gitignored. `results/` is curated. Never treat a `runs/`
artifact as a result of record.

---

## 6. Known defects — do not paper over

- `build_library.py` can overwrite `motifs.yaml` and silently drop transcribed
  or custom motifs (e.g. Pandey). Until fixed, never run it without checking
  what it will destroy. Preferred fix: write generated motifs to a separate file,
  or refuse overwrite without `--force`.
- `make_slab_strain_series.py --only` behaves inconsistently as exact-match vs
  substring filter. Verify what it matched before trusting a campaign is complete.
- `geometry.py` emits NumPy divide/overflow/invalid warnings during generation.
  Final coordinates appear finite, but the warnings are unexplained. Do not
  suppress them; assert finiteness explicitly instead.
- `run_queue.py` has not reliably copied completed slab relaxations into
  `results/slabs/`. Verify the copy rather than assuming it.

---

## 7. Before any QE job is submitted

Run the structural pre-flight gate. It must pass, not be reviewed and waived.
Checks: face-symmetry counts, inversion symmetry where required, no
undercoordinated surface carbon, C-C and C-H distances physically reasonable, no
NaN/inf, vacuum present, no periodic overlap, required pseudopotential files
present, folder name consistent with actual geometry, dipole correction present
for asymmetric structures.

Write a provenance file into every calculation directory before it runs: git
commit and branch, generator version, source motif, top and bottom terminations,
symmetric/asymmetric flag, orientation, layer count, vacuum, in-plane repeat,
lattice constant, pseudopotential filenames, cutoffs, k-mesh, strain mode and
value, and the source relaxed geometry. Provenance at this level would have
surfaced the asymmetric error immediately.

---

## 8. Cluster rules (Ginsburg)

- Slurm account: `rent`. Every job needs `--account=rent`.
- **Never run calculations on a login node.** Long-running processes there are a
  policy violation.
- Shared scratch has been observed near-full. QE `outdir` must point at
  node-local scratch (`$TMPDIR`), with only `pw.out` and required artifacts
  copied back and wavefunction/charge-density files discarded unless explicitly
  needed. A few hundred sweep jobs writing to shared storage will fill it.
- Cluster access requires VPN plus interactive Duo. An agent cannot authenticate.
  Cluster-touching operations rely on a pre-authenticated multiplexed SSH socket
  and must **fail loudly** if it is absent — never hang, never retry silently.
- Do not submit batch campaigns until the execution path has been validated by a
  single successful job. N broken jobs cost N times as much as one.

---

## 9. Reporting style

- State the epistemic level (L0-L3) with every quantitative claim.
- State units and sign convention for every stress number.
- Distinguish "the calculation gives X" from "the physics implies X."
- When a result depends on a provisional constant, a proxy model, or a literature
  value, say so in the same sentence as the number.
- Prefer "would need ~0.50% in-plane compression to remove residual mean stress"
  over "is compressed by 0.50%" — the former is what was calculated.
- If asked for a conclusion the pipeline cannot support, say what is missing
  rather than producing the number anyway.
