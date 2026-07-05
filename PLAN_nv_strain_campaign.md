# NV Strain Campaign Plan — v1

Scientific target: experimentally observed ODMR **shift** and **splitting** of the
NV ms=0 → ±1 resonance upon change of diamond surface termination (e.g.
hydrogenation). Question: is the perturbation **surface-induced strain**
(spin–strain coupling) or **surface-induced dipoles/electric fields** (Stark
coupling)? This plan builds the strain arm first, with the dipole arm as the
designed discriminator in Phase 2.

Narrative emphasis: the **zero-field splitting D** (axial) and the **transverse
zero-field splitting E**. Anisotropic surface strain generates nonzero E; E
couples the ms=±1 levels and is tied to NV coherence. Claim to test: *surfaces
limit NV coherence by inducing anisotropic strain* — a mechanism distinct from
the usual surface-spin-bath picture.

---

## Workstream A — NV spin–strain physics layer (code)

**A1. `nv_spin_strain.py` (new; quantitative successor to the ranking in
`nv_strain_model.py`, which stays as a screening tool).**

Hamiltonian: Udvarhelyi, Shkolnikov, Gali, Burkard & Pályi, PRB **98**, 075201
(2018) — six DFT-derived couplings h41, h43, h15, h16, h25, h26 in the NV frame:

```
H_e0/h = [h41 (exx + eyy) + h43 ezz] Sz^2
H_e1/h = ... (h15, h16: couples {Sx,Sz},{Sy,Sz} to exz, eyz, exx−eyy, exy)
H_e2/h = ... (h25, h26: couples Sx^2−Sy^2 and {Sx,Sy})
```

Numeric values transcribed from the paper's table at implementation time, with
Barfuss et al., PRB 99, 174102 (2019) as cross-check (implementation includes a
unit test against a published worked example). Observables per NV orientation:
ΔD, E, zero-field line positions f± = (D0 + ΔD) ± E.

**A2. Orientation projection.** Input: strain tensor in the cubic crystal
frame. The module rotates it into each of the four NV frames (⟨111⟩ family) and
evaluates A1 for each, then groups degenerate orientations to emit a predicted
zero-field ODMR **stick spectrum** per (face, termination, thickness).

Falsifiable per-face signatures under the slab's own equilibrium biaxial strain:
- **(111) slab**: the surface-normal NV → pure ΔD, E = 0; the three inclined
  NVs → common ΔD plus nonzero E (splitting). Two-group spectrum.
- **(100) slab**: all four NVs symmetry-equivalent under equal-biaxial strain,
  but the 2×1 dimer anisotropy (σxx ≠ σyy, already large in the data:
  −6 vs −72 kbar at 6L) makes E ≠ 0 for all of them.
- **(110) slab**: intrinsically anisotropic face; two inequivalent NV pairs.

**A3. Strain inputs, two levels of realism.**
1. *Slab-average*: equilibrium in-plane strain ε\*(t) from
   `fit_slab_strain.py` (`zero_stress_epsilon`), εzz from the Poisson response
   using bulk C11/C12 (to be fitted from the existing reference series or a
   small uniaxial bulk add-on).
2. *Layer-resolved* (new `layer_strain.py`): εzz(z) from relaxed interlayer
   spacings vs bulk, giving strain-vs-depth decay → prediction of NV
   perturbation vs implantation depth. This is the quantity to compare across
   thicknesses in the adaptive loop.

**A4. Plotting layer** (`plot_nv_predictions.py`): ε\*(t), τ(t), ΔD(t), E(t)
per face/termination, plus predicted stick spectra overlaid for two
terminations (the "hydrogenation shift" figure, computed).

**A5. Ensemble lineshape.** The experimental reference data are nanodiamond
ensembles (6.7 / 14.8 nm effective diameter), so the comparison target is an
orientation- and facet-averaged lineshape: convolve the four-orientation stick
spectrum (weighted over (100)/(110)/(111) facets) with a Lorentzian of the
measured width. Slab thickness t maps onto particle size heuristically
(surface-to-volume of a particle ≈ 2× a slab of t = diameter); the 4.5/6 nm
ladder anchors double as the direct comparison points for the 6.7 nm sample.
Sign check to resolve early: observed shift is DOWNWARD (~15–25 MHz) while
naive in-plane compression raises D — the h41/h43 balance under biaxial
strain + Poisson εzz determines the sign and is a sharp test of the strain
hypothesis.

---

## Workstream B — thickness ladder (replaces prior 1–20 nm plan)

Rationale: prior 6L runs were pipeline-validation scale, not physics choices.
Surface-induced strain scales as ε\*(t) ≈ −2τ/(Y_b t): the action is at small
t, and the experimental interest is 0–3 nm. The thick end is only needed to
anchor the 1/t extrapolation.

Base ladder (t = carbon-slab thickness; layer counts per face, rounded to
parity constraints — (111) even for SDB/SDB symmetric slabs, symmetric H/H
everywhere for clean stress):

| t target | (111) layers (~2.06 Å/bilayer) | (100) layers (~0.89 Å) | (110) layers (~1.26 Å) |
|---|---|---|---|
| 0.5 nm | 6  | 6  | 4  |
| 0.75 nm| 8  | 8  | 6  |
| 1.0 nm | 10 | 12 | 8  |
| 1.5 nm | 14 | 17→18 | 12 |
| 2.0 nm | 20 | 22 | 16 |
| 2.5 nm | 24 | 28 | 20 |
| 3.0 nm | 30 | 34 | 24 |
| 4.5 nm (anchor) | 44 | 50 | 36 |
| 6.0 nm (anchor) | 58 | 68 | 48 |

Cost note: with 1×1/2×1 lateral cells these are 12–140 atoms — thin half runs
locally, anchors go to Insomnia/Ginsburg. All slabs symmetric-terminated;
per-thickness bundle = relax → stress SCF → 5-point biaxial (+x, +y where
anisotropy matters) strain series, i.e. the workflow already in README.

## Batches (terminations, small cells first; checkpoint after Batch 5)

1. **H** on (111) 1×1, (100) 2×1, (110) 1×1 — full ladder. Direct analog of the
   hydrogenation experiment; extends existing 6L data.
2. **F** on the same three faces — opposite surface-dipole sign vs H at similar
   cell size: the cheapest first probe of strain-vs-dipole correlation.
3. **Bare reconstructed**: C100_2x1_bare, C111_2x1_pandey, C110_1x1_bare — the
   "before termination" baseline. Pandey may need spin polarization checks.
4. **O-ether**: C100_1x1_O_ether, C110_O_ether — full ladder.
5. **O-ketone**: C100_1x1_O_ketone, C111_1x1_O_ketone — truncated ladder
   (3–4 t points; metastable/comparison models, C111 ketone needs nspin=2).

— CHECKPOINT: review ε\*(t), E(t) trends; re-plan before committing to the
large cells below. —

6. **Seiwatz family** (TDB face; asymmetric-only, odd layers, dipole correction
   on): seiwatz, seiwatz_H, Sei-1ML-ketone.
7. **Graphitic** (111) 2×1/4×1 metastables.
8. **Pending-literature**: methoxyacetone (100), Sei-1ML-canted — when author
   coordinates arrive.

## Workstream C — adaptive thickness selection

After each completed bundle, the scheduler: (1) fits ε\*(t) = ε∞ − 2τ/(Y_eff t)
and E(t) per series; (2) computes leave-one-out residuals; (3) proposes the
next thickness at the largest-uncertainty gap in 1/t space, or where predicted
E(t) crosses the experimental linewidth (config parameter; default 20 MHz to
match the observed nanodiamond-ensemble ODMR linewidth — set to ~0.1 MHz only
for single-NV comparisons);
(4) stops when the model interpolates all sampled t within tolerance.
Deterministic rule; Fable-5 layer writes the human-readable proposal; `--go`
approval gate unchanged. Implementation: `propose_next_thickness.py` +
scheduler hook in the queue/campaign tooling.

## Phase 2 — hypothesis discrimination (dipole arm; second, as agreed)

- **Asymmetric two-termination slabs** (e.g. H top / F bottom), dipole
  correction enabled: extract per-face surface dipole density from the
  planar-averaged electrostatic potential step; convert to E-field vs depth.
- Stark couplings (d∥, d⊥ from the literature, e.g. Van Oort & Glasbeek; values
  verified at implementation) → predicted Stark shift/splitting vs depth.
- Deliverable: side-by-side strain-predicted vs dipole-predicted (ΔD, E) for
  each termination change → attribution of the observed ODMR shift/splitting.

## Near-term implementation order

1. `nv_spin_strain.py` + unit tests (A1, A2)
2. `layer_strain.py` (A3.2) + C11/C12 fit from reference series
3. Ladder config + thickness-series generation wrapper (B)
4. `plot_nv_predictions.py` (A4)
5. `propose_next_thickness.py` (C)
6. Batch-1 (H, full ladder) input generation for review
