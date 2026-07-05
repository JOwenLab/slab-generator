# Review notes: issues in the completed runs & changes on the branch

Audience: whoever is running/extending the QE slab campaign.
Branch with all updates: **`claude/nv-strain-campaign-plan`**
(GitHub → branch dropdown, or open a PR against `main` to see the full diff.)

---

## Part 1 — Issues found in the calculations completed so far

The 45 completed QE runs (4 relax, 3 stress-SCF, 33 strained SCF, 5 bulk EOS
points) proved the whole run → parse → fit → predict chain works, and the
bulk reference (a0 = 3.5736 Å from DFT, correctly used as the strain
reference) is solid. The issues below don't undo that — but they mean the
slab *numbers* should be treated as pipeline diagnostics, not physics
results.

1. **All slab runs used asymmetric slabs (H top / bare bottom).** Check any
   `results/slabs/*/pw.in` header: `bottom=bare`. The measured surface
   stress τ and equilibrium strain ε\* therefore average over two *different*
   surfaces — an H-terminated face and a dangling-bond bare face — so they
   cannot be attributed to H-termination alone. The bare (111) 1×1 bottom is
   especially problematic (unreconstructed radical surface).

2. **Those runs also had no dipole correction.** Two different surfaces ⇒
   net slab dipole ⇒ spurious field between periodic images, contaminating
   energies, forces, and the stress tensor. Root cause was a generator bug:
   dipole flags (`tefield`/`dipfield`/`edir`) were emitted only for
   `cell2d` runs, never for ordinary `relax` runs. (Fixed on the branch —
   see Part 3.)

3. **(111)-H has only a biaxial strain series** (5 points; no uniaxial x/y,
   unlike (100) and (110)). Defensible by symmetry (3-fold axis ⇒ isotropic
   in-plane τ), but it leaves no anisotropy consistency check and no
   direct data for the transverse-strain channel that controls the
   inclined-NV splitting E.

4. **(100) uniaxial fits extrapolate their zero-stress crossing** (−2.6 %
   and −3.6 %, far outside the ±1 % sampled window; the fit summary carries
   `zero_stress_outside_sampled_range` warnings). If uniaxial zero
   crossings are ever quoted, extend the sampled strain range instead of
   trusting the extrapolation.

5. **Single thickness only (6L ≈ 0.5–0.9 nm of C).** No thickness
   dependence exists yet; 6L was chosen for ~2-minute laptop runs, not
   physics. All τ, ε\*, and derived NV numbers are single-point.

6. **`C110_1x1_H_6L` (pre-SSSP) is superseded** by `C110_1x1_H_6L_SSSP`;
   don't mix them in analysis.

7. **Workflow footgun:** `build_library.py` regenerates `motifs.yaml` and
   silently drops the transcribed Pandey entry. Always run
   `transcribe_petukhov.py` after it (the documented pipeline order).

8. **Minor:** the NV model currently takes εzz from *literature* elastic
   constants (C11 = 1076, C12 = 125, C44 = 576 GPa) rather than our own
   PBE values. Consistent enough for now; a 2-point uniaxial bulk add-on
   would make it self-contained.

## Part 2 — Suggested changes (in priority order)

1. **Re-run the 6L baselines with symmetric slabs** — inputs are already
   generated and committed: `batches/differential_6L/` (9 jobs: H / bare /
   O-ether at 6L on (100), (110), (111); Pandey + H pair at 20L).
   `python3 run_queue.py --runs-root batches/differential_6L --run`, then
   the usual stress-SCF → strain-series → `update_slab_analysis.py` chain.
2. **Add uniaxial x/y series for (111)-H** in the re-run (τxx = τyy is then
   a symmetry check; the anisotropic response feeds the NV E predictions).
3. **Extend the (100) uniaxial strain range** toward the zero crossing
   (add −2 %, −3 % points) if uniaxial ε\* is needed.
4. **Report the differential quantities**: ΔD(H) − ΔD(X) and ΔE(H) − ΔE(X)
   per face — the experiment measures shifts *between* surface states, so
   the strain hypothesis is tested by these differences (prediction: the
   starting termination must be more compressive than H for the observed
   downshift). See "Differential framing" in PLAN_nv_strain_campaign.md.
5. **Then the thickness ladder** (0.5–3 nm + 4.5/6 nm anchors, per plan)
   and, later, layer-resolved strain (`layer_strain.py`, not yet written).

## Part 3 — What's already implemented on the branch

Commit trail (newest first): b4437e4, fd440af, 6508f89, 2071a91, d07f63d,
59a1fef, b4faa07, e59238a.

- **Symmetric-always policy in the generator.** New
  `geometry.invert_symmetrize()` builds both-faces-identical slabs from
  single-face motifs by inversion through a bulk C–C bond midpoint (an
  exact inversion center of diamond ⇒ identical reconstruction on both
  faces, zero net dipole). Refuses centers inside the reconstruction
  region; verifies inversion symmetry atom-by-atom; reports achievable
  layer counts when a request is unrealizable (symmetric Pandey needs
  ≥ 20 layers). CLI now defaults to symmetric; asymmetric generation
  requires explicit `--asymmetric` + prints a warning.
- **Dipole-correction bug fixed**: asymmetric slabs now get
  `tefield`/`dipfield`/`edir` in *all* relax modes, not just `cell2d`.
- **`nv_spin_strain.py`**: exact 6-parameter Udvarhelyi (PRB 98, 075201)
  spin-strain Hamiltonian, anisotropic traction-free elasticity for the
  slab strain tensor, projection onto all four NV orientations, exact
  diagonalization → ΔD, E, f± per orientation. Swappable parameter set
  (`--params dft|barson`). First predictions from the existing 6L fits are
  in `results/nv/` (carrying the Part-1 caveats).
- **Tests**: 10 passing (6 physics validations incl. reproducing the
  paper's a1 = −2.66 MHz/GPa; 4 for the symmetrizer incl. both-face
  coordination checks). Library validation still 16/16.
- **`PLAN_nv_strain_campaign.md`**: full campaign plan — thickness ladder,
  batch order with post-batch-5 checkpoint, adaptive thickness selection
  rule (proposed, not yet implemented), Phase-2 dipole/Stark discrimination
  arm.
- **`batches/differential_6L/`**: the 9 ready-to-run symmetric inputs +
  README with per-job caveats.
