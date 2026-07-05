# Review notes: what we found in the first round of calculations, and what changed

Audience: the student running/extending the QE slab campaign.

**Where the updates live:** everything described here sits on a new Git
branch called `claude/nv-strain-campaign-plan`. It has NOT been merged into
`main` yet — `main` is untouched. To see it on GitHub, use the branch
dropdown above the file list, or open a pull request from that branch to
review every change line by line before anything becomes official.

---

## Part 1 — Issues found in the completed calculations

The first round (45 finished QE runs: 4 slab relaxations, 3 stress
calculations, 33 strained-slab calculations, 5 bulk crystal points) did its
real job: it proved the whole workflow runs end to end, from generating
inputs to fitting the results. The bulk diamond reference (lattice constant
3.5736 Å from our own DFT) is solid and stays. But the slab numbers
themselves have problems and should be treated as workflow tests, not
physics results:

1. **Every slab had two different surfaces.** The top face carried the
   intended termination (hydrogen), but the bottom face was left as bare,
   unterminated carbon. You can see this in any of the old input files —
   the header says `bottom=bare`. Because the calculation measures the two
   surfaces together, the extracted surface stress is a blend of the
   hydrogen surface and the bare surface, and can't be attributed to
   hydrogen termination alone.

2. **Those slabs also needed, and lacked, a "dipole correction."** When the
   two faces of a slab are chemically different, the slab as a whole acts
   like a tiny capacitor with an electric field around it. In a periodic
   calculation the slab is repeated infinitely, so each copy feels the
   field of its neighbors — an artificial effect that contaminates the
   energies, forces, and stresses. Quantum ESPRESSO has a standard fix (a
   compensating field in the vacuum region, switched on with the
   `tefield`/`dipfield` settings), but a bug in our input generator only
   added it for one type of run, and none of the completed runs got it.

3. **The (111) slab was only strained equally in both in-plane directions**
   (a "biaxial" series). The (100) and (110) slabs were additionally
   strained along x only and along y only, which reveals whether the
   surface responds differently in different directions. For (111) the
   symmetry of the surface says the response should be direction-
   independent, so nothing is strictly missing — but measuring it anyway
   is a cheap sanity check, and the direction-dependent response is exactly
   what feeds the NV "transverse splitting" predictions.

4. **Two of the (100) fits extrapolate beyond their data.** The strain
   value where the stress crosses zero came out at −2.6 % and −3.6 %, but
   the calculations only sampled strains between −1 % and +1 %. The fitting
   code flagged this itself. If those crossing points ever matter, add
   calculation points at larger compression rather than trusting the
   extrapolation.

5. **Everything is one thickness.** All slabs were 6 atomic layers
   (roughly half a nanometer of carbon) — chosen so each run finished in
   ~2 minutes on a laptop, not for physics. How the strain evolves with
   thickness, which is the scientific question, hasn't been touched yet.

6. **One duplicate run.** The (110) slab was calculated twice: once with an
   older set of pseudopotentials, then again with the community-standard
   SSSP set (which the whole project now uses). Only the SSSP version
   counts; the older one must never be mixed into an analysis, because
   energies and stresses from different pseudopotentials aren't
   comparable.

7. **An easy mistake to be aware of in the build scripts:**
   `build_library.py` rewrites the motif library file (`motifs.yaml`) and,
   on its own, silently drops the Pandey surface entry that comes from the
   literature-transcription step. Always run `transcribe_petukhov.py`
   afterward — that's the intended order.

8. **Minor:** the NV prediction code currently takes diamond's elastic
   constants from the literature rather than from our own DFT. That's
   acceptable for now; computing them ourselves later (two small extra
   bulk calculations) would make the chain fully self-contained.

## Part 2 — Recommended next steps, in order

1. **Re-run the 6-layer baselines with correct (symmetric) slabs.** The
   inputs are already generated, checked, and committed:
   `batches/differential_6L/` — nine jobs covering hydrogen, bare, and
   oxygen(ether) terminations at 6 layers on all three faces, plus a
   Pandey/hydrogen pair at 20 layers. Run with
   `python3 run_queue.py --runs-root batches/differential_6L --run`, then
   the usual stress → strain-series → `update_slab_analysis.py` chain.
2. **Include x-only and y-only strain series for (111)** in the re-run
   (the direction-independence of the result is then a built-in check).
3. **Extend the (100) strain range** (add −2 % and −3 % points) if the
   zero-crossing values are needed.
4. **Report differences between terminations** — ΔD(H) − ΔD(bare), etc.,
   per face. The experiment measures the shift *between* two surface
   states, so these differences, not absolute numbers, are what test the
   strain hypothesis. Details in PLAN_nv_strain_campaign.md
   ("Differential framing").
5. **Then start the thickness series** (0.5–3 nm plus two thicker anchor
   points, per the plan document).

## Part 3 — What has already been fixed or added (all on the branch)

- **The generator now makes symmetric slabs by default.** Both faces
  identical — same termination, same reconstruction — which removes issues
  1 and 2 at the source: equal surfaces mean the surface stress is
  well-defined and there is no electric-field artifact to correct.
  Reconstructions that are only published for one face (like Pandey) are
  mirrored onto the other face using an exact symmetry of the diamond
  crystal, and the code verifies the result atom by atom. Making an
  asymmetric slab now requires an explicit `--asymmetric` flag, prints a
  warning, and automatically switches the dipole correction on (the bug in
  issue 2 is fixed for every run type).
- **New physics module `nv_spin_strain.py`**: converts a slab's strain
  state into predicted NV quantities — the shift of the 2.87 GHz
  zero-field splitting (ΔD) and the transverse splitting (E) — for all
  four NV orientations, using published coupling constants (Udvarhelyi et
  al., Phys. Rev. B 98, 075201). Ten automated tests pass, including
  reproducing a published benchmark number.
- **Campaign plan** (`PLAN_nv_strain_campaign.md`): thickness ladder, the
  order in which terminations get calculated, and the longer-term ideas
  (adaptive thickness selection; electric-field/dipole comparison arm).
- **Ready-to-run inputs** in `batches/differential_6L/` (Part 2, step 1).

## Part 4 — Cleanup performed (also on the branch, reviewable in the PR)

To make sure the problematic numbers can't accidentally be mixed into
future analysis, the old results have been moved out of the active results
area rather than deleted:

- All 45 old slab run directories, their fitted summaries, and the NV
  predictions computed from them now live in
  **`results/archive_asymmetric_6L/`**, with a README explaining exactly
  why they were retired and what replaces them.
- `results/slabs/` is now empty and will be repopulated by the symmetric
  re-runs; the analysis scripts scan that folder, so the archive is
  invisible to them.
- The bulk diamond reference (`results/reference_diamond/`) is unaffected
  and remains the strain reference for everything.

Nothing was deleted: the archive preserves the record of the work (and the
runs' value as workflow validation), while the active project only contains
data we stand behind. If you'd rather delete the archive outright once the
re-runs finish, that's a one-line decision to make at merge time.
