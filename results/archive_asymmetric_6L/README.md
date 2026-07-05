# Archived: the original 6-layer slab runs (2026, asymmetric)

These 45 completed QE runs and their derived summaries are preserved here
for provenance but should NOT be used for physics results, because:

1. Every slab had two different surfaces (terminated top, bare bottom), so
   the measured surface stress mixes two surfaces.
2. None used the dipole correction that two-different-surface slabs need.
3. One directory (C110_1x1_H_6L, no _SSSP suffix) additionally used an
   older pseudopotential set and is superseded by the _SSSP rerun.

They remain valuable as the end-to-end validation of the workflow (input
generation -> local runs -> parsing -> stress/strain fits), and as timing
references. The replacement calculations are the symmetric slabs in
batches/differential_6L/; once those are run and analyzed, fresh summaries
will be written to results/slabs/.

The NV predictions computed from these runs are archived alongside
(nv_predictions_from_asymmetric/) for the same reason.
