# Slab Stress Summary

This report estimates vacuum-corrected 2D surface stress from relaxed slab QE stress tensors.

Important caveat: QE slab stress is averaged over the full vacuum-containing supercell. The values below multiply by the cell height and divide by two slab faces, giving a first surface-stress diagnostic. The more rigorous next step is an explicit in-plane strain series.

## Clean H-Terminated Baseline

| Run | Orient. | Formula | sigma_xx (kbar) | sigma_yy (kbar) | mean sigma (kbar) | tau_mean (N/m) | effective pressure (kbar) | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C100_2x1_H_6L_stress_scf | (100) | C12H2 | -6.150 | -72.020 | -39.085 | -4.0031 | 39.085 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_stress_scf | (110) | C12H2 | -22.180 | 5.080 | -8.550 | -0.9494 | 8.550 | compressive_inplane_pressure_like |
| C111_1x1_H_6L_stress_scf | (111) | C6H | -2.440 | -2.440 | -2.440 | -0.2529 | 2.440 | compressive_inplane_pressure_like |

## Interpretation

- `tau_mean_n_per_m` is the approximate per-surface in-plane stress after correcting for vacuum dilution.
- `effective_inplane_pressure_kbar = -mean(sigma_xx, sigma_yy)` uses QE stress sign convention to provide pressure-like language.
- Large positive effective pressure indicates a compressive pressure-like surface contribution; large negative values indicate tensile stress-like behavior.
- These values are best used to rank orientations and functionalizations before doing explicit strain fits.

## Excluded or Flagged Rows

| Run | Reason |
|---|---|
| C110_1x1_H_6L | exclude_from_comparison_pseudo_mismatch |

## Next Step

Generate fixed-cell in-plane strain series for the clean H-terminated slabs and fit energy/stress versus strain to obtain robust surface stress tensors.
