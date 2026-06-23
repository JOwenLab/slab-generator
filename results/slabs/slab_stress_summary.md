# Slab Stress Summary

This report estimates vacuum-corrected 2D surface stress from relaxed slab QE stress tensors.

Important caveat: QE slab stress is averaged over the full vacuum-containing supercell. The values below multiply by the cell height and divide by two slab faces, giving a first surface-stress diagnostic. The more rigorous next step is an explicit in-plane strain series.

## Clean H-Terminated Baseline

| Run | Orient. | Formula | sigma_xx (kbar) | sigma_yy (kbar) | mean sigma (kbar) | tau_mean (N/m) | effective pressure (kbar) | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C100_2x1_H_6L_strain_biaxial_m0p005_scf | (100) | C12H2 | 9.360 | -61.420 | -26.030 | -2.6660 | 26.030 | compressive_inplane_pressure_like |
| C100_2x1_H_6L_strain_biaxial_m0p010_scf | (100) | C12H2 | 25.520 | -50.330 | -12.405 | -1.2705 | 12.405 | compressive_inplane_pressure_like |
| C100_2x1_H_6L_strain_biaxial_m0p013_scf | (100) | C12H2 | 33.870 | -44.590 | -5.360 | -0.5490 | 5.360 | compressive_inplane_pressure_like |
| C100_2x1_H_6L_strain_biaxial_m0p015_scf | (100) | C12H2 | 42.390 | -38.710 | 1.840 | 0.1885 | -1.840 | tensile_inplane_stress_like |
| C100_2x1_H_6L_strain_biaxial_m0p020_scf | (100) | C12H2 | 60.000 | -26.500 | 16.750 | 1.7155 | -16.750 | tensile_inplane_stress_like |
| C100_2x1_H_6L_strain_biaxial_p0p000_scf | (100) | C12H2 | -6.150 | -72.020 | -39.085 | -4.0031 | 39.085 | compressive_inplane_pressure_like |
| C100_2x1_H_6L_strain_biaxial_p0p005_scf | (100) | C12H2 | -21.030 | -82.170 | -51.600 | -5.2849 | 51.600 | compressive_inplane_pressure_like |
| C100_2x1_H_6L_strain_biaxial_p0p010_scf | (100) | C12H2 | -35.310 | -91.890 | -63.600 | -6.5140 | 63.600 | compressive_inplane_pressure_like |
| C100_2x1_H_6L_stress_scf | (100) | C12H2 | -6.150 | -72.020 | -39.085 | -4.0031 | 39.085 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_strain_biaxial_m0p005_scf | (110) | C12H2 | 0.430 | 23.030 | 11.730 | 1.3025 | -11.730 | tensile_inplane_stress_like |
| C110_1x1_H_6L_SSSP_strain_biaxial_m0p010_scf | (110) | C12H2 | 24.250 | 41.740 | 32.995 | 3.6637 | -32.995 | tensile_inplane_stress_like |
| C110_1x1_H_6L_SSSP_strain_biaxial_p0p000_scf | (110) | C12H2 | -22.180 | 5.080 | -8.550 | -0.9494 | 8.550 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_strain_biaxial_p0p005_scf | (110) | C12H2 | -43.610 | -12.130 | -27.870 | -3.0946 | 27.870 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_strain_biaxial_p0p010_scf | (110) | C12H2 | -63.920 | -28.620 | -46.270 | -5.1377 | 46.270 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_stress_scf | (110) | C12H2 | -22.180 | 5.080 | -8.550 | -0.9494 | 8.550 | compressive_inplane_pressure_like |
| C111_1x1_H_6L_strain_biaxial_m0p005_scf | (111) | C6H | 16.610 | 16.610 | 16.610 | 1.7218 | -16.610 | tensile_inplane_stress_like |
| C111_1x1_H_6L_strain_biaxial_m0p010_scf | (111) | C6H | 36.560 | 36.560 | 36.560 | 3.7899 | -36.560 | tensile_inplane_stress_like |
| C111_1x1_H_6L_strain_biaxial_p0p000_scf | (111) | C6H | -2.440 | -2.440 | -2.440 | -0.2529 | 2.440 | compressive_inplane_pressure_like |
| C111_1x1_H_6L_strain_biaxial_p0p005_scf | (111) | C6H | -20.620 | -20.620 | -20.620 | -2.1375 | 20.620 | compressive_inplane_pressure_like |
| C111_1x1_H_6L_strain_biaxial_p0p010_scf | (111) | C6H | -37.970 | -37.970 | -37.970 | -3.9360 | 37.970 | compressive_inplane_pressure_like |
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
