# Slab Stress Summary

This report estimates vacuum-corrected 2D surface stress from relaxed slab QE stress tensors.

Important caveat: QE slab stress is averaged over the full vacuum-containing supercell. The values below multiply by the cell height and divide by two slab faces, giving a first surface-stress diagnostic. The more rigorous next step is an explicit in-plane strain series.

## Clean H-Terminated Baseline

| Run | Orient. | Formula | sigma_xx (kbar) | sigma_yy (kbar) | mean sigma (kbar) | tau_mean (N/m) | effective pressure (kbar) | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C100_2x1_H_6L_sym_strain_biaxial_m0p005_scf | (100) | C24H8 | 28.870 | -29.560 | -0.345 | -0.0250 | -0.345 | near_zero_inplane_mean_stress |
| C100_2x1_H_6L_sym_strain_biaxial_m0p010_scf | (100) | C24H8 | 52.590 | -7.470 | 22.560 | 1.6357 | 22.560 | compressive_inplane_pressure_like |
| C100_2x1_H_6L_sym_strain_biaxial_p0p000_scf | (100) | C24H8 | 6.110 | -50.710 | -22.300 | -1.6169 | -22.300 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_strain_biaxial_p0p005_scf | (100) | C24H8 | -15.690 | -70.940 | -43.315 | -3.1406 | -43.315 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_strain_biaxial_p0p010_scf | (100) | C24H8 | -36.570 | -90.300 | -63.435 | -4.5994 | -63.435 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_strain_x_m0p005_scf | (100) | C24H8 | 28.150 | -50.380 | -11.115 | -0.8059 | -11.115 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_strain_x_m0p010_scf | (100) | C24H8 | 50.950 | -50.060 | 0.445 | 0.0323 | 0.445 | near_zero_inplane_mean_stress |
| C100_2x1_H_6L_sym_strain_x_p0p000_scf | (100) | C24H8 | 6.110 | -50.710 | -22.300 | -1.6169 | -22.300 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_strain_x_p0p005_scf | (100) | C24H8 | -15.170 | -51.040 | -33.105 | -2.4003 | -33.105 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_strain_x_p0p010_scf | (100) | C24H8 | -35.730 | -51.380 | -43.555 | -3.1580 | -43.555 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_strain_y_m0p005_scf | (100) | C24H8 | 6.730 | -29.990 | -11.630 | -0.8433 | -11.630 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_strain_y_m0p010_scf | (100) | C24H8 | 7.340 | -8.520 | -0.590 | -0.0428 | -0.590 | near_zero_inplane_mean_stress |
| C100_2x1_H_6L_sym_strain_y_p0p000_scf | (100) | C24H8 | 6.110 | -50.710 | -22.300 | -1.6169 | -22.300 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_strain_y_p0p005_scf | (100) | C24H8 | 5.500 | -70.700 | -32.600 | -2.3637 | -32.600 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_strain_y_p0p010_scf | (100) | C24H8 | 4.890 | -90.000 | -42.555 | -3.0855 | -42.555 | tensile_inplane_stress_like |
| C100_2x1_H_6L_sym_stress_scf | (100) | C24H8 | 6.110 | -50.710 | -22.300 | -1.6169 | -22.300 | tensile_inplane_stress_like |
| C110_1x1_H_6L_SSSP_sym_strain_biaxial_m0p005_scf | (110) | C12H4 | 55.250 | 79.890 | 67.570 | 5.4385 | 67.570 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_biaxial_m0p010_scf | (110) | C12H4 | 89.530 | 110.040 | 99.785 | 8.0314 | 99.785 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_biaxial_p0p000_scf | (110) | C12H4 | 22.700 | 50.850 | 36.775 | 2.9599 | 36.775 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_biaxial_p0p005_scf | (110) | C12H4 | -8.210 | 22.890 | 7.340 | 0.5908 | 7.340 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_biaxial_p0p010_scf | (110) | C12H4 | -37.550 | -4.050 | -20.800 | -1.6741 | -20.800 | tensile_inplane_stress_like |
| C110_1x1_H_6L_SSSP_sym_strain_x_m0p005_scf | (110) | C12H4 | 51.150 | 54.820 | 52.985 | 4.2646 | 52.985 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_x_m0p010_scf | (110) | C12H4 | 80.560 | 58.920 | 69.740 | 5.6131 | 69.740 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_x_p0p000_scf | (110) | C12H4 | 22.700 | 50.850 | 36.775 | 2.9599 | 36.775 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_x_p0p005_scf | (110) | C12H4 | -4.800 | 47.020 | 21.110 | 1.6991 | 21.110 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_x_p0p010_scf | (110) | C12H4 | -31.370 | 43.310 | 5.970 | 0.4805 | 5.970 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_y_m0p005_scf | (110) | C12H4 | 26.570 | 75.600 | 51.085 | 4.1117 | 51.085 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_y_m0p010_scf | (110) | C12H4 | 30.660 | 100.680 | 65.670 | 5.2856 | 65.670 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_y_p0p000_scf | (110) | C12H4 | 22.700 | 50.850 | 36.775 | 2.9599 | 36.775 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_y_p0p005_scf | (110) | C12H4 | 19.060 | 26.420 | 22.740 | 1.8303 | 22.740 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_strain_y_p0p010_scf | (110) | C12H4 | 15.640 | 2.290 | 8.965 | 0.7216 | 8.965 | compressive_inplane_pressure_like |
| C110_1x1_H_6L_SSSP_sym_stress_scf | (110) | C12H4 | 22.700 | 50.850 | 36.775 | 2.9599 | 36.775 | compressive_inplane_pressure_like |
| C111_1x1_H_6L_sym_strain_biaxial_m0p005_scf | (111) | C6H2 | 30.520 | 30.520 | 30.520 | 2.2619 | 30.520 | compressive_inplane_pressure_like |
| C111_1x1_H_6L_sym_strain_biaxial_m0p010_scf | (111) | C6H2 | 59.180 | 59.180 | 59.180 | 4.3859 | 59.180 | compressive_inplane_pressure_like |
| C111_1x1_H_6L_sym_strain_biaxial_p0p000_scf | (111) | C6H2 | 3.190 | 3.190 | 3.190 | 0.2364 | 3.190 | compressive_inplane_pressure_like |
| C111_1x1_H_6L_sym_strain_biaxial_p0p005_scf | (111) | C6H2 | -22.810 | -22.810 | -22.810 | -1.6905 | -22.810 | tensile_inplane_stress_like |
| C111_1x1_H_6L_sym_strain_biaxial_p0p010_scf | (111) | C6H2 | -47.550 | -47.550 | -47.550 | -3.5240 | -47.550 | tensile_inplane_stress_like |
| C111_1x1_H_6L_sym_stress_scf | (111) | C6H2 | 3.190 | 3.190 | 3.190 | 0.2364 | 3.190 | compressive_inplane_pressure_like |

## Interpretation

- `tau_mean_n_per_m` is the approximate per-surface in-plane stress after correcting for vacuum dilution.
- Sign convention (CLAUDE.md section 2): positive `sigma` means the cell is COMPRESSED and wants to expand; negative `sigma` means it is in TENSION. Anchored on bulk diamond at -1% strain, which gives sigma = P = +162.86 kbar.
- `effective_inplane_pressure_kbar = +mean(sigma_xx, sigma_yy)`, the in-plane analogue of the QE pressure `P = +(1/3)tr(sigma)`. It shares `sigma`'s sign, so it is positive under compression.
- Accordingly, large positive effective pressure indicates a compressive, pressure-like in-plane surface contribution; large negative values indicate a tensile one.
- These values are best used to rank orientations and functionalizations before doing explicit strain fits.

## Excluded or Flagged Rows

| Run | Reason |
|---|---|
| none | |

## Next Step

Generate fixed-cell in-plane strain series for the clean H-terminated slabs and fit energy/stress versus strain to obtain robust surface stress tensors.
