# Slab Summary

Parsed completed and available diamond slab Quantum ESPRESSO calculations.

- Slab calculations found: 25
- Completed: 25
- Needs attention: 1

## Slab Runs

| Run | Formula | Orient. | Term. | Energy (Ry) | Area (A^2) | Force (Ry/Bohr) | Pseudos | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| C100_2x1_H_6L | C12H2 | (100) | H | -223.02770009 | 12.771 | 0.000151 | ok | JOB DONE |
| C100_2x1_H_6L_strain_biaxial_m0p005_scf | C12H2 | (100) | H | -223.03158185 | 12.644 | 0.006583 | ok | JOB DONE |
| C100_2x1_H_6L_strain_biaxial_m0p010_scf | C12H2 | (100) | H | -223.03385490 | 12.517 | 0.013356 | ok | JOB DONE |
| C100_2x1_H_6L_strain_biaxial_m0p013_scf | C12H2 | (100) | H | -223.03436997 | 12.454 | 0.016869 | ok | JOB DONE |
| C100_2x1_H_6L_strain_biaxial_m0p015_scf | C12H2 | (100) | H | -223.03446389 | 12.391 | 0.020416 | ok | JOB DONE |
| C100_2x1_H_6L_strain_biaxial_m0p020_scf | C12H2 | (100) | H | -223.03335270 | 12.265 | 0.027736 | ok | JOB DONE |
| C100_2x1_H_6L_strain_biaxial_p0p000_scf | C12H2 | (100) | H | -223.02770009 | 12.771 | 0.000150 | ok | JOB DONE |
| C100_2x1_H_6L_strain_biaxial_p0p005_scf | C12H2 | (100) | H | -223.02226068 | 12.899 | 0.006190 | ok | JOB DONE |
| C100_2x1_H_6L_strain_biaxial_p0p010_scf | C12H2 | (100) | H | -223.01531198 | 13.028 | 0.012202 | ok | JOB DONE |
| C100_2x1_H_6L_stress_scf | C12H2 | (100) | H | -223.02770009 | 12.771 | 0.000150 | ok | JOB DONE |
| C110_1x1_H_6L | C12H2 | (110) | H | -138.82985237 | 8.997 | 0.000119 | C_pseudo_mismatch; H_pseudo_not_project_SSSP; ecutwfc_below_reference; ecutrho_below_reference | JOB DONE |
| C110_1x1_H_6L_SSSP | C12H2 | (110) | H | -223.29801692 | 9.030 | 0.000164 | ok | JOB DONE |
| C110_1x1_H_6L_SSSP_strain_biaxial_m0p005_scf | C12H2 | (110) | H | -223.29785804 | 8.940 | 0.005667 | ok | JOB DONE |
| C110_1x1_H_6L_SSSP_strain_biaxial_m0p010_scf | C12H2 | (110) | H | -223.29580184 | 8.851 | 0.011843 | ok | JOB DONE |
| C110_1x1_H_6L_SSSP_strain_biaxial_p0p000_scf | C12H2 | (110) | H | -223.29801692 | 9.030 | 0.000163 | ok | JOB DONE |
| C110_1x1_H_6L_SSSP_strain_biaxial_p0p005_scf | C12H2 | (110) | H | -223.29635194 | 9.121 | 0.005147 | ok | JOB DONE |
| C110_1x1_H_6L_SSSP_strain_biaxial_p0p010_scf | C12H2 | (110) | H | -223.29292928 | 9.212 | 0.009873 | ok | JOB DONE |
| C110_1x1_H_6L_SSSP_stress_scf | C12H2 | (110) | H | -223.29801692 | 9.030 | 0.000163 | ok | JOB DONE |
| C111_1x1_H_6L | C6H | (111) | H | -111.61393168 | 5.530 | 0.000016 | ok | JOB DONE |
| C111_1x1_H_6L_strain_biaxial_m0p005_scf | C6H | (111) | H | -111.61355343 | 5.475 | 0.001239 | ok | JOB DONE |
| C111_1x1_H_6L_strain_biaxial_m0p010_scf | C6H | (111) | H | -111.61215978 | 5.420 | 0.002643 | ok | JOB DONE |
| C111_1x1_H_6L_strain_biaxial_p0p000_scf | C6H | (111) | H | -111.61393168 | 5.530 | 0.000019 | ok | JOB DONE |
| C111_1x1_H_6L_strain_biaxial_p0p005_scf | C6H | (111) | H | -111.61333022 | 5.585 | 0.001091 | ok | JOB DONE |
| C111_1x1_H_6L_strain_biaxial_p0p010_scf | C6H | (111) | H | -111.61178470 | 5.641 | 0.002075 | ok | JOB DONE |
| C111_1x1_H_6L_stress_scf | C6H | (111) | H | -111.61393168 | 5.530 | 0.000019 | ok | JOB DONE |

## Interpretation Notes

- H-terminated surface energies are not final yet because the hydrogen chemical potential is not defined.
- `carbon_bulk_subtracted_energy_ry` subtracts only the fitted bulk carbon reference and should be treated as a preliminary excess-energy diagnostic.
- Rows with pseudo/cutoff mismatches should not be used for direct orientation comparisons.

## Next Analysis Step

Add a consistent hydrogen chemical-potential reference, then compute area-normalized surface energies using:

```text
gamma = (E_slab - N_C * mu_C_bulk - N_H * mu_H) / (2A)
```
