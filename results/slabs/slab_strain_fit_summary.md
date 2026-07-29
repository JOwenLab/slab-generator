# Slab Strain Fit Summary

Fits the slab's in-plane strain response from strain-series SCF calculations. The primary result is the vacuum-corrected 2D surface stress tau (N/m); the raw QE cell-averaged stress sigma (kbar) is retained as a diagnostic cross-check. Both are strain-linear fits of the same underlying data, so their zero-stress strains must agree (see Consistency Check below); tau changes units and physical interpretation, not the equilibrium strain.

## Surface stress fit (primary)

```text
tau_mean_N_per_m(epsilon) = m_tau * epsilon + b_tau
```

| Series | Orient. | Mode | Points | tau slope (N/m/strain) | tau at zero (N/m) | Zero-tau eps | R^2 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C100_2x1_H_6L_sym | (100) | biaxial | 5 | -311.7199 | -1.54925 | -0.004970 | 0.99935 | good |
| C100_2x1_H_6L_sym | (100) | x | 5 | -159.4998 | -1.58978 | -0.009967 | 0.99960 | good |
| C100_2x1_H_6L_sym | (100) | y | 5 | -152.1186 | -1.59043 | -0.010455 | 0.99958 | warning |
| C110_1x1_H_6L_SSSP_sym | (110) | biaxial | 5 | -485.1734 | 3.06928 | 0.006326 | 0.99929 | good |
| C110_1x1_H_6L_SSSP_sym | (110) | x | 5 | -256.6155 | 3.00344 | 0.011704 | 0.99960 | warning |
| C110_1x1_H_6L_SSSP_sym | (110) | y | 5 | -228.1877 | 2.98179 | 0.013067 | 0.99987 | warning |
| C111_1x1_H_6L_sym | (111) | biaxial | 5 | -395.4440 | 0.33395 | 0.000844 | 0.99916 | good |

## Diagnostic QE stress fit (sigma, vacuum-dependent)

```text
mean_sigma_kbar(epsilon) = m_sigma * epsilon + b_sigma
```

| Series | Orient. | Mode | sigma slope (kbar/strain) | sigma at zero (kbar) | Zero-sigma eps | R^2 |
|---|---:|---:|---:|---:|---:|---:|
| C100_2x1_H_6L_sym | (100) | biaxial | -4299.20 | -21.367 | -0.004970 | 0.99935 |
| C100_2x1_H_6L_sym | (100) | x | -2199.80 | -21.926 | -0.009967 | 0.99960 |
| C100_2x1_H_6L_sym | (100) | y | -2098.00 | -21.935 | -0.010455 | 0.99958 |
| C110_1x1_H_6L_SSSP_sym | (110) | biaxial | -6028.00 | 38.134 | 0.006326 | 0.99929 |
| C110_1x1_H_6L_SSSP_sym | (110) | x | -3188.30 | 37.316 | 0.011704 | 0.99960 |
| C110_1x1_H_6L_SSSP_sym | (110) | y | -2835.10 | 37.047 | 0.013067 | 0.99987 |
| C111_1x1_H_6L_sym | (111) | biaxial | -5335.80 | 4.506 | 0.000844 | 0.99916 |

## Consistency check: zero-sigma vs zero-tau strain

All series: `zero_stress_epsilon` (sigma) and `zero_tau_strain` (tau) agree within tolerance, as expected since tau = sigma * Lz * 0.005 pointwise with Lz fixed within each strain series.

## Interpretation

- `tau_at_zero_n_per_m` / `stress_at_zero_kbar` is the residual in-plane surface stress / QE stress at the relaxed generated geometry (epsilon = 0).
- `zero_tau_strain` / `zero_stress_epsilon` estimate the in-plane strain that would null the surface stress / QE stress; these are the same physical strain expressed via two equivalent linear fits.
- tau (N/m) is vacuum-thickness independent and directly comparable to surface-stress literature; sigma (kbar) depends on the arbitrary supercell vacuum and is retained here only as a diagnostic cross-check.
- Large slopes indicate a stiff slab response to imposed in-plane strain.
- This fit is a more reliable diagnostic than a single stress-SCF point, but final surface stress should also consider ionic relaxation under strain.
