# Slab Strain Fit Summary

Fits in-plane slab stress response from strain-series SCF calculations.

Model:

```text
mean_sigma_kbar(epsilon) = m * epsilon + b
```

| Series | Orient. | Mode | Points | Stress slope (kbar/strain) | Stress at zero (kbar) | Zero-stress eps | R^2 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C100_2x1_H_6L_sym | (100) | biaxial | 5 | -4299.20 | -21.367 | -0.004970 | 0.99935 | good |
| C100_2x1_H_6L_sym | (100) | x | 5 | -2199.80 | -21.926 | -0.009967 | 0.99960 | good |
| C100_2x1_H_6L_sym | (100) | y | 5 | -2098.00 | -21.935 | -0.010455 | 0.99958 | warning |
| C110_1x1_H_6L_SSSP_sym | (110) | biaxial | 5 | -6028.00 | 38.134 | 0.006326 | 0.99929 | good |
| C110_1x1_H_6L_SSSP_sym | (110) | x | 5 | -3188.30 | 37.316 | 0.011704 | 0.99960 | warning |
| C110_1x1_H_6L_SSSP_sym | (110) | y | 5 | -2835.10 | 37.047 | 0.013067 | 0.99987 | warning |
| C111_1x1_H_6L_sym | (111) | biaxial | 5 | -5335.80 | 4.506 | 0.000844 | 0.99916 | good |

## Interpretation

- `stress_at_zero_kbar` is the residual in-plane mean stress at the relaxed generated geometry.
- `zero_stress_epsilon` estimates the in-plane strain that would null the mean slab stress.
- Large stress slopes indicate a stiff slab response to imposed in-plane strain.
- This fit is a more reliable stress diagnostic than a single stress-SCF point, but final surface stress should also consider ionic relaxation under strain.
