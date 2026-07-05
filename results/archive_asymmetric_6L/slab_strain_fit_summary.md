# Slab Strain Fit Summary

Fits in-plane slab stress response from strain-series SCF calculations.

Model:

```text
mean_sigma_kbar(epsilon) = m * epsilon + b
```

| Series | Orient. | Mode | Points | Stress slope (kbar/strain) | Stress at zero (kbar) | Zero-stress eps | R^2 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C100_2x1_H_6L | (100) | biaxial | 8 | -2658.28 | -38.386 | -0.014440 | 0.99814 | good |
| C100_2x1_H_6L | (100) | x | 5 | -1487.00 | -38.836 | -0.026117 | 0.99961 | warning |
| C100_2x1_H_6L | (100) | y | 5 | -1070.90 | -38.898 | -0.036323 | 0.99957 | warning |
| C110_1x1_H_6L_SSSP | (110) | biaxial | 5 | -3962.60 | -7.593 | -0.001916 | 0.99919 | good |
| C110_1x1_H_6L_SSSP | (110) | x | 5 | -2215.80 | -8.162 | -0.003684 | 0.99957 | good |
| C110_1x1_H_6L_SSSP | (110) | y | 5 | -1744.10 | -8.354 | -0.004790 | 0.99982 | good |
| C111_1x1_H_6L | (111) | biaxial | 5 | -3725.80 | -1.572 | -0.000422 | 0.99924 | good |

## Interpretation

- `stress_at_zero_kbar` is the residual in-plane mean stress at the relaxed generated geometry.
- `zero_stress_epsilon` estimates the in-plane strain that would null the mean slab stress.
- Large stress slopes indicate a stiff slab response to imposed in-plane strain.
- This fit is a more reliable stress diagnostic than a single stress-SCF point, but final surface stress should also consider ionic relaxation under strain.
