# Slab Strain Fit Summary

Fits in-plane slab stress response from strain-series SCF calculations.

Model:

```text
mean_sigma_kbar(epsilon) = m * epsilon + b
```

| Series | Orient. | Mode | Points | Stress slope (kbar/strain) | Stress at zero (kbar) | Zero-stress eps | R^2 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C111_1x1_H_6L | (111) | biaxial | 5 | -3725.80 | -1.572 | -0.000422 | 0.99924 | good |

## Interpretation

- `stress_at_zero_kbar` is the residual in-plane mean stress at the relaxed generated geometry.
- `zero_stress_epsilon` estimates the in-plane strain that would null the mean slab stress.
- Large stress slopes indicate a stiff slab response to imposed in-plane strain.
- This fit is a more reliable stress diagnostic than a single stress-SCF point, but final surface stress should also consider ionic relaxation under strain.
