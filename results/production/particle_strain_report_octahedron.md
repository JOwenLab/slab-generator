# Particle interior strain and NV zero-field splitting

**Epistemic level: L1** (CLAUDE.md sec 4). This is a continuum model with LITERATURE cubic elastic constants (C11 = 1076.0, C12 = 125.0, C44 = 576.0 GPa, source `literature`), no explicit NV defect, no depth model, and published spin-strain parameter sets (`udvarhelyi2018_dft`, `barson2017_scaled`). The defensible outputs are the ranking across shapes and the D/E channel separation, not the absolute MHz values.

## Sign conventions

tau in tau_infinity.csv is in the PROJECT convention, where positive tau is COMPRESSIVE surface stress (the released cell expands; verified against free 2D vc-relax on all three surfaces, six axes). load_taus() negates it once to obtain the continuum surface stress f, positive = TENSILE. sigma_*_project: positive = COMPRESSED (CLAUDE.md sec 2, QE convention). sigma_*_mech: positive = TENSION (continuum convention, used internally and by nv_spin_strain). pressure_gpa: positive = compressive.

## Coupling-set uncertainty

The spin-strain coupling set is the dominant uncertainty in these MHz values: the two published sets differ by ~1.65x on the axial channel for identical strain. delta_D_band_lo/hi span ALL emitted sets for this NV axis and are the honest quantity; any single delta_D_mhz is one corner of that band.

Emitted parameter sets: `udvarhelyi2018_dft`; `barson2017_scaled`. Ratio of the largest to the smallest |Delta D| across sets: **1.65x**.

## Inputs

* Shape: **octahedron**, facet area fractions (111) 1.0000
* Effective radius R = 3V/A_total = 1.500 nm
* Surface stress from `results/production/tau_infinity.csv` (epistemic level L2, fitted layers 8L 10L 12L 16L, excluded 6L)

| family | tau_xx (N/m) | tau_yy (N/m) | surface x | surface y |
| --- | ---: | ---: | --- | --- |
| (111) | -0.4833 | -0.4833 | [1-10] | [11-2] |

## Validation: Laplace limit

A sphere with isotropic tau must give a compressive pressure P = 2*tau/R. If it does not, the geometric factors are wrong and nothing below is trustworthy.

* model 1.333333 GPa vs Laplace 1.333333 GPa (relative error 1.67e-16, 2000 directions)
* residual anisotropy of the sphere result: 7.82e-07 GPa diagonal, 1.74e-05 GPa off-diagonal

## Interior stress and strain

Volume-averaged interior stress, continuum convention (positive = tension), GPa:

```
    +0.64445    -0.00000    -0.00000
    -0.00000    +0.64445    -0.00000
    -0.00000    -0.00000    +0.64445
```

Hydrostatic pressure (compressive-positive): **-0.64445 GPa**
In the project/QE convention (positive = compressed) the same tensor is diag(-0.64445, -0.64445, -0.64445) GPa.

Interior strain (dimensionless, negative = compression):

```
    +4.860e-04    +0.000e+00    +0.000e+00
    +0.000e+00    +4.860e-04    -0.000e+00
    +0.000e+00    -0.000e+00    +4.860e-04
```

## NV zero-field splitting

Delta D and E are given as a BAND across the emitted spin-strain parameter sets. The band, not either endpoint, is the result.

| NV axis | eps_axial | Delta D band (MHz) | E band (MHz) |
| --- | ---: | ---: | ---: |
| [111] | +4.860e-04 | -8.473 .. -5.123 | 0.0000 .. 0.0000 |
| [1-1-1] | +4.860e-04 | -8.473 .. -5.123 | 0.0000 .. 0.0000 |
| [-11-1] | +4.860e-04 | -8.473 .. -5.123 | 0.0000 .. 0.0000 |
| [-1-11] | +4.860e-04 | -8.473 .. -5.123 | 0.0000 .. 0.0000 |

### Per parameter set

| param set | NV axis | Delta D (MHz) | E (MHz) | f+ (MHz) | f- (MHz) |
| --- | --- | ---: | ---: | ---: | ---: |
| udvarhelyi2018_dft | [111] | -5.123 | 0.0000 | 2864.88 | 2864.88 |
| udvarhelyi2018_dft | [1-1-1] | -5.123 | 0.0000 | 2864.88 | 2864.88 |
| udvarhelyi2018_dft | [-11-1] | -5.123 | 0.0000 | 2864.88 | 2864.88 |
| udvarhelyi2018_dft | [-1-11] | -5.123 | 0.0000 | 2864.88 | 2864.88 |
| barson2017_scaled | [111] | -8.473 | 0.0000 | 2861.53 | 2861.53 |
| barson2017_scaled | [1-1-1] | -8.473 | 0.0000 | 2861.53 | 2861.53 |
| barson2017_scaled | [-11-1] | -8.473 | 0.0000 | 2861.53 | 2861.53 |
| barson2017_scaled | [-1-11] | -8.473 | 0.0000 | 2861.53 | 2861.53 |

* Delta D over all axes and all parameter sets: **-8.473 .. -5.123 MHz** (1.65x from the coupling set alone)
* Largest |E| across the four NV orientations and all parameter sets: **0.0000 MHz**
* Largest spread in Delta D across orientations within one parameter set: 0.0000 MHz

* `udvarhelyi2018_dft`: Udvarhelyi et al., PRB 98, 075201 (2018), Table I
* `barson2017_scaled`: Udvarhelyi tensor rescaled to Barson et al., Nano Lett. 17, 1496 (2017) axial/transverse magnitudes

## Symmetry check

* Every facet of every family carries equal area, so the shape has the full Td symmetry of the crystal. Summing an anisotropic facet tau over a full Td orbit gives a Td-invariant tensor, and the only Td-invariant symmetric rank-2 tensor is isotropic. The interior stress is therefore purely hydrostatic and **E = 0 exactly** -- as it must be. Note this holds for ANY number of facet families, not only one: mixing (100) and (111) does not by itself produce E.
* Nonzero E requires genuinely broken symmetry: unequal areas among the facets of a family (a non-equilibrium shape, specified here with a per-facet `--fraction` override), or an NV shallow enough to see local rather than volume-averaged strain -- which this model does not treat at all.

## What this cannot support

* No explicit NV defect is present anywhere in this chain. D and E come from a published spin-strain tensor applied to a volume-averaged continuum strain.
* The elastic constants are literature values, not fitted from this project's DFT (CLAUDE.md sec 3). Every number above inherits that.
* Facet tau values are infinite-plane quantities. Edges, corners, and facet-size effects are not modelled and are not obviously small for a particle of this size.
* `layer_profile.py` shows the surface perturbation decays within a few Angstrom, which is what licenses a uniform interior strain for a particle this size. It does not license applying that average to a near-surface NV.

