# Particle interior strain and NV zero-field splitting

**Epistemic level: L1** (CLAUDE.md sec 4). This is a continuum model with LITERATURE cubic elastic constants (C11 = 1076.0, C12 = 125.0, C44 = 576.0 GPa, source `literature`), no explicit NV defect, no depth model, and published spin-strain parameter sets (`udvarhelyi2018_dft`, `barson2017_scaled`). The defensible outputs are the ranking across shapes and the D/E channel separation, not the absolute MHz values.

## Sign conventions

tau in tau_infinity.csv is in the PROJECT convention, where positive tau is COMPRESSIVE surface stress (the released cell expands; verified against free 2D vc-relax on all three surfaces, six axes). load_taus() negates it once to obtain the continuum surface stress f, positive = TENSILE. sigma_*_project: positive = COMPRESSED (CLAUDE.md sec 2, QE convention). sigma_*_mech: positive = TENSION (continuum convention, used internally and by nv_spin_strain). pressure_gpa: positive = compressive.

## Coupling-set uncertainty

The spin-strain coupling set is the dominant uncertainty in these MHz values: the two published sets differ by ~1.65x on the axial channel for identical strain. delta_D_band_lo/hi span ALL emitted sets for this NV axis and are the honest quantity; any single delta_D_mhz is one corner of that band.

Emitted parameter sets: `udvarhelyi2018_dft`; `barson2017_scaled`. Ratio of the largest to the smallest |Delta D| across sets: **1.65x**.

## The testable prediction: which way the lattice moves

A {100}-dominated particle **contracts**; {110}- and {111}-dominated particles **expand**. The direction alone identifies the dominant facet family.

| dominant facet | mean f (N/m) | lattice strain at R=1.5 nm | direction | da/d(1/R) (A per nm^-1) |
| --- | ---: | ---: | :---: | ---: |
| (100) | +1.996 | -0.2007 % | **contract** | -0.010756 |
| (110) | -3.271 | +0.3289 % | **expand** | +0.017628 |
| (111) | -0.483 | +0.0486 % | **expand** | +0.002605 |

This is the most directly testable output of the whole pipeline, and it is robust twice over:

1. **No spin-strain coupling enters it.** The ~1.65x spread between published coupling sets, which dominates every MHz number below, is simply absent here.
2. **The direction does not depend on the absolute tau scale.** Scaling every tau by a common positive factor changes the magnitude and not the sign. Only the sign of each family's mean f matters, and (100) has the opposite sign to (110) and (111).

Nanodiamond lattice parameter versus particle size is a standard powder-XRD measurement. A size-resolved series therefore reads off the dominant facet family from the SIGN of the shift, and then tests the magnitude against the 1/R slope above. Both comparisons are independent of the couplings.

### This shape (mixture)

* **Lattice strain = +0.1161 %** (linear, tr(eps)/3) at R = 1.500 nm -> **expands**
* **Lattice parameter a = 3.577144 A** vs unstrained a0 = 3.572997 A (delta_a = +0.004147 A)
* Interior pressure = -1.53889 GPa (compressive-positive)

Every quantity here scales as 1/R exactly; the companion `..._size_sweep_*.csv` tabulates that curve.

## Inputs

* Shape: **mixture**, facet area fractions (100) 0.1648, (110) 0.3872, (111) 0.4480
* Effective radius R = 3V/A_total = 1.500 nm
* Surface stress from `results/production/tau_infinity.csv` (epistemic level L2, fitted layers 8L 10L 12L 16L, excluded 6L)
* Surface energies from `config/surface_energies_h.json` (source_type `project_dft_fit`) evaluated at delta_mu_H = 0.0000 eV below the H-rich limit: (100) +0.0501, (110) -0.8265, (111) -0.9761 J/m^2

| family | tau_xx (N/m) | tau_yy (N/m) | surface x | surface y |
| --- | ---: | ---: | --- | --- |
| (100) | -1.0323 | +5.0241 | [110] | [1-10] |
| (110) | -2.1445 | -4.3975 | [1-10] | [001] |
| (111) | -0.4833 | -0.4833 | [1-10] | [11-2] |

## D and E are not equally constrained

Measured by perturbing each coupling by 5% and recording the response (computed here, not asserted):

| coupling | delta_D response | E response |
| --- | ---: | ---: |
| `h41` | -6.09 % | -- |
| `h43` | +1.09 % | -- |
| `h15` | +0.00 % | -- |
| `h16` | -0.00 % | -- |
| `h25` | -0.00 % | -- |
| `h26` | -0.00 % | -- |

The two channels use **disjoint** couplings: `h41`/`h43` set delta_D, `h15`/`h16` set E, and `h25`/`h26` enter only at second order.

For the hydrostatic interior stress of a symmetry-complete particle, delta_D collapses onto the single combination **2*h41 + h43 = -10540.0 MHz/strain** — which is precisely what a hydrostatic-pressure ODMR experiment measures. That is why the axial channel, for all its 1.65x spread between parameter sets, rests on a directly measured quantity.

E does not. `h15`/`h16` are untouched by any hydrostatic measurement, and E carries a second, larger problem:

> **E is identically zero for this shape.** Every symmetry-complete facet set gives a hydrostatic interior stress, and a hydrostatic strain produces no transverse splitting for *any* values of the couplings. The E reported below is zero by symmetry, not by cancellation.

**Consequence for the (100) anisotropy result.** The (100) surface stress anisotropy is real and is the largest in the set, but this model cannot presently turn it into a predicted E: the volume average it computes has no deviatoric part for a symmetric particle. The ranking of surfaces by anisotropy is the defensible output; a predicted E in MHz is not.

### Open question: is the volume average the right object for E?

**Hypothesis, not a result.** The volume-averaged interior stress this module computes is exact for its TRACE — that is a consequence of the divergence theorem and needs no assumption about how stress is distributed inside the particle. The trace is what delta_D responds to, which is why the axial channel is on firm ground.

E responds to the DEVIATORIC part, and the volume average of the deviatoric stress is not the deviatoric stress anywhere in particular. Inside a faceted particle the deviatoric field is not uniform: it varies with position, and near a facet it reflects that facet's own anisotropic tau rather than the orientation average. In a 3 nm particle every NV sits within ~1.5 nm of a surface, so no NV samples the average.

It is therefore possible that solving the elasticity boundary-value problem for the actual polyhedron and evaluating the strain at realistic NV depths yields a non-zero E from computed geometry, with no assumed shape asymmetry anywhere. That would put E on the same footing as delta_D instead of resting on a free parameter.

This is untested. It is recorded here as a question to settle, not as a claim: the boundary-value problem has not been solved, the depth dependence has not been computed, and it is not established that the result is non-zero. Until it is, this module reports E = 0 for symmetric particles and says why.


## Validation: Laplace limit

A sphere with isotropic tau must give a compressive pressure P = 2*tau/R. If it does not, the geometric factors are wrong and nothing below is trustworthy.

* model 1.333333 GPa vs Laplace 1.333333 GPa (relative error 1.67e-16, 2000 directions)
* residual anisotropy of the sphere result: 7.82e-07 GPa diagonal, 1.74e-05 GPa off-diagonal

## Interior stress and strain

Volume-averaged interior stress, continuum convention (positive = tension), GPa:

```
    +1.53889    -0.00000    -0.00000
    -0.00000    +1.53889    -0.00000
    -0.00000    -0.00000    +1.53889
```

Hydrostatic pressure (compressive-positive): **-1.53889 GPa**
In the project/QE convention (positive = compressed) the same tensor is diag(-1.53889, -1.53889, -1.53889) GPa.

Interior strain (dimensionless, negative = compression):

```
    +1.161e-03    +0.000e+00    +0.000e+00
    +0.000e+00    +1.161e-03    -0.000e+00
    +0.000e+00    -0.000e+00    +1.161e-03
```

## NV zero-field splitting

Delta D and E are given as a BAND across the emitted spin-strain parameter sets. The band, not either endpoint, is the result.

| NV axis | eps_axial | Delta D band (MHz) | E band (MHz) |
| --- | ---: | ---: | ---: |
| [111] | +1.161e-03 | -20.232 .. -12.232 | 0.0000 .. 0.0000 |
| [1-1-1] | +1.161e-03 | -20.232 .. -12.232 | 0.0000 .. 0.0000 |
| [-11-1] | +1.161e-03 | -20.232 .. -12.232 | 0.0000 .. 0.0000 |
| [-1-11] | +1.161e-03 | -20.232 .. -12.232 | 0.0000 .. 0.0000 |

### Per parameter set

| param set | NV axis | Delta D (MHz) | E (MHz) | f+ (MHz) | f- (MHz) |
| --- | --- | ---: | ---: | ---: | ---: |
| udvarhelyi2018_dft | [111] | -12.232 | 0.0000 | 2857.77 | 2857.77 |
| udvarhelyi2018_dft | [1-1-1] | -12.232 | 0.0000 | 2857.77 | 2857.77 |
| udvarhelyi2018_dft | [-11-1] | -12.232 | 0.0000 | 2857.77 | 2857.77 |
| udvarhelyi2018_dft | [-1-11] | -12.232 | 0.0000 | 2857.77 | 2857.77 |
| barson2017_scaled | [111] | -20.232 | 0.0000 | 2849.77 | 2849.77 |
| barson2017_scaled | [1-1-1] | -20.232 | 0.0000 | 2849.77 | 2849.77 |
| barson2017_scaled | [-11-1] | -20.232 | 0.0000 | 2849.77 | 2849.77 |
| barson2017_scaled | [-1-11] | -20.232 | 0.0000 | 2849.77 | 2849.77 |

* Delta D over all axes and all parameter sets: **-20.232 .. -12.232 MHz** (1.65x from the coupling set alone)
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

