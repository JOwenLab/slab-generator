# Depth profile of surface relaxation

Geometric profile: **L2**. Exponential decay length: **L1** (few-point fit; see caveats). CLAUDE.md sec 4.

This answers how deep the surface perturbation reaches, and therefore whether the interior of a nanoparticle may be treated as uniformly strained bulk in `particle_strain.py`.

**Scope.** Out-of-plane profile only. A periodic slab has a uniform in-plane lattice by construction, so there is no depth-dependent in-plane strain in these calculations and none is reported.

## Summary

| surface | run | layers | eps_zz (surface gap) | decay length | penetration (<0.1%) | max rumpling |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| (100) | `thick_a0corr~C100_16L` | 16 | -9.211% | 0.65 A | 3.95 A | 0.1864 A |
| (110) | `thick_a0corr~C110_16L` | 16 | -2.228% | 0.54 A | 3.13 A | 0.0000 A |
| (111) | `final~C111_24L` | 24 | -5.376% | 0.62 A | 3.33 A | 0.0000 A |

## Per-surface detail

### (100) — `thick_a0corr~C100_16L`

* 32 C (2/layer) + 4 H, cell area 12.7663 A^2, Lz 23.4329 A
* Bulk reference C-C bond 1.54715 A at a0 = 3.572997 A; largest layer-mean deviation +2.349%
* Largest intralayer rumpling 0.18638 A
* Outermost interlayer gap: eps_zz = -9.211%
* Fold asymmetry between the two halves: 3.98e-15 in eps_zz (should be numerical noise; a nonzero value would break CLAUDE.md invariant 1)
* Noise floor used for the fit: |eps_zz| > 4.08e-04
* Exponential fit over all gaps: lambda = 0.652 A, R^2 = 0.7824, 4 points (ok)
* Penetration depth (|eps_zz| stays below 0.10%): 3.948 A

### (110) — `thick_a0corr~C110_16L`

* 32 C (2/layer) + 4 H, cell area 9.0271 A^2, Lz 28.7286 A
* Bulk reference C-C bond 1.54715 A at a0 = 3.572997 A; largest layer-mean deviation +1.023%
* Largest intralayer rumpling 0.00000 A
* Outermost interlayer gap: eps_zz = -2.228%
* Fold asymmetry between the two halves: 7.92e-09 in eps_zz (should be numerical noise; a nonzero value would break CLAUDE.md invariant 1)
* Noise floor used for the fit: |eps_zz| > 1.13e-04
* Exponential fit over all gaps: lambda = 0.539 A, R^2 = 1.0000, 3 points (ok)
* Penetration depth (|eps_zz| stays below 0.10%): 3.133 A

### (111) — `final~C111_24L`

* 24 C (1/layer) + 2 H, cell area 5.5280 A^2, Lz 33.3873 A
* Bulk reference C-C bond 1.54715 A at a0 = 3.572997 A; largest layer-mean deviation +0.583%
* Largest intralayer rumpling 0.00000 A
* Outermost interlayer gap: eps_zz = -5.376%
* Fold asymmetry between the two halves: 6.89e-15 in eps_zz (should be numerical noise; a nonzero value would break CLAUDE.md invariant 1)
* Noise floor used for the fit: |eps_zz| > 8.42e-04
* Exponential fit over all gaps: lambda = 0.622 A, R^2 = 0.9202, 3 points (ok)
* Class-resolved fit — gap class 0 (ideal 0.5157 A): lambda = 0.621 A, R^2 = 1.0000, 2 points (two points only: lambda is exact, not fitted)
* Class-resolved fit — gap class 1 (ideal 1.5472 A): lambda = nan A, R^2 = nan, 1 points (too few points above the noise floor (1))
* Penetration depth (|eps_zz| stays below 0.10%): 3.331 A

## Caveats on the decay length

* On (111) the interlayer gaps alternate between two ideal classes (a0*sqrt3/12 and a0*sqrt3/4) whose relaxations differ in both sign and magnitude. A single exponential through all gaps is an envelope, not a law; the class-resolved fits above are the more meaningful numbers, and the penetration depth is convention-free.
* The fits use only the points above the noise floor, which is a handful. R^2 measures how well a line fits those few log values; it is not evidence that the decay is exponential (CLAUDE.md sec 4).
* The profile is a fixed-cell, relaxed-ion geometry at the bulk in-plane lattice constant. It is not a relaxed-cell result.

## Consequence for the continuum model

* The deepest penetration across the three surfaces is 3.95 Angstrom.
* A 3 nm particle has roughly 15 Angstrom of half-thickness, so the perturbed shell occupies a small fraction of the radius and the core is bulk-like. A uniform interior strain in `particle_strain.py` is therefore a defensible leading approximation for particles of this size and larger.
* It is NOT defensible for an NV sited within the perturbed shell itself: such a centre sees local, not volume-averaged, strain. `particle_strain.py` models the volume average only.

