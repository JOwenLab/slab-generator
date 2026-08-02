# Thickness-extrapolated surface stress (tau_inf)

Epistemic level: **L2** (CLAUDE.md sec 4). Converged production stress SCFs at 90/720 Ry, a0 = 3.572997 Angstrom.

## Conventions

* Stress sign: sigma>0 compressed (CLAUDE.md sec 2); tau = sigma*Lz/2 inherits that, so tau>0 is COMPRESSIVE surface stress: the surface pushes outward and a released cell expands. This equals MINUS the continuum surface stress f (for which positive is tensile); consumers wanting f must negate. Verified against free 2D vc-relax (cell_dofree='2Dxy') at 16L on all three surfaces: positive tau expanded the cell in 6 of 6 axes ((100)[110] +0.139%, (110)[1-10] +0.145%, (110)[001] +0.429%, (111) +0.052% both axes), negative tau contracted it ((100)[1-10] -0.664%)..
* `tau[N/m] = 0.01 * (sigma[kbar] * Lz[Angstrom]) / 2`; the factor 2 is the slab's two surfaces.
* Bulk-equivalent thickness: `t = N_C * a0^3/8 / A (bulk-equivalent, atom-counted)`, with Omega_bulk = 5.701750 Angstrom^3 per carbon.
* `tau_aniso = tau_xx - tau_yy`; its sign is meaningless without the axis assignment given per surface below.

## Fit set

* Included: 8L 10L 12L 16L
* Excluded: 6L — outside the asymptotic regime on all three surfaces (parity oscillation on C100; the outlier in energy slope, surface energy, and tau_aniso).

## Results

| surface | axes (x, y) | tau_xx | tau_yy | tau_mean | tau_aniso | sigma_res | rms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (100) | [110], [1-10] | +1.0323 | -5.0241 | -1.9959 | +6.0564 | -0.283 | 5.795 |
| (110) | [1-10], [001] | +2.1445 | +4.3975 | +3.2710 | -2.2530 | -0.257 | 0.276 |
| (111) | [1-10], [11-2] | +0.4833 | +0.4833 | +0.4833 | +0.0000 | -0.362 | 0.429 |

tau in N/m, sigma_res in kbar, rms in kbar*Angstrom (worse of the two components).

## Per-surface detail

### (100) (C100)

* Surface-frame axes: x = [110], y = [1-10]. 2x1 dimer bond lies along y = [1-10]; dimer rows run along x = [110]
* Layers in fit: 8L 10L 12L 16L; excluded: 6L
* tau_xx = +1.0323 N/m, tau_yy = -5.0241 N/m
* sigma_res = -0.425 (xx) / -0.141 (yy) kbar
* fit rms = 2.032 / 5.795 kbar*Angstrom, equivalent to 0.0102 / 0.0290 N/m on tau
* 6L residual against the fit: +73.205 (xx) / -287.431 (yy) kbar*Angstrom

### (110) (C110)

* Surface-frame axes: x = [1-10], y = [001]. 1x1; x is the zig-zag chain direction
* Layers in fit: 8L 10L 12L 16L; excluded: 6L
* tau_xx = +2.1445 N/m, tau_yy = +4.3975 N/m
* sigma_res = -0.179 (xx) / -0.335 (yy) kbar
* fit rms = 0.126 / 0.276 kbar*Angstrom, equivalent to 0.0006 / 0.0014 N/m on tau
* 6L residual against the fit: +0.139 (xx) / -0.033 (yy) kbar*Angstrom

### (111) (C111)

* Surface-frame axes: x = [1-10], y = [11-2]. 1x1 H-terminated; in-plane isotropic by 3-fold symmetry
* Layers in fit: 8L 10L 12L 16L; excluded: 6L
* tau_xx = +0.4833 N/m, tau_yy = +0.4833 N/m
* sigma_res = -0.362 (xx) / -0.362 (yy) kbar
* fit rms = 0.429 / 0.429 kbar*Angstrom, equivalent to 0.0021 / 0.0021 N/m on tau
* 6L residual against the fit: +3.545 (xx) / +3.545 (yy) kbar*Angstrom

## sigma_res consistency

sigma_res is a bulk property at the reference lattice constant and must agree across surfaces. Its magnitude is a direct diagnostic of a0: a large value means the reference lattice constant is wrong.

* Consistent within the 0.300 kbar tolerance.

## What this is and is not

* This is the surface stress of the H-terminated slab extrapolated to infinite thickness. It is a slab-cell quantity: the interior elastic response has been separated out by the fit, not by an independent bulk calculation.
* It is a clamped-cell, relaxed-ion quantity: ions were relaxed at the fixed bulk in-plane lattice constant. It is not a relaxed-cell surface stress and says nothing about the strain response (CLAUDE.md sec 3).
* The fit is four points and two parameters per component. The residuals above, not the closeness of the numbers to expectation, are the evidence of fit quality.

