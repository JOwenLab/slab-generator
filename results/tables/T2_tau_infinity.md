# Table T2 - Surface stress per facet (N/m; sigma_res in kbar)

| facet | axes x / y      | tau_xx | tau_yy | tau_mean | tau_aniso | sigma_res | fit rms | level |
| ----- | --------------- | -----: | -----: | -------: | --------: | --------: | ------: | ----- |
| (100) | [110] / [1-10]  | +1.032 | -5.024 |   -1.996 |    +6.056 |    -0.283 |  0.0290 | L2    |
| (110) | [1-10] / [001]  | +2.144 | +4.397 |   +3.271 |    -2.253 |    -0.257 |  0.0014 | L2    |
| (111) | [1-10] / [11-2] | +0.483 | +0.483 |   +0.483 |    +0.000 |    -0.362 |  0.0021 | L2    |

**Table T2.** Thickness-extrapolated surface stress per facet, the one L2 quantity in this chain and the input every later number depends on. tau_xx and tau_yy are the two in-plane axes named in the second column; the axis assignment is not conventional and the anisotropy sign is meaningless without it. POSITIVE tau IS COMPRESSIVE (CLAUDE.md sec 2): it equals MINUS the continuum surface stress f, so a facet with positive tau relaxes by EXPANDING, which was verified directly against free 2D vc-relax on all three surfaces and six axes. sigma_res is the residual interior stress the ladder fit leaves behind, within the 0.3 kbar tolerance on all three facets. The rms column is the larger of the two per-axis fit residuals expressed as a tau, i.e. the fit's own scatter, and is not an estimate of the systematic error in the DFT. (111) is isotropic identically, not numerically: its 3-fold symmetry forbids an in-plane anisotropy.


- Thickness definition: t = N_C * a0^3/8 / A (bulk-equivalent, atom-counted).
- Fit uses 8L 10L 12L 16L; 6L excluded.
- a0 = 3.572997 A at 90/720 Ry.
