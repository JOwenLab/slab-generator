# H-terminated diamond surface energies vs hydrogen chemical potential

gamma = [E_slab - N_C*mu_C - N_H*mu_H] / (2A), per face, from the production thickness ladder.

Epistemic level is per surface (see the table); it is not uniform.

## Chemical potentials

* mu_H(H-rich) = E(H2)/2 = -1.166457483 Ry (E(H2) = -2.332914967 Ry, `results/reference_90_720/H2`)
* mu_C = -18.434076958 Ry, the Birch-Murnaghan E0 per carbon at the fitted equilibrium lattice constant (`results/reference_90_720/bulk_fit_summary.json`). The sampled eps=0 run sits at a0 = 3.567 Angstrom and is NOT used.
* Pseudopotentials: C `C.pbe-n-kjpaw_psl.1.0.0.UPF`, H `H_ONCV_PBE-1.0.oncvpsp.upf`; 90/720 Ry throughout, checked against every slab.

## Surface energies at the H-rich limit

| surface | level | gamma (J/m^2) | scatter | dgamma/d(-mu_H) | N_H/2A | mu_C drift | fit rms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (100) | L1 | +0.0501 | 0.0241 | 2.5100 | 0.1567 | -0.0524 mRy | 1151.68 uRy |
| (110) | L2 | -0.8265 | 0.0003 | 3.5497 | 0.2216 | +0.0017 mRy | 0.19 uRy |
| (111) | L2 | -0.9761 | 0.0006 | 2.8983 | 0.1809 | +0.0032 mRy | 5.41 uRy |

gamma in J/m^2; dgamma/d(-mu_H) in (J/m^2)/eV; N_H/2A in Angstrom^-2. `scatter` is the spread of the per-slab gamma across the fitted ladder using the independent bulk mu_C, and is the honest uncertainty on each number.

## Dependence on mu_H

Lowering mu_H below the H-rich limit raises every gamma, at a rate set by that facet's hydrogen coverage N_H/2A. Because the rates differ, the stability ORDERING is a function of mu_H, not a fixed fact.

Scanned range: delta_mu = mu_H(H-rich) - mu_H from 0.00 to 3.00 eV. The lower end of that range is a CONVENTION, not a computed bound: fixing the H-poor limit needs a CH4 or graphite+H2 reference, which this repository does not have.

### Where each gamma turns positive

* C100: delta_mu = -0.0200 eV (outside the scanned range)
* C110: delta_mu = +0.2328 eV (inside the scanned range)
* C111: delta_mu = +0.3368 eV (inside the scanned range)

**A Wulff construction exists only for delta_mu >= 0.3368 eV**, where all three gammas are positive.

### Where the ordering changes

* C100 vs C110: occurs at delta_mu = +0.8432 eV
* C100 vs C111: occurs at delta_mu = +2.6429 eV
* C110 vs C111: would occur at (UNPHYSICAL: needs mu_H ABOVE the H-rich limit, where H2 condenses) delta_mu = -0.2297 eV

### Stability ordering across the range

| delta_mu (eV) | ordering (most stable first) | all gamma > 0 |
| ---: | --- | --- |
| 0.0000 | C111 < C110 < C100 | no |
| 0.1164 | C111 < C110 < C100 | no |
| 0.2328 | C111 < C110 < C100 | no |
| 0.2848 | C111 < C110 < C100 | no |
| 0.3368 | C111 < C110 < C100 | yes |
| 0.5900 | C111 < C110 < C100 | yes |
| 0.8432 | C111 < C100 < C110 | yes |
| 1.7430 | C111 < C100 < C110 | yes |
| 2.6429 | C111 < C100 < C110 | yes |
| 2.8215 | C100 < C111 < C110 | yes |
| 3.0000 | C100 < C111 < C110 | yes |

## Per-slab cross-check

gamma computed for each individual slab using the independent bulk mu_C rather than the fitted slope. Constancy across the ladder is the evidence that the interior is bulk-like.

* (100): 6L +0.2917  8L +0.0333  10L +0.0575  12L +0.0342  16L +0.0343
* (110): 6L -0.8262  8L -0.8262  10L -0.8261  12L -0.8260  16L -0.8259
* (111): 6L -0.9744  8L -0.9755  10L -0.9756  12L -0.9755  16L -0.9750

## Caveats

* The H-poor end of the mu_H range is not bounded by any calculation here. Statements about very hydrogen-poor conditions are extrapolations of a straight line, nothing more.
* gamma is a fixed-cell, relaxed-ion quantity at the bulk in-plane lattice constant.
* Zero-point and finite-temperature contributions to mu_H are omitted entirely; at room temperature and realistic H2 partial pressures these shift mu_H by a few tenths of an eV, which is the same scale as the crossings above.
* k-meshes differ between the slabs (9x5x1, 9x7x1, 9x9x1) and the bulk reference (8x8x8). The fitted-slope construction cancels the leading effect of that mismatch; it does not cancel it exactly.

## Warnings

* C100: fitted mu_C = -18.434129349 Ry drifts -0.0524 mRy from the bulk reference -18.434076958 Ry. The slab interior is not perfectly bulk-like, or the ladder is not in the asymptotic regime. gamma for this surface is downgraded to L1; per-slab scatter is 0.0241 J/m^2.

