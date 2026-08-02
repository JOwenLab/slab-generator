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

## Translating delta_mu into (T, p_H2)

delta_mu is an abstract axis until it is anchored to conditions a furnace can reach. The mapping below uses the ideal-gas chemical potential of H2:

    delta_mu(T, p) = -[ mu_H2(T,p) - E(H2) ] / 2

with mu_H2 built from the standard translational, rotational and vibrational terms.

**What is computed here versus what is textbook thermodynamics:**

* COMPUTED (this project's DFT): E(H2); the slab energies; hence gamma at the H-rich limit and its slope N_H/2A.
* NOT COMPUTED (ideal-gas statistical thermodynamics with literature spectroscopic constants of H2): every T- and p-dependent term below. Rigid rotor with an explicit level sum, harmonic oscillator, ideal gas. No anharmonicity, no real-gas correction.

The model reproduces the NIST-JANAF tabulation of G(T) - H(0) for H2 to within 12 meV between 298 K and 2000 K, which is the dominant systematic on any temperature quoted below (of order +/-15 K).

| T (K) | this model (eV) | NIST-JANAF (eV) | deviation (meV) |
| ---: | ---: | ---: | ---: |
| 298.15 | -0.3154 | -0.3160 | +0.6 |
| 500.00 | -0.6051 | -0.6065 | +1.4 |
| 1000.00 | -1.4169 | -1.4207 | +3.8 |
| 1500.00 | -2.3092 | -2.3053 | -3.9 |
| 2000.00 | -3.2566 | -3.2690 | +12.4 |

Zero-point energy is EXCLUDED. gamma was derived from DFT total energies with no vibrational term on either side; adding the H2 ZPE (0.136 eV per H) without the adsorbed-H ZPE would be an unbalanced correction, and the adsorbed-H ZPE needs a phonon calculation this repository does not have. The two are of similar size and partially cancel.

### Conditions for each landmark

| landmark | delta_mu (eV) | p_H2 (bar) | p_H2 (Torr) | T (K) | T (C) |
| --- | ---: | ---: | ---: | ---: | ---: |
| wulff_available | 0.337 | 1e+00 | 7.5e+02 | 545 | 272 |
| wulff_available | 0.337 | 1e-03 | 7.5e-01 | 389 | 116 |
| wulff_available | 0.337 | 1e-06 | 7.5e-04 | 299 | 26 |
| wulff_available | 0.337 | 1e-09 | 7.5e-07 | 242 | -31 |
| wulff_available | 0.337 | 1e-12 | 7.5e-10 | 203 | -71 |
| zero C110 | 0.233 | 1e+00 | 7.5e+02 | 406 | 132 |
| zero C110 | 0.233 | 1e-03 | 7.5e-01 | 284 | 11 |
| zero C110 | 0.233 | 1e-06 | 7.5e-04 | 216 | -57 |
| zero C110 | 0.233 | 1e-09 | 7.5e-07 | 173 | -100 |
| zero C110 | 0.233 | 1e-12 | 7.5e-10 | unreachable below 4000 K |  |
| zero C111 | 0.337 | 1e+00 | 7.5e+02 | 545 | 272 |
| zero C111 | 0.337 | 1e-03 | 7.5e-01 | 389 | 116 |
| zero C111 | 0.337 | 1e-06 | 7.5e-04 | 299 | 26 |
| zero C111 | 0.337 | 1e-09 | 7.5e-07 | 242 | -31 |
| zero C111 | 0.337 | 1e-12 | 7.5e-10 | 203 | -71 |
| ordering C100+C110 | 0.843 | 1e+00 | 7.5e+02 | 1155 | 882 |
| ordering C100+C110 | 0.843 | 1e-03 | 7.5e-01 | 858 | 585 |
| ordering C100+C110 | 0.843 | 1e-06 | 7.5e-04 | 677 | 404 |
| ordering C100+C110 | 0.843 | 1e-09 | 7.5e-07 | 557 | 284 |
| ordering C100+C110 | 0.843 | 1e-12 | 7.5e-10 | 472 | 199 |
| ordering C100+C111 | 2.643 | 1e+00 | 7.5e+02 | 3006 | 2733 |
| ordering C100+C111 | 2.643 | 1e-03 | 7.5e-01 | 2327 | 2054 |
| ordering C100+C111 | 2.643 | 1e-06 | 7.5e-04 | 1887 | 1614 |
| ordering C100+C111 | 2.643 | 1e-09 | 7.5e-07 | 1582 | 1309 |
| ordering C100+C111 | 2.643 | 1e-12 | 7.5e-10 | 1359 | 1086 |


## Caveats

* **The H-terminated surface is ASSUMED to remain the stable termination at every mu_H.** Only H-terminated facets were calculated, so gamma_H can be extrapolated to arbitrarily hydrogen-poor conditions without anything stopping it. In reality the surface dehydrogenates or reconstructs once gamma_H rises above the bare or reconstructed surface energy, and that bound cannot be computed from this data set. It is the single largest limitation on everything above, and it bites hardest exactly where delta_mu is large.
* Competing kinetics are not modelled at all. Nanodiamond surfaces graphitize on annealing in vacuum, and that process is not in this thermodynamic picture. A (T, p) point being thermodynamically reachable does not mean the H-terminated diamond surface survives the trip.
* The H-poor end of the mu_H range is not bounded by any calculation here. Statements about very hydrogen-poor conditions are extrapolations of a straight line, nothing more.
* gamma is a fixed-cell, relaxed-ion quantity at the bulk in-plane lattice constant.
* Zero-point and finite-temperature contributions to mu_H are omitted entirely; at room temperature and realistic H2 partial pressures these shift mu_H by a few tenths of an eV, which is the same scale as the crossings above.
* k-meshes differ between the slabs (9x5x1, 9x7x1, 9x9x1) and the bulk reference (8x8x8). The fitted-slope construction cancels the leading effect of that mismatch; it does not cancel it exactly.

## Warnings

* C100: fitted mu_C = -18.434129349 Ry drifts -0.0524 mRy from the bulk reference -18.434076958 Ry. The slab interior is not perfectly bulk-like, or the ladder is not in the asymptotic regime. gamma for this surface is downgraded to L1; per-slab scatter is 0.0241 J/m^2.
* dehydrogenation bound MISSING: no bare-facet ladder found. gamma_H(mu_H) is therefore unbounded above in this model and every hydrogen-poor statement is an extrapolation with no stop. Generate the campaign with make_bare_slabs.py and rerun with --bare-runs-dir.

