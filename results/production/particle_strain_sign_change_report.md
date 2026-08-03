# Interior-pressure sign change versus hydrogen chemical potential

**Epistemic level: L1.** Continuum model, literature elastic constants, no explicit NV defect. See `particle_strain_report_*.md` for the full caveat list.

## The prediction

The equilibrium (Wulff) habit is a function of the hydrogen chemical potential. Lowering mu_H raises every surface energy, but at facet-dependent rates, so the {100} area fraction grows. (100)-H carries a net TENSILE surface stress (f = +2.0 N/m) while (111)-H carries a compressive one (f = -0.48 N/m), so growing the {100} fraction drives the particle interior from tension towards compression -- and through zero.

At the crossing the ODMR shift changes sign: Delta D passes through zero and reverses. That is a far sharper experimental signature than any single magnitude this model produces, because it does not depend on the spin-strain coupling set, on the particle radius, or on the absolute size of tau -- only on the shape at which the two facet contributions balance.

## Where it happens

### delta_mu_H = 1.295 eV (tension to compression)

Equilibrium shape at the crossing: {100} 0.195, {111} 0.805

| p_H2 (bar) | p_H2 (Torr) | T (K) | T (C) | ZPE-corrected T (C) | ZPE shift (K) | ideal-gas resid (K) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e+00 | 7.5e+02 | 1651 | 1378 | 1164 | -214 | 8 |
| 1e-03 | 7.5e-01 | 1246 | 973 | 805 | -168 | 6 |
| 1e-06 | 7.5e-04 | 994 | 721 | 583 | -138 | 5 |
| 1e-09 | 7.5e-07 | 824 | 550 | 435 | -116 | 4 |
| 1e-12 | 7.5e-10 | 702 | 428 | 328 | -100 | 4 |

**The ZPE column is the honest number, and the ZPE shift is the dominant uncertainty** -- see below. The ideal-gas residual is roughly thirty times smaller and is not the limiting term.

## What is computed and what is thermodynamics

* **Computed here (DFT):** DFT (this project): E(H2), slab total energies, hence gamma(H-rich) and its slope N_H/2A; tau per facet; the continuum interior stress and its sign.
* **Not computed (textbook):** ideal-gas statistical thermodynamics with literature H2 spectroscopic constants: the entire T and p dependence (rigid rotor with explicit level sum, harmonic oscillator, no ZPE, no anharmonicity).

The ideal-gas model reproduces the NIST-JANAF tabulation for H2 to 12 meV over 298-2000 K, which is the `ideal-gas resid` column above (a few K). That is NOT the dominant uncertainty.

## The dominant uncertainty: missing zero-point energy

gamma was built from DFT total energies with no vibrational term on either side. Restoring zero-point energy consistently adds N_H*ZPE_ads to E_slab and ZPE(H2)/2 to mu_H, which is algebraically a rigid shift of the delta_mu axis by

    DELTA_ZPE = ZPE_ads(per H) - ZPE(H2)/2 ~ 0.198 eV

using literature monohydride C-H frequencies (stretch ~2900 cm^-1, two bends ~1250 cm^-1) against ZPE(H2)/2 = 0.136 eV. Because the C-H zero-point energy is nearly facet-independent, it enters through N_H/2A exactly as mu_H does: the SHAPE at a given delta_mu is unchanged, but the (T, p) needed to reach it moves.

DELTA_ZPE is positive, so the correction moves every crossing to LOWER temperature -- the effect is more accessible than the uncorrected numbers say, not less. The shift is pressure dependent (d(delta_mu)/dT carries a (k/2)ln(p0/p) term), which is why the column above varies with pressure.

It is QUOTED, NOT APPLIED. Applying the H2 side alone would be an unbalanced correction, and the adsorbed-H side needs a phonon calculation. `make_h_phonons.py` generates that campaign and `analyze_h_phonons.py` turns it into a computed DELTA_ZPE, at which point this systematic becomes a number rather than an estimate.

The DFT surface energies and tau carry their own uncertainties, not included in any column here; (100)'s gamma alone scatters by 0.024 J/m^2.

## What would falsify this, and what could stop it happening

* **The H-terminated surface is assumed stable at every mu_H.** Only H-terminated facets were calculated, so nothing in this model stops gamma_H rising indefinitely. In reality the surface dehydrogenates or reconstructs once gamma_H exceeds the bare or reconstructed surface energy, and that bound CANNOT be computed from this data set -- it needs bare and reconstructed facet calculations. This is the largest single caveat, and it bites hardest at exactly the hydrogen-poor end where the crossing sits.
* **The CH4 / graphite+H2 bound on mu_H is still missing** and is deliberately not invented here.
* **Kinetics are absent.** Nanodiamond surfaces graphitize on vacuum annealing. A (T, p) point being thermodynamically reachable does not mean the H-terminated particle survives the anneal, and graphitization is not in this picture at all.
* **The particle is assumed to re-equilibrate its shape.** A real particle whose habit is kinetically frozen will not follow the Wulff locus, in which case the sign change tracks whatever the actual facet fractions are rather than the equilibrium ones.
* The (100) surface energy is L1 with 0.024 J/m^2 of scatter (`surface_energy.py`), and the crossing position depends on it.

