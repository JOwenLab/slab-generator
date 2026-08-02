# Bulk Diamond Reference — Equation-of-State Fit

## Project Context

This report summarises the equation-of-state analysis of the PBE/SSSP
bulk diamond reference series computed with Quantum ESPRESSO `pw.x`.
A 5-point hydrostatic strain series (ε = -0.0100 … +0.0100) was fitted
to extract the equilibrium lattice constant and bulk modulus.  These
values define the zero-strain baseline for downstream slab
surface-energy, surface-stress, Raman-shift, and NV-centre analyses.

**Method.** The primary fit is a 3rd-order Birch-Murnaghan EOS on E(V).
A quadratic P(ε) fit is reported as an independent cross-check.  Both
are shown; disagreement is flagged rather than silently resolved.
A *linear* P(ε) fit — used by this script before 2026-08 — is biased,
because P(ε) is strongly curved over ±1 %; it is retained below under
'Superseded fits' so the size of that bias stays visible.

**ε is computed from the cell volume**, ε = (V/V_ref)^(1/3) − 1, not
read from the run folder name (CLAUDE.md §1: read the geometry).  The
reference is V_ref = 45.3856 Å³, a_ref = 3.56702 Å.  The
fitted a₀ does not depend on that choice; only the reported ε₀ does.

## Input Data

| folder | ε (from V) | a (Å) | V (Å³) | E (Ry) | P (kbar) |
|--------|-----------|-------|--------|--------|---------|
| eps_-0.010 | -0.010014 | 3.53131 | 44.0358 | -147.46673674 | 162.53 |
| eps_-0.005 | -0.005014 | 3.54914 | 44.7063 | -147.47062646 | 89.94 |
| eps_+0.000 | +0.000000 | 3.56702 | 45.3856 | -147.47236441 | 21.55 |
| eps_+0.005 | +0.004999 | 3.58486 | 46.0698 | -147.47203546 | -42.84 |
| eps_+0.010 | +0.009999 | 3.60269 | 46.7607 | -147.46972352 | -103.41 |

## 1. PRIMARY — 3rd-order Birch-Murnaghan on E(V)

BM3 is exactly a cubic polynomial in x = V^(−2/3), so this is a linear
least-squares fit; V₀, B₀ and B₀′ follow analytically from the
polynomial's derivatives at its minimum.

| Parameter | Value |
|-----------|-------|
| V₀ (Å³) | 45.61398 |
| **a₀ (Å)** | **3.572997** |
| **B₀ (GPa)** | **433.67** |
| B₀′ | 3.765 |
| E₀ (Ry) | -147.47248042 |
| fit residual rms (Ry) | 8.624e-07 |
| V₀ inside sampled range | True |

**Stability of the 4th parameter.** With 5 points a BM3 fit has one
residual degree of freedom, so B₀′ is the least-constrained quantity.
Refitting with B₀′ fixed at 4 (2nd-order BM, 3 parameters) gives:

| | a₀ (Å) | B₀ (GPa) | rms (Ry) |
|---|--------|----------|----------|
| BM3 (B₀′ free = 3.765) | 3.572997 | 433.67 | 8.624e-07 |
| BM2 (B₀′ ≡ 4) | 3.572965 | 433.14 | 2.233e-06 |
| difference | 0.000032 | 0.53 | — |

## 2. CROSS-CHECK — quadratic P(ε)

B from the cross-check uses V = V_ref(1+ε)³, so B = −(1/3) dP/dε|₍ε₀₎.

| Parameter | Value |
|-----------|-------|
| ε₀ | +0.001634 |
| a₀ (Å) | 3.572855 |
| B (GPa) | 434.06 |
| dP/dε at ε₀ (kbar) | -13021.8 |
| fit residual rms (kbar) | 0.0884 |

## 3. Method agreement

| Quantity | BM3 E(V) | quadratic P(ε) | |Δ| |
|----------|----------|----------------|-----|
| a₀ (Å) | 3.572997 | 3.572855 | 0.000142 |
| B (GPa) | 433.67 | 434.06 | 0.39 |

**Status: CONSISTENT**

## 4. Superseded fits (biased — recorded, not used)

| Fit | ε₀ | a₀ (Å) | B (GPa) | rms |
|-----|----|--------|---------|-----|
| P(ε) **linear** | +0.001918 | 3.573866 | — | 3.329 kbar |
| P(ε) quadratic (cross-check above) | +0.001634 | 3.572855 | 434.06 | 0.0884 kbar |
| E(V) **quadratic** minimum | — | 3.573641 | 445.12 (from linear P(V)) | — |
| E(ε) quadratic minimum | +0.001779 | 3.573371 | — | — |

The linear P(ε) rms is 3.33 kbar against 0.088 kbar for the quadratic — a factor of 38. That residual structure is the curvature the linear fit cannot
represent, and it biases ε₀ by 0.000283 in strain.

## Interpretation

The recommended **PBE/SSSP bulk reference lattice constant** is
**a₀ = 3.572997 Å** (3rd-order Birch-Murnaghan on E(V), V₀ = 45.6140 Å³), with the quadratic P(ε)
cross-check giving 3.572855 Å.

The bulk modulus is **B₀ = 433.67 GPa** with B₀′ = 3.77; the cross-check gives 434.06 GPa.

a_ref = 3.56702 Å is smaller than the fitted a₀, i.e. the input lattice constant sits on the compressed side of the PBE/SSSP equilibrium; removing that residual would need ~0.167% isotropic expansion.

B₀ is in the expected range for PBE diamond (~430–445 GPa depending on EOS form and strain window).

## Warnings

None.
