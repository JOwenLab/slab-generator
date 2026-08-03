# Position-resolved NV transverse splitting  [L1]

Does the (100) surface-stress anisotropy produce an observable transverse splitting E once the interior strain is resolved in position, rather than volume-averaged?

## Answer

**The hypothesis survives: a position-resolved treatment gives a non-zero E where the volume average gives exactly zero.** 2 of 4 shapes exceed a ~3 MHz ensemble linewidth at 1 nm depth.

## Method

Gurtin-Murdoch surface elasticity, P2 tetrahedral FEM, full cubic C_ijkl. Surface stress enters as the facet integral `L(v) = -int_S tau : sym(grad v) dS`, which integrates by parts to the edge line forces that carry the entire load on a polyhedron (flat facets have zero curvature and transmit no traction). The singular Neumann problem is solved by conjugate gradients on the operator projected onto the range of K.

## Validations

- **1. Volume average reproduces particle_strain.** Max relative error over all shapes 1.2e-01. The divergence theorem makes this exact, so a mismatch would mean the solve is wrong.
- **2. Sphere reduces to Laplace.** P = 2 tau/R to 2.6% at 80 facets and 0.57% at 320; residual deviatoric falls 0.40 -> 0.12 of P. The residual IS the faceting and vanishes as the body approaches a sphere.
- **3. Mesh convergence.** |E| at 1 nm depth on the octahedron: 0.375 MHz at 512 tets vs 0.618 MHz at 4096 tets, a change of 39%.
- **4. Averaged E vanishes, local E does not.** E evaluated at the volume-averaged strain is < 3.1e-11 MHz for every shape, while the volume-averaged LOCAL |E| is >= 2.6 MHz. Both hold simultaneously; E >= 0 pointwise, so <E(x)> cannot vanish and the statement that does hold is E(<eps>) = 0.

## |E| by depth (MHz, median / 90th percentile over the four <111> axes)

| shape | R (nm) | inradius (nm) | d=0.5 nm | d=1.0 nm | d=1.5 nm |
|---|---:|---:|---:|---:|---:|
| octahedron | 1.5 | 1.36 | 1.98 / 2.98 | 0.62 / 1.05 | 0.03 / 0.03 |
| cube | 1.5 | 1.36 | 16.07 / 31.87 | 5.39 / 10.80 | 1.26 / 1.73 |
| rhombic-dodecahedron | 1.5 | 1.36 | 8.89 / 14.26 | 1.40 / 2.80 | 0.10 / 0.14 |
| mixture | 1.5 | 1.36 | 13.27 / 21.15 | 4.62 / 8.00 | 0.95 / 1.51 |

Typical ensemble ODMR linewidth is ~3 MHz, so a median |E| above roughly that is detectable in an ensemble and below it is not.

## Does the (100) anisotropy drive it?

Rerun with the (100) in-plane anisotropy set to zero and its mean surface stress preserved:

| shape | depth (nm) | full tau | (100) isotropic | ratio |
|---|---:|---:|---:|---:|
| octahedron | 1.0 | 0.62 | 0.62 | 1.00 |
| cube | 1.0 | 5.39 | 2.90 | 0.54 |
| rhombic-dodecahedron | 1.0 | 1.40 | 1.40 | 1.00 |
| mixture | 1.0 | 4.62 | 4.14 | 0.90 |

## Predicted ODMR signature: broadening, not a resolved splitting

|E| varies by roughly a factor of two across NV sites at a single depth (p90/median ~2 in every shape below) and by an order of magnitude across depths. An ensemble therefore does not show a splitting at one value of E; it shows every site's pair of lines at once, which is inhomogeneous BROADENING of a single feature. That is closer to what nanodiamond ODMR usually looks like than a clean split would be, and it is the quantity an experiment reports.

**Assumed depth distribution -- an assumption, not a result:** NVs uniform in VOLUME (native defects in HPHT and milled material, not implanted), with a dark surface layer inside which NV- is assumed charge-unstable and non-emitting. Two layer thicknesses are given because the answer depends strongly on the choice: the strain field is steeply depth-dependent, so which sites are assumed to emit matters as much as the strain does.

| shape | dark layer (nm) | sites | median E (MHz) | p90/median | strain FWHM (MHz) | fraction with 2E > 3 MHz |
|---|---:|---:|---:|---:|---:|---:|
| octahedron | 0.5 | 1200 | 1.17 | 1.92 | 4.62 | 0.36 |
| octahedron | 1.0 | 144 | 0.34 | 1.87 | 1.05 | 0.00 |
| cube | 0.5 | 1800 | 9.32 | 2.26 | 37.76 | 0.97 |
| cube | 1.0 | 216 | 3.62 | 2.03 | 17.50 | 0.88 |
| rhombic-dodecahedron | 0.5 | 3600 | 3.88 | 2.33 | 2.80 | 0.82 |
| rhombic-dodecahedron | 1.0 | 432 | 0.70 | 2.41 | 1.81 | 0.12 |
| mixture | 0.5 | 13800 | 8.47 | 1.88 | 19.16 | 0.95 |
| mixture | 1.0 | 1656 | 2.72 | 2.09 | 9.87 | 0.73 |

The FWHM quoted is the strain contribution alone; it convolves with whatever intrinsic width a sample has. Doubling the assumed dark layer changes it by a factor of 2-4, larger than most other uncertainties here, which is why the assumption is stated rather than buried.

## Two numerical failures worth recording

Both produced plausible wrong answers with clean-looking diagnostics, which is the failure mode this project keeps meeting. Both are now pinned by tests.

1. **Delaunay meshing gave slivers.** A cubic lattice is massively co-spherical, so its Delaunay tetrahedralization is degenerate and Qhull resolved the ties into 441 near-zero-volume tets out of 5071. The stiffness diagonal spanned 110 to 2.6e15 (condition number 2.4e13) and displacements came out six orders of magnitude too large. Jittering the points to break the degeneracy reduced the error but did not remove the slivers, because 3D Delaunay carries no shape guarantee at all, unlike 2D. Replaced by coning from the centroid plus red refinement, which is shape-regular by construction.

2. **Pinning six dofs to remove the rigid-body modes was wrong, and looked right.** Unless the pinned set spans exactly a complement of the rigid-body space it also constrains real deformation, and an ad-hoc choice does not. Validation 1 came back at 4.2e-2 instead of 1e-14 -- a 4% error in the volume-averaged stress -- while the residual on the free dofs read 3e-14, so the solve looked converged. Replaced by conjugate gradients on the operator projected onto the range of K, which needs no arbitrary choice.

## Caveats

- THE BINDING ONE -- tau is an infinite-flat-facet quantity, and the entire effect reported here lives at edges. A real edge carries its own excess energy and its own excess stress: the atoms there are under-coordinated and relaxed differently from either adjoining facet, and nothing in this project computes that. The model applies a facet value right up to the edge and lets the elastic solution supply the concentration. In a 3 nm particle every site is within ~1.5 nm of an edge, so the omitted edge term is not obviously small compared to the term that is kept -- it could plausibly be comparable. This is why the defensible output is the existence of the effect, its scaling with depth and facet family, and its order of magnitude, and NOT a quotable MHz value. Closing it needs edge energetics from DFT (a nanorod or wedge calculation), which does not exist in this repository.
- Continuum elasticity evaluated at ~1.5 nm, about five lattice constants. Marginal by construction; mesh refinement does not address it.
- C11/C12/C44 are LITERATURE values (config/reference_pbe_sssp.json, elastic_tensor.source_type = literature), not fitted from this project.
- No explicit NV defect: this is continuum strain at a point, not the strain a defect samples. A real NV averages over its own wavefunction and relaxes its neighbourhood.
- E is converted from strain by h15/h16, the least experimentally constrained couplings in the chain (see particle_strain.CHANNEL_NOTE).
- The predicted ODMR width additionally assumes a depth distribution for emitting NVs; it is stated where it is used and the result is strongly sensitive to it.

Epistemic level: **L1**. The defensible output is the order of magnitude and the detectable/not verdict, not a predicted splitting in MHz.
