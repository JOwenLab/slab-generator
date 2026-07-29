# NV Spin-Strain Predictions

Spin-strain parameters: udvarhelyi2018_dft (Udvarhelyi et al., PRB 98, 075201 (2018), Table I)

## Elastic reference

**Lattice constant:**
  3.573641 Å
  source: project PBE/SSSP bulk fit

**Hydrostatic bulk modulus:**
  445.1 GPa
  source: project PBE/SSSP bulk fit

**Cubic stiffness tensor:**
  C11 = 1076.0 GPa
  C12 = 125.0 GPa
  C44 = 576.0 GPa
  source: literature (Diamond cubic stiffness constants as used by Udvarhelyi, Shkolnikov, Gali, Burkard & Palyi, Phys. Rev. B 98, 075201 (2018), attributed therein to E. Kaxiras, Atomic and Electronic Structure of Solids (Cambridge University Press, 2003).)

**Tensor-implied bulk modulus:**
  442.0 GPa

**Difference from project hydrostatic fit:**
  -3.1 GPa (-0.7%)

**Status:**
  mixed-source elastic reference; full DFT Cij not yet computed
  CLI override used: False
  config: config/reference_pbe_sssp.json

See `nv_spin_strain.py` for the authoritative exact NV physics model; `nv_strain_model.py` is a screening tool only.

