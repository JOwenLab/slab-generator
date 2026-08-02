# slab-generator

**Diamond surface slab construction for periodic DFT, built for NV-center surface science.**

`slab-generator` builds symmetric, dipole-free diamond slab models across the (111), (110), and (100) orientations with a curated library of surface terminations and reconstructions, and emits ready-to-run Quantum ESPRESSO inputs. It was developed in the Owen Group (Columbia University, Department of Chemistry) to support a systematic first-principles screen of surface terminations for nanodiamond-hosted nitrogen-vacancy (NV) quantum sensors — identifying surface chemistries that stabilize the NV⁻ charge state while minimizing surface stress and strain.

Development is active; interfaces may change ahead of the accompanying publication.

## What it does

- **Slab construction** — symmetric slabs (matching top/bottom terminations) at user-specified thickness, built on ASE `Atoms` objects, with the DFT bulk lattice constant as the strain reference to avoid Pulay-stress artifacts.
- **Termination library** — 19 validated surface motifs across (111), (110), and (100): hydrogen, fluorine, oxygen (ether and ketone), hydroxyl, and clean reconstructed surfaces, including reconstructions transcribed from published coordinates. Two additional literature motifs are included as placeholder stubs pending author-provided coordinates.
- **Quantum ESPRESSO input generation** — `pw.x` inputs with sensible defaults for surface work (`tstress=.true.`, Methfessel-Paxton smearing, thickness-scaled k-grids), including constrained in-plane variable-cell relaxation (`calculation='vc-relax'`, `cell_dofree='2Dxy'`) for surface-stress extraction.
- **Validation** — the library build pipeline runs 16 structural and chemical validation checks (stoichiometry, symmetry, bond-length sanity, termination coverage) on every motif.

## Installation

```bash
git clone https://github.com/JOwenLab/slab-generator.git
cd slab-generator
pip install -e .
```

Requires Python ≥3.10 and ASE. <!-- TODO: confirm minimum versions and any other dependencies from pyproject.toml -->

## Quick start

<!-- TODO: replace with the actual CLI/API invocation — the snippet below is a placeholder to be checked against the current interface -->

```python
from diamond_slabs import build_slab

slab = build_slab(
    orientation="111",
    termination="H",
    thickness_nm=3.0,
)
slab.write_espresso_input("C111_H_3nm.in", relax_mode="cell2d")
```

Rebuild and validate the full motif library:

```bash
python build_library.py
```

## Physics notes

- **Symmetric slabs** (identical top and bottom terminations) are required for clean surface-stress measurements; asymmetric slabs introduce dipole artifacts and complicate interpretation of the stress tensor.
- **Surface stress** is extracted from the in-plane stress of a 2D-relaxed slab as *f*<sub>αβ</sub> = ½ · *L*<sub>z</sub> · σ<sub>αβ</sub> (1 kbar·Å ≈ 0.01 N/m), referenced to the DFT equilibrium lattice constant at the same cutoffs.
- **Pseudopotentials are not distributed** with this repository. Inputs are written for SSSP-style UPF files (PBE); download them from the [SSSP library](https://www.materialscloud.org/discover/sssp) and set the pseudopotential directory in your configuration.

## Development

This package was developed iteratively with **Claude (Anthropic)** as an active co-developer: implementation briefs are handed to Claude Code for well-scoped features, and the accompanying DFT campaign pipeline uses the Anthropic API for automated triage of failed HPC jobs. The repository history reflects that workflow.

## Citation

A publication describing the termination library and the surface-stress screening campaign is in preparation. Until then, please cite this repository:

> Owen Group, Columbia University. *slab-generator: diamond surface slab construction for periodic DFT.* https://github.com/JOwenLab/slab-generator (2026).

## License

MIT — see [LICENSE](LICENSE).

## Contact

Jonathan S. Owen — jso2115@columbia.edu
Department of Chemistry, Columbia University
