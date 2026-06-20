#!/usr/bin/env python3
"""Print the authoritative PBE/SSSP bulk diamond reference parameters."""
import json
import pathlib

CONFIG = pathlib.Path(__file__).parent / "config" / "reference_pbe_sssp.json"

with CONFIG.open() as f:
    cfg = json.load(f)

ref = cfg["bulk_reference"]
pseudo = cfg["pseudopotentials"]

print(f"Functional   : {cfg['functional']}")
print(f"Pseudo set   : {cfg['pseudo_set']}")
print(f"C pseudo     : {pseudo['C']}")
print(f"Fitted a0    : {ref['a0_fit_angstrom']:.6f} Å  (E–V fit)")
print(f"Bulk modulus : {ref['bulk_modulus_gpa']:.1f} GPa")
print(f"Fit status   : {ref['fit_status']}")
print(f"Source report: {ref['source_report']}")
