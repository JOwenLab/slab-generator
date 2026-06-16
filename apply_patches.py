#!/usr/bin/env python3
"""Apply the two output-hygiene patches from PATCHES_slabgen.md to slabgen.py.

Usage: put this file in the same folder as slabgen.py and run once:

    python3 apply_patches.py

Safe to re-run: already-patched files are detected and left unchanged.
A backup of the original is written to slabgen.py.bak before modifying.
"""
import shutil
import sys
from pathlib import Path

path = Path(__file__).resolve().parent / "slabgen.py"
if not path.exists():
    sys.exit("slabgen.py not found next to this script — run it from the "
             "folder that contains the tool.")

src = path.read_text()

PATCHES = [
    (
        "CIF wrap/snap fix",
        '''    for i in range(slab.n):
        f = inv @ pos[i]
        lines.append(f"{slab.el[i]} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f}")''',
        '''    for i in range(slab.n):
        f = inv @ pos[i]
        for k in (0, 1):                  # wrap in-plane fractions to [0,1)
            f[k] %= 1.0
            if f[k] > 1.0 - 5e-9:         # snap print-precision residue
                f[k] = 0.0
        f += 0.0                          # normalize -0.0 -> 0.0
        lines.append(f"{slab.el[i]} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f}")''',
        "snap print-precision residue",
    ),
    (
        "QE -0.0 fix",
        '''    cell, pos = _cell3d(slab, vacuum)
    species = sorted(set(slab.el), key=lambda e: ("C", "O", "H", "F").index(e))''',
        '''    cell, pos = _cell3d(slab, vacuum)
    pos = pos.copy()
    pos[abs(pos) < 1e-8] = 0.0            # avoid printing -0.00000000
    species = sorted(set(slab.el), key=lambda e: ("C", "O", "H", "F").index(e))''',
        "avoid printing -0.00000000",
    ),
]

changed = False
for name, find, repl, signature in PATCHES:
    if signature in src:
        print(f"[ok]   {name}: already applied")
        continue
    n = src.count(find)
    if n == 0:
        sys.exit(f"[FAIL] {name}: expected code block not found — your "
                 "slabgen.py differs from the expected version. Patch by "
                 "hand following PATCHES_slabgen.md.")
    if n > 1:
        sys.exit(f"[FAIL] {name}: code block found {n} times — ambiguous. "
                 "Patch by hand following PATCHES_slabgen.md.")
    src = src.replace(find, repl)
    changed = True
    print(f"[done] {name}: applied")

if changed:
    shutil.copy(path, path.with_name("slabgen.py.bak"))
    path.write_text(src)
    print("Patched slabgen.py written. Original saved as slabgen.py.bak.")
else:
    print("Nothing to do — slabgen.py is already current.")
