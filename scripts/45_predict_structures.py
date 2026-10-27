"""Predict a 3D structure for every antibody with ABodyBuilder2 (cache/structures/).

Needs ImmuneBuilder plus OpenMM and PDBFixer (see README). About 6 seconds per
antibody on a laptop. Existing structures that parse and match their sequence
are kept.
"""

import time
from pathlib import Path

from able.data import load_jain
from able.structure import check_matches_sequence, predict_structure, read_residues

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "structures"

df = load_jain(ROOT / "data" / "jain.csv")
predicted, cached, failed = 0, 0, []
start = time.time()
for name, row in df.iterrows():
    path = OUT / f"{name}.pdb"
    if path.exists():
        try:
            check_matches_sequence(read_residues(path), row["vh"], row["vl"], name)
            cached += 1
            continue
        except Exception:
            pass  # unreadable or stale: predict again
    try:
        predict_structure(row["vh"], row["vl"], path)
        check_matches_sequence(read_residues(path), row["vh"], row["vl"], name)
        predicted += 1
    except Exception as err:  # one bad antibody should not stop the other 136
        failed.append(f"{name}: {err}")

print(f"{predicted} predicted, {cached} cached, {len(failed)} failed in {time.time() - start:.0f} s")
for line in failed:
    print("  " + line)
