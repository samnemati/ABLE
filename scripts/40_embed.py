"""Embed every antibody with ESM-2 650M and AbLang2, caching results in cache/."""

import time
from pathlib import Path

import numpy as np

from able.data import load_jain
from able.embeddings import ablang2_embed, esm2_embed, esm2_per_residue, load_cache, save_cache

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

df = load_jain(ROOT / "data" / "jain.csv")
names = list(df.index)


def esm2_fv():
    # ESM-2 is a single-chain model, so embed VH and VL separately and join
    # the two vectors: the first 1280 columns are VH, the next 1280 are VL.
    return np.hstack([esm2_embed(list(df["vh"])), esm2_embed(list(df["vl"]))])


def ablang2_fv():
    return ablang2_embed(list(df["vh"]), list(df["vl"]))


for label, embed in (("esm2_650m", esm2_fv), ("ablang2", ablang2_fv)):
    path = CACHE / f"{label}.npz"
    try:
        X = load_cache(path, names)
        print(f"{label} {X.shape} (cached)")
        continue
    except (FileNotFoundError, ValueError):
        pass
    start = time.time()
    X = embed()
    if np.isnan(X).any():
        raise SystemExit(f"{label}: embeddings contain NaN")
    save_cache(path, names, X)
    print(f"{label} {X.shape} in {time.time() - start:.0f} s")


# Per-residue vectors for the stage 4 graph node features. Stored as one tall
# array of all residues plus the offsets that split it back per antibody,
# because .npz cannot hold ragged arrays.
residue_path = CACHE / "esm2_650m_residues.npz"
try:
    stored = np.load(residue_path)
    if list(stored["names"]) == names:
        print(f"esm2 per-residue {stored['X'].shape} (cached)")
    else:
        raise ValueError("order changed")
except (FileNotFoundError, ValueError):
    start = time.time()
    heavy = esm2_per_residue(list(df["vh"]))
    light = esm2_per_residue(list(df["vl"]))
    per_antibody = [np.vstack([h, l]) for h, l in zip(heavy, light)]
    offsets = np.cumsum([0] + [len(a) for a in per_antibody])
    stacked = np.vstack(per_antibody)
    np.savez(residue_path, names=np.array(names), X=stacked, offsets=offsets)
    print(f"esm2 per-residue {stacked.shape} in {time.time() - start:.0f} s")
