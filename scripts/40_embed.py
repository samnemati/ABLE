"""Embed every antibody with ESM-2 650M and AbLang2, caching results in cache/."""

import time
from pathlib import Path

import numpy as np

from able.data import load_jain
from able.embeddings import ablang2_embed, esm2_embed, load_cache, save_cache

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
