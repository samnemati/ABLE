import numpy as np
import pytest
import torch

from able.embeddings import load_cache, mean_pool, save_cache


def test_mean_pool_excludes_special_and_padding():
    # Row 0: BOS, residues 1 2 3, EOS.  Row 1: BOS, residues 4 6, EOS, pad.
    # Special tokens carry 100 so any leak would be obvious in the mean.
    hidden = torch.tensor(
        [[100.0, 1.0, 2.0, 3.0, 100.0], [100.0, 4.0, 6.0, 100.0, 999.0]]
    ).unsqueeze(-1)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    pooled = mean_pool(hidden, mask)
    assert pooled.tolist() == [[2.0], [5.0]]


def test_cache_roundtrip_and_order_check(tmp_path):
    path = tmp_path / "emb.npz"
    X = np.arange(6, dtype=np.float32).reshape(3, 2)
    save_cache(path, ["a", "b", "c"], X)
    assert np.array_equal(load_cache(path, ["a", "b", "c"]), X)
    with pytest.raises(ValueError):
        load_cache(path, ["b", "a", "c"])
