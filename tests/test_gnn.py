from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from able.gnn import GraphRegressor, shuffle_edges, to_pyg
from able.graphs import residue_graph
from able.structure import read_residues

PDB = Path(__file__).parent / "fixtures" / "trastuzumab_abb2.pdb"


def fixture_graph():
    return residue_graph(read_residues(PDB))


def test_to_pyg_shapes():
    data = to_pyg(fixture_graph(), 5.0)
    assert data.x.shape == (227, 23)
    assert data.edge_index.shape[0] == 2
    assert data.edge_weight.shape[0] == data.edge_index.shape[1]
    assert 0 < data.edge_weight.max() <= 1
    assert data.y.item() == 5.0


def test_to_pyg_with_embedding():
    data = to_pyg(fixture_graph(), 1.0, node_embedding=np.zeros((227, 4), dtype=np.float32))
    assert data.x.shape == (227, 27)


def test_shuffle_edges_preserves_count_and_symmetry():
    data = to_pyg(fixture_graph(), 1.0)
    shuffled = shuffle_edges(data, seed=0)
    assert shuffled.edge_index.shape == data.edge_index.shape
    src, dst = shuffled.edge_index.tolist()
    edges = set(zip(src, dst))
    assert not any(i == j for i, j in edges)
    assert all((j, i) in edges for i, j in edges)
    assert torch.allclose(shuffled.edge_weight.sort().values, data.edge_weight.sort().values)
    assert edges != set(zip(*data.edge_index.tolist()))


def small_graph(n: int, value: float) -> dict:
    rng = np.random.default_rng(0)
    src, dst = [], []
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    return {
        "node_aa": rng.integers(0, 20, n),
        "node_chain": np.zeros(n, dtype=int),
        "node_cdr": np.zeros(n, dtype=int),
        "node_error": np.zeros(n),
        "edge_index": np.array([src, dst]),
        "edge_dist": np.full(len(src), 4.0),
        "keys": [("H", i + 1, "") for i in range(n)],
    }


def test_forward_matches_batch_order():
    data = [to_pyg(small_graph(6, y), float(y)) for y in (1, 2, 3)]
    batch = next(iter(DataLoader(data, batch_size=3)))
    model = GraphRegressor(in_dim=23, hidden=8).eval()
    assert model(batch).shape == (3,)
    assert batch.y.tolist() == [1.0, 2.0, 3.0]


def test_forward_is_permutation_invariant():
    torch.manual_seed(0)
    data = to_pyg(fixture_graph(), 1.0)
    model = GraphRegressor(in_dim=23, hidden=8).eval()
    perm = torch.randperm(data.num_nodes)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(data.num_nodes)
    permuted = data.clone()
    permuted.x = data.x[perm]
    permuted.edge_index = inverse[data.edge_index]
    with torch.no_grad():
        first = model(next(iter(DataLoader([data], batch_size=1))))
        second = model(next(iter(DataLoader([permuted], batch_size=1))))
    assert torch.allclose(first, second, atol=1e-5)
