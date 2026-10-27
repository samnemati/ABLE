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


def synthetic_dataset(n_graphs: int = 40, n_nodes: int = 12, signal: bool = True, seed: int = 0):
    """Small graphs whose target is the fraction of nodes of amino acid 0."""
    rng = np.random.default_rng(seed)
    dataset, targets = [], []
    for _ in range(n_graphs):
        graph = small_graph(n_nodes, 0.0)
        graph["node_aa"] = rng.integers(0, 2, n_nodes)
        y = float((graph["node_aa"] == 0).mean()) if signal else float(rng.normal())
        dataset.append(to_pyg(graph, y))
        targets.append(y)
    return dataset, np.array(targets)


SYNTH_GROUPS = np.repeat(np.arange(10), 4)


def test_inner_split_is_grouped():
    from able.gnn import _inner_split

    groups = np.repeat(np.arange(12), 3)
    train_idx, val_idx = _inner_split(groups, seed=0)
    assert not set(groups[train_idx]) & set(groups[val_idx])
    assert len(val_idx) > 0


def test_noise_target_not_positively_predicted():
    from able.gnn import gnn_repeated_cv

    dataset, y = synthetic_dataset(signal=False)
    result = gnn_repeated_cv(dataset, y, SYNTH_GROUPS, n_repeats=2, max_epochs=30, hidden=8)
    # Small positive values occur by chance on 40 graphs; leakage through the
    # scaler, the target z-scoring or the early-stopping split would push this
    # far higher, which is what the bound is here to catch.
    assert result["rho_mean"] < 0.25


def test_deterministic_with_seed():
    from able.gnn import gnn_oof

    dataset, y = synthetic_dataset()
    first = gnn_oof(dataset, y, SYNTH_GROUPS, seed=0, max_epochs=3, hidden=8)
    second = gnn_oof(dataset, y, SYNTH_GROUPS, seed=0, max_epochs=3, hidden=8)
    assert np.allclose(first, second)


def test_learns_simple_graph_signal():
    from able.gnn import gnn_repeated_cv

    dataset, y = synthetic_dataset()
    result = gnn_repeated_cv(dataset, y, SYNTH_GROUPS, n_repeats=1, max_epochs=150, hidden=8)
    assert result["rho_mean"] > 0.5
