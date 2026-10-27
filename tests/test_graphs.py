from pathlib import Path

import numpy as np
import pandas as pd

from able.graphs import residue_graph
from able.structure import read_residues

PDB = Path(__file__).parent / "fixtures" / "trastuzumab_abb2.pdb"


def line_of_residues(n: int, spacing: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chain": ["H"] * n,
            "position": range(1, n + 1),
            "insertion": [""] * n,
            "aa": ["A"] * n,
            "x": np.arange(n) * spacing,
            "y": 0.0,
            "z": 0.0,
            "error": 0.0,
        }
    )


def test_line_graph_neighbors():
    graph = residue_graph(line_of_residues(5, 4.0), cutoff=10.0)
    src, dst = graph["edge_index"]
    # 4 A apart: neighbors 1 and 2 steps away are within 10 A, 3 steps is 12 A.
    assert set(zip(src.tolist(), dst.tolist())) == {
        (i, j) for i in range(5) for j in range(5) if 0 < abs(i - j) <= 2
    }
    assert len(src) == 14


def test_edges_symmetric_no_self_loops():
    graph = residue_graph(read_residues(PDB))
    src, dst = graph["edge_index"]
    edges = set(zip(src.tolist(), dst.tolist()))
    assert all((j, i) in edges for i, j in edges)
    assert not any(i == j for i, j in edges)
    assert (graph["edge_dist"] <= 10.0).all()


def test_node_features():
    graph = residue_graph(read_residues(PDB))
    assert graph["node_cdr"].sum() == 8 + 8 + 13 + 6 + 3 + 9
    assert (graph["node_chain"] == 0).sum() == 120
    assert (graph["node_chain"] == 1).sum() == 107
    assert len(graph["keys"]) == 227
