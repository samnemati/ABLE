"""Graph neural network on antibody residue graphs.

The model is deliberately close to the GCN used on brain connectomes in
GraphBRAIN: GraphConv layers, batch normalization, dropout, global mean
pooling, a linear head. What differs is the data. Brain graphs have a fixed
atlas, so node 50 is the same region in every subject. Antibodies have
different numbers of residues and no correspondence between them, so the
model must learn from local neighborhoods (what kind of residue, surrounded
by what, at what distances) rather than from node identity.
"""

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_mean_pool

N_AMINO_ACIDS = 20
BASIC_FEATURES = N_AMINO_ACIDS + 3  # one-hot residue, chain, CDR flag, predicted error


def to_pyg(
    graph: dict,
    y: float,
    node_embedding: np.ndarray | None = None,
    distance_scale: float = 4.0,
) -> Data:
    """Turn a residue graph into a PyTorch Geometric graph.

    Edge weight is exp(-distance / scale), so a residue's close neighbors
    count for more than ones at the far end of the 10 Å cutoff, instead of
    every contact counting the same. The scale (4 Å) is a fixed choice.
    """
    n = len(graph["node_aa"])
    one_hot = np.zeros((n, N_AMINO_ACIDS), dtype=np.float32)
    one_hot[np.arange(n), graph["node_aa"]] = 1.0
    features = [
        one_hot,
        graph["node_chain"].reshape(-1, 1).astype(np.float32),
        graph["node_cdr"].reshape(-1, 1).astype(np.float32),
        graph["node_error"].reshape(-1, 1).astype(np.float32),
    ]
    if node_embedding is not None:
        features.append(np.asarray(node_embedding, dtype=np.float32))
    return Data(
        x=torch.from_numpy(np.hstack(features)),
        edge_index=torch.from_numpy(np.asarray(graph["edge_index"], dtype=np.int64)),
        edge_weight=torch.from_numpy(np.exp(-graph["edge_dist"] / distance_scale).astype(np.float32)),
        y=torch.tensor([float(y)], dtype=torch.float32),
    )


def shuffle_edges(data: Data, seed: int) -> Data:
    """Rewire a graph at random, keeping its edge count and weights.

    The control condition: if a model does as well on rewired graphs, then
    the 3D structure contributed nothing and only the node features mattered.
    """
    rng = np.random.default_rng(seed)
    n, n_directed = data.num_nodes, data.edge_index.shape[1]
    wanted = n_directed // 2
    # Draw unordered pairs: storing i < j keeps (i, j) and (j, i) from being
    # drawn as two separate edges, which would break the symmetry below.
    edges = set()
    while len(edges) < wanted:
        i, j = rng.integers(0, n, 2)
        if i != j:
            edges.add((int(min(i, j)), int(max(i, j))))
    src, dst = zip(*sorted(edges))
    shuffled = data.clone()
    shuffled.edge_index = torch.from_numpy(np.array([src + dst, dst + src], dtype=np.int64))
    # Reuse the real distance weights, one per undirected edge, so only the
    # wiring is randomized and the weight distribution is unchanged.
    per_edge = data.edge_weight.sort().values[::2]
    shuffled.edge_weight = torch.cat([per_edge, per_edge])
    return shuffled


class GraphRegressor(nn.Module):
    """GraphConv layers, mean pooling over residues, a linear head.

    GraphConv rather than GCNConv: GCNConv folds a node into the normalized
    average of its neighbors, while GraphConv keeps a separate transform of
    the node itself. A residue's own identity should not be averaged away by
    its eighteen neighbors.
    """

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(layers):
            self.convs.append(GraphConv(in_dim if layer == 0 else hidden, hidden))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        for conv, norm in zip(self.convs, self.norms):
            x = self.dropout(torch.relu(norm(conv(x, data.edge_index, data.edge_weight))))
        # One vector per graph, in the order the graphs were batched, which is
        # the order of data.y.
        return self.head(global_mean_pool(x, data.batch)).squeeze(-1)
