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


def shuffle_edges(data: Data, seed: int, n_swaps_per_edge: int = 10) -> Data:
    """Rewire a graph at random while keeping every residue's number of contacts.

    The control condition: if a model does as well on rewired graphs, then the
    3D arrangement contributed nothing and only the node features mattered.

    Rewiring is done by repeated double-edge swaps: take edges (a, b) and
    (c, d) and replace them with (a, d) and (c, b). Every node keeps its
    degree, so this is the configuration-model null used for connectomes,
    not a plain random graph. That matters because GraphConv sums over
    neighbours, so a node whose degree changed would see a different signal
    scale for a reason that has nothing to do with wiring. Edge weights are
    carried along with their slots and then shuffled, so the weight
    distribution is unchanged and no weight lines up with node position.
    """
    rng = np.random.default_rng(seed)
    src, dst = data.edge_index.numpy()
    upper = src < dst  # one row per undirected edge
    edges = list(zip(src[upper].tolist(), dst[upper].tolist()))
    present = set(edges) | {(j, i) for i, j in edges}
    for _ in range(n_swaps_per_edge * len(edges)):
        x, y = rng.integers(0, len(edges), 2)
        if x == y:
            continue
        a, b = edges[x]
        c, d = edges[y]
        if len({a, b, c, d}) < 4:
            continue
        new_a, new_c = (a, d), (c, b)
        if new_a in present or new_c in present:
            continue
        for old in ((a, b), (b, a), (c, d), (d, c)):
            present.discard(old)
        for fresh in (new_a, (d, a), new_c, (b, c)):
            present.add(fresh)
        edges[x], edges[y] = new_a, new_c
    weights = data.edge_weight.numpy()[upper]
    rng.shuffle(weights)
    new_src, new_dst = zip(*edges)
    shuffled = data.clone()
    shuffled.edge_index = torch.from_numpy(
        np.array([new_src + new_dst, new_dst + new_src], dtype=np.int64)
    )
    shuffled.edge_weight = torch.from_numpy(np.concatenate([weights, weights]).astype(np.float32))
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


def _inner_split(groups: np.ndarray, seed: int, val_fraction: float = 0.2):
    """Hold out whole identity groups for early stopping.

    Early stopping needs a validation set that the model does not train on.
    It must be grouped for the same reason the outer folds are: a close
    relative in the validation set would make the stopping point optimistic.
    """
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_groups = set(unique[:n_val].tolist())
    is_val = np.array([g in val_groups for g in groups])
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def train_one(
    train_data: list[Data],
    val_data: list[Data],
    in_dim: int,
    seed: int,
    max_epochs: int = 200,
    patience: int = 20,
    hidden: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
) -> tuple[nn.Module, dict]:
    """Train until the validation loss stops improving, then restore the best epoch."""
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)
    model = GraphRegressor(in_dim=in_dim, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=len(val_data))
    best_loss, best_state, best_epoch, since_best = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(batch), batch.y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(nn.functional.mse_loss(model(b), b.y).item() for b in val_loader)
        if val_loss < best_loss:
            best_loss, best_epoch, since_best = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), {"best_epoch": best_epoch, "best_val_loss": best_loss}


def _standardize(dataset: list[Data], train_idx: np.ndarray) -> list[Data]:
    """Z-score node features using training-fold statistics only."""
    train_x = torch.cat([dataset[i].x for i in train_idx])
    mean, std = train_x.mean(0), train_x.std(0)
    std[std < 1e-8] = 1.0
    out = []
    for data in dataset:
        scaled = data.clone()
        scaled.x = (data.x - mean) / std
        out.append(scaled)
    return out


def gnn_oof(dataset, y, groups, seed: int, **kwargs) -> tuple[np.ndarray, list[int]]:
    """Out-of-fold predictions, with scaling and early stopping inside each fold.

    Returns the predictions and the stopping epoch of each fold, so a run that
    never converged (or stopped at the epoch cap) is visible afterwards.
    """
    from sklearn.model_selection import GroupKFold
    from torch_geometric.loader import DataLoader

    y = np.asarray(y, dtype=float)
    pred = np.empty(len(y))
    best_epochs = []
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), groups=groups)):
        scaled = _standardize(dataset, train_idx)
        # Z-score the target on the training fold so the loss scale does not
        # depend on the assay's units, then undo it for the predictions.
        mu, sigma = y[train_idx].mean(), y[train_idx].std() or 1.0
        for i in range(len(scaled)):
            scaled[i].y = torch.tensor([(y[i] - mu) / sigma], dtype=torch.float32)
        inner_train, inner_val = _inner_split(groups[train_idx], seed=seed + fold)
        model, history = train_one(
            [scaled[train_idx[i]] for i in inner_train],
            [scaled[train_idx[i]] for i in inner_val],
            in_dim=scaled[0].x.shape[1],
            seed=seed + fold,
            **kwargs,
        )
        best_epochs.append(history["best_epoch"])
        with torch.no_grad():
            batch = next(iter(DataLoader([scaled[i] for i in test_idx], batch_size=len(test_idx))))
            pred[test_idx] = model(batch).numpy() * sigma + mu
    return pred, best_epochs


def gnn_repeated_cv(dataset, y, groups, n_repeats: int = 5, n_boot: int = 2000, **kwargs) -> dict:
    """Repeat grouped cross-validation, reported like able.evaluate.repeated_cv.

    `dataset` may be a list of graphs, or a callable taking the repeat seed and
    returning one. The callable form exists for the rewired control, which must
    be redrawn each repeat so the reported spread includes the randomness of
    the rewiring itself.
    """
    from able.evaluate import _bootstrap_indices, _mean_rho, _spearman

    y = np.asarray(y, dtype=float)
    oof, epochs = [], []
    for seed in range(n_repeats):
        graphs = dataset(seed) if callable(dataset) else dataset
        predictions, best_epochs = gnn_oof(graphs, y, groups, seed=seed, **kwargs)
        oof.append(predictions)
        epochs.extend(best_epochs)
    oof = np.array(oof)
    rhos = np.array([_spearman(y, p) for p in oof])
    boot = np.array([_mean_rho(y[i], oof[:, i]) for i in _bootstrap_indices(len(y), n_boot, seed=0)])
    return {
        "rho_per_repeat": rhos,
        "rho_mean": float(rhos.mean()),
        "rho_sd": float(rhos.std(ddof=1)) if n_repeats > 1 else 0.0,
        "oof": oof,
        "best_epoch_mean": float(np.mean(epochs)),
        "best_epoch_max": int(np.max(epochs)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }
