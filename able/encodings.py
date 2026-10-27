"""IMGT-aligned encodings of antibodies, and groups of similar antibodies.

IMGT numbering gives every residue an address, (chain, position, insertion
code), that means the same thing in every antibody. Lining antibodies up by
address turns sequences of different lengths into rows of one table, which
is what a one-hot encoding and a sequence-identity score both need.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from able.numbering import number_chain

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"


def aligned_positions(vh: str, vl: str, name: str) -> dict[tuple[str, int, str], str]:
    """Map each IMGT address (chain, position, insertion) to its amino acid."""
    positions = {}
    for chain, seq in (("H", vh), ("L", vl)):
        numbered, _ = number_chain(seq, name)
        for position, insertion, aa in numbered:
            positions[(chain, position, insertion)] = aa
    return positions


def _sorted_union(maps: list[dict]) -> list[tuple[str, int, str]]:
    keys = set().union(*maps)
    # Sort by chain, then position, then insertion code, so columns read in
    # sequence order. (IMGT orders some CDR3 insertions symmetrically; for an
    # encoding only a consistent order matters, not the biological one.)
    return sorted(keys)


def onehot_aligned(maps: list[dict]) -> tuple[np.ndarray, list[str]]:
    """One-hot encode aligned antibodies: 21 columns (20 amino acids + gap) per address.

    An address an antibody does not use (a shorter loop) is encoded as a gap,
    so every antibody has exactly one hot column per address and the columns
    stay aligned across antibodies.
    """
    keys = _sorted_union(maps)
    symbols = AMINO_ACIDS + GAP
    columns = [f"{c}_{p}{i}_{s}" for c, p, i in keys for s in symbols]
    X = np.zeros((len(maps), len(columns)), dtype=np.float32)
    for row, positions in enumerate(maps):
        for k, key in enumerate(keys):
            aa = positions.get(key, GAP)
            X[row, k * len(symbols) + symbols.index(aa)] = 1.0
    return X, columns


def identity_matrix(maps: list[dict]) -> np.ndarray:
    """Pairwise sequence identity over the union of IMGT addresses.

    An address present in only one of the two antibodies counts as a mismatch,
    so a length difference lowers identity.
    """
    n = len(maps)
    ident = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            keys = set(maps[a]) | set(maps[b])
            same = sum(maps[a].get(k) == maps[b].get(k) for k in keys)
            ident[a, b] = ident[b, a] = same / len(keys)
    return ident


def identity_groups(maps: list[dict], min_identity: float = 0.8) -> np.ndarray:
    """Cluster antibodies so that close relatives share a group label.

    Average-linkage clustering on (1 - identity), cut so that groups hold
    antibodies at roughly min_identity or more. Cross-validation then keeps
    each group entirely in train or entirely in test.
    """
    distance = 1 - identity_matrix(maps)
    tree = linkage(squareform(distance, checks=False), "average")
    return fcluster(tree, 1 - min_identity, criterion="distance")
