"""Residue graphs from predicted antibody structures.

Same idea as a connectome graph in GraphBRAIN, with different parts: nodes
are residues instead of atlas regions, and an edge means two residues are
physically close in 3D instead of two regions having correlated activity.
Distance is in absolute units (Ångströms), so a fixed cutoff means the same
thing in every antibody and no proportional thresholding is needed. Unlike a
brain atlas, the number of nodes differs between antibodies.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from able.encodings import AMINO_ACIDS

CHAIN_INDEX = {"H": 0, "L": 1}
CDR_RANGES = ((27, 38), (56, 65), (105, 117))


def _is_cdr(position: int) -> bool:
    return any(first <= position <= last for first, last in CDR_RANGES)


def residue_graph(residues: pd.DataFrame, cutoff: float = 10.0) -> dict:
    """Connect every pair of residues whose Cα atoms are within `cutoff` Å.

    10 Å between Cα atoms is a common contact definition: it links residues
    whose side chains can touch, plus their close neighbors, so each residue
    sees its local 3D environment. Edges are stored in both directions
    (i to j and j to i), the convention graph libraries use for undirected
    graphs, and a residue is never connected to itself.
    """
    coords = residues[["x", "y", "z"]].to_numpy()
    dist = cdist(coords, coords)
    close = (dist <= cutoff) & ~np.eye(len(coords), dtype=bool)
    src, dst = np.nonzero(close)
    return {
        "node_aa": np.array([AMINO_ACIDS.index(aa) for aa in residues["aa"]]),
        "node_chain": residues["chain"].map(CHAIN_INDEX).to_numpy(),
        "node_cdr": np.array([int(_is_cdr(p)) for p in residues["position"]]),
        "node_error": residues["error"].to_numpy(dtype=float),
        "edge_index": np.stack([src, dst]),
        "edge_dist": dist[src, dst],
        "keys": list(zip(residues["chain"], residues["position"], residues["insertion"])),
    }
