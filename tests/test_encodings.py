from pathlib import Path

import numpy as np

from able.data import load_jain
from able.encodings import aligned_positions, identity_groups, identity_matrix, onehot_aligned

FIXTURE = Path(__file__).parent / "fixtures" / "jain_head.csv"

TWO_POSITIONS = {("H", 1, ""): "A", ("H", 2, ""): "C"}
ONE_POSITION = {("H", 1, ""): "A"}


def test_onehot_gap_column():
    X, columns = onehot_aligned([TWO_POSITIONS, ONE_POSITION])
    assert X[1, columns.index("H_2_-")] == 1
    assert X[0, columns.index("H_2_C")] == 1
    # Every position contributes exactly one "hot" column per antibody.
    assert (X.sum(axis=1) == 2).all()


def test_identity_values():
    ident = identity_matrix([TWO_POSITIONS, TWO_POSITIONS, ONE_POSITION])
    assert ident[0, 1] == 1.0
    assert ident[0, 2] == 0.5
    assert (np.diag(ident) == 1.0).all()


def test_groups_cluster_identical():
    other = {("H", 1, ""): "W", ("H", 2, ""): "Y"}
    groups = identity_groups([TWO_POSITIONS, TWO_POSITIONS, other])
    assert len(set(groups)) == 2
    assert groups[0] == groups[1] != groups[2]


def test_trastuzumab_positions():
    row = load_jain(FIXTURE).loc["trastuzumab"]
    positions = aligned_positions(row["vh"], row["vl"], "trastuzumab")
    assert sum(1 for key in positions if key[0] == "H") == 120
    assert sum(1 for key in positions if key[0] == "L") == 107


def _variant(base: dict, changes: dict) -> dict:
    new = dict(base)
    new.update(changes)
    return new


def test_no_cross_group_pair_reaches_threshold():
    # A chain of relatives: B is 85% identical to A, C is 85% identical to B,
    # but only 70% identical to A. Average linkage can split C off from B even
    # though they are close; single linkage must keep all three together.
    a = {("H", p, ""): "A" for p in range(1, 21)}
    b = _variant(a, {("H", p, ""): "W" for p in (1, 2, 3)})
    c = _variant(b, {("H", p, ""): "Y" for p in (4, 5, 6)})
    maps = [a, b, c]
    groups = identity_groups(maps, min_identity=0.8)
    ident = identity_matrix(maps)
    for i in range(3):
        for j in range(3):
            if groups[i] != groups[j]:
                assert ident[i, j] < 0.8
