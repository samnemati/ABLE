from pathlib import Path

import pytest

from able.data import load_jain
from able.encodings import aligned_positions
from able.structure import check_matches_sequence, read_residues

FIXTURES = Path(__file__).parent / "fixtures"
PDB = FIXTURES / "trastuzumab_abb2.pdb"


@pytest.fixture(scope="module")
def trastuzumab():
    row = load_jain(FIXTURES / "jain_head.csv").loc["trastuzumab"]
    return row["vh"], row["vl"]


@pytest.fixture(scope="module")
def residues():
    return read_residues(PDB)


def test_fixture_residue_counts(residues):
    assert (residues["chain"] == "H").sum() == 120
    assert (residues["chain"] == "L").sum() == 107
    assert (residues["error"] >= 0).all()


def test_fixture_matches_sequence(residues, trastuzumab):
    check_matches_sequence(residues, *trastuzumab, "trastuzumab")


def test_mismatch_raises(residues, trastuzumab):
    vh, vl = trastuzumab
    with pytest.raises(ValueError, match="trastuzumab"):
        check_matches_sequence(residues, vh[:-1] + "A", vl, "trastuzumab")


def test_keys_match_imgt_numbering(residues, trastuzumab):
    keys = set(zip(residues["chain"], residues["position"], residues["insertion"]))
    assert keys == set(aligned_positions(*trastuzumab, "trastuzumab"))


def test_cdrh3_has_higher_error_than_framework(residues):
    heavy = residues[residues["chain"] == "H"]
    in_cdrs = heavy["position"].between(27, 38) | heavy["position"].between(56, 65) | heavy["position"].between(105, 117)
    cdrh3 = heavy["position"].between(105, 117)
    assert heavy.loc[cdrh3, "error"].mean() > heavy.loc[~in_cdrs, "error"].mean()
