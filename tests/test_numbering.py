"""Expected CDRs are IMGT definitions (CDR1 27-38, CDR2 56-65, CDR3 105-117).

If ANARCI ever disagrees with one of these strings, the scheme or the region
boundaries are wrong. Investigate; do not edit the expected value to match.
"""

from pathlib import Path

import pytest

from fvcast.data import load_jain
from fvcast.numbering import number_antibody, number_chain

FIXTURE = Path(__file__).parent / "fixtures" / "jain_head.csv"


@pytest.fixture(scope="module")
def jain():
    return load_jain(FIXTURE)


@pytest.fixture(scope="module")
def trastuzumab(jain):
    row = jain.loc["trastuzumab"]
    return row["vh"], row["vl"], number_antibody(row["vh"], row["vl"], "trastuzumab")


def test_trastuzumab_cdrs(trastuzumab):
    _, _, regions = trastuzumab
    assert regions["H_CDR1"] == "GFNIKDTY"
    assert regions["H_CDR2"] == "IYPTNGYT"
    assert regions["H_CDR3"] == "SRWGGDGFYAMDY"
    assert regions["L_CDR1"] == "QDVNTA"
    assert regions["L_CDR2"] == "SAS"
    assert regions["L_CDR3"] == "QQHYTTPPT"


def test_regions_reassemble_sequence(trastuzumab):
    vh, vl, regions = trastuzumab
    order = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
    assert "".join(regions[f"H_{r}"] for r in order) == vh
    assert "".join(regions[f"L_{r}"] for r in order) == vl


def test_lambda_light_chain_accepted(jain):
    row = jain.loc["belimumab"]
    regions = number_antibody(row["vh"], row["vl"], "belimumab")
    assert regions["L_CDR3"] == "SSRDSSGNHWV"


def test_long_cdrh3_insertions(jain):
    row = jain.loc["cixutumumab"]
    regions = number_antibody(row["vh"], row["vl"], "cixutumumab")
    assert regions["H_CDR3"] == "ARAPLRFLEWSTQDHYYYYYMDV"
    assert len(regions["H_CDR3"]) == 23


def test_unnumberable_sequence_raises():
    with pytest.raises(ValueError, match="junk"):
        number_chain("ACDEFGHIKLMNPQRSTVWY", "junk")


def test_swapped_chains_raise(jain):
    row = jain.loc["trastuzumab"]
    with pytest.raises(ValueError):
        number_antibody(row["vl"], row["vh"], "trastuzumab")


# Real-world sequences sometimes carry a signal peptide in front or part of
# the constant domain behind. ANARCI numbers only the variable domain and
# drops the rest, so number_antibody must refuse rather than return regions
# that no longer match the sequence the features are computed from.
LEADER = "MGWSCIILFLVATATGVHS"
CH1_START = "ASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKV"


def test_extra_residues_outside_variable_domain_raise(jain):
    row = jain.loc["trastuzumab"]
    with pytest.raises(ValueError, match="trastuzumab"):
        number_antibody(LEADER + row["vh"] + CH1_START, row["vl"], "trastuzumab")


def test_truncated_variable_domain_raises(jain):
    row = jain.loc["trastuzumab"]
    with pytest.raises(ValueError, match="trastuzumab"):
        number_antibody(row["vh"][30:], row["vl"], "trastuzumab")
