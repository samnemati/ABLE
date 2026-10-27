from pathlib import Path

import pytest

from fvcast.data import load_jain
from fvcast.features import sequence_features
from fvcast.numbering import IMGT_REGIONS, number_antibody

FIXTURE = Path(__file__).parent / "fixtures" / "jain_head.csv"
CDRS = ["H_CDR1", "H_CDR2", "H_CDR3", "L_CDR1", "L_CDR2", "L_CDR3"]


def make_regions(**overrides):
    """All 14 regions empty, except the ones given."""
    regions = {f"{c}_{r}": "" for c in "HL" for r in IMGT_REGIONS}
    regions.update(overrides)
    return regions


def test_motif_counts():
    feats = sequence_features("", "", make_regions(**{cdr: "NGS" for cdr in CDRS}))
    # Each "NGS" holds one N-glycosylation site and one deamidation site.
    assert feats["n_glyc"] == 6
    assert feats["deamidation"] == 6
    assert feats["isomerization"] == 0


def test_motif_counts_single_cdr():
    feats = sequence_features("", "", make_regions(H_CDR3="NGS"))
    assert feats["n_glyc"] == 1
    assert feats["deamidation"] == 1
    assert feats["isomerization"] == 0


def test_overlapping_motifs_counted():
    feats = sequence_features("", "", make_regions(H_CDR3="NGNGS"))
    assert feats["deamidation"] == 2


def test_charge_sign():
    regions = make_regions()
    assert sequence_features("KKKKK", "", regions)["charge_fv_ph7_4"] > 0
    assert sequence_features("DDDDD", "", regions)["charge_fv_ph7_4"] < 0


def test_gravy_known_value():
    feats = sequence_features("", "", make_regions(H_CDR3="IIII"))
    assert feats["gravy_cdrh3"] == pytest.approx(4.5)


def test_empty_regions_give_zero_not_error():
    feats = sequence_features("", "", make_regions())
    assert feats["gravy_cdrs"] == 0.0
    assert feats["charge_cdrs_ph7_4"] == 0.0


def test_trastuzumab_lengths():
    row = load_jain(FIXTURE).loc["trastuzumab"]
    regions = number_antibody(row["vh"], row["vl"], "trastuzumab")
    feats = sequence_features(row["vh"], row["vl"], regions)
    assert feats["len_cdrh3"] == 13
    assert feats["len_cdrl2"] == 3
    assert feats["len_vh"] == 120
