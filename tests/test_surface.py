from pathlib import Path

from able.surface import _exposed, surface_features

PDB = Path(__file__).parent / "fixtures" / "trastuzumab_abb2.pdb"


def test_surface_values_sane():
    feats = surface_features(PDB)
    assert feats["sasa_total"] > 0
    assert 0 < feats["frac_sasa_hydrophobic"] < 1
    assert feats["sasa_hydrophobic_cdrs"] <= feats["sasa_hydrophobic"]
    assert feats["sasa_positive_cdrs"] <= feats["sasa_positive"]
    assert feats["sasa_positive"] + feats["sasa_negative"] + feats["sasa_hydrophobic"] < feats["sasa_total"]


def test_relative_sasa_clipped():
    # Terminal residues can exceed the theoretical maximum; still "exposed".
    assert _exposed(1.3)
    assert not _exposed(0.1)
