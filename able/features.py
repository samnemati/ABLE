"""Baseline sequence features for antibody developability.

These are the kinds of hand-computed descriptors protein engineers already
use to flag risky antibodies. They are the bar that learned representations
(language models, structure graphs) have to beat in later stages.
"""

import re

from Bio.SeqUtils.ProtParam import ProteinAnalysis

CDR_KEYS = ["H_CDR1", "H_CDR2", "H_CDR3", "L_CDR1", "L_CDR2", "L_CDR3"]

# Chemical liability motifs, counted inside the CDRs only: the CDRs are where
# the antibody binds its target, so a modification there is most likely to
# hurt binding. Patterns use lookahead so overlapping matches all count.
LIABILITY_MOTIFS = {
    # N, then anything but P, then S or T: a site where the cell can attach a
    # sugar chain, making the product heterogeneous.
    "n_glyc": r"(?=N[^P][ST])",
    # N followed by G or S: asparagine deamidation, which changes charge and
    # can reduce binding.
    "deamidation": r"(?=N[GS])",
    # D followed by G or S: aspartate isomerization, which bends the backbone.
    "isomerization": r"(?=D[GS])",
    # Methionine and tryptophan oxidize during manufacturing and storage.
    "met_oxidation": r"M",
    "trp_oxidation": r"W",
    # A cysteine in a CDR is usually unpaired and can form wrong disulfides.
    "cys_in_cdrs": r"C",
}


def _charge(seq: str, ph: float = 7.4) -> float:
    # Net charge from the Henderson-Hasselbalch equation with standard pKa
    # values. Highly charged antibodies tend to stick to things nonspecifically.
    return ProteinAnalysis(seq).charge_at_pH(ph) if seq else 0.0


def _gravy(seq: str) -> float:
    # Kyte-Doolittle hydropathy averaged over residues; positive = hydrophobic.
    # Hydrophobic surfaces drive aggregation and long HIC retention.
    return ProteinAnalysis(seq).gravy() if seq else 0.0


def sequence_features(vh: str, vl: str, regions: dict[str, str]) -> dict[str, float]:
    """Compute length, charge, hydrophobicity, CDR-length and liability features."""
    fv = vh + vl
    cdrs = "".join(regions[key] for key in CDR_KEYS)
    features = {
        "len_vh": len(vh),
        "len_vl": len(vl),
        "charge_fv_ph7_4": _charge(fv),
        "charge_cdrs_ph7_4": _charge(cdrs),
        "gravy_fv": _gravy(fv),
        "gravy_cdrs": _gravy(cdrs),
        "gravy_cdrh3": _gravy(regions["H_CDR3"]),
    }
    for key in CDR_KEYS:
        chain, cdr_number = key[0].lower(), key[-1]
        features[f"len_cdr{chain}{cdr_number}"] = len(regions[key])
    for motif, pattern in LIABILITY_MOTIFS.items():
        # Count per CDR and sum, so a motif cannot be created by gluing the
        # end of one CDR to the start of the next.
        features[motif] = sum(len(re.findall(pattern, regions[key])) for key in CDR_KEYS)
    return {name: float(value) for name, value in features.items()}


def charge_counts(vh: str, vl: str) -> dict[str, float]:
    """Count positively (K, R) and negatively (D, E) charged residues in the Fv.

    A sequence-only control for the structure-based surface charge features:
    if surface area adds nothing beyond how many charged residues there are,
    these counts should predict as well.
    """
    fv = vh + vl
    return {
        "n_positive": float(sum(fv.count(aa) for aa in "KR")),
        "n_negative": float(sum(fv.count(aa) for aa in "DE")),
    }
