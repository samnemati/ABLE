"""Solvent-accessible surface features from predicted antibody structures.

HIC and SMAC measure how much hydrophobic surface an antibody exposes, and
polyreactivity assays respond to exposed charge. Sequence features can only
guess which residues are on the surface; a structure tells us. Solvent-
accessible surface area (SASA) is the area a water-sized probe can touch
when rolled over the protein.
"""

from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import index_to_one, three_to_index
from Bio.PDB.SASA import ShrakeRupley

# Theoretical maximum SASA of each residue in a Gly-X-Gly peptide, in Å²
# (Tien et al., 2013, PLoS ONE). Dividing by it gives relative exposure.
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}
HYDROPHOBIC = set("AVILMFW")
POSITIVE = set("KR")
NEGATIVE = set("DE")
EXPOSED_THRESHOLD = 0.25
CDR_RANGES = ((27, 38), (56, 65), (105, 117))


def _exposed(relative_sasa: float) -> bool:
    # Terminal residues can exceed the Gly-X-Gly maximum (no neighbor on one
    # side), giving a relative SASA above 1. Clip so they still count simply
    # as exposed.
    return min(relative_sasa, 1.0) >= EXPOSED_THRESHOLD


def surface_features(pdb_path: Path) -> dict[str, float]:
    """Total, hydrophobic and charged exposed surface, overall and in the CDRs."""
    model = PDBParser(QUIET=True).get_structure("ab", str(pdb_path))[0]
    ShrakeRupley().compute(model, level="R")
    feats = dict.fromkeys(
        ["sasa_total", "sasa_hydrophobic", "sasa_positive", "sasa_negative",
         "sasa_hydrophobic_cdrs", "sasa_positive_cdrs"],
        0.0,
    )
    exposed_charge, exposed_hydrophobic_h3 = 0, 0
    for chain in model:
        for residue in chain:
            aa = index_to_one(three_to_index(residue.get_resname()))
            sasa = residue.sasa
            position = residue.id[1]
            in_cdr = any(first <= position <= last for first, last in CDR_RANGES)
            exposed = _exposed(sasa / MAX_ASA[aa])
            feats["sasa_total"] += sasa
            if aa in HYDROPHOBIC:
                feats["sasa_hydrophobic"] += sasa
                if in_cdr:
                    feats["sasa_hydrophobic_cdrs"] += sasa
                if exposed and chain.id == "H" and 105 <= position <= 117:
                    exposed_hydrophobic_h3 += 1
            elif aa in POSITIVE:
                feats["sasa_positive"] += sasa
                if in_cdr:
                    feats["sasa_positive_cdrs"] += sasa
                exposed_charge += exposed
            elif aa in NEGATIVE:
                feats["sasa_negative"] += sasa
                exposed_charge -= exposed
    feats["frac_sasa_hydrophobic"] = feats["sasa_hydrophobic"] / feats["sasa_total"]
    feats["exposed_net_charge"] = float(exposed_charge)
    feats["exposed_hydrophobic_cdrh3"] = float(exposed_hydrophobic_h3)
    return feats
