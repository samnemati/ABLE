"""IMGT numbering of antibody chains and extraction of framework and CDR regions.

Antibody sequences differ in length, mostly because the CDR loops differ in
length. Numbering puts every chain on a shared coordinate system, so that
position 105 means the start of CDR3 in every antibody, the way a brain atlas
makes region N mean the same thing in every subject. Extra residues in a long
loop get insertion codes (111A, 112A, ...) instead of shifting everything after.
"""

from anarci import anarci

# IMGT region boundaries, inclusive, by position number. Insertion codes
# (111A, 112B, ...) belong to the region of their position number.
IMGT_REGIONS = {
    "FR1": (1, 26),
    "CDR1": (27, 38),
    "FR2": (39, 55),
    "CDR2": (56, 65),
    "FR3": (66, 104),
    "CDR3": (105, 117),
    "FR4": (118, 128),
}


def number_chain(seq: str, name: str) -> tuple[list[tuple[int, str, str]], str]:
    """Number one variable-domain sequence with ANARCI (IMGT scheme).

    Returns the numbered residues as (position, insertion_code, amino_acid)
    and the chain type: "H" for heavy, "L" for light. ANARCI calls kappa light
    chains "K" and lambda light chains "L"; both are light chains here.
    """
    numbering, alignment_details, _ = anarci([(name, seq)], scheme="imgt", output=False)
    if numbering[0] is None:
        raise ValueError(f"ANARCI could not number {name}")
    domain_numbering, _start, _end = numbering[0][0]
    chain_type = alignment_details[0][0]["chain_type"]
    if chain_type == "K":
        chain_type = "L"
    # ANARCI marks positions the sequence skips with "-"; drop them so only
    # real residues remain.
    residues = [
        (position, insertion.strip(), aa)
        for (position, insertion), aa in domain_numbering
        if aa != "-"
    ]
    return residues, chain_type


def extract_regions(numbered: list[tuple[int, str, str]]) -> dict[str, str]:
    """Split numbered residues into FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4."""
    regions = {region: "" for region in IMGT_REGIONS}
    for position, _insertion, aa in numbered:
        for region, (first, last) in IMGT_REGIONS.items():
            if first <= position <= last:
                regions[region] += aa
                break
    return regions


def number_antibody(vh: str, vl: str, name: str) -> dict[str, str]:
    """Number both chains and return 14 regions keyed H_FR1 ... L_FR4.

    Raises ValueError if the heavy chain is not numbered as heavy or the light
    chain as light, which catches swapped or mislabeled sequences.
    """
    regions = {}
    for prefix, seq, expected_type in (("H", vh, "H"), ("L", vl, "L")):
        numbered, chain_type = number_chain(seq, name)
        if chain_type != expected_type:
            raise ValueError(
                f"{name}: expected a {expected_type} chain, ANARCI found {chain_type}"
            )
        for region, residues in extract_regions(numbered).items():
            regions[f"{prefix}_{region}"] = residues
    return regions
