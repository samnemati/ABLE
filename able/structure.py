"""Predicted antibody structures: ABodyBuilder2 prediction and PDB parsing.

ABodyBuilder2 writes a PDB file with the heavy chain as chain H and the light
chain as chain L, residues numbered in the IMGT scheme (the same addresses as
able/encodings.py), and its predicted error for each residue, in Ångströms,
in the B-factor column.
"""

from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import index_to_one, three_to_index


def predict_structure(vh: str, vl: str, out_path: Path) -> Path:
    """Predict and refine the Fv structure of one antibody, saved as PDB."""
    # Imported here so the rest of the package (and the tests) do not need
    # ImmuneBuilder and OpenMM installed.
    from ImmuneBuilder import ABodyBuilder2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ABodyBuilder2().predict({"H": vh, "L": vl}).save(str(out_path))
    return out_path


def read_residues(pdb_path: Path) -> pd.DataFrame:
    """One row per residue: chain, IMGT address, amino acid, Cα position, error."""
    structure = PDBParser(QUIET=True).get_structure("ab", str(pdb_path))
    rows = []
    for chain in structure[0]:
        for residue in chain:
            ca = residue["CA"]
            x, y, z = ca.coord
            rows.append(
                {
                    "chain": chain.id,
                    "position": int(residue.id[1]),
                    # The insertion code is what separates 111A from 111;
                    # dropping it would merge different residues.
                    "insertion": residue.id[2].strip(),
                    "aa": index_to_one(three_to_index(residue.get_resname())),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "error": float(ca.bfactor),
                }
            )
    return pd.DataFrame(rows)


def check_matches_sequence(residues: pd.DataFrame, vh: str, vl: str, name: str) -> None:
    """Refuse a structure whose residues do not spell the input sequences.

    Every later step pairs structure residues with sequence positions, so a
    dropped or changed residue would silently misalign everything after it.
    """
    for chain, seq in (("H", vh), ("L", vl)):
        spelled = "".join(residues.loc[residues["chain"] == chain, "aa"])
        if spelled != seq:
            raise ValueError(f"{name}: structure chain {chain} does not match the input sequence")
