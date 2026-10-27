"""Download and load the Jain et al. (2017) antibody developability data."""

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

# PNAS blocks scripted downloads of the supplementary tables, so we use a
# cleaned redistribution of the same data (MIT license). The checksum pins the
# exact file every result in this repo was computed from.
JAIN_URL = (
    "https://huggingface.co/datasets/hugging-science/"
    "jain-developability-clean/resolve/main/jain.csv"
)
JAIN_SHA256 = "67105123c4cf79358aeec43ffb264e660e10bd0d7a72625782669b3c141b0109"

# The 12 assays, keyed by a short name used everywhere else in the code.
# higher_is_better: whether a larger value means a more developable antibody.
# quality: "continuous" is a normal numeric readout. Two assays are not:
#   SGAC-SINS is read in 100 mM salt steps and capped at 0 and 1000, so it is
#   ordinal and censored; CSI-BLI sits near zero for most antibodies (a floor).
#   Later stages should not treat those two as clean regression targets.
ASSAYS = {
    "titer": {
        "column": "hek_titer_mg_l",
        "label": "HEK expression titer (mg/L)",
        "higher_is_better": True,
        "quality": "continuous",
    },
    "fab_tm": {
        "column": "fab_tm_by_dsf_deg_c",
        "label": "Fab melting temperature (°C)",
        "higher_is_better": True,
        "quality": "continuous",
    },
    "sgac_sins": {
        "column": "sgac_sins_as100_nh4_2so4_mm",
        "label": "SGAC-SINS (mM ammonium sulfate)",
        "higher_is_better": True,
        "quality": "ordinal_censored",
    },
    "hic": {
        "column": "hic_retention_time_min_a",
        "label": "HIC retention time (min)",
        "higher_is_better": False,
        "quality": "continuous",
    },
    "smac": {
        "column": "smac_retention_time_min_a",
        "label": "SMAC retention time (min)",
        "higher_is_better": False,
        "quality": "continuous",
    },
    "slope_as": {
        "column": "slope_for_accelerated_stability",
        "label": "Accelerated stability slope",
        "higher_is_better": False,
        "quality": "continuous",
    },
    "psr": {
        "column": "poly_specificity_reagent_psr_smp_score_0_1",
        "label": "Polyspecificity reagent score (0-1)",
        "higher_is_better": False,
        "quality": "continuous",
    },
    "acsins": {
        "column": (
            "affinity_capture_self_interaction_nanoparticle_spectroscopy_"
            "ac_sins_delta_lambda_max_nm_average"
        ),
        "label": "AC-SINS shift (nm)",
        "higher_is_better": False,
        "quality": "continuous",
    },
    "cic": {
        "column": "cic_retention_time_min",
        "label": "CIC retention time (min)",
        "higher_is_better": False,
        "quality": "continuous",
    },
    "csi_bli": {
        "column": "csi_bli_delta_response_nm",
        "label": "CSI-BLI response (nm)",
        "higher_is_better": False,
        "quality": "floor",
    },
    "elisa": {
        "column": "elisa",
        "label": "ELISA (fold over background)",
        "higher_is_better": False,
        "quality": "continuous",
    },
    "bvp_elisa": {
        "column": "bvp_elisa",
        "label": "BVP ELISA (fold over background)",
        "higher_is_better": False,
        "quality": "continuous",
    },
}

DESCRIPTIVE_COLUMNS = ["name", "vh", "vl", "lc_class", "type", "clinical_status"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_jain(dest: Path) -> Path:
    """Download jain.csv into the folder `dest` and verify its checksum.

    If a file with the right checksum is already there, nothing is downloaded.
    A checksum mismatch means the upstream file changed; we stop rather than
    silently analyze different data.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "jain.csv"
    if path.exists() and _sha256(path) == JAIN_SHA256:
        return path
    urllib.request.urlretrieve(JAIN_URL, path)
    actual = _sha256(path)
    if actual != JAIN_SHA256:
        path.unlink()
        raise ValueError(
            f"SHA-256 mismatch for {JAIN_URL}: expected {JAIN_SHA256}, got {actual}"
        )
    return path


def load_jain(path: Path) -> pd.DataFrame:
    """Load the Jain table with short assay names, one row per antibody."""
    raw = pd.read_csv(path)
    df = raw[DESCRIPTIVE_COLUMNS].copy()
    for short_name, info in ASSAYS.items():
        df[short_name] = raw[info["column"]].astype(float)
    return df.set_index("name")
