from pathlib import Path

import pytest

from able import data
from able.data import ASSAYS, download_jain, load_jain

FIXTURE = Path(__file__).parent / "fixtures" / "jain_head.csv"


def test_load_jain_columns_and_index():
    df = load_jain(FIXTURE)
    assert len(df) == 3
    assert "trastuzumab" in df.index
    for short_name in ASSAYS:
        assert df[short_name].dtype == float


def test_load_jain_values():
    df = load_jain(FIXTURE)
    assert df.loc["trastuzumab", "vh"].startswith("EVQLVESGGGLVQPGGSLRLSCAAS")
    assert df.loc["belimumab", "lc_class"] == "lambda"


def test_assay_quality_flags():
    assert ASSAYS["sgac_sins"]["quality"] == "ordinal_censored"
    assert ASSAYS["csi_bli"]["quality"] == "floor"
    assert ASSAYS["hic"]["higher_is_better"] is False
    assert ASSAYS["fab_tm"]["higher_is_better"] is True


def test_download_rejects_bad_checksum(tmp_path, monkeypatch):
    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(b"not the file")

    monkeypatch.setattr(data.urllib.request, "urlretrieve", fake_urlretrieve)
    with pytest.raises(ValueError, match="SHA-256"):
        download_jain(tmp_path)
