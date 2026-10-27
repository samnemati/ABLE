"""Download the Jain 2017 data to data/jain.csv and report what was loaded."""

from pathlib import Path

from able.data import ASSAYS, download_jain, load_jain

ROOT = Path(__file__).resolve().parents[1]

path = download_jain(ROOT / "data")
df = load_jain(path)
print(f"{len(df)} antibodies, {len(ASSAYS)} assays")
