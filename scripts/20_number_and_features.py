"""Number every antibody, extract its regions and compute baseline features.

Writes outputs/t_regions.csv (14 region strings per antibody) and
outputs/t_features.csv (baseline features plus the 12 assays).
"""

from pathlib import Path

import pandas as pd

from fvcast.data import ASSAYS, load_jain
from fvcast.features import sequence_features
from fvcast.numbering import number_antibody

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = load_jain(ROOT / "data" / "jain.csv")

region_rows, feature_rows, failed = [], [], []
for name, row in df.iterrows():
    try:
        regions = number_antibody(row["vh"], row["vl"], name)
    except ValueError as err:
        failed.append(f"{name}: {err}")
        continue
    region_rows.append({"name": name, **regions})
    feature_rows.append({"name": name, **sequence_features(row["vh"], row["vl"], regions)})

print(f"Numbering failed for {len(failed)} antibodies")
for line in failed:
    print("  " + line)

pd.DataFrame(region_rows).to_csv(OUT / "t_regions.csv", index=False)
features = pd.DataFrame(feature_rows).set_index("name").join(df[list(ASSAYS)])
features.to_csv(OUT / "t_features.csv")
print(f"Wrote {len(features)} rows to outputs/t_features.csv")

if failed:
    raise SystemExit(1)
