#!/usr/bin/env bash
# Run the full stage 1 pipeline in order, stopping at the first error.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/10_download.py
python scripts/20_number_and_features.py
python scripts/30_explore.py
python scripts/40_embed.py
python scripts/50_evaluate.py
# Structure prediction needs ImmuneBuilder + OpenMM (see README); skip if absent
# and use whatever structures are already in cache/structures/.
if python -c "import ImmuneBuilder" 2>/dev/null; then
  python scripts/45_predict_structures.py
else
  echo "ImmuneBuilder not installed: skipping structure prediction"
fi
python scripts/55_structure_analysis.py
