#!/usr/bin/env bash
# Run the full stage 1 pipeline in order, stopping at the first error.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/10_download.py
python scripts/20_number_and_features.py
python scripts/30_explore.py
python scripts/40_embed.py
python scripts/50_evaluate.py
