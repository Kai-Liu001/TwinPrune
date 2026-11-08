#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"
python scripts/test_unified_contracts.py
python -m twinprune.evaluate \
  --cell internvl35_instruct_prefill_b2_cac_v4_textvqa \
  --model "${TWINPRUNE_MODEL:-OpenGVLab/InternVL3_5-8B-Instruct}" \
  --data-config "${TWINPRUNE_DATA_CONFIG:-configs/data.example.yaml}" \
  --backend legacy --max-samples 20 --output-dir "${TWINPRUNE_OUTPUT_DIR:-outputs/quickstart}"
echo "Completed 20-sample smoke evaluation. Use REPRODUCE.md for full metric checks."
