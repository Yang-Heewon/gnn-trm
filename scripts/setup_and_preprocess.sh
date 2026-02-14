#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-all}"
MAX_STEPS="${MAX_STEPS:-4}"
MAX_PATHS="${MAX_PATHS:-4}"
MINE_MAX_NEIGHBORS="${MINE_MAX_NEIGHBORS:-128}"

echo "[step] download + map data"
DATASET="$DATASET" bash scripts/download_data.sh

echo "[step] preprocess"
if [ "$DATASET" = "all" ]; then
  MAX_STEPS="$MAX_STEPS" MAX_PATHS="$MAX_PATHS" MINE_MAX_NEIGHBORS="$MINE_MAX_NEIGHBORS" \
    bash scripts/preprocess_cwq_then_webqsp.sh
else
  python -m trm_rag_style.run \
    --dataset "$DATASET" \
    --stage preprocess \
    --override \
      max_steps="$MAX_STEPS" \
      max_paths="$MAX_PATHS" \
      mine_max_neighbors="$MINE_MAX_NEIGHBORS"
fi

echo "[done] preprocess complete"
