#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-webqsp}" # cwq | webqsp | all
DATA_SOURCE="${DATA_SOURCE:-rog_hf}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
ROG_CWQ_DATASET="${ROG_CWQ_DATASET:-rmanluo/RoG-cwq}"
ROG_WEBQSP_DATASET="${ROG_WEBQSP_DATASET:-rmanluo/RoG-webqsp}"

if [ "$DATASET" != "cwq" ] && [ "$DATASET" != "webqsp" ] && [ "$DATASET" != "all" ]; then
  echo "[err] DATASET must be cwq, webqsp, or all (got: $DATASET)"
  exit 2
fi

echo "[step] download + map data (DATASET=$DATASET, DATA_SOURCE=$DATA_SOURCE)"
DATASET="$DATASET" \
DATA_SOURCE="$DATA_SOURCE" \
HF_CACHE_DIR="$HF_CACHE_DIR" \
ROG_CWQ_DATASET="$ROG_CWQ_DATASET" \
ROG_WEBQSP_DATASET="$ROG_WEBQSP_DATASET" \
bash scripts/download_data.sh

echo "[done] download complete"
