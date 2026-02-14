#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-webqsp}"

echo "[step] download + map data"
bash scripts/download_data.sh

echo "[step] preprocess"
if [ "$DATASET" = "all" ]; then
  python -m trm_rag_style.run --dataset webqsp --stage preprocess
  python -m trm_rag_style.run --dataset cwq --stage preprocess
else
  python -m trm_rag_style.run --dataset "$DATASET" --stage preprocess
fi

echo "[done] preprocess complete"
