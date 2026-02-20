#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATASET="cwq" \
CWQ_VOCAB_ONLY="1" \
DATA_SOURCE="${DATA_SOURCE:-rog_hf}" \
ROG_CWQ_DATASET="${ROG_CWQ_DATASET:-rmanluo/RoG-cwq}" \
HF_CACHE_DIR="${HF_CACHE_DIR:-}" \
bash scripts/download_data.sh

echo "[done] CWQ vocab-only download complete"
echo "  - data/CWQ/entities.txt"
echo "  - data/CWQ/relations.txt"
