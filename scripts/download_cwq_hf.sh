#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATASET="cwq" \
CWQ_VOCAB_ONLY="0" \
DATA_SOURCE="${DATA_SOURCE:-rog_hf}" \
ROG_CWQ_DATASET="${ROG_CWQ_DATASET:-rmanluo/RoG-cwq}" \
HF_CACHE_DIR="${HF_CACHE_DIR:-}" \
bash scripts/download_data.sh

echo "[done] CWQ HF download complete"
echo "  - data/CWQ/train_split.jsonl"
echo "  - data/CWQ/dev_split.jsonl"
echo "  - data/CWQ/test_split.jsonl"
echo "  - data/CWQ/entities.txt"
echo "  - data/CWQ/relations.txt"
echo "  - data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt"
echo "  - data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt"
