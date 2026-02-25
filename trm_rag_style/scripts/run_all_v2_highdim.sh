#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-cwq}"
EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"
EMB_TAG="${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-4}"
EMBED_GPUS="${EMBED_GPUS:-}"

echo "[step 1/3] download"
DATASET="$DATASET" \
bash trm_rag_style/scripts/run_download.sh

echo "[step 2/3] preprocess + embed"
DATASET="$DATASET" \
EMB_MODEL="$EMB_MODEL" \
EMB_TAG="$EMB_TAG" \
EMB_DIR="$EMB_DIR" \
RUN_PREPROCESS=1 \
PREPROCESS_WORKERS="$PREPROCESS_WORKERS" \
EMBED_GPUS="$EMBED_GPUS" \
bash trm_rag_style/scripts/run_embed.sh

echo "[step 3/3] train(v2 highdim) + auto-test"
DATASET="$DATASET" \
EMB_MODEL="$EMB_MODEL" \
EMB_TAG="$EMB_TAG" \
EMB_DIR="$EMB_DIR" \
bash trm_rag_style/scripts/run_train_subgraph_v2_highdim.sh

echo "[done] v2 highdim full pipeline finished"
