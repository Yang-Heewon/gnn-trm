#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET="${DATASET:-webqsp}" # cwq | webqsp
if [ "$DATASET" != "cwq" ] && [ "$DATASET" != "webqsp" ]; then
  echo "[err] DATASET must be cwq or webqsp (got: $DATASET)"
  exit 2
fi

EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"
EMB_TAG="${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}"
EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
EMBED_GPUS="${EMBED_GPUS:-}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-64}"
EMBED_MAX_LENGTH="${EMBED_MAX_LENGTH:-128}"
EMBED_STYLE="${EMBED_STYLE:-gnn_rag_gnn_exact}"
EMBED_BACKEND="${EMBED_BACKEND:-sentence_transformers}"
ENTITY_NAMES_JSON="${ENTITY_NAMES_JSON:-data/data/entities_names.json}"

DOWNLOAD_FIRST="${DOWNLOAD_FIRST:-1}"
RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
DATA_SOURCE="${DATA_SOURCE:-rog_hf}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
ROG_CWQ_DATASET="${ROG_CWQ_DATASET:-rmanluo/RoG-cwq}"
ROG_WEBQSP_DATASET="${ROG_WEBQSP_DATASET:-rmanluo/RoG-webqsp}"

MAX_STEPS="${MAX_STEPS:-4}"
MAX_PATHS="${MAX_PATHS:-4}"
MINE_MAX_NEIGHBORS="${MINE_MAX_NEIGHBORS:-128}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-0}"

if [ "$DOWNLOAD_FIRST" = "1" ]; then
  echo "[step] download + map data"
  DATASET="$DATASET" \
  DATA_SOURCE="$DATA_SOURCE" \
  HF_CACHE_DIR="$HF_CACHE_DIR" \
  ROG_CWQ_DATASET="$ROG_CWQ_DATASET" \
  ROG_WEBQSP_DATASET="$ROG_WEBQSP_DATASET" \
  bash scripts/download_data.sh
fi

COMMON_OVR=(
  "emb_tag=$EMB_TAG"
  "embed_device=$EMBED_DEVICE"
  "embed_gpus=$EMBED_GPUS"
  "embed_batch_size=$EMBED_BATCH_SIZE"
  "embed_max_length=$EMBED_MAX_LENGTH"
  "embed_style=$EMBED_STYLE"
  "embed_backend=$EMBED_BACKEND"
)
if [ -n "$ENTITY_NAMES_JSON" ]; then
  COMMON_OVR+=("entity_names_json=$ENTITY_NAMES_JSON")
fi

if [ "$RUN_PREPROCESS" = "1" ]; then
  echo "[step] preprocess"
  $PYTHON_BIN -m trm_agent.run \
    --dataset "$DATASET" \
    --stage preprocess \
    --embedding_model "$EMB_MODEL" \
    --override \
      max_steps="$MAX_STEPS" \
      max_paths="$MAX_PATHS" \
      mine_max_neighbors="$MINE_MAX_NEIGHBORS" \
      preprocess_workers="$PREPROCESS_WORKERS" \
      "${COMMON_OVR[@]}"
fi

echo "[step] embed"
$PYTHON_BIN -m trm_agent.run \
  --dataset "$DATASET" \
  --stage embed \
  --embedding_model "$EMB_MODEL" \
  --override "${COMMON_OVR[@]}"

echo "[done] download + preprocess + embed complete"
echo "       emb dir: trm_agent/emb/${DATASET}_${EMB_TAG}"
