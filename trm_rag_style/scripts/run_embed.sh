#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET="${DATASET:-cwq}" # cwq | webqsp | all
if [ "$DATASET" != "cwq" ] && [ "$DATASET" != "webqsp" ] && [ "$DATASET" != "all" ]; then
  echo "[err] DATASET must be cwq, webqsp, or all (got: $DATASET)"
  exit 2
fi

EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"
EMB_TAG="${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}"
EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
EMBED_GPUS="${EMBED_GPUS:-}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"
EMBED_MAX_LENGTH="${EMBED_MAX_LENGTH:-128}"
EMBED_STYLE="${EMBED_STYLE:-gnn_rag_gnn_exact}"
EMBED_BACKEND="${EMBED_BACKEND:-sentence_transformers}"
ENTITY_NAMES_JSON="${ENTITY_NAMES_JSON:-data/data/entities_names.json}"

RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
MAX_STEPS="${MAX_STEPS:-4}"
MAX_PATHS="${MAX_PATHS:-4}"
MINE_MAX_NEIGHBORS="${MINE_MAX_NEIGHBORS:-128}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-0}"
TRAIN_PATH_POLICY="${TRAIN_PATH_POLICY:-all}"
TRAIN_SHORTEST_K="${TRAIN_SHORTEST_K:-1}"

TARGET_DATASETS=()
if [ "$DATASET" = "all" ]; then
  TARGET_DATASETS=("cwq" "webqsp")
else
  TARGET_DATASETS=("$DATASET")
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

for ds in "${TARGET_DATASETS[@]}"; do
  if [ "$ds" = "cwq" ]; then
    REQUIRED_RAW=(
      "data/CWQ/train_split.jsonl"
      "data/CWQ/dev_split.jsonl"
      "data/CWQ/entities.txt"
      "data/CWQ/relations.txt"
    )
  else
    REQUIRED_RAW=(
      "data/webqsp/train.json"
      "data/webqsp/dev.json"
      "data/webqsp/entities.txt"
      "data/webqsp/relations.txt"
    )
  fi

  for p in "${REQUIRED_RAW[@]}"; do
    if [ ! -f "$p" ]; then
      echo "[err][$ds] required data file not found: $p"
      echo "           run: bash trm_rag_style/scripts/run_download.sh"
      exit 2
    fi
  done

  if [ "$RUN_PREPROCESS" = "1" ]; then
    echo "[step][$ds] preprocess"
    $PYTHON_BIN -m trm_agent.run \
      --dataset "$ds" \
      --stage preprocess \
      --embedding_model "$EMB_MODEL" \
      --override \
        max_steps="$MAX_STEPS" \
        max_paths="$MAX_PATHS" \
        mine_max_neighbors="$MINE_MAX_NEIGHBORS" \
        preprocess_workers="$PREPROCESS_WORKERS" \
        train_path_policy="$TRAIN_PATH_POLICY" \
        train_shortest_k="$TRAIN_SHORTEST_K" \
        "${COMMON_OVR[@]}"
  fi

  if [ "$RUN_PREPROCESS" != "1" ]; then
    if [ ! -f "trm_agent/processed/${ds}/train.jsonl" ] || [ ! -f "trm_agent/processed/${ds}/dev.jsonl" ]; then
      echo "[err][$ds] processed files not found under trm_agent/processed/${ds}"
      echo "           set RUN_PREPROCESS=1 or run preprocess stage first."
      exit 2
    fi
  fi

  echo "[step][$ds] embed"
  $PYTHON_BIN -m trm_agent.run \
    --dataset "$ds" \
    --stage embed \
    --embedding_model "$EMB_MODEL" \
    --override "${COMMON_OVR[@]}"

  echo "[done][$ds] preprocess + embed complete"
  echo "          emb dir: trm_agent/emb/${ds}_${EMB_TAG}"
done
