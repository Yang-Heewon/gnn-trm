#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"
EMB_TAG="${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"
CKPT_DIR="${CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_subgraph_3gpu_v2_highdim}"

# High-dim / high-recursion preset (v2 objective 유지)
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1.5e-5}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
SUBGRAPH_RECURSION_STEPS="${SUBGRAPH_RECURSION_STEPS:-16}"
SUBGRAPH_LR_SCHEDULER="${SUBGRAPH_LR_SCHEDULER:-cosine}"
SUBGRAPH_LR_MIN="${SUBGRAPH_LR_MIN:-1e-6}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DATASET}_v2_highdim_scratch}"

DATASET="$DATASET" \
MODEL_IMPL="$MODEL_IMPL" \
EMB_MODEL="$EMB_MODEL" \
EMB_TAG="$EMB_TAG" \
EMB_DIR="$EMB_DIR" \
CKPT_DIR="$CKPT_DIR" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
LR="$LR" \
HIDDEN_SIZE="$HIDDEN_SIZE" \
SUBGRAPH_RECURSION_STEPS="$SUBGRAPH_RECURSION_STEPS" \
SUBGRAPH_LR_SCHEDULER="$SUBGRAPH_LR_SCHEDULER" \
SUBGRAPH_LR_MIN="$SUBGRAPH_LR_MIN" \
WANDB_RUN_NAME="$WANDB_RUN_NAME" \
bash trm_rag_style/scripts/run_train_subgraph_v2_resume.sh
