#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Paper-style phase1 preset (legacy-like):
# - Supervised relation CE as main objective
# - No halt/endpoint auxiliary loss
# - Legacy-friendly train/eval defaults

export DATASET="${DATASET:-cwq}"
export MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
export EMB_MODEL="${EMB_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
export EMB_TAG="${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}"
export EPOCHS="${EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export LR="${LR:-2e-4}"
export MAX_STEPS="${MAX_STEPS:-4}"
export BEAM="${BEAM:-10}"
export START_TOPK="${START_TOPK:-5}"

export EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-4}"
export EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-1}"
export EVAL_START_EPOCH="${EVAL_START_EPOCH:-1}"
export EVAL_LIMIT="${EVAL_LIMIT:-200}"
export EVAL_PRED_TOPK="${EVAL_PRED_TOPK:-10}"
export EVAL_USE_HALT="${EVAL_USE_HALT:-false}"
export EVAL_NO_CYCLE="${EVAL_NO_CYCLE:-false}"
export EVAL_BEAM="${EVAL_BEAM:-10}"
export EVAL_START_TOPK="${EVAL_START_TOPK:-5}"
export TRAIN_STYLE="${TRAIN_STYLE:-gnn_rag}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}"

# Objective alignment to legacy/paper-style phase1.
export ENDPOINT_LOSS_MODE="${ENDPOINT_LOSS_MODE:-aux}"
export RELATION_AUX_WEIGHT="${RELATION_AUX_WEIGHT:-1.0}"
export ENDPOINT_AUX_WEIGHT="${ENDPOINT_AUX_WEIGHT:-0.0}"
export METRIC_ALIGN_AUX_WEIGHT="${METRIC_ALIGN_AUX_WEIGHT:-0.0}"
export HALT_AUX_WEIGHT="${HALT_AUX_WEIGHT:-0.0}"
export TRAIN_ACC_MODE="${TRAIN_ACC_MODE:-relation}"
export TRAIN_SANITY_EVAL_EVERY_PCT="${TRAIN_SANITY_EVAL_EVERY_PCT:-10}"
export TRAIN_SANITY_EVAL_LIMIT="${TRAIN_SANITY_EVAL_LIMIT:-5}"
export TRAIN_SANITY_EVAL_BEAM="${TRAIN_SANITY_EVAL_BEAM:-5}"
export TRAIN_SANITY_EVAL_START_TOPK="${TRAIN_SANITY_EVAL_START_TOPK:-5}"
export TRAIN_SANITY_EVAL_PRED_TOPK="${TRAIN_SANITY_EVAL_PRED_TOPK:-1}"
export TRAIN_SANITY_EVAL_NO_CYCLE="${TRAIN_SANITY_EVAL_NO_CYCLE:-false}"
export TRAIN_SANITY_EVAL_USE_HALT="${TRAIN_SANITY_EVAL_USE_HALT:-false}"
export PHASE2_START_EPOCH="${PHASE2_START_EPOCH:-0}"
export PHASE2_AUTO_ENABLED="${PHASE2_AUTO_ENABLED:-false}"

# DDP/W&B defaults.
export DDP_FIND_UNUSED="${DDP_FIND_UNUSED:-true}"
export TORCHRUN="${TORCHRUN:-/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

export CKPT="${CKPT:-}"
export CKPT_DIR="${CKPT_DIR:-trm_rag_style/ckpt/${DATASET}_${MODEL_IMPL}_phase1_paperstyle}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
export WANDB_ENTITY="${WANDB_ENTITY:-heewon6205-chung-ang-university}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DATASET}_phase1_paperstyle_ep${EPOCHS}}"

bash trm_rag_style/scripts/run_train.sh
