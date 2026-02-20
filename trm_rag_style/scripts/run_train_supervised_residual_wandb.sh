#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Supervised multi-hop preset with query residual update.
# - Keep candidate search inside per-example subgraph (already default in collate/eval).
# - Emphasize endpoint distribution (multi-answer friendly) + relation supervision.
# - Enable halt/no-cycle for cleaner endpoint behavior.

export DATASET="${DATASET:-cwq}"
export MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
export EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"
export EMB_TAG="${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}"

export EPOCHS="${EPOCHS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-6}"
export LR="${LR:-2e-4}"
export MAX_STEPS="${MAX_STEPS:-4}"
export MAX_NEIGHBORS="${MAX_NEIGHBORS:-128}"
export PRUNE_KEEP="${PRUNE_KEEP:-64}"
export PRUNE_RAND="${PRUNE_RAND:-32}"

# Core objective: entity distribution main (multi-answer friendly),
# plus relation CE auxiliary.
export ENDPOINT_LOSS_MODE="${ENDPOINT_LOSS_MODE:-entity_dist_main}"
export RELATION_AUX_WEIGHT="${RELATION_AUX_WEIGHT:-0.5}"
export ENDPOINT_AUX_WEIGHT="${ENDPOINT_AUX_WEIGHT:-0.0}"
export METRIC_ALIGN_AUX_WEIGHT="${METRIC_ALIGN_AUX_WEIGHT:-0.0}"
export HALT_AUX_WEIGHT="${HALT_AUX_WEIGHT:-0.1}"

# Query residual update: q_{t+1} <- norm(q_t - alpha * r_t)
export QUERY_RESIDUAL_ENABLED="${QUERY_RESIDUAL_ENABLED:-true}"
export QUERY_RESIDUAL_ALPHA="${QUERY_RESIDUAL_ALPHA:-0.15}"
export QUERY_RESIDUAL_MODE="${QUERY_RESIDUAL_MODE:-sub_rel}"

# Evaluation / search behavior for Hit@1 + F1.
export EVAL_NO_CYCLE="${EVAL_NO_CYCLE:-true}"
export EVAL_USE_HALT="${EVAL_USE_HALT:-true}"
export EVAL_MIN_HOPS_BEFORE_STOP="${EVAL_MIN_HOPS_BEFORE_STOP:-1}"
export EVAL_PRED_TOPK="${EVAL_PRED_TOPK:-10}"
export EVAL_BEAM="${EVAL_BEAM:-10}"
export START_TOPK="${START_TOPK:-5}"
export EVAL_START_TOPK="${EVAL_START_TOPK:-5}"
export TRAIN_ACC_MODE="${TRAIN_ACC_MODE:-endpoint_proxy}"
export EVAL_LIMIT="${EVAL_LIMIT:-200}"
export EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-1}"
export EVAL_START_EPOCH="${EVAL_START_EPOCH:-1}"

export TORCHRUN="${TORCHRUN:-/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
export WANDB_ENTITY="${WANDB_ENTITY:-heewon6205-chung-ang-university}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DATASET}_sup_residual_ep${EPOCHS}}"

export CKPT_DIR="${CKPT_DIR:-trm_rag_style/ckpt/${DATASET}_${MODEL_IMPL}_sup_residual}"

bash trm_rag_style/scripts/run_train.sh
