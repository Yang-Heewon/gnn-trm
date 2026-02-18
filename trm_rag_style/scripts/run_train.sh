#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
DATASET=${DATASET:-webqsp}
MODEL_IMPL=${MODEL_IMPL:-trm_hier6}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-4}
MAX_STEPS=${MAX_STEPS:-4}
MAX_NEIGHBORS=${MAX_NEIGHBORS:-256}
PRUNE_KEEP=${PRUNE_KEEP:-64}
PRUNE_RAND=${PRUNE_RAND:-64}
BEAM=${BEAM:-8}
START_TOPK=${START_TOPK:-5}
EVAL_LIMIT=${EVAL_LIMIT:-200}
DEBUG_EVAL_N=${DEBUG_EVAL_N:-5}
EVAL_EVERY_EPOCHS=${EVAL_EVERY_EPOCHS:-2}
EVAL_START_EPOCH=${EVAL_START_EPOCH:-2}
EVAL_NO_CYCLE=${EVAL_NO_CYCLE:-true}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-4}
EVAL_MAX_NEIGHBORS=${EVAL_MAX_NEIGHBORS:-256}
EVAL_PRUNE_KEEP=${EVAL_PRUNE_KEEP:-64}
EVAL_BEAM=${EVAL_BEAM:-8}
EVAL_START_TOPK=${EVAL_START_TOPK:-5}
EVAL_PRED_TOPK=${EVAL_PRED_TOPK:-5}
EVAL_USE_HALT=${EVAL_USE_HALT:-true}
EVAL_MIN_HOPS_BEFORE_STOP=${EVAL_MIN_HOPS_BEFORE_STOP:-2}
ENDPOINT_AUX_WEIGHT=${ENDPOINT_AUX_WEIGHT:-0.2}
HALT_AUX_WEIGHT=${HALT_AUX_WEIGHT:-0.1}
ORACLE_DIAG_ENABLED=${ORACLE_DIAG_ENABLED:-true}
ORACLE_DIAG_LIMIT=${ORACLE_DIAG_LIMIT:-500}
ORACLE_DIAG_FAIL_THRESHOLD=${ORACLE_DIAG_FAIL_THRESHOLD:--1}
ORACLE_DIAG_ONLY=${ORACLE_DIAG_ONLY:-false}
WANDB_MODE=${WANDB_MODE:-disabled}
WANDB_PROJECT=${WANDB_PROJECT:-graph-traverse}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
DDP_FIND_UNUSED=${DDP_FIND_UNUSED:-true}

TORCHRUN=${TORCHRUN:-torchrun}
$TORCHRUN --nproc_per_node=3 --master_port=29500 -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --embedding_model "$EMB_MODEL" \
  --stage train \
  --override \
    epochs="$EPOCHS" \
    batch_size="$BATCH_SIZE" \
    lr="$LR" \
    max_steps="$MAX_STEPS" \
    max_neighbors="$MAX_NEIGHBORS" \
    prune_keep="$PRUNE_KEEP" \
    prune_rand="$PRUNE_RAND" \
    beam="$BEAM" \
    start_topk="$START_TOPK" \
    eval_limit="$EVAL_LIMIT" \
    debug_eval_n="$DEBUG_EVAL_N" \
    eval_every_epochs="$EVAL_EVERY_EPOCHS" \
    eval_start_epoch="$EVAL_START_EPOCH" \
    eval_no_cycle="$EVAL_NO_CYCLE" \
    eval_max_steps="$EVAL_MAX_STEPS" \
    eval_max_neighbors="$EVAL_MAX_NEIGHBORS" \
    eval_prune_keep="$EVAL_PRUNE_KEEP" \
    eval_beam="$EVAL_BEAM" \
    eval_start_topk="$EVAL_START_TOPK" \
    eval_pred_topk="$EVAL_PRED_TOPK" \
    eval_use_halt="$EVAL_USE_HALT" \
    eval_min_hops_before_stop="$EVAL_MIN_HOPS_BEFORE_STOP" \
    endpoint_aux_weight="$ENDPOINT_AUX_WEIGHT" \
    halt_aux_weight="$HALT_AUX_WEIGHT" \
    oracle_diag_enabled="$ORACLE_DIAG_ENABLED" \
    oracle_diag_limit="$ORACLE_DIAG_LIMIT" \
    oracle_diag_fail_threshold="$ORACLE_DIAG_FAIL_THRESHOLD" \
    oracle_diag_only="$ORACLE_DIAG_ONLY" \
    ddp_find_unused="$DDP_FIND_UNUSED" \
    wandb_mode="$WANDB_MODE" \
    wandb_project="$WANDB_PROJECT" \
    wandb_entity="$WANDB_ENTITY" \
    wandb_run_name="$WANDB_RUN_NAME"
