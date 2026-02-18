#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

if [ -n "${PHASE1_CKPT:-}" ]; then
  CKPT="$PHASE1_CKPT"
else
  CKPT="$(ls -1 "$CKPT_DIR_PHASE1"/model_ep*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then
  echo "Phase1 checkpoint not found. Run 02_train_phase1.sh first or set PHASE1_CKPT=/path/to/model_epX.pt"
  exit 1
fi

echo "Using phase1 checkpoint: $CKPT"

DATASET="$DATASET" \
MODEL_IMPL="$MODEL_IMPL" \
CKPT="$CKPT" \
CKPT_DIR="$CKPT_DIR_PHASE2" \
EPOCHS="$EPOCHS_PHASE2" \
PHASE2_START_EPOCH=1 \
BATCH_SIZE="$BATCH_SIZE" \
LR="$LR" \
MAX_STEPS="$MAX_STEPS" \
EVAL_MAX_STEPS="$MAX_STEPS" \
TORCHRUN="$TORCHRUN" \
NPROC_PER_NODE="$NPROC_PER_NODE_PHASE2" \
MASTER_PORT="$MASTER_PORT" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
DDP_FIND_UNUSED="$DDP_FIND_UNUSED" \
DDP_TIMEOUT_MINUTES="$DDP_TIMEOUT_MINUTES" \
WANDB_MODE="$WANDB_MODE" \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_ENTITY="$WANDB_ENTITY" \
WANDB_RUN_NAME="$WANDB_RUN_NAME_PHASE2" \
bash trm_rag_style/scripts/run_train_2phase_rl_wandb.sh
