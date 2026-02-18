#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Dataset/model
export DATASET="${DATASET:-cwq}"
export MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"

# Preprocess settings (paper-style supervision: keep all mined paths)
export MAX_STEPS="${MAX_STEPS:-4}"
export MAX_PATHS="${MAX_PATHS:-4}"
export MINE_MAX_NEIGHBORS="${MINE_MAX_NEIGHBORS:-128}"
export TRAIN_PATH_POLICY="${TRAIN_PATH_POLICY:-all}"
export TRAIN_SHORTEST_K="${TRAIN_SHORTEST_K:-1}"

# Train settings
export BATCH_SIZE="${BATCH_SIZE:-6}"
export LR="${LR:-1e-4}"
export EPOCHS_PHASE1="${EPOCHS_PHASE1:-30}"
export EPOCHS_PHASE2="${EPOCHS_PHASE2:-20}"

# Distributed settings
export TORCHRUN="${TORCHRUN:-/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export DDP_FIND_UNUSED="${DDP_FIND_UNUSED:-true}"

# W&B settings
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
export WANDB_ENTITY="${WANDB_ENTITY:-heewon6205-chung-ang-university}"
export WANDB_RUN_NAME_PHASE1="${WANDB_RUN_NAME_PHASE1:-${DATASET}_phase1_paperstyle_ep${EPOCHS_PHASE1}}"
export WANDB_RUN_NAME_PHASE2="${WANDB_RUN_NAME_PHASE2:-${DATASET}_phase2_rl_from_phase1_ep${EPOCHS_PHASE2}}"

# Output dirs
export CKPT_DIR_PHASE1="${CKPT_DIR_PHASE1:-trm_rag_style/ckpt/${DATASET}_${MODEL_IMPL}_phase1_paperstyle}"
export CKPT_DIR_PHASE2="${CKPT_DIR_PHASE2:-trm_rag_style/ckpt/${DATASET}_${MODEL_IMPL}_phase2_rl_from_phase1}"
