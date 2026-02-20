#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Dataset/model
export DATASET="${DATASET:-cwq}"
export MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
# Match GNN-RAG gnn defaults.
export EMB_MODEL="${EMB_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
export EMB_TAG="${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}"
export DATA_SOURCE="${DATA_SOURCE:-rog_hf}"
export ROG_CWQ_DATASET="${ROG_CWQ_DATASET:-rmanluo/RoG-cwq}"
export ROG_WEBQSP_DATASET="${ROG_WEBQSP_DATASET:-rmanluo/RoG-webqsp}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-}"

# Preprocess settings (paper-style supervision: keep all mined paths)
export MAX_STEPS="${MAX_STEPS:-4}"
export MAX_PATHS="${MAX_PATHS:-4}"
export MINE_MAX_NEIGHBORS="${MINE_MAX_NEIGHBORS:-128}"
export TRAIN_PATH_POLICY="${TRAIN_PATH_POLICY:-all}"
export TRAIN_SHORTEST_K="${TRAIN_SHORTEST_K:-1}"

# Embedding settings (GNN-RAG gnn-compatible style)
export EMBED_STYLE="${EMBED_STYLE:-gnn_rag_gnn_exact}"
export EMBED_BACKEND="${EMBED_BACKEND:-sentence_transformers}"
export EMBED_QUERY_PREFIX="${EMBED_QUERY_PREFIX:-}"
export EMBED_PASSAGE_PREFIX="${EMBED_PASSAGE_PREFIX:-}"
export ENTITY_NAMES_JSON="${ENTITY_NAMES_JSON:-data/data/entities_names.json}"
export RUN_PREPROCESS="${RUN_PREPROCESS:-1}"

# Train settings (phase1/phase2 split defaults for 10~12GB GPUs)
export BATCH_SIZE="${BATCH_SIZE:-6}"
export BATCH_SIZE_PHASE1="${BATCH_SIZE_PHASE1:-$BATCH_SIZE}"
export BATCH_SIZE_PHASE2="${BATCH_SIZE_PHASE2:-2}"
export TRAIN_ACC_MODE="${TRAIN_ACC_MODE:-relation}"
export TRAIN_SANITY_EVAL_EVERY_PCT="${TRAIN_SANITY_EVAL_EVERY_PCT:-10}"
export TRAIN_SANITY_EVAL_LIMIT="${TRAIN_SANITY_EVAL_LIMIT:-5}"
export TRAIN_SANITY_EVAL_BEAM="${TRAIN_SANITY_EVAL_BEAM:-5}"
export TRAIN_SANITY_EVAL_START_TOPK="${TRAIN_SANITY_EVAL_START_TOPK:-5}"
export TRAIN_SANITY_EVAL_PRED_TOPK="${TRAIN_SANITY_EVAL_PRED_TOPK:-1}"
export TRAIN_SANITY_EVAL_NO_CYCLE="${TRAIN_SANITY_EVAL_NO_CYCLE:-false}"
export TRAIN_SANITY_EVAL_USE_HALT="${TRAIN_SANITY_EVAL_USE_HALT:-false}"
export LR="${LR:-2e-4}"
export EPOCHS_PHASE1="${EPOCHS_PHASE1:-5}"
export EPOCHS_PHASE2="${EPOCHS_PHASE2:-20}"
export EVAL_LIMIT="${EVAL_LIMIT:-200}"
export TRAIN_STYLE="${TRAIN_STYLE:-gnn_rag}"

# Distributed settings
export TORCHRUN="${TORCHRUN:-/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export NPROC_PER_NODE_PHASE2="${NPROC_PER_NODE_PHASE2:-1}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export DDP_FIND_UNUSED="${DDP_FIND_UNUSED:-true}"
# Full-dev eval can exceed 60 minutes on multi-GPU DDP.
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}"

# W&B settings
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
export WANDB_ENTITY="${WANDB_ENTITY:-heewon6205-chung-ang-university}"
export WANDB_RUN_NAME_PHASE1="${WANDB_RUN_NAME_PHASE1:-${DATASET}_phase1_paperstyle_ep${EPOCHS_PHASE1}}"
export WANDB_RUN_NAME_PHASE2="${WANDB_RUN_NAME_PHASE2:-${DATASET}_phase2_rl_from_phase1_ep${EPOCHS_PHASE2}}"

# Output dirs
export CKPT_DIR_PHASE1="${CKPT_DIR_PHASE1:-trm_rag_style/ckpt/${DATASET}_${MODEL_IMPL}_phase1_paperstyle}"
export CKPT_DIR_PHASE2="${CKPT_DIR_PHASE2:-trm_rag_style/ckpt/${DATASET}_${MODEL_IMPL}_phase2_rl_from_phase1}"
export RESULTS_DIR="${RESULTS_DIR:-experiments/paperstyle_rl/results}"
export TEST_EVAL_LIMIT="${TEST_EVAL_LIMIT:--1}"
export TEST_DEBUG_EVAL_N="${TEST_DEBUG_EVAL_N:-5}"
export TEST_EVAL_NO_CYCLE="${TEST_EVAL_NO_CYCLE:-true}"
export TEST_EVAL_PRED_TOPK="${TEST_EVAL_PRED_TOPK:-1}"
