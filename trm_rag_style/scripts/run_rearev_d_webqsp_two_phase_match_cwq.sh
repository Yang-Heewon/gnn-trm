#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# One-command launcher:
# phase1 (final-KL, CWQ-aligned settings) -> auto best ckpt by dev_hit1 -> phase2.
export RUN_TAG="${RUN_TAG:-webqsp_rearevD_matchcwq_$(date +%Y%m%d_%H%M%S)}"

export DATASET="${DATASET:-webqsp}"
export MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
export EMB_TAG="${EMB_TAG:-e5}"
export EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"

# Use 4 GPUs for both phases.
export PRIMARY_GPUS="${PRIMARY_GPUS:-0,1,2,3}"
export PRIMARY_NPROC="${PRIMARY_NPROC:-4}"
export FALLBACK_GPUS="${FALLBACK_GPUS:-0,1,2,3}"
export FALLBACK_NPROC="${FALLBACK_NPROC:-4}"
export PHASE2_GPUS="${PHASE2_GPUS:-0,1,2,3}"
export PHASE2_NPROC_PER_NODE="${PHASE2_NPROC_PER_NODE:-4}"

# Ports (override if conflicts).
export MASTER_PORT="${MASTER_PORT:-29991}"
export FALLBACK_MASTER_PORT="${FALLBACK_MASTER_PORT:-29992}"
export PHASE2_MASTER_PORT="${PHASE2_MASTER_PORT:-29993}"

# Phase1: align with the CWQ final-KL recipe.
export RUN_PHASE1="${RUN_PHASE1:-true}"
export PHASE1_EPOCHS="${PHASE1_EPOCHS:-16}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export SUBGRAPH_GRAD_ACCUM_STEPS="${SUBGRAPH_GRAD_ACCUM_STEPS:-4}"
export LR="${LR:-1e-4}"
export PHASE1_SUBGRAPH_KL_NO_POSITIVE_MODE="${PHASE1_SUBGRAPH_KL_NO_POSITIVE_MODE:-skip}"
export PHASE1_SUBGRAPH_KL_SUPERVISION_MODE="${PHASE1_SUBGRAPH_KL_SUPERVISION_MODE:-final}"

# Phase2 defaults.
export PHASE2_EPOCHS="${PHASE2_EPOCHS:-16}"
export PHASE2_BATCH_SIZE="${PHASE2_BATCH_SIZE:-1}"
export PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS="${PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS:-8}"
export PHASE2_SUBGRAPH_MAX_NODES="${PHASE2_SUBGRAPH_MAX_NODES:-1536}"
export PHASE2_SUBGRAPH_MAX_EDGES="${PHASE2_SUBGRAPH_MAX_EDGES:-6144}"
export PHASE2_SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS="${PHASE2_SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS:-false}"
export PHASE2_BEST_METRIC="${PHASE2_BEST_METRIC:-dev_hit1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[run] webqsp two-phase auto: phase1 -> best ckpt -> phase2"
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
