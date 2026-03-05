#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PRIMARY_GPUS="${PRIMARY_GPUS:-0,1,2,3}"
PRIMARY_NPROC="${PRIMARY_NPROC:-4}"
FALLBACK_GPUS="${FALLBACK_GPUS:-0,1}"
FALLBACK_NPROC="${FALLBACK_NPROC:-2}"

RUN_SCRIPT="${RUN_SCRIPT:-trm_rag_style/scripts/run_rearev_d_gpu01_hit06.sh}"
LOG_DIR="${LOG_DIR:-logs/rearev_d_auto_fallback}"
mkdir -p "$LOG_DIR"

# Shared training knobs (can be overridden by env before launching this wrapper).
export BATCH_SIZE="${BATCH_SIZE:-1}"
export SUBGRAPH_GRAD_ACCUM_STEPS="${SUBGRAPH_GRAD_ACCUM_STEPS:-8}"
export SUBGRAPH_MAX_NODES="${SUBGRAPH_MAX_NODES:-4096}"
export SUBGRAPH_MAX_EDGES="${SUBGRAPH_MAX_EDGES:-16384}"
export SUBGRAPH_LOSS_MODE="${SUBGRAPH_LOSS_MODE:-rearev_kl_rank}"
export SUBGRAPH_KL_NO_POSITIVE_MODE="${SUBGRAPH_KL_NO_POSITIVE_MODE:-uniform}"
export SUBGRAPH_KL_SUPERVISION_MODE="${SUBGRAPH_KL_SUPERVISION_MODE:-final}"
export SUBGRAPH_RANKING_ENABLED="${SUBGRAPH_RANKING_ENABLED:-true}"
export SUBGRAPH_RANKING_WEIGHT="${SUBGRAPH_RANKING_WEIGHT:-0.5}"
export SUBGRAPH_RANKING_MARGIN="${SUBGRAPH_RANKING_MARGIN:-0.2}"
export SUBGRAPH_HARD_NEGATIVE_TOPK="${SUBGRAPH_HARD_NEGATIVE_TOPK:-64}"
export SUBGRAPH_REAREV_NORMALIZED_GNN="${SUBGRAPH_REAREV_NORMALIZED_GNN:-false}"
export SUBGRAPH_EARLY_STOP_ENABLED="${SUBGRAPH_EARLY_STOP_ENABLED:-false}"
export SUBGRAPH_EARLY_STOP_METRIC="${SUBGRAPH_EARLY_STOP_METRIC:-dev_hit1}"
export SUBGRAPH_EARLY_STOP_PATIENCE="${SUBGRAPH_EARLY_STOP_PATIENCE:-0}"
export SUBGRAPH_EARLY_STOP_MIN_DELTA="${SUBGRAPH_EARLY_STOP_MIN_DELTA:-1e-4}"
export SUBGRAPH_EARLY_STOP_MIN_EPOCHS="${SUBGRAPH_EARLY_STOP_MIN_EPOCHS:-1}"
export SUBGRAPH_DEEP_SUPERVISION_ENABLED="${SUBGRAPH_DEEP_SUPERVISION_ENABLED:-false}"
export SUBGRAPH_DEEP_SUPERVISION_WEIGHT="${SUBGRAPH_DEEP_SUPERVISION_WEIGHT:-0.0}"
export SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT="${SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT:-1.0}"
export SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT="${SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT:-1.0}"
export SUBGRAPH_REAREV_TRM_TMINUS1_NO_GRAD="${SUBGRAPH_REAREV_TRM_TMINUS1_NO_GRAD:-true}"
export SUBGRAPH_REAREV_TRM_DETACH_CARRY="${SUBGRAPH_REAREV_TRM_DETACH_CARRY:-true}"
export SUBGRAPH_REAREV_TRM_SUPERVISE_ALL_STAGES="${SUBGRAPH_REAREV_TRM_SUPERVISE_ALL_STAGES:-false}"
export SUBGRAPH_REAREV_ACT_STOP_IN_TRAIN="${SUBGRAPH_REAREV_ACT_STOP_IN_TRAIN:-false}"
export SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED="${SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED:-false}"

run_with() {
  local gpus="$1"
  local nproc="$2"
  local master_port="$3"
  local tag="$4"
  local log_file="$LOG_DIR/${tag}.log"
  echo "[run] tag=$tag gpus=$gpus nproc=$nproc grad_accum=${SUBGRAPH_GRAD_ACCUM_STEPS} log=$log_file"
  CUDA_VISIBLE_DEVICES="$gpus" \
  NPROC_PER_NODE="$nproc" \
  MASTER_PORT="$master_port" \
  bash "$RUN_SCRIPT" 2>&1 | tee "$log_file"
}

# Attempt 1: 4-GPU run.
if run_with "$PRIMARY_GPUS" "$PRIMARY_NPROC" "${MASTER_PORT:-29675}" "primary_4gpu"; then
  echo "[done] primary run finished."
  exit 0
fi

if ! grep -Eiq "outofmemoryerror|cuda out of memory|cudnn_status_alloc_failed" "$LOG_DIR/primary_4gpu.log"; then
  echo "[fail] primary run failed, but no OOM signature found. skip fallback."
  exit 1
fi

echo "[oom] detected. fallback to 2-GPU run."

# Keep effective batch roughly stable when dropping from 4->2 GPUs.
old_accum="${SUBGRAPH_GRAD_ACCUM_STEPS}"
new_accum=$(( old_accum * PRIMARY_NPROC / FALLBACK_NPROC ))
if (( new_accum < 1 )); then
  new_accum=1
fi
export SUBGRAPH_GRAD_ACCUM_STEPS="$new_accum"

run_with "$FALLBACK_GPUS" "$FALLBACK_NPROC" "${FALLBACK_MASTER_PORT:-29676}" "fallback_2gpu"

echo "[done] fallback run finished."
