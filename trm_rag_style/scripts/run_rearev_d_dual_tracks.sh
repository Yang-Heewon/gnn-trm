#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-both}" # baseline | asym | both

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"

# Common topology defaults (can be overridden per track below)
PRIMARY_GPUS="${PRIMARY_GPUS:-0,1,2,3}"
PRIMARY_NPROC="${PRIMARY_NPROC:-4}"
FALLBACK_GPUS="${FALLBACK_GPUS:-0,1}"
FALLBACK_NPROC="${FALLBACK_NPROC:-2}"

run_baseline() {
  local tag="${BASELINE_RUN_TAG:-baseline_${RUN_TAG}}"
  echo "[dual] run baseline two-phase (tag=${tag})"
  RUN_TAG="$tag" \
  WANDB_MODE="$WANDB_MODE" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  PRIMARY_GPUS="${BASELINE_PRIMARY_GPUS:-$PRIMARY_GPUS}" \
  PRIMARY_NPROC="${BASELINE_PRIMARY_NPROC:-$PRIMARY_NPROC}" \
  FALLBACK_GPUS="${BASELINE_FALLBACK_GPUS:-$FALLBACK_GPUS}" \
  FALLBACK_NPROC="${BASELINE_FALLBACK_NPROC:-$FALLBACK_NPROC}" \
  PHASE2_GPUS="${BASELINE_PHASE2_GPUS:-${BASELINE_PRIMARY_GPUS:-$PRIMARY_GPUS}}" \
  PHASE2_NPROC_PER_NODE="${BASELINE_PHASE2_NPROC_PER_NODE:-${BASELINE_PRIMARY_NPROC:-$PRIMARY_NPROC}}" \
  MASTER_PORT="${BASELINE_MASTER_PORT:-29781}" \
  FALLBACK_MASTER_PORT="${BASELINE_FALLBACK_MASTER_PORT:-29782}" \
  PHASE2_MASTER_PORT="${BASELINE_PHASE2_MASTER_PORT:-29791}" \
  EPOCHS="${BASELINE_EPOCHS:-16}" \
  PHASE2_EPOCHS="${BASELINE_PHASE2_EPOCHS:-5}" \
  bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
}

run_asym() {
  local tag="${ASYM_RUN_TAG:-asym_${RUN_TAG}}"
  echo "[dual] run asym phase1-only (tag=${tag})"
  RUN_TAG="$tag" \
  WANDB_MODE="$WANDB_MODE" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  PRIMARY_GPUS="${ASYM_PRIMARY_GPUS:-$PRIMARY_GPUS}" \
  PRIMARY_NPROC="${ASYM_PRIMARY_NPROC:-$PRIMARY_NPROC}" \
  FALLBACK_GPUS="${ASYM_FALLBACK_GPUS:-$FALLBACK_GPUS}" \
  FALLBACK_NPROC="${ASYM_FALLBACK_NPROC:-$FALLBACK_NPROC}" \
  MASTER_PORT="${ASYM_MASTER_PORT:-29801}" \
  FALLBACK_MASTER_PORT="${ASYM_FALLBACK_MASTER_PORT:-29802}" \
  EPOCHS="${ASYM_EPOCHS:-16}" \
  BATCH_SIZE="${ASYM_BATCH_SIZE:-1}" \
  SUBGRAPH_GRAD_ACCUM_STEPS="${ASYM_SUBGRAPH_GRAD_ACCUM_STEPS:-8}" \
  bash trm_rag_style/scripts/run_rearev_d_phase1_asym_only.sh
}

case "$MODE" in
  baseline)
    run_baseline
    ;;
  asym)
    run_asym
    ;;
  both)
    run_baseline
    run_asym
    ;;
  *)
    echo "[err] invalid MODE: ${MODE} (expected: baseline|asym|both)"
    exit 2
    ;;
esac

echo "[done] dual-track launcher finished (mode=${MODE})"
