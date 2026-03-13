#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TORCHRUN_BIN_DEFAULT="torchrun"
if [[ -x "/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun" ]]; then
  TORCHRUN_BIN_DEFAULT="/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun"
fi
TORCHRUN_BIN="${TORCHRUN_BIN:-$TORCHRUN_BIN_DEFAULT}"
if ! command -v "$TORCHRUN_BIN" >/dev/null 2>&1; then
  echo "[err] torchrun not found: $TORCHRUN_BIN"
  echo "      set TORCHRUN_BIN explicitly (e.g., /data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun)."
  exit 1
fi

unset PYTORCH_CUDA_ALLOC_CONF || true
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_TAG="${EMB_TAG:-e5_w4_g4}"
PHASE1_CKPT="${PHASE1_CKPT:-/data2/workspace/heewon/KGQA/trm_agent/ckpt/cwq_trm_hier6_rearev_D_phase1_abl_cwq_d_recSweep_r9_i3_20260312_122635/model_ep25.pt}"

if [[ ! -f "$PHASE1_CKPT" ]]; then
  echo "[err] phase1 checkpoint not found: $PHASE1_CKPT"
  echo "      set PHASE1_CKPT to your phase1 checkpoint path."
  exit 2
fi

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-abl_${DATASET}_d_recSweep_r9_i3_rerun_${TS}}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-paper_final}"
WANDB_GROUP="${WANDB_GROUP:-ablation_${DATASET}_d_recSweep_r9_i3}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ablation_${DATASET}_d_recSweep_r9_i3_phase2_rerun_${TS}}"

MASTER_PORT="${MASTER_PORT:-30212}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-30300}"

SUBGRAPH_RECURSION_STEPS="${SUBGRAPH_RECURSION_STEPS:-9}"
SUBGRAPH_REAREV_NUM_INS="${SUBGRAPH_REAREV_NUM_INS:-3}"
SUBGRAPH_MAX_NODES="${SUBGRAPH_MAX_NODES:-1024}"
SUBGRAPH_MAX_EDGES="${SUBGRAPH_MAX_EDGES:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-25}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

echo "[plan] phase1_ckpt=$PHASE1_CKPT"
echo "[plan] phase2_run_tag=$RUN_TAG"
echo "[plan] phase2 rec/ins=${SUBGRAPH_RECURSION_STEPS}/${SUBGRAPH_REAREV_NUM_INS}"
echo "[plan] phase2 batch_size=$BATCH_SIZE epochs=$EPOCHS nproc=$NPROC_PER_NODE"
echo "[plan] wandb project=$WANDB_PROJECT group=$WANDB_GROUP"

echo ""
echo "[1/2] start phase2 rerun from phase1 checkpoint"
CKPT="$PHASE1_CKPT" \
DATASET="$DATASET" \
MODEL_IMPL="$MODEL_IMPL" \
EMB_TAG="$EMB_TAG" \
RUN_TAG="$RUN_TAG" \
MASTER_PORT="$MASTER_PORT" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
BATCH_SIZE="$BATCH_SIZE" \
EPOCHS="$EPOCHS" \
SUBGRAPH_RECURSION_STEPS="$SUBGRAPH_RECURSION_STEPS" \
SUBGRAPH_REAREV_NUM_INS="$SUBGRAPH_REAREV_NUM_INS" \
SUBGRAPH_MAX_NODES="$SUBGRAPH_MAX_NODES" \
SUBGRAPH_MAX_EDGES="$SUBGRAPH_MAX_EDGES" \
WANDB_MODE="$WANDB_MODE" \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_GROUP="$WANDB_GROUP" \
WANDB_RUN_NAME="$WANDB_RUN_NAME" \
TORCHRUN_BIN="$TORCHRUN_BIN" \
bash trm_rag_style/scripts/run_rearev_d_phase2_resume.sh

echo ""
echo "[2/2] run remaining ablation cycles (skip completed)"
ONLY_REMAINING=true \
WANDB_MODE="$WANDB_MODE" \
WANDB_PROJECT="$WANDB_PROJECT" \
TORCHRUN_BIN="$TORCHRUN_BIN" \
BASE_MASTER_PORT="$BASE_MASTER_PORT" \
bash trm_rag_style/scripts/run_rearev_d_dplus_ablation_remaining.sh

echo ""
echo "[done] phase2 rerun + remaining ablations completed."
