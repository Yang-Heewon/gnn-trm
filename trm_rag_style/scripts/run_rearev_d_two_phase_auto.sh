#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# Phase1 (coarse) directories/names
PHASE1_CKPT_DIR="${PHASE1_CKPT_DIR:-trm_agent/ckpt/cwq_trm_hier6_rearev_D_phase1_${RUN_TAG}}"
PHASE1_WANDB_RUN_NAME="${PHASE1_WANDB_RUN_NAME:-cwq-rearev-D-phase1-${RUN_TAG}}"
PHASE1_LOG_DIR="${PHASE1_LOG_DIR:-logs/rearev_d_auto_fallback_${RUN_TAG}}"

# Phase2 (fine) directories/names
PHASE2_CKPT_DIR="${PHASE2_CKPT_DIR:-trm_agent/ckpt/cwq_trm_hier6_rearev_D_phase2_${RUN_TAG}}"
PHASE2_WANDB_RUN_NAME="${PHASE2_WANDB_RUN_NAME:-cwq-rearev-D-phase2-${RUN_TAG}}"

# Control flags
RUN_PHASE1="${RUN_PHASE1:-true}"  # false => skip phase1 and use PHASE1_BEST_CKPT
PHASE1_BEST_CKPT="${PHASE1_BEST_CKPT:-}"
SUBGRAPH_REAREV_NORMALIZED_GNN="${SUBGRAPH_REAREV_NORMALIZED_GNN:-false}"

# Optional parallel TRM-style training during Phase2.
# Phase2 can use 2 GPUs while TRM-style uses the remaining 2 GPUs.
RUN_TRM_PARALLEL_ON_PHASE2="${RUN_TRM_PARALLEL_ON_PHASE2:-false}"
TRM_PARALLEL_GPUS="${TRM_PARALLEL_GPUS:-2,3}"
TRM_PARALLEL_NPROC="${TRM_PARALLEL_NPROC:-2}"
TRM_PARALLEL_MASTER_PORT="${TRM_PARALLEL_MASTER_PORT:-29688}"
TRM_PARALLEL_RUN_TAG="${TRM_PARALLEL_RUN_TAG:-${RUN_TAG}_trm_parallel}"
TRM_PARALLEL_CKPT_DIR="${TRM_PARALLEL_CKPT_DIR:-trm_agent/ckpt/cwq_trm_hier6_rearev_trm_isolated_${TRM_PARALLEL_RUN_TAG}}"
TRM_PARALLEL_WANDB_RUN_NAME="${TRM_PARALLEL_WANDB_RUN_NAME:-cwq-rearev-trm-parallel-${TRM_PARALLEL_RUN_TAG}}"
TRM_PARALLEL_LOG_FILE="${TRM_PARALLEL_LOG_FILE:-${PHASE1_LOG_DIR}/trm_parallel.log}"
TRM_PARALLEL_SUBGRAPH_LOSS_MODE="${TRM_PARALLEL_SUBGRAPH_LOSS_MODE:-rearev_kl_trm}"
TRM_PARALLEL_SUBGRAPH_REAREV_TRM_WEIGHT="${TRM_PARALLEL_SUBGRAPH_REAREV_TRM_WEIGHT:-0.1}"
TRM_PARALLEL_SUBGRAPH_REAREV_TRM_CE_WEIGHT="${TRM_PARALLEL_SUBGRAPH_REAREV_TRM_CE_WEIGHT:-1.0}"
TRM_PARALLEL_SUBGRAPH_REAREV_TRM_HALT_BCE_WEIGHT="${TRM_PARALLEL_SUBGRAPH_REAREV_TRM_HALT_BCE_WEIGHT:-1.0}"

mkdir -p "$PHASE1_LOG_DIR"

if [[ "$RUN_PHASE1" == "true" ]]; then
  echo "[phase1] start coarse training with fallback"
  PRIMARY_GPUS="${PRIMARY_GPUS:-0,1,2,3}" \
  PRIMARY_NPROC="${PRIMARY_NPROC:-4}" \
  FALLBACK_GPUS="${FALLBACK_GPUS:-2,3}" \
  FALLBACK_NPROC="${FALLBACK_NPROC:-2}" \
  LOG_DIR="$PHASE1_LOG_DIR" \
  CKPT_DIR="$PHASE1_CKPT_DIR" \
  WANDB_RUN_NAME="$PHASE1_WANDB_RUN_NAME" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}" \
  BATCH_SIZE="${BATCH_SIZE:-1}" \
  SUBGRAPH_GRAD_ACCUM_STEPS="${SUBGRAPH_GRAD_ACCUM_STEPS:-8}" \
  SUBGRAPH_MAX_NODES="${SUBGRAPH_MAX_NODES:-4096}" \
  SUBGRAPH_MAX_EDGES="${SUBGRAPH_MAX_EDGES:-16384}" \
  SUBGRAPH_REAREV_NORMALIZED_GNN="$SUBGRAPH_REAREV_NORMALIZED_GNN" \
  SUBGRAPH_LOSS_MODE="${SUBGRAPH_LOSS_MODE:-rearev_kl}" \
  SUBGRAPH_RANKING_ENABLED="${SUBGRAPH_RANKING_ENABLED:-false}" \
  SUBGRAPH_DEEP_SUPERVISION_ENABLED="${PHASE1_SUBGRAPH_DEEP_SUPERVISION_ENABLED:-false}" \
  SUBGRAPH_DEEP_SUPERVISION_WEIGHT="${PHASE1_SUBGRAPH_DEEP_SUPERVISION_WEIGHT:-0.0}" \
  SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT="${PHASE1_SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT:-1.0}" \
  SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT="${PHASE1_SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT:-1.0}" \
  SUBGRAPH_REAREV_TRM_TMINUS1_NO_GRAD="${PHASE1_SUBGRAPH_REAREV_TRM_TMINUS1_NO_GRAD:-true}" \
  SUBGRAPH_REAREV_TRM_DETACH_CARRY="${PHASE1_SUBGRAPH_REAREV_TRM_DETACH_CARRY:-true}" \
  bash trm_rag_style/scripts/run_rearev_d_auto_fallback_4to2.sh
fi

if [[ -z "$PHASE1_BEST_CKPT" ]]; then
  echo "[phase1] select best ckpt from logs by dev Hit@1"
  PHASE1_BEST_CKPT="$(
    PHASE1_CKPT_DIR="$PHASE1_CKPT_DIR" PHASE1_LOG_DIR="$PHASE1_LOG_DIR" \
    /data2/workspace/heewon/anaconda3/envs/taiLab/bin/python - <<'PY'
import os, re
from pathlib import Path

ckpt_dir = Path(os.environ["PHASE1_CKPT_DIR"])
log_dir = Path(os.environ["PHASE1_LOG_DIR"])
log_files = [log_dir / "primary_4gpu.log", log_dir / "fallback_2gpu.log"]

best = None  # (hit, ep, ckpt_path)
dev_ep_pat = re.compile(r"Dev ep(\d+) \\[Subgraph\\]")
dev_hit_pat = re.compile(r"\\[Dev-Subgraph\\] Hit@1=([0-9.]+)")

for lf in log_files:
    if not lf.exists():
        continue
    cur_ep = None
    with lf.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_ep = dev_ep_pat.search(line)
            if m_ep:
                cur_ep = int(m_ep.group(1))
                continue
            m_hit = dev_hit_pat.search(line)
            if m_hit is None or cur_ep is None:
                continue
            hit = float(m_hit.group(1))
            ckpt = ckpt_dir / f"model_ep{cur_ep}.pt"
            if not ckpt.exists():
                continue
            cand = (hit, cur_ep, str(ckpt))
            if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand

if best is None:
    # Fallback: latest checkpoint if logs are missing.
    cands = sorted(ckpt_dir.glob("model_ep*.pt"), key=lambda p: int(re.search(r"model_ep(\\d+)\\.pt$", p.name).group(1)))
    if not cands:
        raise SystemExit("ERROR: no checkpoint found for phase1")
    print(str(cands[-1]))
else:
    print(best[2])
PY
  )"
fi

if [[ ! -f "$PHASE1_BEST_CKPT" ]]; then
  echo "[err] phase1 best ckpt not found: $PHASE1_BEST_CKPT"
  exit 2
fi
echo "[phase1] best ckpt: $PHASE1_BEST_CKPT"

TRM_PARALLEL_PID=""
if [[ "$RUN_TRM_PARALLEL_ON_PHASE2" == "true" ]]; then
  mkdir -p "$(dirname "$TRM_PARALLEL_LOG_FILE")"
  echo "[parallel] launch TRM-style on GPUs ${TRM_PARALLEL_GPUS} (nproc=${TRM_PARALLEL_NPROC})"
  (
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$TRM_PARALLEL_GPUS" \
    NPROC_PER_NODE="$TRM_PARALLEL_NPROC" \
    MASTER_PORT="$TRM_PARALLEL_MASTER_PORT" \
    RUN_TAG="$TRM_PARALLEL_RUN_TAG" \
    CKPT_DIR="$TRM_PARALLEL_CKPT_DIR" \
    WANDB_MODE="${WANDB_MODE:-online}" \
    WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}" \
    WANDB_RUN_NAME="$TRM_PARALLEL_WANDB_RUN_NAME" \
    BATCH_SIZE="${TRM_PARALLEL_BATCH_SIZE:-2}" \
    EPOCHS="${TRM_PARALLEL_EPOCHS:-16}" \
    LR="${TRM_PARALLEL_LR:-2e-4}" \
    SUBGRAPH_RECURSION_STEPS="${TRM_PARALLEL_SUBGRAPH_RECURSION_STEPS:-4}" \
    SUBGRAPH_REAREV_ADAPT_STAGES="${TRM_PARALLEL_SUBGRAPH_REAREV_ADAPT_STAGES:-2}" \
    SUBGRAPH_GRAD_ACCUM_STEPS="${TRM_PARALLEL_SUBGRAPH_GRAD_ACCUM_STEPS:-4}" \
    SUBGRAPH_MAX_NODES="${TRM_PARALLEL_SUBGRAPH_MAX_NODES:-2048}" \
    SUBGRAPH_MAX_EDGES="${TRM_PARALLEL_SUBGRAPH_MAX_EDGES:-8192}" \
    SUBGRAPH_LOSS_MODE="$TRM_PARALLEL_SUBGRAPH_LOSS_MODE" \
    SUBGRAPH_REAREV_TRM_WEIGHT="$TRM_PARALLEL_SUBGRAPH_REAREV_TRM_WEIGHT" \
    SUBGRAPH_REAREV_TRM_CE_WEIGHT="$TRM_PARALLEL_SUBGRAPH_REAREV_TRM_CE_WEIGHT" \
    SUBGRAPH_REAREV_TRM_HALT_BCE_WEIGHT="$TRM_PARALLEL_SUBGRAPH_REAREV_TRM_HALT_BCE_WEIGHT" \
    bash trm_rag_style/scripts/run_rearev_trm_isolated.sh
  ) >"$TRM_PARALLEL_LOG_FILE" 2>&1 &
  TRM_PARALLEL_PID="$!"
  echo "[parallel] TRM-style PID=$TRM_PARALLEL_PID log=$TRM_PARALLEL_LOG_FILE"
fi

echo "[phase2] start fine-tuning from phase1 best"
CUDA_VISIBLE_DEVICES="${PHASE2_GPUS:-0,1}" \
NPROC_PER_NODE="${PHASE2_NPROC_PER_NODE:-2}" \
MASTER_PORT="${PHASE2_MASTER_PORT:-29679}" \
CKPT="$PHASE1_BEST_CKPT" \
CKPT_DIR="$PHASE2_CKPT_DIR" \
WANDB_MODE="${WANDB_MODE:-online}" \
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}" \
WANDB_RUN_NAME="$PHASE2_WANDB_RUN_NAME" \
EPOCHS="${PHASE2_EPOCHS:-5}" \
LR="${PHASE2_LR:-5e-5}" \
BATCH_SIZE="${PHASE2_BATCH_SIZE:-2}" \
SUBGRAPH_GRAD_ACCUM_STEPS="${PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS:-4}" \
SUBGRAPH_REAREV_NORMALIZED_GNN="$SUBGRAPH_REAREV_NORMALIZED_GNN" \
bash trm_rag_style/scripts/run_rearev_d_phase2_resume.sh

if [[ -n "$TRM_PARALLEL_PID" ]]; then
  echo "[parallel] waiting TRM-style process to finish (PID=$TRM_PARALLEL_PID)"
  wait "$TRM_PARALLEL_PID"
  echo "[parallel] TRM-style finished"
fi

echo "[done] two-phase pipeline finished"
