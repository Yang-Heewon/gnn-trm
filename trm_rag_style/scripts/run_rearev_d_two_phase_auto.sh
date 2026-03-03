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

echo "[done] two-phase pipeline finished"
