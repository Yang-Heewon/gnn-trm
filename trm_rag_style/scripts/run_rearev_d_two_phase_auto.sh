#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_TAG="${EMB_TAG:-e5_w4_g4}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"
if [[ ! -d "$EMB_DIR" && -d "trm_agent/emb/${DATASET}_e5" ]]; then
  EMB_TAG="e5"
  EMB_DIR="trm_agent/emb/${DATASET}_${EMB_TAG}"
fi

# Phase1 (coarse) directories/names
PHASE1_CKPT_DIR="${PHASE1_CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_rearev_D_phase1_${RUN_TAG}}"
PHASE1_WANDB_RUN_NAME="${PHASE1_WANDB_RUN_NAME:-${DATASET}-rearev-D-phase1-${RUN_TAG}}"
PHASE1_LOG_DIR="${PHASE1_LOG_DIR:-logs/rearev_d_auto_fallback_${RUN_TAG}}"

# Phase2 (fine) directories/names
PHASE2_CKPT_DIR="${PHASE2_CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_rearev_D_phase2_${RUN_TAG}}"
PHASE2_WANDB_RUN_NAME="${PHASE2_WANDB_RUN_NAME:-${DATASET}-rearev-D-phase2-${RUN_TAG}}"
PHASE2_LOG_DIR="${PHASE2_LOG_DIR:-logs/rearev_d_phase2_${RUN_TAG}}"
PHASE2_LOG_FILE="${PHASE2_LOG_FILE:-${PHASE2_LOG_DIR}/phase2.log}"
PHASE2_BEST_METRIC="${PHASE2_BEST_METRIC:-dev_hit1}" # dev_hit1 | dev_f1
PHASE2_BEST_CKPT="${PHASE2_BEST_CKPT:-}"
AUTO_RUN_LATENT_ABLATION="${AUTO_RUN_LATENT_ABLATION:-false}"
ABLATION_SPLIT="${ABLATION_SPLIT:-test}"

# Control flags
RUN_PHASE1="${RUN_PHASE1:-true}"  # false => skip phase1 and use PHASE1_BEST_CKPT
PHASE1_BEST_CKPT="${PHASE1_BEST_CKPT:-}"
SUBGRAPH_REAREV_NORMALIZED_GNN="${SUBGRAPH_REAREV_NORMALIZED_GNN:-false}"

mkdir -p "$PHASE1_LOG_DIR"

if [[ "$RUN_PHASE1" == "true" ]]; then
  echo "[phase1] start coarse training with fallback"
  PRIMARY_GPUS="${PRIMARY_GPUS:-0,1,2,3}" \
  PRIMARY_NPROC="${PRIMARY_NPROC:-4}" \
  FALLBACK_GPUS="${FALLBACK_GPUS:-0,1,2,3}" \
  FALLBACK_NPROC="${FALLBACK_NPROC:-4}" \
  LOG_DIR="$PHASE1_LOG_DIR" \
  CKPT_DIR="$PHASE1_CKPT_DIR" \
  WANDB_RUN_NAME="$PHASE1_WANDB_RUN_NAME" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}" \
  EPOCHS="${PHASE1_EPOCHS:-16}" \
  BATCH_SIZE="${BATCH_SIZE:-1}" \
  SUBGRAPH_GRAD_ACCUM_STEPS="${SUBGRAPH_GRAD_ACCUM_STEPS:-8}" \
  SUBGRAPH_MAX_NODES="${SUBGRAPH_MAX_NODES:-4096}" \
  SUBGRAPH_MAX_EDGES="${SUBGRAPH_MAX_EDGES:-16384}" \
  DATASET="$DATASET" \
  MODEL_IMPL="$MODEL_IMPL" \
  EMB_TAG="$EMB_TAG" \
  EMB_DIR="$EMB_DIR" \
  SEED="${SEED:-42}" \
  DETERMINISTIC="${DETERMINISTIC:-false}" \
  SUBGRAPH_REAREV_NORMALIZED_GNN="$SUBGRAPH_REAREV_NORMALIZED_GNN" \
  SUBGRAPH_LOSS_MODE="${SUBGRAPH_LOSS_MODE:-rearev_kl}" \
  SUBGRAPH_KL_NO_POSITIVE_MODE="${PHASE1_SUBGRAPH_KL_NO_POSITIVE_MODE:-${SUBGRAPH_KL_NO_POSITIVE_MODE:-uniform}}" \
  SUBGRAPH_KL_SUPERVISION_MODE="${PHASE1_SUBGRAPH_KL_SUPERVISION_MODE:-${SUBGRAPH_KL_SUPERVISION_MODE:-final}}" \
  SUBGRAPH_RANKING_ENABLED="${SUBGRAPH_RANKING_ENABLED:-false}" \
  SUBGRAPH_EARLY_STOP_ENABLED="${PHASE1_SUBGRAPH_EARLY_STOP_ENABLED:-true}" \
  SUBGRAPH_EARLY_STOP_METRIC="${PHASE1_SUBGRAPH_EARLY_STOP_METRIC:-dev_hit1}" \
  SUBGRAPH_EARLY_STOP_PATIENCE="${PHASE1_SUBGRAPH_EARLY_STOP_PATIENCE:-4}" \
  SUBGRAPH_EARLY_STOP_MIN_DELTA="${PHASE1_SUBGRAPH_EARLY_STOP_MIN_DELTA:-1e-3}" \
  SUBGRAPH_EARLY_STOP_MIN_EPOCHS="${PHASE1_SUBGRAPH_EARLY_STOP_MIN_EPOCHS:-8}" \
  SUBGRAPH_REAREV_LATENT_REASONING_ENABLED="${PHASE1_SUBGRAPH_REAREV_LATENT_REASONING_ENABLED:-true}" \
  SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA="${PHASE1_SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA:-0.25}" \
  SUBGRAPH_REAREV_LATENT_UPDATE_MODE="${PHASE1_SUBGRAPH_REAREV_LATENT_UPDATE_MODE:-${SUBGRAPH_REAREV_LATENT_UPDATE_MODE:-gru}}" \
  SUBGRAPH_REAREV_GLOBAL_GATE_ENABLED="${PHASE1_SUBGRAPH_REAREV_GLOBAL_GATE_ENABLED:-true}" \
  SUBGRAPH_REAREV_LOGIT_GLOBAL_FUSION_ENABLED="${PHASE1_SUBGRAPH_REAREV_LOGIT_GLOBAL_FUSION_ENABLED:-true}" \
  SUBGRAPH_REAREV_DYNAMIC_HALTING_ENABLED="${PHASE1_SUBGRAPH_REAREV_DYNAMIC_HALTING_ENABLED:-true}" \
  SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS="${PHASE1_SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS:-3}" \
  SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD="${PHASE1_SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD:-0.95}" \
  SUBGRAPH_DEEP_SUPERVISION_ENABLED="${PHASE1_SUBGRAPH_DEEP_SUPERVISION_ENABLED:-false}" \
  SUBGRAPH_DEEP_SUPERVISION_WEIGHT="${PHASE1_SUBGRAPH_DEEP_SUPERVISION_WEIGHT:-0.0}" \
  SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT="${PHASE1_SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT:-1.0}" \
  SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT="${PHASE1_SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT:-1.0}" \
  SUBGRAPH_REAREV_TRM_TMINUS1_NO_GRAD="${PHASE1_SUBGRAPH_REAREV_TRM_TMINUS1_NO_GRAD:-true}" \
  SUBGRAPH_REAREV_TRM_DETACH_CARRY="${PHASE1_SUBGRAPH_REAREV_TRM_DETACH_CARRY:-true}" \
  SUBGRAPH_REAREV_TRM_SUPERVISE_ALL_STAGES="${PHASE1_SUBGRAPH_REAREV_TRM_SUPERVISE_ALL_STAGES:-false}" \
  SUBGRAPH_REAREV_ACT_STOP_IN_TRAIN="${PHASE1_SUBGRAPH_REAREV_ACT_STOP_IN_TRAIN:-false}" \
  SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED="${PHASE1_SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED:-false}" \
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
dev_ep_pat = re.compile(r"Dev ep(\d+) \[Subgraph\]")
dev_hit_pat = re.compile(r"\[Dev-Subgraph\] Hit@1=([0-9.]+)")

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
mkdir -p "$PHASE2_LOG_DIR"
CUDA_VISIBLE_DEVICES="${PHASE2_GPUS:-0,1,2,3}" \
NPROC_PER_NODE="${PHASE2_NPROC_PER_NODE:-4}" \
MASTER_PORT="${PHASE2_MASTER_PORT:-29679}" \
DATASET="$DATASET" \
MODEL_IMPL="$MODEL_IMPL" \
EMB_TAG="$EMB_TAG" \
EMB_DIR="$EMB_DIR" \
CKPT="$PHASE1_BEST_CKPT" \
CKPT_DIR="$PHASE2_CKPT_DIR" \
SEED="${SEED:-42}" \
DETERMINISTIC="${DETERMINISTIC:-false}" \
WANDB_MODE="${WANDB_MODE:-online}" \
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}" \
WANDB_RUN_NAME="$PHASE2_WANDB_RUN_NAME" \
EPOCHS="${PHASE2_EPOCHS:-16}" \
LR="${PHASE2_LR:-5e-5}" \
BATCH_SIZE="${PHASE2_BATCH_SIZE:-2}" \
SUBGRAPH_GRAD_ACCUM_STEPS="${PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS:-4}" \
SUBGRAPH_MAX_NODES="${PHASE2_SUBGRAPH_MAX_NODES:-2048}" \
SUBGRAPH_MAX_EDGES="${PHASE2_SUBGRAPH_MAX_EDGES:-8192}" \
SUBGRAPH_RANKING_WEIGHT="${PHASE2_SUBGRAPH_RANKING_WEIGHT:-0.15}" \
SUBGRAPH_RANKING_MARGIN="${PHASE2_SUBGRAPH_RANKING_MARGIN:-0.2}" \
SUBGRAPH_HARD_NEGATIVE_TOPK="${PHASE2_SUBGRAPH_HARD_NEGATIVE_TOPK:-32}" \
SUBGRAPH_BCE_HARD_NEGATIVE_TOPK="${PHASE2_SUBGRAPH_BCE_HARD_NEGATIVE_TOPK:-64}" \
SUBGRAPH_POS_WEIGHT_MAX="${PHASE2_SUBGRAPH_POS_WEIGHT_MAX:-256}" \
SUBGRAPH_REAREV_NORMALIZED_GNN="$SUBGRAPH_REAREV_NORMALIZED_GNN" \
SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS="${PHASE2_SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS:-${SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS:-false}}" \
SUBGRAPH_REAREV_LATENT_REASONING_ENABLED="${PHASE2_SUBGRAPH_REAREV_LATENT_REASONING_ENABLED:-true}" \
SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA="${PHASE2_SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA:-0.25}" \
SUBGRAPH_REAREV_LATENT_UPDATE_MODE="${PHASE2_SUBGRAPH_REAREV_LATENT_UPDATE_MODE:-${SUBGRAPH_REAREV_LATENT_UPDATE_MODE:-gru}}" \
SUBGRAPH_REAREV_GLOBAL_GATE_ENABLED="${PHASE2_SUBGRAPH_REAREV_GLOBAL_GATE_ENABLED:-true}" \
SUBGRAPH_REAREV_LOGIT_GLOBAL_FUSION_ENABLED="${PHASE2_SUBGRAPH_REAREV_LOGIT_GLOBAL_FUSION_ENABLED:-true}" \
SUBGRAPH_REAREV_DYNAMIC_HALTING_ENABLED="${PHASE2_SUBGRAPH_REAREV_DYNAMIC_HALTING_ENABLED:-true}" \
SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS="${PHASE2_SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS:-3}" \
SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD="${PHASE2_SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD:-0.95}" \
SUBGRAPH_RECURSION_STEPS="${PHASE2_SUBGRAPH_RECURSION_STEPS:-${SUBGRAPH_RECURSION_STEPS:-4}}" \
SUBGRAPH_REAREV_ADAPT_STAGES="${PHASE2_SUBGRAPH_REAREV_ADAPT_STAGES:-${SUBGRAPH_REAREV_ADAPT_STAGES:-2}}" \
SUBGRAPH_REAREV_TRM_SUPERVISE_ALL_STAGES="${PHASE2_SUBGRAPH_REAREV_TRM_SUPERVISE_ALL_STAGES:-${PHASE1_SUBGRAPH_REAREV_TRM_SUPERVISE_ALL_STAGES:-false}}" \
SUBGRAPH_REAREV_ACT_STOP_IN_TRAIN="${PHASE2_SUBGRAPH_REAREV_ACT_STOP_IN_TRAIN:-${PHASE1_SUBGRAPH_REAREV_ACT_STOP_IN_TRAIN:-false}}" \
SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED="${PHASE2_SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED:-${PHASE1_SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED:-false}}" \
bash trm_rag_style/scripts/run_rearev_d_phase2_resume.sh 2>&1 | tee "$PHASE2_LOG_FILE"

if [[ -z "$PHASE2_BEST_CKPT" ]]; then
  echo "[phase2] select best ckpt by ${PHASE2_BEST_METRIC}"
  PHASE2_BEST_CKPT="$(
    PHASE2_CKPT_DIR="$PHASE2_CKPT_DIR" PHASE2_LOG_FILE="$PHASE2_LOG_FILE" PHASE2_BEST_METRIC="$PHASE2_BEST_METRIC" \
    /data2/workspace/heewon/anaconda3/envs/taiLab/bin/python - <<'PY'
import os
import re
from pathlib import Path

ckpt_dir = Path(os.environ["PHASE2_CKPT_DIR"])
log_file = Path(os.environ["PHASE2_LOG_FILE"])
metric_name = str(os.environ.get("PHASE2_BEST_METRIC", "dev_hit1")).strip().lower()
if metric_name not in {"dev_hit1", "dev_f1"}:
    metric_name = "dev_hit1"

if not log_file.exists():
    raise SystemExit("")

dev_ep_pat = re.compile(r"Dev ep(\d+) \[Subgraph\]")
hit_pat = re.compile(r"Hit@1=([0-9.]+)")
f1_pat = re.compile(r"F1=([0-9.]+)")

best = None  # (metric, ep, ckpt_path)
cur_ep = None
with log_file.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m_ep = dev_ep_pat.search(line)
        if m_ep:
            cur_ep = int(m_ep.group(1))
            continue
        if "[Dev-Subgraph]" not in line or cur_ep is None:
            continue
        m_hit = hit_pat.search(line)
        m_f1 = f1_pat.search(line)
        if m_hit is None or m_f1 is None:
            continue
        hit = float(m_hit.group(1))
        f1 = float(m_f1.group(1))
        metric = hit if metric_name == "dev_hit1" else f1
        ckpt = ckpt_dir / f"model_ep{cur_ep}.pt"
        if not ckpt.exists():
            continue
        cand = (metric, cur_ep, str(ckpt))
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand

if best is None:
    cands = sorted(
        ckpt_dir.glob("model_ep*.pt"),
        key=lambda p: int(re.search(r"model_ep(\d+)\.pt$", p.name).group(1))
    )
    if not cands:
        raise SystemExit("")
    print(str(cands[-1]))
else:
    print(best[2])
PY
  )"
fi

if [[ -n "$PHASE2_BEST_CKPT" && -f "$PHASE2_BEST_CKPT" ]]; then
  echo "[phase2] best ckpt: $PHASE2_BEST_CKPT"
  printf "%s\n" "$PHASE2_BEST_CKPT" > "${PHASE2_CKPT_DIR}/best_${PHASE2_BEST_METRIC}.txt"
else
  echo "[warn] could not determine phase2 best checkpoint."
fi

if [[ "$AUTO_RUN_LATENT_ABLATION" == "true" ]]; then
  if [[ -n "$PHASE2_BEST_CKPT" && -f "$PHASE2_BEST_CKPT" ]]; then
    echo "[phase2] run latent ablation on best checkpoint"
    CKPT="$PHASE2_BEST_CKPT" \
    SPLIT="$ABLATION_SPLIT" \
    CUDA_VISIBLE_DEVICES="${PHASE2_GPUS:-0}" \
    bash trm_rag_style/scripts/run_rearev_d_latent_ablation_eval.sh
  else
    echo "[warn] skip ablation: best checkpoint missing"
  fi
fi

echo "[done] two-phase pipeline finished"
