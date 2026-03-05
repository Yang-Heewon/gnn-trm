#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Latent-maximization protocol:
# - Phase1: KL + deep supervision + conservative halting + latent-dominant alpha
# - Phase2: BCE + ranking + hard negatives tuned for Hit@1 separation
# - Multi-seed loop (default: 42,43,44)
# - Per-seed test eval + summary(max/mean) for reproducible "best" measurement

PROTOCOL_TAG="${PROTOCOL_TAG:-latentmax_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-42,43,44}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"

# Use 4 GPUs for both phases by default.
PRIMARY_GPUS="${PRIMARY_GPUS:-0,1,2,3}"
PRIMARY_NPROC="${PRIMARY_NPROC:-4}"
FALLBACK_GPUS="${FALLBACK_GPUS:-0,1,2,3}"
FALLBACK_NPROC="${FALLBACK_NPROC:-4}"
PHASE2_GPUS="${PHASE2_GPUS:-0,1,2,3}"
PHASE2_NPROC_PER_NODE="${PHASE2_NPROC_PER_NODE:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29820}"
PHASE2_MASTER_PORT_BASE="${PHASE2_MASTER_PORT_BASE:-29920}"

# Shared graph size
SUBGRAPH_MAX_NODES="${SUBGRAPH_MAX_NODES:-4096}"
SUBGRAPH_MAX_EDGES="${SUBGRAPH_MAX_EDGES:-16384}"

# Phase1 (latent-dominant)
PHASE1_EPOCHS="${PHASE1_EPOCHS:-16}"
PHASE1_BATCH_SIZE="${PHASE1_BATCH_SIZE:-1}"
PHASE1_GRAD_ACCUM="${PHASE1_GRAD_ACCUM:-8}"
PHASE1_LR="${PHASE1_LR:-2e-4}"
PHASE1_LATENT_ALPHA="${PHASE1_LATENT_ALPHA:-0.6}"
PHASE1_HALT_MIN="${PHASE1_HALT_MIN:-4}"
PHASE1_HALT_THR="${PHASE1_HALT_THR:-0.98}"
PHASE1_DS_ENABLED="${PHASE1_DS_ENABLED:-true}"
PHASE1_DS_W="${PHASE1_DS_W:-0.05}"
PHASE1_DS_CE_W="${PHASE1_DS_CE_W:-1.0}"
PHASE1_DS_HALT_W="${PHASE1_DS_HALT_W:-0.3}"
PHASE1_ESTOP_ENABLED="${PHASE1_ESTOP_ENABLED:-true}"
PHASE1_ESTOP_METRIC="${PHASE1_ESTOP_METRIC:-dev_hit1}"
PHASE1_ESTOP_PATIENCE="${PHASE1_ESTOP_PATIENCE:-4}"
PHASE1_ESTOP_MIN_DELTA="${PHASE1_ESTOP_MIN_DELTA:-1e-3}"
PHASE1_ESTOP_MIN_EPOCHS="${PHASE1_ESTOP_MIN_EPOCHS:-8}"
PHASE1_KL_NO_POSITIVE_MODE="${PHASE1_KL_NO_POSITIVE_MODE:-skip}"

# Phase2 (hit@1-focused separation)
PHASE2_EPOCHS="${PHASE2_EPOCHS:-8}"
PHASE2_BATCH_SIZE="${PHASE2_BATCH_SIZE:-1}"
PHASE2_GRAD_ACCUM="${PHASE2_GRAD_ACCUM:-8}"
PHASE2_LR="${PHASE2_LR:-3e-5}"
PHASE2_LATENT_ALPHA="${PHASE2_LATENT_ALPHA:-0.6}"
PHASE2_HALT_MIN="${PHASE2_HALT_MIN:-4}"
PHASE2_HALT_THR="${PHASE2_HALT_THR:-0.98}"
PHASE2_RANKING_WEIGHT="${PHASE2_RANKING_WEIGHT:-0.25}"
PHASE2_RANKING_MARGIN="${PHASE2_RANKING_MARGIN:-0.30}"
PHASE2_HARDNEG_TOPK="${PHASE2_HARDNEG_TOPK:-96}"
PHASE2_BCE_HARDNEG_TOPK="${PHASE2_BCE_HARDNEG_TOPK:-128}"
PHASE2_POS_WEIGHT_MAX="${PHASE2_POS_WEIGHT_MAX:-64}"
PHASE2_BEST_METRIC="${PHASE2_BEST_METRIC:-dev_hit1}"

AUTO_RUN_LATENT_ABLATION="${AUTO_RUN_LATENT_ABLATION:-false}"
ABLATION_SPLIT="${ABLATION_SPLIT:-test}"

# Test summary options
EVAL_ON_TEST="${EVAL_ON_TEST:-true}"
TEST_GPU="${TEST_GPU:-0}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-2}"
TEST_SPLIT="${TEST_SPLIT:-test}"

LOG_ROOT="${LOG_ROOT:-logs/latent_max_protocol_${PROTOCOL_TAG}}"
mkdir -p "$LOG_ROOT"
SUMMARY_FILE="$LOG_ROOT/summary.txt"
: > "$SUMMARY_FILE"
METRICS_TSV="$LOG_ROOT/metrics.tsv"
echo -e "seed\trun_tag\tckpt\thit1\tf1\tprecision\trecall" > "$METRICS_TSV"

IFS=',' read -r -a seed_list <<< "$SEEDS"
idx=0
for seed in "${seed_list[@]}"; do
  seed="$(echo "$seed" | xargs)"
  if [[ -z "$seed" ]]; then
    continue
  fi
  idx=$((idx + 1))
  run_tag="${PROTOCOL_TAG}_s${seed}"
  mport=$((MASTER_PORT_BASE + idx))
  p2port=$((PHASE2_MASTER_PORT_BASE + idx))
  seed_log="$LOG_ROOT/${run_tag}.log"

  echo "[protocol] seed=$seed run_tag=$run_tag" | tee -a "$SUMMARY_FILE"

  RUN_TAG="$run_tag" \
  WANDB_MODE="$WANDB_MODE" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  SEED="$seed" \
  DETERMINISTIC=false \
  PRIMARY_GPUS="$PRIMARY_GPUS" \
  PRIMARY_NPROC="$PRIMARY_NPROC" \
  FALLBACK_GPUS="$FALLBACK_GPUS" \
  FALLBACK_NPROC="$FALLBACK_NPROC" \
  PHASE2_GPUS="$PHASE2_GPUS" \
  PHASE2_NPROC_PER_NODE="$PHASE2_NPROC_PER_NODE" \
  MASTER_PORT="$mport" \
  PHASE2_MASTER_PORT="$p2port" \
  EPOCHS="$PHASE1_EPOCHS" \
  BATCH_SIZE="$PHASE1_BATCH_SIZE" \
  SUBGRAPH_GRAD_ACCUM_STEPS="$PHASE1_GRAD_ACCUM" \
  LR="$PHASE1_LR" \
  SUBGRAPH_MAX_NODES="$SUBGRAPH_MAX_NODES" \
  SUBGRAPH_MAX_EDGES="$SUBGRAPH_MAX_EDGES" \
  PHASE1_SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA="$PHASE1_LATENT_ALPHA" \
  PHASE1_SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS="$PHASE1_HALT_MIN" \
  PHASE1_SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD="$PHASE1_HALT_THR" \
  PHASE1_SUBGRAPH_DEEP_SUPERVISION_ENABLED="$PHASE1_DS_ENABLED" \
  PHASE1_SUBGRAPH_DEEP_SUPERVISION_WEIGHT="$PHASE1_DS_W" \
  PHASE1_SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT="$PHASE1_DS_CE_W" \
  PHASE1_SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT="$PHASE1_DS_HALT_W" \
  PHASE1_SUBGRAPH_EARLY_STOP_ENABLED="$PHASE1_ESTOP_ENABLED" \
  PHASE1_SUBGRAPH_EARLY_STOP_METRIC="$PHASE1_ESTOP_METRIC" \
  PHASE1_SUBGRAPH_EARLY_STOP_PATIENCE="$PHASE1_ESTOP_PATIENCE" \
  PHASE1_SUBGRAPH_EARLY_STOP_MIN_DELTA="$PHASE1_ESTOP_MIN_DELTA" \
  PHASE1_SUBGRAPH_EARLY_STOP_MIN_EPOCHS="$PHASE1_ESTOP_MIN_EPOCHS" \
  PHASE1_SUBGRAPH_KL_NO_POSITIVE_MODE="$PHASE1_KL_NO_POSITIVE_MODE" \
  PHASE2_EPOCHS="$PHASE2_EPOCHS" \
  PHASE2_BATCH_SIZE="$PHASE2_BATCH_SIZE" \
  PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS="$PHASE2_GRAD_ACCUM" \
  PHASE2_LR="$PHASE2_LR" \
  PHASE2_SUBGRAPH_MAX_NODES="$SUBGRAPH_MAX_NODES" \
  PHASE2_SUBGRAPH_MAX_EDGES="$SUBGRAPH_MAX_EDGES" \
  PHASE2_SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA="$PHASE2_LATENT_ALPHA" \
  PHASE2_SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS="$PHASE2_HALT_MIN" \
  PHASE2_SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD="$PHASE2_HALT_THR" \
  PHASE2_SUBGRAPH_RANKING_WEIGHT="$PHASE2_RANKING_WEIGHT" \
  PHASE2_SUBGRAPH_RANKING_MARGIN="$PHASE2_RANKING_MARGIN" \
  PHASE2_SUBGRAPH_HARD_NEGATIVE_TOPK="$PHASE2_HARDNEG_TOPK" \
  PHASE2_SUBGRAPH_BCE_HARD_NEGATIVE_TOPK="$PHASE2_BCE_HARDNEG_TOPK" \
  PHASE2_SUBGRAPH_POS_WEIGHT_MAX="$PHASE2_POS_WEIGHT_MAX" \
  PHASE2_BEST_METRIC="$PHASE2_BEST_METRIC" \
  AUTO_RUN_LATENT_ABLATION="$AUTO_RUN_LATENT_ABLATION" \
  ABLATION_SPLIT="$ABLATION_SPLIT" \
  bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh 2>&1 | tee "$seed_log"

  best_file="trm_agent/ckpt/cwq_trm_hier6_rearev_D_phase2_${run_tag}/best_${PHASE2_BEST_METRIC}.txt"
  if [[ -f "$best_file" ]]; then
    best_ckpt="$(cat "$best_file" | head -n1)"
    echo "[protocol] seed=$seed best=${best_ckpt}" | tee -a "$SUMMARY_FILE"
    if [[ "$EVAL_ON_TEST" == "true" ]]; then
      test_log="$LOG_ROOT/${run_tag}.test.log"
      set +e
      CUDA_VISIBLE_DEVICES="$TEST_GPU" \
      CKPT="$best_ckpt" \
      BATCH_SIZE="$TEST_BATCH_SIZE" \
      SUBGRAPH_MAX_NODES="$SUBGRAPH_MAX_NODES" \
      SUBGRAPH_MAX_EDGES="$SUBGRAPH_MAX_EDGES" \
      bash trm_rag_style/scripts/run_test.sh 2>&1 | tee "$test_log"
      test_rc=${PIPESTATUS[0]}
      set -e
      if [[ $test_rc -ne 0 ]]; then
        echo "[protocol] seed=$seed test_failed rc=$test_rc log=$test_log" | tee -a "$SUMMARY_FILE"
      else
        parsed="$(
          TEST_LOG="$test_log" /data2/workspace/heewon/anaconda3/envs/taiLab/bin/python - <<'PY'
import os, re
from pathlib import Path
lf = Path(os.environ["TEST_LOG"])
pat = re.compile(r"\[Test-Subgraph\] Hit@1=([0-9.]+) F1=([0-9.]+) Precision=([0-9.]+) Recall=([0-9.]+)")
last = None
for line in lf.open("r", encoding="utf-8", errors="ignore"):
    m = pat.search(line)
    if m:
        last = m.groups()
if last is None:
    print("")
else:
    print("\t".join(last))
PY
        )"
        if [[ -n "$parsed" ]]; then
          hit1="$(echo "$parsed" | cut -f1)"
          f1="$(echo "$parsed" | cut -f2)"
          prec="$(echo "$parsed" | cut -f3)"
          rec="$(echo "$parsed" | cut -f4)"
          echo -e "${seed}\t${run_tag}\t${best_ckpt}\t${hit1}\t${f1}\t${prec}\t${rec}" >> "$METRICS_TSV"
          echo "[protocol] seed=$seed test hit1=${hit1} f1=${f1}" | tee -a "$SUMMARY_FILE"
        else
          echo "[protocol] seed=$seed test_parse_failed log=$test_log" | tee -a "$SUMMARY_FILE"
        fi
      fi
    fi
  else
    echo "[protocol] seed=$seed best=(not found)" | tee -a "$SUMMARY_FILE"
  fi
done

if [[ "$EVAL_ON_TEST" == "true" ]]; then
  agg="$(
    METRICS_TSV="$METRICS_TSV" /data2/workspace/heewon/anaconda3/envs/taiLab/bin/python - <<'PY'
import os, csv
from statistics import mean
rows=[]
with open(os.environ["METRICS_TSV"], "r", encoding="utf-8") as f:
    rd=csv.DictReader(f, delimiter="\t")
    for r in rd:
        try:
            r["hit1"]=float(r["hit1"]); r["f1"]=float(r["f1"])
            rows.append(r)
        except Exception:
            pass
if not rows:
    print("")
else:
    best=max(rows, key=lambda x: x["hit1"])
    print(f"max_hit1_seed={best['seed']} max_hit1={best['hit1']:.4f} max_f1={best['f1']:.4f}")
    print(f"mean_hit1={mean([r['hit1'] for r in rows]):.4f} mean_f1={mean([r['f1'] for r in rows]):.4f} n={len(rows)}")
PY
  )"
  if [[ -n "$agg" ]]; then
    echo "$agg" | tee -a "$SUMMARY_FILE"
  fi
fi

echo "[done] latent-max protocol finished. summary=$SUMMARY_FILE metrics=$METRICS_TSV"
