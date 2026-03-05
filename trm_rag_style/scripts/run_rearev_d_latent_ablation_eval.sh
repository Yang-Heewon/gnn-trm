#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Run latent ablation on the same checkpoint to measure causal contribution.
# Usage example:
#   CKPT=trm_agent/ckpt/.../model_ep30.pt SPLIT=test CUDA_VISIBLE_DEVICES=2 \
#   bash trm_rag_style/scripts/run_rearev_d_latent_ablation_eval.sh

PYTHON_BIN="${PYTHON_BIN:-/data2/workspace/heewon/anaconda3/envs/taiLab/bin/python}"
DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_TAG="${EMB_TAG:-e5_w4_g4}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"
CKPT="${CKPT:-}"
SPLIT="${SPLIT:-dev}" # dev|test
EVAL_LIMIT="${EVAL_LIMIT:--1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MODES="${MODES:-full,no_latent,no_global,no_ins_delta}"
LOG_DIR="${LOG_DIR:-logs/latent_ablation_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "$CKPT" ]]; then
  echo "[err] set CKPT=/path/to/model_epX.pt"
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "[err] checkpoint not found: $CKPT"
  exit 2
fi

mkdir -p "$LOG_DIR"

case "$SPLIT" in
  dev)
    EVAL_JSON="trm_agent/processed/${DATASET}/dev.jsonl"
    QUERY_EMB="trm_agent/emb/${DATASET}_${EMB_TAG}/query_dev.npy"
    ;;
  test)
    EVAL_JSON="trm_agent/processed/${DATASET}/test.jsonl"
    QUERY_EMB="trm_agent/emb/${DATASET}_${EMB_TAG}/query_test.npy"
    ;;
  *)
    echo "[err] SPLIT must be dev or test. got: $SPLIT"
    exit 2
    ;;
esac

if [[ ! -f "$EVAL_JSON" ]]; then
  echo "[err] eval file missing: $EVAL_JSON"
  exit 2
fi
if [[ ! -f "$QUERY_EMB" ]]; then
  echo "[err] query emb missing: $QUERY_EMB"
  exit 2
fi

run_case() {
  local name="$1"
  local latent="$2"
  local gate="$3"
  local fusion="$4"
  local halt="$5"
  local alpha="$6"
  local log_file="$LOG_DIR/${name}.log"

  echo "[run] case=$name latent=$latent gate=$gate fusion=$fusion halt=$halt alpha=$alpha"
  "$PYTHON_BIN" -m trm_agent.run \
    --dataset "$DATASET" \
    --model_impl "$MODEL_IMPL" \
    --stage test \
    --ckpt "$CKPT" \
    --override \
      emb_tag="$EMB_TAG" \
      emb_dir="$EMB_DIR" \
      eval_json="$EVAL_JSON" \
      query_emb_eval_npy="$QUERY_EMB" \
      eval_limit="$EVAL_LIMIT" \
      batch_size="$BATCH_SIZE" \
      wandb_mode=offline \
      subgraph_reader_enabled=true \
      subgraph_rearev_latent_reasoning_enabled="$latent" \
      subgraph_rearev_global_gate_enabled="$gate" \
      subgraph_rearev_logit_global_fusion_enabled="$fusion" \
      subgraph_rearev_dynamic_halting_enabled="$halt" \
      subgraph_rearev_latent_residual_alpha="$alpha" \
    2>&1 | tee "$log_file"
}

IFS=',' read -r -a mode_list <<< "$MODES"
for m in "${mode_list[@]}"; do
  case "$m" in
    full)
      run_case "full" "true" "true" "true" "true" "0.25"
      ;;
    no_latent)
      # Keep global modules ON but disable latent-state updates.
      run_case "no_latent" "false" "true" "true" "true" "0.25"
      ;;
    no_global)
      # Disable latent and global control path entirely.
      run_case "no_global" "false" "false" "false" "false" "0.25"
      ;;
    no_ins_delta)
      # Keep latent recurrence, remove instruction residual injection only.
      run_case "no_ins_delta" "true" "true" "true" "true" "0.0"
      ;;
    *)
      echo "[warn] unknown mode: $m (skip)"
      ;;
  esac
done

echo
echo "[summary] split=$SPLIT ckpt=$CKPT"
printf "%-14s  %s\n" "case" "metrics"
printf "%-14s  %s\n" "--------------" "---------------------------------------------"
for lf in "$LOG_DIR"/*.log; do
  case_name="$(basename "$lf" .log)"
  metric_line="$(grep -E "\\[Test-Subgraph\\]" "$lf" | tail -1 || true)"
  if [[ -z "$metric_line" ]]; then
    metric_line="(no metric line found)"
  fi
  printf "%-14s  %s\n" "$case_name" "$metric_line"
done

echo
echo "[done] logs: $LOG_DIR"
