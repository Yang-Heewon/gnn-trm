#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

mkdir -p "$RESULTS_DIR"

if [ -n "${PHASE2_CKPT:-}" ]; then
  CKPT="$PHASE2_CKPT"
else
  CKPT="$(ls -1 "$CKPT_DIR_PHASE2"/model_ep*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then
  echo "Phase2 checkpoint not found. Run 03_train_phase2_rl.sh first or set PHASE2_CKPT=/path/to/model_epX.pt"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$RESULTS_DIR/${DATASET}_phase2_test_${TS}.log"
SUMMARY_PATH="$RESULTS_DIR/${DATASET}_phase2_test_${TS}.summary.txt"

echo "Using phase2 checkpoint: $CKPT"

set -o pipefail
DATASET="$DATASET" \
MODEL_IMPL="$MODEL_IMPL" \
EMB_MODEL="$EMB_MODEL" \
EMB_TAG="$EMB_TAG" \
CKPT="$CKPT" \
EVAL_LIMIT="$TEST_EVAL_LIMIT" \
DEBUG_EVAL_N="$TEST_DEBUG_EVAL_N" \
EVAL_MAX_STEPS="$MAX_STEPS" \
EVAL_NO_CYCLE="$TEST_EVAL_NO_CYCLE" \
EVAL_PRED_TOPK="$TEST_EVAL_PRED_TOPK" \
bash trm_rag_style/scripts/run_test.sh | tee "$LOG_PATH"

LAST_TEST_LINE="$(grep -E '\[Test\] Hit@1=' "$LOG_PATH" | tail -n 1 || true)"
if [ -z "$LAST_TEST_LINE" ]; then
  echo "Could not find final [Test] metric line in $LOG_PATH" | tee "$SUMMARY_PATH"
  exit 2
fi

{
  echo "dataset=$DATASET"
  echo "ckpt=$CKPT"
  echo "log=$LOG_PATH"
  echo "$LAST_TEST_LINE"
} | tee "$SUMMARY_PATH"

echo "Saved summary: $SUMMARY_PATH"
