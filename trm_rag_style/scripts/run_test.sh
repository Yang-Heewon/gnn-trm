#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET=${DATASET:-webqsp}
MODEL_IMPL=${MODEL_IMPL:-trm_hier6}
CKPT=${CKPT:-}
EVAL_LIMIT=${EVAL_LIMIT:--1}
DEBUG_EVAL_N=${DEBUG_EVAL_N:-5}
BATCH_SIZE=${BATCH_SIZE:-6}
EVAL_NO_CYCLE=${EVAL_NO_CYCLE:-true}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-4}
EVAL_MAX_NEIGHBORS=${EVAL_MAX_NEIGHBORS:-256}
EVAL_PRUNE_KEEP=${EVAL_PRUNE_KEEP:-64}
EVAL_BEAM=${EVAL_BEAM:-8}
EVAL_START_TOPK=${EVAL_START_TOPK:-5}
EVAL_PRED_TOPK=${EVAL_PRED_TOPK:-5}
EVAL_USE_HALT=${EVAL_USE_HALT:-true}
EVAL_MIN_HOPS_BEFORE_STOP=${EVAL_MIN_HOPS_BEFORE_STOP:-2}

if [ -z "$CKPT" ]; then
  echo "Set CKPT=/path/to/model_epX.pt"
  exit 1
fi

$PYTHON_BIN -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage test \
  --ckpt "$CKPT" \
  --override \
    eval_limit="$EVAL_LIMIT" \
    debug_eval_n="$DEBUG_EVAL_N" \
    batch_size="$BATCH_SIZE" \
    eval_no_cycle="$EVAL_NO_CYCLE" \
    eval_max_steps="$EVAL_MAX_STEPS" \
    eval_max_neighbors="$EVAL_MAX_NEIGHBORS" \
    eval_prune_keep="$EVAL_PRUNE_KEEP" \
    eval_beam="$EVAL_BEAM" \
    eval_start_topk="$EVAL_START_TOPK" \
    eval_pred_topk="$EVAL_PRED_TOPK" \
    eval_use_halt="$EVAL_USE_HALT" \
    eval_min_hops_before_stop="$EVAL_MIN_HOPS_BEFORE_STOP"
