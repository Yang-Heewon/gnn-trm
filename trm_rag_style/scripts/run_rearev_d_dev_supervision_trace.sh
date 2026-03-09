#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data2/workspace/heewon/anaconda3/envs/taiLab/bin/python}"
DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_TAG="${EMB_TAG:-e5_w4_g4}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"
CKPT="${CKPT:-}"

if [ -z "$CKPT" ]; then
  echo "Set CKPT=/path/to/model_epX.pt"
  exit 1
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EVAL_JSON="${EVAL_JSON:-trm_agent/processed/${DATASET}/dev.jsonl}"
QUERY_EMB_EVAL_NPY="${QUERY_EMB_EVAL_NPY:-${EMB_DIR}/query_dev.npy}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_LIMIT="${EVAL_LIMIT:--1}"

SUBGRAPH_MAX_NODES="${SUBGRAPH_MAX_NODES:-4096}"
SUBGRAPH_MAX_EDGES="${SUBGRAPH_MAX_EDGES:-16384}"
SUBGRAPH_RECURSION_STEPS="${SUBGRAPH_RECURSION_STEPS:-1}"
SUBGRAPH_REAREV_ADAPT_STAGES="${SUBGRAPH_REAREV_ADAPT_STAGES:-4}"
SUBGRAPH_REAREV_TRM_STYLE_ENABLED="${SUBGRAPH_REAREV_TRM_STYLE_ENABLED:-true}"
SUBGRAPH_REAREV_TRM_SUPERVISE_ALL_STAGES="${SUBGRAPH_REAREV_TRM_SUPERVISE_ALL_STAGES:-true}"
SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED="${SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED:-true}"
SUBGRAPH_REAREV_ASYM_INNER_Y_EMA_ENABLED="${SUBGRAPH_REAREV_ASYM_INNER_Y_EMA_ENABLED:-true}"
SUBGRAPH_REAREV_ASYM_INNER_Y_EMA_ALPHA="${SUBGRAPH_REAREV_ASYM_INNER_Y_EMA_ALPHA:-0.25}"

TRACE_EXAMPLES="${TRACE_EXAMPLES:-5}"
TRACE_DUMP_JSONL="${TRACE_DUMP_JSONL:-logs/trace/dev_supervision_trace_${RUN_TAG}.jsonl}"
TRACE_PLOT_PNG="${TRACE_PLOT_PNG:-logs/trace/dev_supervision_trace_${RUN_TAG}.png}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$PYTHON_BIN" -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage test \
  --ckpt "$CKPT" \
  --override \
    emb_tag="$EMB_TAG" \
    emb_dir="$EMB_DIR" \
    eval_json="$EVAL_JSON" \
    query_emb_eval_npy="$QUERY_EMB_EVAL_NPY" \
    batch_size="$BATCH_SIZE" \
    eval_limit="$EVAL_LIMIT" \
    wandb_mode=disabled \
    subgraph_reader_enabled=true \
    subgraph_max_nodes="$SUBGRAPH_MAX_NODES" \
    subgraph_max_edges="$SUBGRAPH_MAX_EDGES" \
    subgraph_recursion_steps="$SUBGRAPH_RECURSION_STEPS" \
    subgraph_rearev_adapt_stages="$SUBGRAPH_REAREV_ADAPT_STAGES" \
    subgraph_rearev_trm_style_enabled="$SUBGRAPH_REAREV_TRM_STYLE_ENABLED" \
    subgraph_rearev_trm_supervise_all_stages="$SUBGRAPH_REAREV_TRM_SUPERVISE_ALL_STAGES" \
    subgraph_rearev_asymmetric_yz_enabled="$SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED" \
    subgraph_rearev_asym_inner_y_ema_enabled="$SUBGRAPH_REAREV_ASYM_INNER_Y_EMA_ENABLED" \
    subgraph_rearev_asym_inner_y_ema_alpha="$SUBGRAPH_REAREV_ASYM_INNER_Y_EMA_ALPHA" \
    subgraph_trace_supervision_enabled=true \
    subgraph_trace_supervision_examples="$TRACE_EXAMPLES" \
    subgraph_trace_supervision_dump_jsonl="$TRACE_DUMP_JSONL" \
    subgraph_trace_supervision_plot_png="$TRACE_PLOT_PNG"

echo "[done] supervision trace dump: $TRACE_DUMP_JSONL"
echo "[done] supervision trace plot: $TRACE_PLOT_PNG"
