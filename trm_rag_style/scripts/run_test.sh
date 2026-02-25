#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

if [ "${CONDA_DEFAULT_ENV:-}" = "base" ]; then
  echo "[warn] current conda env is 'base'."
  echo "       If checkpoint/tokenizer was trained in another env, set PYTHON_BIN explicitly."
fi

DATASET=${DATASET:-cwq}
MODEL_IMPL=${MODEL_IMPL:-trm_hier6}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}
EMB_TAG=${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}
EMB_DIR=${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}
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
EVAL_RANDOM_SAMPLE_SIZE=${EVAL_RANDOM_SAMPLE_SIZE:-0}
EVAL_RANDOM_SEED=${EVAL_RANDOM_SEED:-42}
QUERY_RESIDUAL_ENABLED=${QUERY_RESIDUAL_ENABLED:-false}
QUERY_RESIDUAL_ALPHA=${QUERY_RESIDUAL_ALPHA:-0.0}
QUERY_RESIDUAL_MODE=${QUERY_RESIDUAL_MODE:-sub_rel}
EVAL_DUMP_JSONL=${EVAL_DUMP_JSONL:-}
SUBGRAPH_HOPS=${SUBGRAPH_HOPS:-3}
SUBGRAPH_MAX_NODES=${SUBGRAPH_MAX_NODES:-2048}
SUBGRAPH_MAX_EDGES=${SUBGRAPH_MAX_EDGES:-8192}
SUBGRAPH_RECURSION_STEPS=${SUBGRAPH_RECURSION_STEPS:-12}
SUBGRAPH_PRED_THRESHOLD=${SUBGRAPH_PRED_THRESHOLD:-0.5}
SUBGRAPH_SPLIT_REVERSE_RELATIONS=${SUBGRAPH_SPLIT_REVERSE_RELATIONS:-false}
SUBGRAPH_DIRECTION_EMBEDDING_ENABLED=${SUBGRAPH_DIRECTION_EMBEDDING_ENABLED:-false}

if [ -z "$CKPT" ]; then
  echo "Set CKPT=/path/to/model_epX.pt"
  exit 1
fi

_to_bool() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "$CKPT" == *"_sup_residual"* ]] && ! _to_bool "$QUERY_RESIDUAL_ENABLED"; then
  echo "[warn] CKPT path suggests residual-trained model, but QUERY_RESIDUAL_ENABLED=false."
  echo "       For consistent eval, set:"
  echo "       QUERY_RESIDUAL_ENABLED=true QUERY_RESIDUAL_ALPHA=0.15 QUERY_RESIDUAL_MODE=sub_rel"
fi

if [ ! -f "$EMB_DIR/entity_embeddings.npy" ] || [ ! -f "$EMB_DIR/relation_embeddings.npy" ] || [ ! -f "$EMB_DIR/query_test.npy" ]; then
  echo "[err] required embedding files are missing in $EMB_DIR"
  echo "      expected: entity_embeddings.npy, relation_embeddings.npy, query_test.npy"
  echo "      run embed first with same EMB_MODEL/EMB_TAG."
  exit 2
fi

$PYTHON_BIN -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage test \
  --ckpt "$CKPT" \
  --override \
    emb_tag="$EMB_TAG" \
    emb_dir="$EMB_DIR" \
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
    eval_min_hops_before_stop="$EVAL_MIN_HOPS_BEFORE_STOP" \
    eval_random_sample_size="$EVAL_RANDOM_SAMPLE_SIZE" \
    eval_random_seed="$EVAL_RANDOM_SEED" \
    query_residual_enabled="$QUERY_RESIDUAL_ENABLED" \
    query_residual_alpha="$QUERY_RESIDUAL_ALPHA" \
    query_residual_mode="$QUERY_RESIDUAL_MODE" \
    eval_dump_jsonl="$EVAL_DUMP_JSONL" \
    subgraph_reader_enabled=true \
    subgraph_hops="$SUBGRAPH_HOPS" \
    subgraph_max_nodes="$SUBGRAPH_MAX_NODES" \
    subgraph_max_edges="$SUBGRAPH_MAX_EDGES" \
    subgraph_recursion_steps="$SUBGRAPH_RECURSION_STEPS" \
    subgraph_pred_threshold="$SUBGRAPH_PRED_THRESHOLD" \
    subgraph_split_reverse_relations="$SUBGRAPH_SPLIT_REVERSE_RELATIONS" \
    subgraph_direction_embedding_enabled="$SUBGRAPH_DIRECTION_EMBEDDING_ENABLED"
