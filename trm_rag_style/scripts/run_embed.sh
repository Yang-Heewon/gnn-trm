#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET=${DATASET:-webqsp}
EMBED_DEVICE=${EMBED_DEVICE:-cuda}
EMBED_GPUS=${EMBED_GPUS:-}
# Default to GNN-RAG gnn-compatible embedding path.
EMBED_STYLE=${EMBED_STYLE:-gnn_rag_gnn_exact}
EMBED_BACKEND=${EMBED_BACKEND:-sentence_transformers}

if [ -z "${EMB_MODEL:-}" ]; then
  case "$EMBED_STYLE" in
    gnn_exact|gnn-rag-gnn|gnn_rag_gnn|gnn_rag_gnn_exact|gnn_gnn_exact)
      EMB_MODEL='sentence-transformers/all-MiniLM-L6-v2'
      ;;
    *)
      EMB_MODEL='intfloat/multilingual-e5-large'
      ;;
  esac
fi

case "$EMBED_STYLE" in
  gnn_exact|gnn-rag-gnn|gnn_rag_gnn|gnn_rag_gnn_exact|gnn_gnn_exact)
    EMBED_QUERY_PREFIX=${EMBED_QUERY_PREFIX:-}
    EMBED_PASSAGE_PREFIX=${EMBED_PASSAGE_PREFIX:-}
    ;;
  *)
    EMBED_QUERY_PREFIX=${EMBED_QUERY_PREFIX:-query: }
    EMBED_PASSAGE_PREFIX=${EMBED_PASSAGE_PREFIX:-passage: }
    ;;
esac

ENTITY_NAMES_JSON=${ENTITY_NAMES_JSON:-data/data/entities_names.json}
OVR=(embed_device="$EMBED_DEVICE" embed_gpus="$EMBED_GPUS")
if [ -n "$EMBED_STYLE" ]; then
  OVR+=(embed_style="$EMBED_STYLE")
fi
if [ -n "$EMBED_BACKEND" ]; then
  OVR+=(embed_backend="$EMBED_BACKEND")
fi
if [ -n "$EMBED_QUERY_PREFIX" ] || [ "$EMBED_STYLE" = "gnn_exact" ] || [ "$EMBED_STYLE" = "gnn-rag-gnn" ] || [ "$EMBED_STYLE" = "gnn_rag_gnn" ] || [ "$EMBED_STYLE" = "gnn_rag_gnn_exact" ] || [ "$EMBED_STYLE" = "gnn_gnn_exact" ]; then
  OVR+=(embed_query_prefix="$EMBED_QUERY_PREFIX")
fi
if [ -n "$EMBED_PASSAGE_PREFIX" ] || [ "$EMBED_STYLE" = "gnn_exact" ] || [ "$EMBED_STYLE" = "gnn-rag-gnn" ] || [ "$EMBED_STYLE" = "gnn_rag_gnn" ] || [ "$EMBED_STYLE" = "gnn_rag_gnn_exact" ] || [ "$EMBED_STYLE" = "gnn_gnn_exact" ]; then
  OVR+=(embed_passage_prefix="$EMBED_PASSAGE_PREFIX")
fi
if [ -n "$ENTITY_NAMES_JSON" ]; then
  OVR+=(entity_names_json="$ENTITY_NAMES_JSON")
fi
$PYTHON_BIN -m trm_agent.run --dataset "$DATASET" --stage embed --embedding_model "$EMB_MODEL" \
  --override "${OVR[@]}"
