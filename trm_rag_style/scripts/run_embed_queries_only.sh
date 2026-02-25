#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET="${DATASET:-cwq}"
EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"
EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
EMBED_GPUS="${EMBED_GPUS:-0,1,2,3}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"
EMBED_MAX_LENGTH="${EMBED_MAX_LENGTH:-128}"
EMBED_BACKEND="${EMBED_BACKEND:-transformers}"
EMBED_STYLE="${EMBED_STYLE:-gnn_rag_gnn_exact}"
QUERY_PREFIX="${QUERY_PREFIX:-}"
SAVE_LISTS="${SAVE_LISTS:-true}"

SRC_EMB_TAG="${SRC_EMB_TAG:-e5}"
DST_EMB_TAG="${DST_EMB_TAG:-${SRC_EMB_TAG}_qonly_g4}"
SRC_EMB_DIR="${SRC_EMB_DIR:-trm_agent/emb/${DATASET}_${SRC_EMB_TAG}}"
DST_EMB_DIR="${DST_EMB_DIR:-trm_agent/emb/${DATASET}_${DST_EMB_TAG}}"
PROCESSED_DIR="${PROCESSED_DIR:-trm_agent/processed/${DATASET}}"

TRAIN_JSONL="${TRAIN_JSONL:-${PROCESSED_DIR}/train.jsonl}"
DEV_JSONL="${DEV_JSONL:-${PROCESSED_DIR}/dev.jsonl}"
TEST_JSONL="${TEST_JSONL:-${PROCESSED_DIR}/test.jsonl}"

if [ ! -f "$SRC_EMB_DIR/entity_embeddings.npy" ] || [ ! -f "$SRC_EMB_DIR/relation_embeddings.npy" ]; then
  echo "[err] source entity/relation embeddings not found in $SRC_EMB_DIR"
  exit 2
fi
if [ ! -f "$TRAIN_JSONL" ] || [ ! -f "$DEV_JSONL" ]; then
  echo "[err] processed train/dev jsonl not found under $PROCESSED_DIR"
  exit 2
fi

mkdir -p "$DST_EMB_DIR"
if [ ! -s "$DST_EMB_DIR/entity_embeddings.npy" ]; then
  cp -f "$SRC_EMB_DIR/entity_embeddings.npy" "$DST_EMB_DIR/entity_embeddings.npy"
fi
if [ ! -s "$DST_EMB_DIR/relation_embeddings.npy" ]; then
  cp -f "$SRC_EMB_DIR/relation_embeddings.npy" "$DST_EMB_DIR/relation_embeddings.npy"
fi

if [ -f "$SRC_EMB_DIR/entity_ids.txt" ]; then cp -f "$SRC_EMB_DIR/entity_ids.txt" "$DST_EMB_DIR/entity_ids.txt"; fi
if [ -f "$SRC_EMB_DIR/relation_ids.txt" ]; then cp -f "$SRC_EMB_DIR/relation_ids.txt" "$DST_EMB_DIR/relation_ids.txt"; fi
if [ -f "$SRC_EMB_DIR/entity_names_used.txt" ]; then cp -f "$SRC_EMB_DIR/entity_names_used.txt" "$DST_EMB_DIR/entity_names_used.txt"; fi
if [ -f "$SRC_EMB_DIR/relation_names_used.txt" ]; then cp -f "$SRC_EMB_DIR/relation_names_used.txt" "$DST_EMB_DIR/relation_names_used.txt"; fi

export EMB_MODEL EMBED_BATCH_SIZE EMBED_MAX_LENGTH EMBED_DEVICE EMBED_GPUS EMBED_BACKEND EMBED_STYLE QUERY_PREFIX
export TRAIN_JSONL DEV_JSONL TEST_JSONL DST_EMB_DIR SAVE_LISTS

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

from trm_unified.embedder import collect_questions_and_ids, encode_texts

model_name = os.environ["EMB_MODEL"]
batch_size = int(os.environ["EMBED_BATCH_SIZE"])
max_length = int(os.environ["EMBED_MAX_LENGTH"])
device = os.environ["EMBED_DEVICE"]
embed_gpus = os.environ["EMBED_GPUS"]
backend = os.environ["EMBED_BACKEND"]
style = os.environ["EMBED_STYLE"].strip().lower()
query_prefix = os.environ.get("QUERY_PREFIX", "")
train_jsonl = os.environ["TRAIN_JSONL"]
dev_jsonl = os.environ["DEV_JSONL"]
test_jsonl = os.environ.get("TEST_JSONL", "")
out_dir = Path(os.environ["DST_EMB_DIR"])
save_lists = str(os.environ.get("SAVE_LISTS", "true")).strip().lower() in {"1", "true", "yes", "y", "on"}

gnn_style = style in {"gnn_rag", "gnn-rag", "paperstyle", "legacy"}
gnn_exact_style = style in {
    "gnn_exact",
    "gnn-rag-gnn",
    "gnn_rag_gnn",
    "gnn_rag_gnn_exact",
    "gnn_gnn_exact",
}
if gnn_exact_style:
    query_prefix = ""
elif gnn_style and not query_prefix:
    query_prefix = "query: "

q_train, q_train_ids = collect_questions_and_ids(train_jsonl)
q_dev, q_dev_ids = collect_questions_and_ids(dev_jsonl)
q_test, q_test_ids = ([], [])
if test_jsonl and os.path.exists(test_jsonl):
    q_test, q_test_ids = collect_questions_and_ids(test_jsonl)

qtr = encode_texts(
    model_name, q_train, batch_size, max_length, device,
    embed_gpus=embed_gpus, prefix=query_prefix, backend=backend, desc="embed:query_train"
)
qdv = encode_texts(
    model_name, q_dev, batch_size, max_length, device,
    embed_gpus=embed_gpus, prefix=query_prefix, backend=backend, desc="embed:query_dev"
)
np.save(out_dir / "query_train.npy", qtr)
np.save(out_dir / "query_dev.npy", qdv)

meta_path = out_dir / "embedding_meta.json"
meta = {}
if meta_path.exists():
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
meta.update(
    {
        "model_name": model_name,
        "embed_style": style,
        "embed_backend": backend,
        "query_prefix": query_prefix,
        "query_train_shape": list(qtr.shape),
        "query_dev_shape": list(qdv.shape),
    }
)

if q_test:
    qte = encode_texts(
        model_name, q_test, batch_size, max_length, device,
        embed_gpus=embed_gpus, prefix=query_prefix, backend=backend, desc="embed:query_test"
    )
    np.save(out_dir / "query_test.npy", qte)
    meta["query_test_shape"] = list(qte.shape)

meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

if save_lists:
    def save_lines(path: Path, items):
        with path.open("w", encoding="utf-8") as f:
            for x in items:
                f.write(f"{x}\n")

    save_lines(out_dir / "query_train.txt", q_train)
    save_lines(out_dir / "query_dev.txt", q_dev)
    save_lines(out_dir / "query_train_ids.txt", q_train_ids)
    save_lines(out_dir / "query_dev_ids.txt", q_dev_ids)
    if q_test:
        save_lines(out_dir / "query_test.txt", q_test)
        save_lines(out_dir / "query_test_ids.txt", q_test_ids)

print(
    "✅ query-only embed done:",
    {
        "out_dir": str(out_dir),
        "query_train_shape": list(qtr.shape),
        "query_dev_shape": list(qdv.shape),
        "query_test_shape": meta.get("query_test_shape", []),
        "embed_backend": backend,
        "embed_gpus": embed_gpus,
    },
)
PY
