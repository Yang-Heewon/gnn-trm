#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"

DATA_DIR="$REPO_ROOT/data"
mkdir -p "$DATA_DIR" "$DATA_DIR/.downloads"

TARGET_DATASET="$(echo "${DATASET:-all}" | tr '[:upper:]' '[:lower:]')"
DATA_SOURCE="$(echo "${DATA_SOURCE:-rog_hf}" | tr '[:upper:]' '[:lower:]')"
ROG_CWQ_DATASET="${ROG_CWQ_DATASET:-rmanluo/RoG-cwq}"
ROG_WEBQSP_DATASET="${ROG_WEBQSP_DATASET:-rmanluo/RoG-webqsp}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"

require_path() {
  local p="$1"
  if [ ! -f "$p" ]; then
    echo "[missing] $p"
    return 1
  fi
  echo "[ok] $p"
  return 0
}

if [ "$DATA_SOURCE" != "rog_hf" ] && [ "$DATA_SOURCE" != "hf_rog" ] && [ "$DATA_SOURCE" != "rog" ]; then
  echo "[err] this pipeline now supports only RoG HF data source."
  echo "      set DATA_SOURCE=rog_hf"
  exit 2
fi

echo "[step] preparing RoG datasets from Hugging Face"
PREP_ARGS=(
  --dataset "$TARGET_DATASET"
  --cwq_name "$ROG_CWQ_DATASET"
  --webqsp_name "$ROG_WEBQSP_DATASET"
  --out_root "$REPO_ROOT"
)
if [ -n "$HF_CACHE_DIR" ]; then
  PREP_ARGS+=(--cache_dir "$HF_CACHE_DIR")
fi
"$PYTHON_BIN" scripts/prepare_rog_hf_data.py "${PREP_ARGS[@]}"

set +e
status=0
if [ "$TARGET_DATASET" = "webqsp" ] || [ "$TARGET_DATASET" = "all" ]; then
  require_path "$DATA_DIR/webqsp/train.json" || status=1
  require_path "$DATA_DIR/webqsp/dev.json" || status=1
  require_path "$DATA_DIR/webqsp/test.json" || status=1
  require_path "$DATA_DIR/webqsp/entities.txt" || status=1
  require_path "$DATA_DIR/webqsp/relations.txt" || status=1
fi
if [ "$TARGET_DATASET" = "cwq" ] || [ "$TARGET_DATASET" = "all" ]; then
  require_path "$DATA_DIR/CWQ/train_split.jsonl" || status=1
  require_path "$DATA_DIR/CWQ/dev_split.jsonl" || status=1
  require_path "$DATA_DIR/CWQ/test_split.jsonl" || status=1
  require_path "$DATA_DIR/CWQ/embeddings_output/CWQ/e5/entity_ids.txt" || status=1
  require_path "$DATA_DIR/CWQ/embeddings_output/CWQ/e5/relation_ids.txt" || status=1
  require_path "$DATA_DIR/data/CWQ/test.json" || status=1
fi
set -e

if [ "$status" -ne 0 ]; then
  echo "[done] RoG HF data preparation incomplete."
  exit 2
fi

echo "[done] RoG HF data is ready"
