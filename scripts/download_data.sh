#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
TMP_DIR="$DATA_DIR/.downloads"

mkdir -p "$DATA_DIR" "$TMP_DIR" "$DATA_DIR/webqsp" "$DATA_DIR/CWQ/embeddings_output/CWQ/e5"

GDRIVE_FOLDER_URL="${GDRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp?usp=sharing}"
SKIP_GDRIVE="${SKIP_GDRIVE:-0}"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

require_tool() {
  local t="$1"
  have_cmd "$t" || { echo "[err] missing tool: $t"; exit 1; }
}

download_file() {
  local url="$1" out="$2"
  [ -z "$url" ] && return 0
  if [ -f "$out" ]; then
    echo "[skip] exists: $out"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  echo "[dl] $url -> $out"
  if have_cmd curl; then
    curl -L --fail --retry 3 "$url" -o "$out"
  elif have_cmd wget; then
    wget -O "$out" "$url"
  else
    echo "[err] curl/wget not found"
    exit 1
  fi
}

extract_if_archive() {
  local path="$1" target="$2"
  case "$path" in
    *.zip) require_tool unzip; unzip -o "$path" -d "$target" >/dev/null ;;
    *.tar) tar -xf "$path" -C "$target" ;;
    *.tar.gz|*.tgz) tar -xzf "$path" -C "$target" ;;
    *) return 1 ;;
  esac
  return 0
}

copy_first_match() {
  local target="$1"
  shift
  [ -f "$target" ] && { echo "[skip] exists: $target"; return 0; }

  local found=""
  for pat in "$@"; do
    found="$(find "$DATA_DIR" "$TMP_DIR" -type f -iname "$pat" 2>/dev/null | head -n 1 || true)"
    if [ -n "$found" ]; then
      break
    fi
  done

  if [ -z "$found" ]; then
    echo "[miss] could not find source for $target"
    return 1
  fi

  mkdir -p "$(dirname "$target")"
  cp -f "$found" "$target"
  echo "[map] $found -> $target"
  return 0
}

require_path() {
  local p="$1"
  if [ ! -f "$p" ]; then
    echo "[missing] $p"
    return 1
  fi
  echo "[ok] $p"
  return 0
}

# Optional direct URLs
WEBQSP_URL="${WEBQSP_URL:-}"
CWQ_URL="${CWQ_URL:-}"
WEBQSP_TRAIN_URL="${WEBQSP_TRAIN_URL:-}"
WEBQSP_DEV_URL="${WEBQSP_DEV_URL:-}"
WEBQSP_ENTITIES_URL="${WEBQSP_ENTITIES_URL:-}"
WEBQSP_RELATIONS_URL="${WEBQSP_RELATIONS_URL:-}"
CWQ_TRAIN_URL="${CWQ_TRAIN_URL:-}"
CWQ_DEV_URL="${CWQ_DEV_URL:-}"
CWQ_ENTITY_IDS_URL="${CWQ_ENTITY_IDS_URL:-}"
CWQ_RELATION_IDS_URL="${CWQ_RELATION_IDS_URL:-}"

# 1) Google Drive folder download (default on)
if [ "$SKIP_GDRIVE" != "1" ] && [ -n "$GDRIVE_FOLDER_URL" ]; then
  if ! have_cmd gdown; then
    echo "[err] gdown not found. Install with: pip install gdown"
    exit 1
  fi
  GDRIVE_DIR="$TMP_DIR/gdrive_folder"
  mkdir -p "$GDRIVE_DIR"
  echo "[dl] gdrive folder -> $GDRIVE_DIR"
  gdown --folder --fuzzy "$GDRIVE_FOLDER_URL" -O "$GDRIVE_DIR"
fi

# 2) Optional bundle download/extract via URL
if [ -n "$WEBQSP_URL" ]; then
  webqsp_bundle="$TMP_DIR/$(basename "$WEBQSP_URL")"
  download_file "$WEBQSP_URL" "$webqsp_bundle"
  extract_if_archive "$webqsp_bundle" "$DATA_DIR" || true
fi
if [ -n "$CWQ_URL" ]; then
  cwq_bundle="$TMP_DIR/$(basename "$CWQ_URL")"
  download_file "$CWQ_URL" "$cwq_bundle"
  extract_if_archive "$cwq_bundle" "$DATA_DIR" || true
fi

# 3) Optional per-file direct URLs
[ -n "$WEBQSP_TRAIN_URL" ] && download_file "$WEBQSP_TRAIN_URL" "$DATA_DIR/webqsp/train.json"
[ -n "$WEBQSP_DEV_URL" ] && download_file "$WEBQSP_DEV_URL" "$DATA_DIR/webqsp/dev.json"
[ -n "$WEBQSP_ENTITIES_URL" ] && download_file "$WEBQSP_ENTITIES_URL" "$DATA_DIR/webqsp/entities.txt"
[ -n "$WEBQSP_RELATIONS_URL" ] && download_file "$WEBQSP_RELATIONS_URL" "$DATA_DIR/webqsp/relations.txt"

[ -n "$CWQ_TRAIN_URL" ] && download_file "$CWQ_TRAIN_URL" "$DATA_DIR/CWQ/train_split.jsonl"
[ -n "$CWQ_DEV_URL" ] && download_file "$CWQ_DEV_URL" "$DATA_DIR/CWQ/dev_split.jsonl"
[ -n "$CWQ_ENTITY_IDS_URL" ] && download_file "$CWQ_ENTITY_IDS_URL" "$DATA_DIR/CWQ/embeddings_output/CWQ/e5/entity_ids.txt"
[ -n "$CWQ_RELATION_IDS_URL" ] && download_file "$CWQ_RELATION_IDS_URL" "$DATA_DIR/CWQ/embeddings_output/CWQ/e5/relation_ids.txt"

# 4) Auto-map from downloaded folder content
copy_first_match "$DATA_DIR/webqsp/train.json" "train.json" "*webqsp*train*.json"
copy_first_match "$DATA_DIR/webqsp/dev.json" "dev.json" "*webqsp*dev*.json"
copy_first_match "$DATA_DIR/webqsp/entities.txt" "entities.txt" "*webqsp*entities*.txt" "entity_ids.txt"
copy_first_match "$DATA_DIR/webqsp/relations.txt" "relations.txt" "*webqsp*relations*.txt" "relation_ids.txt"

copy_first_match "$DATA_DIR/CWQ/train_split.jsonl" "train_split.jsonl" "*cwq*train*.jsonl" "*cwq*train*.json"
copy_first_match "$DATA_DIR/CWQ/dev_split.jsonl" "dev_split.jsonl" "*cwq*dev*.jsonl" "*cwq*dev*.json"
copy_first_match "$DATA_DIR/CWQ/embeddings_output/CWQ/e5/entity_ids.txt" "entity_ids.txt" "*cwq*entity*ids*.txt"
copy_first_match "$DATA_DIR/CWQ/embeddings_output/CWQ/e5/relation_ids.txt" "relation_ids.txt" "*cwq*relation*ids*.txt"

# 5) Verification
set +e
status=0
require_path "$DATA_DIR/webqsp/train.json" || status=1
require_path "$DATA_DIR/webqsp/dev.json" || status=1
require_path "$DATA_DIR/webqsp/entities.txt" || status=1
require_path "$DATA_DIR/webqsp/relations.txt" || status=1
require_path "$DATA_DIR/CWQ/train_split.jsonl" || status=1
require_path "$DATA_DIR/CWQ/dev_split.jsonl" || status=1
require_path "$DATA_DIR/CWQ/embeddings_output/CWQ/e5/entity_ids.txt" || status=1
require_path "$DATA_DIR/CWQ/embeddings_output/CWQ/e5/relation_ids.txt" || status=1
set -e

if [ "$status" -ne 0 ]; then
  echo "[done] data setup incomplete. Check file names under data/.downloads and update env URLs if needed."
  exit 2
fi

echo "[done] data is ready"
