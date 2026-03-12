#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CKPT_DIR="${CKPT_DIR:-}"
if [[ -z "$CKPT_DIR" ]]; then
  CKPT_DIR="$(
    find trm_agent/ckpt -maxdepth 1 -mindepth 1 -type d -name '*_rearev_D_phase2_*' -printf '%T@ %p\n' \
      | sort -nr \
      | head -n1 \
      | awk '{print $2}'
  )"
  if [[ -z "$CKPT_DIR" ]]; then
    echo "[err] CKPT_DIR not set and no phase2 checkpoint directory found under trm_agent/ckpt"
    exit 2
  fi
  echo "[info] CKPT_DIR not set. use latest phase2 dir: $CKPT_DIR"
fi

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[err] checkpoint directory not found: $CKPT_DIR"
  exit 2
fi

metric_raw="${METRIC:-dev_f1}"
metric="$(echo "$metric_raw" | tr '[:upper:]' '[:lower:]')"
case "$metric" in
  dev_f1|f1)
    metric="dev_f1"
    best_file="$CKPT_DIR/best_dev_f1.txt"
    ;;
  dev_hit1|hit1)
    metric="dev_hit1"
    best_file="$CKPT_DIR/best_dev_hit1.txt"
    ;;
  *)
    echo "[err] unsupported METRIC=$metric_raw (use dev_f1 or dev_hit1)"
    exit 2
    ;;
esac

if [[ -f "$best_file" ]]; then
  CKPT="$(cat "$best_file")"
else
  echo "[warn] best file not found: $best_file"
  CKPT="$(
    ls -1 "$CKPT_DIR"/model_ep*.pt 2>/dev/null \
      | sed -E 's#^.*model_ep([0-9]+)\.pt$#\1\t&#' \
      | sort -n \
      | tail -n1 \
      | cut -f2
  )"
fi

if [[ -z "${CKPT:-}" ]]; then
  echo "[err] could not resolve checkpoint from $CKPT_DIR"
  exit 2
fi
if [[ ! -f "$CKPT" && -f "$REPO_ROOT/$CKPT" ]]; then
  CKPT="$REPO_ROOT/$CKPT"
fi
if [[ ! -f "$CKPT" ]]; then
  echo "[err] checkpoint file not found: $CKPT"
  exit 2
fi

if [[ -z "${DATASET:-}" ]]; then
  ckpt_base="$(basename "$CKPT_DIR")"
  DATASET="${ckpt_base%%_*}"
  export DATASET
  echo "[info] inferred DATASET=$DATASET from ckpt dir"
fi

echo "[test] metric=$metric ckpt=$CKPT"
CKPT="$CKPT" bash trm_rag_style/scripts/run_test.sh
