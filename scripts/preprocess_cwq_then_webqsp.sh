#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

MAX_STEPS="${MAX_STEPS:-4}"
MAX_PATHS="${MAX_PATHS:-4}"
MINE_MAX_NEIGHBORS="${MINE_MAX_NEIGHBORS:-128}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-0}"
TRAIN_PATH_POLICY="${TRAIN_PATH_POLICY:-shortest_only}"
TRAIN_SHORTEST_K="${TRAIN_SHORTEST_K:-1}"

echo "[step] preprocess cwq first (BFS depth=$MAX_STEPS, max_paths=$MAX_PATHS)"
$PYTHON_BIN -m trm_agent_pipeline.run \
  --dataset cwq \
  --stage preprocess \
  --override \
    max_steps="$MAX_STEPS" \
    max_paths="$MAX_PATHS" \
    mine_max_neighbors="$MINE_MAX_NEIGHBORS" \
    preprocess_workers="$PREPROCESS_WORKERS" \
    train_path_policy="$TRAIN_PATH_POLICY" \
    train_shortest_k="$TRAIN_SHORTEST_K"

echo "[step] preprocess webqsp"
$PYTHON_BIN -m trm_agent_pipeline.run \
  --dataset webqsp \
  --stage preprocess \
  --override \
    max_steps="$MAX_STEPS" \
    max_paths="$MAX_PATHS" \
    mine_max_neighbors="$MINE_MAX_NEIGHBORS" \
    preprocess_workers="$PREPROCESS_WORKERS" \
    train_path_policy="$TRAIN_PATH_POLICY" \
    train_shortest_k="$TRAIN_SHORTEST_K"

echo "[done] cwq -> webqsp preprocess complete"
