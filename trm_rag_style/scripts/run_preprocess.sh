#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET=${DATASET:-webqsp}
TRAIN_PATH_POLICY=${TRAIN_PATH_POLICY:-shortest_only}
TRAIN_SHORTEST_K=${TRAIN_SHORTEST_K:-1}
$PYTHON_BIN -m trm_agent.run \
  --dataset "$DATASET" \
  --stage preprocess \
  --override \
    train_path_policy="$TRAIN_PATH_POLICY" \
    train_shortest_k="$TRAIN_SHORTEST_K"
