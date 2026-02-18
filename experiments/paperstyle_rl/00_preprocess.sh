#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

DATASET="$DATASET" \
MAX_STEPS="$MAX_STEPS" \
MAX_PATHS="$MAX_PATHS" \
MINE_MAX_NEIGHBORS="$MINE_MAX_NEIGHBORS" \
TRAIN_PATH_POLICY="$TRAIN_PATH_POLICY" \
TRAIN_SHORTEST_K="$TRAIN_SHORTEST_K" \
bash scripts/setup_and_preprocess.sh
