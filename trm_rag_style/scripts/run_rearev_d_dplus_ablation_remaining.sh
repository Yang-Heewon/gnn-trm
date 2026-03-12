#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Run only unfinished cycles (detected by phase2 best_*.txt marker).
ONLY_REMAINING=true \
bash trm_rag_style/scripts/run_rearev_d_dplus_ablation_2datasets.sh "$@"
