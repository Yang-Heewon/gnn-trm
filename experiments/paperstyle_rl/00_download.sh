#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

DATASET="$DATASET" \
DATA_SOURCE="$DATA_SOURCE" \
HF_CACHE_DIR="$HF_CACHE_DIR" \
ROG_CWQ_DATASET="$ROG_CWQ_DATASET" \
ROG_WEBQSP_DATASET="$ROG_WEBQSP_DATASET" \
bash trm_rag_style/scripts/run_download.sh
