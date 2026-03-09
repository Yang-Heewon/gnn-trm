#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# D+: same two-phase pipeline as D, but latent update uses attention-memory.
export SUBGRAPH_GNN_VARIANT="${SUBGRAPH_GNN_VARIANT:-rearev_dplus}"
export SUBGRAPH_REAREV_LATENT_REASONING_ENABLED="${SUBGRAPH_REAREV_LATENT_REASONING_ENABLED:-true}"
export SUBGRAPH_REAREV_LATENT_UPDATE_MODE="${SUBGRAPH_REAREV_LATENT_UPDATE_MODE:-attn}"
export RUN_TAG="${RUN_TAG:-dplus_$(date +%Y%m%d_%H%M%S)}"

echo "[run] D+ two-phase auto (latent_update=attn, gnn_variant=${SUBGRAPH_GNN_VARIANT})"
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh

