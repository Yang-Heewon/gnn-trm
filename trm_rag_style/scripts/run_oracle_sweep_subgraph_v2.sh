#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-cwq}"
PROCESSED_DIR="${PROCESSED_DIR:-trm_agent/processed/${DATASET}}"
if [ "$DATASET" = "cwq" ]; then
  RELATIONS_TXT_DEFAULT="data/CWQ/relations.txt"
else
  RELATIONS_TXT_DEFAULT="data/webqsp/relations.txt"
fi
RELATIONS_TXT="${RELATIONS_TXT:-$RELATIONS_TXT_DEFAULT}"

SPLITS="${SPLITS:-dev,test}"
HOPS_LIST="${HOPS_LIST:-3,4}"
MAX_NODES_LIST="${MAX_NODES_LIST:-2048,3072}"
MAX_EDGES_LIST="${MAX_EDGES_LIST:-8192,12288}"
ADD_REVERSE_EDGES="${ADD_REVERSE_EDGES:-true}"
SPLIT_REVERSE_RELATIONS="${SPLIT_REVERSE_RELATIONS:-false}"
LIMIT="${LIMIT:--1}"
OUT_CSV="${OUT_CSV:-trm_rag_style/analysis/oracle_sweep_${DATASET}_v2.csv}"

PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=(
  "$PYTHON_BIN" trm_rag_style/scripts/oracle_sweep_subgraph.py
  --dataset "$DATASET"
  --processed-dir "$PROCESSED_DIR"
  --relations-txt "$RELATIONS_TXT"
  --splits "$SPLITS"
  --hops-list "$HOPS_LIST"
  --max-nodes-list "$MAX_NODES_LIST"
  --max-edges-list "$MAX_EDGES_LIST"
  --add-reverse-edges "$ADD_REVERSE_EDGES"
  --split-reverse-relations "$SPLIT_REVERSE_RELATIONS"
  --limit "$LIMIT"
  --out-csv "$OUT_CSV"
)

echo "[run] ${CMD[*]}"
"${CMD[@]}"

