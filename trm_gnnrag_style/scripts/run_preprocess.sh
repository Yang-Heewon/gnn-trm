#!/bin/bash
set -euo pipefail
cd /data2/workspace/heewon/논문작업

DATASET=${DATASET:-webqsp}
python -m trm_gnnrag_style.run --dataset "$DATASET" --stage preprocess
