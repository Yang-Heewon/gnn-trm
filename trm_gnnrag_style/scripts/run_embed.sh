#!/bin/bash
set -euo pipefail
cd /data2/workspace/heewon/논문작업

DATASET=${DATASET:-webqsp}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}
python -m trm_gnnrag_style.run --dataset "$DATASET" --stage embed --embedding_model "$EMB_MODEL"
