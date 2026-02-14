#!/bin/bash
set -euo pipefail
cd /data2/workspace/heewon/논문작업

DATASET=${DATASET:-webqsp}
MODEL_IMPL=${MODEL_IMPL:-trm_hier6}
CKPT=${CKPT:-}

if [ -z "$CKPT" ]; then
  echo "Set CKPT=/path/to/model_epX.pt"
  exit 1
fi

python -m trm_gnnrag_style.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage test \
  --ckpt "$CKPT"
