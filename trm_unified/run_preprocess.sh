#!/bin/bash
set -euo pipefail
cd /data2/workspace/heewon/논문작업

DATASET=${DATASET:-webqsp}
if [ "$DATASET" = "cwq" ]; then
  python -m trm_unified.pipeline preprocess \
    --dataset cwq \
    --train_in data/CWQ/train_split.jsonl \
    --dev_in data/CWQ/dev_split.jsonl \
    --entities_txt data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt \
    --out_dir trm_unified/processed/cwq
else
  python -m trm_unified.pipeline preprocess \
    --dataset webqsp \
    --train_in data/webqsp/train.json \
    --dev_in data/webqsp/dev.json \
    --entities_txt data/webqsp/entities.txt \
    --out_dir trm_unified/processed/webqsp \
    --max_steps 4 --max_paths 4 --mine_max_neighbors 128
fi
