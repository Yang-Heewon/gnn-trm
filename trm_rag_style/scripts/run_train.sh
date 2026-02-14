#!/bin/bash
set -euo pipefail
cd /data2/workspace/heewon/논문작업

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
DATASET=${DATASET:-webqsp}
MODEL_IMPL=${MODEL_IMPL:-trm_hier6}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}

# DDP train
TORCHRUN=${TORCHRUN:-torchrun}
$TORCHRUN --nproc_per_node=3 --master_port=29500 -m trm_rag_style.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --embedding_model "$EMB_MODEL" \
  --stage train
