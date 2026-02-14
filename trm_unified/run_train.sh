#!/bin/bash
set -euo pipefail
cd /data2/workspace/heewon/논문작업

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
DATASET=${DATASET:-webqsp}
MODEL_IMPL=${MODEL_IMPL:-trm_hier6}
BATCH=${BATCH:-6}
EPOCHS=${EPOCHS:-5}

if [ "$DATASET" = "cwq" ]; then
  TRAIN_JSON=trm_unified/processed/cwq/train.jsonl
  ENT_TXT=data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt
  REL_TXT=data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt
  EMB_DIR=trm_unified/emb/cwq
  OUT_DIR=trm_unified/ckpt/cwq_${MODEL_IMPL}
else
  TRAIN_JSON=trm_unified/processed/webqsp/train.jsonl
  ENT_TXT=data/webqsp/entities.txt
  REL_TXT=data/webqsp/relations.txt
  EMB_DIR=trm_unified/emb/webqsp
  OUT_DIR=trm_unified/ckpt/webqsp_${MODEL_IMPL}
fi

mkdir -p "$OUT_DIR"

torchrun --nproc_per_node=3 --master_port=29500 -m trm_unified.pipeline train \
  --model_impl "$MODEL_IMPL" \
  --train_json "$TRAIN_JSON" \
  --entities_txt "$ENT_TXT" \
  --relations_txt "$REL_TXT" \
  --entity_emb_npy "$EMB_DIR/entity_embeddings.npy" \
  --relation_emb_npy "$EMB_DIR/relation_embeddings.npy" \
  --query_emb_train_npy "$EMB_DIR/query_train.npy" \
  --out_dir "$OUT_DIR" \
  --batch_size "$BATCH" \
  --epochs "$EPOCHS"
