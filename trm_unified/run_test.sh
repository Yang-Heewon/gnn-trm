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

if [ "$DATASET" = "cwq" ]; then
  ENT_TXT=data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt
  REL_TXT=data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt
  EMB_DIR=trm_unified/emb/cwq
else
  ENT_TXT=data/webqsp/entities.txt
  REL_TXT=data/webqsp/relations.txt
  EMB_DIR=trm_unified/emb/webqsp
fi

python -m trm_unified.pipeline test \
  --model_impl "$MODEL_IMPL" \
  --ckpt "$CKPT" \
  --entities_txt "$ENT_TXT" \
  --relations_txt "$REL_TXT" \
  --entity_emb_npy "$EMB_DIR/entity_embeddings.npy" \
  --relation_emb_npy "$EMB_DIR/relation_embeddings.npy"
