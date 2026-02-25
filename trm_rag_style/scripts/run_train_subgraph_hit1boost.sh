#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export DDP_TIMEOUT_MINUTES=${DDP_TIMEOUT_MINUTES:-180}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

DATASET=${DATASET:-cwq}
MODEL_IMPL=${MODEL_IMPL:-trm_hier6}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}
EMB_TAG=${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}
EMB_DIR=${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}
CKPT_DIR=${CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_subgraph_hit1boost}
CKPT=${CKPT:-}

EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-2.5e-5}
HIDDEN_SIZE=${HIDDEN_SIZE:-768}
EVAL_LIMIT=${EVAL_LIMIT:--1}
EVAL_EVERY_EPOCHS=${EVAL_EVERY_EPOCHS:-1}
EVAL_START_EPOCH=${EVAL_START_EPOCH:-1}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_PROJECT=${WANDB_PROJECT:-graph-traverse}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}

SUBGRAPH_HOPS=${SUBGRAPH_HOPS:-3}
SUBGRAPH_MAX_NODES=${SUBGRAPH_MAX_NODES:-2048}
SUBGRAPH_MAX_EDGES=${SUBGRAPH_MAX_EDGES:-8192}
SUBGRAPH_RECURSION_STEPS=${SUBGRAPH_RECURSION_STEPS:-12}
SUBGRAPH_DROPOUT=${SUBGRAPH_DROPOUT:-0.1}
SUBGRAPH_POS_WEIGHT_MODE=${SUBGRAPH_POS_WEIGHT_MODE:-auto}
SUBGRAPH_POS_WEIGHT_MAX=${SUBGRAPH_POS_WEIGHT_MAX:-64}
SUBGRAPH_RESUME_EPOCH=${SUBGRAPH_RESUME_EPOCH:--1}

SUBGRAPH_SPLIT_REVERSE_RELATIONS=${SUBGRAPH_SPLIT_REVERSE_RELATIONS:-true}
SUBGRAPH_DIRECTION_EMBEDDING_ENABLED=${SUBGRAPH_DIRECTION_EMBEDDING_ENABLED:-true}
SUBGRAPH_RANKING_ENABLED=${SUBGRAPH_RANKING_ENABLED:-true}
SUBGRAPH_RANKING_WEIGHT=${SUBGRAPH_RANKING_WEIGHT:-0.3}
SUBGRAPH_RANKING_MARGIN=${SUBGRAPH_RANKING_MARGIN:-0.2}
SUBGRAPH_HARD_NEGATIVE_TOPK=${SUBGRAPH_HARD_NEGATIVE_TOPK:-32}
SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED=${SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED:-true}
SUBGRAPH_BCE_HARD_NEGATIVE_TOPK=${SUBGRAPH_BCE_HARD_NEGATIVE_TOPK:-128}

TORCHRUN=${TORCHRUN:-torchrun}
MASTER_PORT=${MASTER_PORT:-29608}
NPROC_PER_NODE=${NPROC_PER_NODE:-3}
LOG_CLEAN_CARRIAGE_RETURN=${LOG_CLEAN_CARRIAGE_RETURN:-auto}

if [ ! -f "$EMB_DIR/entity_embeddings.npy" ] || [ ! -f "$EMB_DIR/relation_embeddings.npy" ] || [ ! -f "$EMB_DIR/query_train.npy" ]; then
  echo "[err] required embedding files are missing in $EMB_DIR"
  echo "      expected: entity_embeddings.npy, relation_embeddings.npy, query_train.npy"
  echo "      run embed stage first."
  exit 2
fi

CMD=(
  "$TORCHRUN"
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  -m trm_agent.run
  --dataset "$DATASET"
  --model_impl "$MODEL_IMPL"
  --stage train
)
if [ -n "$CKPT" ]; then
  CMD+=(--ckpt "$CKPT")
fi
CMD+=(
  --override
  emb_tag="$EMB_TAG"
  emb_dir="$EMB_DIR"
  ckpt_dir="$CKPT_DIR"
  epochs="$EPOCHS"
  batch_size="$BATCH_SIZE"
  lr="$LR"
  hidden_size="$HIDDEN_SIZE"
  eval_limit="$EVAL_LIMIT"
  eval_every_epochs="$EVAL_EVERY_EPOCHS"
  eval_start_epoch="$EVAL_START_EPOCH"
  wandb_mode="$WANDB_MODE"
  wandb_project="$WANDB_PROJECT"
  wandb_entity="$WANDB_ENTITY"
  wandb_run_name="$WANDB_RUN_NAME"
  subgraph_reader_enabled=true
  subgraph_hops="$SUBGRAPH_HOPS"
  subgraph_max_nodes="$SUBGRAPH_MAX_NODES"
  subgraph_max_edges="$SUBGRAPH_MAX_EDGES"
  subgraph_recursion_steps="$SUBGRAPH_RECURSION_STEPS"
  subgraph_dropout="$SUBGRAPH_DROPOUT"
  subgraph_pos_weight_mode="$SUBGRAPH_POS_WEIGHT_MODE"
  subgraph_pos_weight_max="$SUBGRAPH_POS_WEIGHT_MAX"
  subgraph_resume_epoch="$SUBGRAPH_RESUME_EPOCH"
  subgraph_split_reverse_relations="$SUBGRAPH_SPLIT_REVERSE_RELATIONS"
  subgraph_direction_embedding_enabled="$SUBGRAPH_DIRECTION_EMBEDDING_ENABLED"
  subgraph_ranking_enabled="$SUBGRAPH_RANKING_ENABLED"
  subgraph_ranking_weight="$SUBGRAPH_RANKING_WEIGHT"
  subgraph_ranking_margin="$SUBGRAPH_RANKING_MARGIN"
  subgraph_hard_negative_topk="$SUBGRAPH_HARD_NEGATIVE_TOPK"
  subgraph_bce_hard_negative_enabled="$SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED"
  subgraph_bce_hard_negative_topk="$SUBGRAPH_BCE_HARD_NEGATIVE_TOPK"
)

echo "[run] ${CMD[*]}"
if [ "$LOG_CLEAN_CARRIAGE_RETURN" = "true" ]; then
  # Force log-friendly mode: tqdm carriage returns -> line breaks.
  "${CMD[@]}" 2>&1 | stdbuf -oL -eL sed -u 's/\r/\n/g'
elif [ "$LOG_CLEAN_CARRIAGE_RETURN" = "false" ]; then
  # Force native terminal tqdm behavior.
  "${CMD[@]}"
else
  # auto: keep native tqdm on interactive TTY, clean carriage returns for redirected logs.
  if [ -t 1 ]; then
    "${CMD[@]}"
  else
    "${CMD[@]}" 2>&1 | stdbuf -oL -eL sed -u 's/\r/\n/g'
  fi
fi
