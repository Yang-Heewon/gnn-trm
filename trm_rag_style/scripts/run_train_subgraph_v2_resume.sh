#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

infer_resume_epoch() {
  local ckpt_path="$1"
  local b
  b="$(basename "$ckpt_path")"
  if [[ "$b" =~ model_ep([0-9]+)\.pt$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "0"
  fi
}

latest_ckpt_in_dir() {
  local d="$1"
  ls "$d"/model_ep*.pt 2>/dev/null | sort -V | tail -n 1 || true
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-180}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"
EMB_TAG="${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"
CKPT_DIR="${CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_subgraph_3gpu_v2}"
CKPT="${CKPT:-}"
AUTO_RESUME="${AUTO_RESUME:-true}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-2.5e-5}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
EVAL_LIMIT="${EVAL_LIMIT:--1}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-1}"
EVAL_START_EPOCH="${EVAL_START_EPOCH:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

SUBGRAPH_HOPS="${SUBGRAPH_HOPS:-3}"
SUBGRAPH_MAX_NODES="${SUBGRAPH_MAX_NODES:-2048}"
SUBGRAPH_MAX_EDGES="${SUBGRAPH_MAX_EDGES:-8192}"
SUBGRAPH_RECURSION_STEPS="${SUBGRAPH_RECURSION_STEPS:-12}"
SUBGRAPH_DROPOUT="${SUBGRAPH_DROPOUT:-0.1}"
SUBGRAPH_POS_WEIGHT_MODE="${SUBGRAPH_POS_WEIGHT_MODE:-auto}"
SUBGRAPH_POS_WEIGHT_MAX="${SUBGRAPH_POS_WEIGHT_MAX:-64}"
SUBGRAPH_RESUME_EPOCH="${SUBGRAPH_RESUME_EPOCH:--1}"

# v2 defaults (no ranking, no split/direction by default).
SUBGRAPH_SPLIT_REVERSE_RELATIONS="${SUBGRAPH_SPLIT_REVERSE_RELATIONS:-false}"
SUBGRAPH_DIRECTION_EMBEDDING_ENABLED="${SUBGRAPH_DIRECTION_EMBEDDING_ENABLED:-false}"
SUBGRAPH_RANKING_ENABLED="${SUBGRAPH_RANKING_ENABLED:-false}"
SUBGRAPH_RANKING_WEIGHT="${SUBGRAPH_RANKING_WEIGHT:-0.0}"
SUBGRAPH_RANKING_MARGIN="${SUBGRAPH_RANKING_MARGIN:-0.2}"
SUBGRAPH_HARD_NEGATIVE_TOPK="${SUBGRAPH_HARD_NEGATIVE_TOPK:-16}"
SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED="${SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED:-false}"
SUBGRAPH_BCE_HARD_NEGATIVE_TOPK="${SUBGRAPH_BCE_HARD_NEGATIVE_TOPK:-64}"
SUBGRAPH_LR_SCHEDULER="${SUBGRAPH_LR_SCHEDULER:-none}"
SUBGRAPH_LR_MIN="${SUBGRAPH_LR_MIN:-0.0}"
SUBGRAPH_LR_STEP_SIZE="${SUBGRAPH_LR_STEP_SIZE:-5}"
SUBGRAPH_LR_GAMMA="${SUBGRAPH_LR_GAMMA:-0.5}"
SUBGRAPH_LR_PLATEAU_FACTOR="${SUBGRAPH_LR_PLATEAU_FACTOR:-0.5}"
SUBGRAPH_LR_PLATEAU_PATIENCE="${SUBGRAPH_LR_PLATEAU_PATIENCE:-2}"
SUBGRAPH_LR_PLATEAU_THRESHOLD="${SUBGRAPH_LR_PLATEAU_THRESHOLD:-0.0001}"
SUBGRAPH_LR_PLATEAU_METRIC="${SUBGRAPH_LR_PLATEAU_METRIC:-train_loss}"

SEED="${SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-false}"
export PYTHONHASHSEED="$SEED"

TORCHRUN="${TORCHRUN:-torchrun}"
MASTER_PORT="${MASTER_PORT:-29606}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
LOG_CLEAN_CARRIAGE_RETURN="${LOG_CLEAN_CARRIAGE_RETURN:-auto}"

SAVE_MANIFEST="${SAVE_MANIFEST:-true}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [ ! -f "$EMB_DIR/entity_embeddings.npy" ] || [ ! -f "$EMB_DIR/relation_embeddings.npy" ] || [ ! -f "$EMB_DIR/query_train.npy" ]; then
  echo "[err] required embedding files are missing in $EMB_DIR"
  echo "      expected: entity_embeddings.npy, relation_embeddings.npy, query_train.npy"
  echo "      run embed stage first."
  exit 2
fi

if [ -z "$CKPT" ] && [ "$AUTO_RESUME" = "true" ]; then
  CKPT="$(latest_ckpt_in_dir "$CKPT_DIR")"
fi

if [ -n "$CKPT" ] && [ ! -f "$CKPT" ]; then
  echo "[err] ckpt not found: $CKPT"
  exit 2
fi

if [ "$SUBGRAPH_RESUME_EPOCH" -lt 0 ] && [ -n "$CKPT" ]; then
  SUBGRAPH_RESUME_EPOCH="$(infer_resume_epoch "$CKPT")"
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
  seed="$SEED"
  deterministic="$DETERMINISTIC"
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
  subgraph_lr_scheduler="$SUBGRAPH_LR_SCHEDULER"
  subgraph_lr_min="$SUBGRAPH_LR_MIN"
  subgraph_lr_step_size="$SUBGRAPH_LR_STEP_SIZE"
  subgraph_lr_gamma="$SUBGRAPH_LR_GAMMA"
  subgraph_lr_plateau_factor="$SUBGRAPH_LR_PLATEAU_FACTOR"
  subgraph_lr_plateau_patience="$SUBGRAPH_LR_PLATEAU_PATIENCE"
  subgraph_lr_plateau_threshold="$SUBGRAPH_LR_PLATEAU_THRESHOLD"
  subgraph_lr_plateau_metric="$SUBGRAPH_LR_PLATEAU_METRIC"
)

echo "[resume] ckpt=${CKPT:-<none>} subgraph_resume_epoch=$SUBGRAPH_RESUME_EPOCH"
echo "[run] ${CMD[*]}"

if [ "$SAVE_MANIFEST" = "true" ]; then
  mkdir -p "$CKPT_DIR/run_manifests"
  MANIFEST_PATH="$CKPT_DIR/run_manifests/run_${RUN_STAMP}.txt"
  {
    echo "timestamp=$RUN_STAMP"
    echo "repo_root=$REPO_ROOT"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "nproc_per_node=$NPROC_PER_NODE"
    echo "master_port=$MASTER_PORT"
    echo "dataset=$DATASET"
    echo "model_impl=$MODEL_IMPL"
    echo "ckpt_dir=$CKPT_DIR"
    echo "ckpt=${CKPT:-}"
    echo "subgraph_resume_epoch=$SUBGRAPH_RESUME_EPOCH"
    echo "seed=$SEED"
    echo "deterministic=$DETERMINISTIC"
    echo "command=${CMD[*]}"
  } >"$MANIFEST_PATH"
  echo "[manifest] $MANIFEST_PATH"
fi

if [ "$LOG_CLEAN_CARRIAGE_RETURN" = "true" ]; then
  "${CMD[@]}" 2>&1 | stdbuf -oL -eL sed -u 's/\r/\n/g'
elif [ "$LOG_CLEAN_CARRIAGE_RETURN" = "false" ]; then
  "${CMD[@]}"
else
  if [ -t 1 ]; then
    "${CMD[@]}"
  else
    "${CMD[@]}" 2>&1 | stdbuf -oL -eL sed -u 's/\r/\n/g'
  fi
fi
