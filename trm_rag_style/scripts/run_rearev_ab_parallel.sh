#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TORCHRUN_BIN="${TORCHRUN_BIN:-/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun}"

DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_TAG="${EMB_TAG:-e5_w4_g4}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

MASTER_PORT_A="${MASTER_PORT_A:-29640}"
MASTER_PORT_B="${MASTER_PORT_B:-29641}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CKPT_DIR_A="${CKPT_DIR_A:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_rearev_A_gpu01_${RUN_TAG}}"
CKPT_DIR_B="${CKPT_DIR_B:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_rearev_B_gpu23_${RUN_TAG}}"
WANDB_RUN_NAME_A="${WANDB_RUN_NAME_A:-${DATASET}-rearev-A-gpu01-${RUN_TAG}}"
WANDB_RUN_NAME_B="${WANDB_RUN_NAME_B:-${DATASET}-rearev-B-gpu23-${RUN_TAG}}"

LOG_DIR="${LOG_DIR:-logs/rearev_ab_${RUN_TAG}}"
mkdir -p "$LOG_DIR"

COMMON_OVERRIDE=(
  subgraph_reader_enabled=true
  emb_tag="$EMB_TAG"
  emb_dir="$EMB_DIR"
  epochs="$EPOCHS"
  batch_size="$BATCH_SIZE"
  lr="$LR"
  eval_every_epochs=1
  eval_start_epoch=1
  eval_limit=-1
  wandb_mode="$WANDB_MODE"
  wandb_project="$WANDB_PROJECT"
  wandb_entity="$WANDB_ENTITY"
  subgraph_loss_mode=rearev_kl
  subgraph_add_reverse_edges=true
  subgraph_split_reverse_relations=true
  subgraph_direction_embedding_enabled=true
)

echo "[A] GPUs=0,1 latent=off ckpt_dir=$CKPT_DIR_A run=$WANDB_RUN_NAME_A"
echo "[B] GPUs=2,3 latent=on  ckpt_dir=$CKPT_DIR_B run=$WANDB_RUN_NAME_B"
echo "[log] $LOG_DIR"

nohup env CUDA_VISIBLE_DEVICES=0,1 "$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT_A" \
  -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage train \
  --override \
    ckpt_dir="$CKPT_DIR_A" \
    wandb_run_name="$WANDB_RUN_NAME_A" \
    subgraph_rearev_latent_reasoning_enabled=false \
    "${COMMON_OVERRIDE[@]}" \
  >"$LOG_DIR/A_gpu01.log" 2>&1 &
PID_A=$!

nohup env CUDA_VISIBLE_DEVICES=2,3 "$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT_B" \
  -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage train \
  --override \
    ckpt_dir="$CKPT_DIR_B" \
    wandb_run_name="$WANDB_RUN_NAME_B" \
    subgraph_rearev_latent_reasoning_enabled=true \
    subgraph_rearev_latent_residual_alpha=0.25 \
    "${COMMON_OVERRIDE[@]}" \
  >"$LOG_DIR/B_gpu23.log" 2>&1 &
PID_B=$!

echo "$PID_A" >"$LOG_DIR/A.pid"
echo "$PID_B" >"$LOG_DIR/B.pid"

echo "[started] A pid=$PID_A log=$LOG_DIR/A_gpu01.log"
echo "[started] B pid=$PID_B log=$LOG_DIR/B_gpu23.log"
echo "[tail] tail -f $LOG_DIR/A_gpu01.log $LOG_DIR/B_gpu23.log"
