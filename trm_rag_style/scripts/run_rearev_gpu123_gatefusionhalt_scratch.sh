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
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

EPOCHS="${EPOCHS:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
MASTER_PORT="${MASTER_PORT:-29652}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DATASET}-rearev-C-gpu123-gatefusionhalt-${RUN_TAG}}"
CKPT_DIR="${CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_rearev_C_gpu123_gatefusionhalt_${RUN_TAG}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export WANDB_ANONYMOUS="${WANDB_ANONYMOUS:-allow}"

"$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage train \
  --override \
    emb_tag="$EMB_TAG" \
    emb_dir="$EMB_DIR" \
    ckpt_dir="$CKPT_DIR" \
    epochs="$EPOCHS" \
    batch_size="$BATCH_SIZE" \
    lr="$LR" \
    eval_every_epochs=1 \
    eval_start_epoch=1 \
    eval_limit=-1 \
    wandb_mode="$WANDB_MODE" \
    wandb_project="$WANDB_PROJECT" \
    wandb_run_name="$WANDB_RUN_NAME" \
    subgraph_reader_enabled=true \
    subgraph_loss_mode=rearev_kl \
    subgraph_add_reverse_edges=true \
    subgraph_split_reverse_relations=true \
    subgraph_direction_embedding_enabled=true \
    subgraph_rearev_latent_reasoning_enabled=true \
    subgraph_rearev_latent_residual_alpha=0.25 \
    subgraph_rearev_global_gate_enabled=true \
    subgraph_rearev_logit_global_fusion_enabled=true \
    subgraph_rearev_dynamic_halting_enabled=true \
    subgraph_rearev_dynamic_halting_threshold=0.90 \
    subgraph_rearev_dynamic_halting_min_steps=1
