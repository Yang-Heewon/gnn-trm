#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

latest_ckpt_in_dir() {
  local d="$1"
  ls "$d"/model_ep*.pt 2>/dev/null | sort -V | tail -n 1 || true
}

TORCHRUN="${TORCHRUN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29620}"
MASTER_PORT_LATENT="${MASTER_PORT_LATENT:-29621}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_TAG="${EMB_TAG:-e5_w4_g4}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"

BASELINE_CKPT_DIR="${BASELINE_CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_rearev_baseline}"
LATENT_CKPT_DIR="${LATENT_CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_rearev_latent}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"

EPOCHS_BASELINE="${EPOCHS_BASELINE:-12}"
EPOCHS_LATENT="${EPOCHS_LATENT:-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR_BASELINE="${LR_BASELINE:-1e-4}"
LR_LATENT="${LR_LATENT:-5e-5}"
LATENT_ALPHA="${LATENT_ALPHA:-0.25}"

COMMON_OVERRIDE=(
  subgraph_reader_enabled=true
  emb_tag="$EMB_TAG"
  emb_dir="$EMB_DIR"
  subgraph_loss_mode=rearev_kl
  subgraph_add_reverse_edges=true
  subgraph_split_reverse_relations=true
  subgraph_direction_embedding_enabled=true
)

echo "[Phase-1] ReaRev baseline (latent=off)"
"$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT_BASE" \
  -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage train \
  --override \
    ckpt_dir="$BASELINE_CKPT_DIR" \
    epochs="$EPOCHS_BASELINE" \
    batch_size="$BATCH_SIZE" \
    lr="$LR_BASELINE" \
    wandb_mode="$WANDB_MODE" \
    wandb_project="$WANDB_PROJECT" \
    subgraph_rearev_latent_reasoning_enabled=false \
    "${COMMON_OVERRIDE[@]}"

BASELINE_CKPT="$(latest_ckpt_in_dir "$BASELINE_CKPT_DIR")"
if [[ -z "$BASELINE_CKPT" ]]; then
  echo "[err] no baseline checkpoint found in $BASELINE_CKPT_DIR"
  exit 2
fi
echo "[Phase-1] latest ckpt: $BASELINE_CKPT"

echo "[Phase-2] Latent reasoning fine-tune (latent=on)"
"$TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT_LATENT" \
  -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage train \
  --ckpt "$BASELINE_CKPT" \
  --override \
    ckpt_dir="$LATENT_CKPT_DIR" \
    epochs="$EPOCHS_LATENT" \
    batch_size="$BATCH_SIZE" \
    lr="$LR_LATENT" \
    wandb_mode="$WANDB_MODE" \
    wandb_project="$WANDB_PROJECT" \
    subgraph_rearev_latent_reasoning_enabled=true \
    subgraph_rearev_latent_residual_alpha="$LATENT_ALPHA" \
    "${COMMON_OVERRIDE[@]}"

echo "[done] baseline dir: $BASELINE_CKPT_DIR"
echo "[done] latent dir:   $LATENT_CKPT_DIR"
