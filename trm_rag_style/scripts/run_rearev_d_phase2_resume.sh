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

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
if ! command -v "$TORCHRUN_BIN" >/dev/null 2>&1; then
  echo "[err] torchrun not found: $TORCHRUN_BIN"
  echo "      install torch distributed launcher or set TORCHRUN_BIN explicitly."
  exit 1
fi
DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_TAG="${EMB_TAG:-e5_w4_g4}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"

# Phase1 best checkpoint (required)
CKPT="${CKPT:-}"
if [[ -z "$CKPT" ]]; then
  echo "[err] set CKPT to phase1 best checkpoint (e.g., .../model_ep15.pt)"
  exit 2
fi
if [[ ! -f "$CKPT" ]]; then
  echo "[err] checkpoint not found: $CKPT"
  exit 2
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_rearev_D_phase2_${RUN_TAG}}"

# Fine-tuning defaults
EPOCHS="${EPOCHS:-16}"                 # additional epochs after resume point
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-5e-5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29678}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DATASET}-rearev-D-phase2-${RUN_TAG}}"
WANDB_GROUP="${WANDB_GROUP:-}"
SEED="${SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-false}"

SUBGRAPH_RESUME_EPOCH="${SUBGRAPH_RESUME_EPOCH:-$(infer_resume_epoch "$CKPT")}"
SUBGRAPH_MAX_NODES="${SUBGRAPH_MAX_NODES:-2048}"
SUBGRAPH_MAX_EDGES="${SUBGRAPH_MAX_EDGES:-8192}"
SUBGRAPH_RECURSION_STEPS="${SUBGRAPH_RECURSION_STEPS:-4}"
SUBGRAPH_REAREV_NUM_INS="${SUBGRAPH_REAREV_NUM_INS:-3}"
SUBGRAPH_REAREV_ADAPT_STAGES="${SUBGRAPH_REAREV_ADAPT_STAGES:-2}"
SUBGRAPH_GRAD_ACCUM_STEPS="${SUBGRAPH_GRAD_ACCUM_STEPS:-4}"

# Phase2 objective: BCE + ranking + hard negatives
SUBGRAPH_POS_WEIGHT_MODE="${SUBGRAPH_POS_WEIGHT_MODE:-auto}"
SUBGRAPH_POS_WEIGHT_MAX="${SUBGRAPH_POS_WEIGHT_MAX:-256}"
SUBGRAPH_RANKING_ENABLED="${SUBGRAPH_RANKING_ENABLED:-true}"
SUBGRAPH_RANKING_WEIGHT="${SUBGRAPH_RANKING_WEIGHT:-0.15}"
SUBGRAPH_RANKING_MARGIN="${SUBGRAPH_RANKING_MARGIN:-0.2}"
SUBGRAPH_HARD_NEGATIVE_TOPK="${SUBGRAPH_HARD_NEGATIVE_TOPK:-32}"
SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED="${SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED:-true}"
SUBGRAPH_BCE_HARD_NEGATIVE_TOPK="${SUBGRAPH_BCE_HARD_NEGATIVE_TOPK:-64}"
SUBGRAPH_REAREV_NORMALIZED_GNN="${SUBGRAPH_REAREV_NORMALIZED_GNN:-false}"

# Keep latent/recursive modules ON
SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD="${SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD:-0.95}"
SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS="${SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS:-3}"
SUBGRAPH_REAREV_LATENT_REASONING_ENABLED="${SUBGRAPH_REAREV_LATENT_REASONING_ENABLED:-true}"
SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA="${SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA:-0.25}"
SUBGRAPH_REAREV_LATENT_UPDATE_MODE="${SUBGRAPH_REAREV_LATENT_UPDATE_MODE:-gru}"
SUBGRAPH_REAREV_GLOBAL_GATE_ENABLED="${SUBGRAPH_REAREV_GLOBAL_GATE_ENABLED:-true}"
SUBGRAPH_REAREV_LOGIT_GLOBAL_FUSION_ENABLED="${SUBGRAPH_REAREV_LOGIT_GLOBAL_FUSION_ENABLED:-true}"
SUBGRAPH_REAREV_DYNAMIC_HALTING_ENABLED="${SUBGRAPH_REAREV_DYNAMIC_HALTING_ENABLED:-true}"
SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED="${SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED:-false}"
SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS="${SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS:-false}"

SUBGRAPH_LR_SCHEDULER="${SUBGRAPH_LR_SCHEDULER:-plateau}"
SUBGRAPH_LR_MIN="${SUBGRAPH_LR_MIN:-1e-6}"
SUBGRAPH_LR_PLATEAU_FACTOR="${SUBGRAPH_LR_PLATEAU_FACTOR:-0.7}"
SUBGRAPH_LR_PLATEAU_PATIENCE="${SUBGRAPH_LR_PLATEAU_PATIENCE:-1}"
SUBGRAPH_LR_PLATEAU_THRESHOLD="${SUBGRAPH_LR_PLATEAU_THRESHOLD:-1e-4}"
SUBGRAPH_LR_PLATEAU_METRIC="${SUBGRAPH_LR_PLATEAU_METRIC:-dev_hit1}"
SUBGRAPH_EARLY_STOP_ENABLED="${SUBGRAPH_EARLY_STOP_ENABLED:-true}"
SUBGRAPH_EARLY_STOP_METRIC="${SUBGRAPH_EARLY_STOP_METRIC:-dev_hit1}"
SUBGRAPH_EARLY_STOP_PATIENCE="${SUBGRAPH_EARLY_STOP_PATIENCE:-4}"
SUBGRAPH_EARLY_STOP_MIN_DELTA="${SUBGRAPH_EARLY_STOP_MIN_DELTA:-1e-3}"
SUBGRAPH_EARLY_STOP_MIN_EPOCHS="${SUBGRAPH_EARLY_STOP_MIN_EPOCHS:-8}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_ANONYMOUS="${WANDB_ANONYMOUS:-allow}"

"$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage train \
  --ckpt "$CKPT" \
  --override \
    emb_tag="$EMB_TAG" \
    emb_dir="$EMB_DIR" \
    ckpt_dir="$CKPT_DIR" \
    seed="$SEED" \
    deterministic="$DETERMINISTIC" \
    epochs="$EPOCHS" \
    batch_size="$BATCH_SIZE" \
    lr="$LR" \
    eval_every_epochs=1 \
    eval_start_epoch=1 \
    eval_limit=-1 \
    auto_test_after_train=false \
    wandb_mode="$WANDB_MODE" \
    wandb_project="$WANDB_PROJECT" \
    wandb_run_name="$WANDB_RUN_NAME" \
    wandb_group="$WANDB_GROUP" \
    subgraph_reader_enabled=true \
    subgraph_loss_mode=bce \
    subgraph_max_nodes="$SUBGRAPH_MAX_NODES" \
    subgraph_max_edges="$SUBGRAPH_MAX_EDGES" \
    subgraph_add_reverse_edges=true \
    subgraph_split_reverse_relations=true \
    subgraph_direction_embedding_enabled=true \
    subgraph_rearev_num_ins="$SUBGRAPH_REAREV_NUM_INS" \
    subgraph_rearev_adapt_stages="$SUBGRAPH_REAREV_ADAPT_STAGES" \
    subgraph_recursion_steps="$SUBGRAPH_RECURSION_STEPS" \
    subgraph_grad_accum_steps="$SUBGRAPH_GRAD_ACCUM_STEPS" \
    subgraph_resume_epoch="$SUBGRAPH_RESUME_EPOCH" \
    subgraph_pos_weight_mode="$SUBGRAPH_POS_WEIGHT_MODE" \
    subgraph_pos_weight_max="$SUBGRAPH_POS_WEIGHT_MAX" \
    subgraph_ranking_enabled="$SUBGRAPH_RANKING_ENABLED" \
    subgraph_ranking_weight="$SUBGRAPH_RANKING_WEIGHT" \
    subgraph_ranking_margin="$SUBGRAPH_RANKING_MARGIN" \
    subgraph_hard_negative_topk="$SUBGRAPH_HARD_NEGATIVE_TOPK" \
    subgraph_bce_hard_negative_enabled="$SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED" \
    subgraph_bce_hard_negative_topk="$SUBGRAPH_BCE_HARD_NEGATIVE_TOPK" \
    subgraph_rearev_normalized_gnn="$SUBGRAPH_REAREV_NORMALIZED_GNN" \
    subgraph_lr_scheduler="$SUBGRAPH_LR_SCHEDULER" \
    subgraph_lr_min="$SUBGRAPH_LR_MIN" \
    subgraph_lr_plateau_factor="$SUBGRAPH_LR_PLATEAU_FACTOR" \
    subgraph_lr_plateau_patience="$SUBGRAPH_LR_PLATEAU_PATIENCE" \
    subgraph_lr_plateau_threshold="$SUBGRAPH_LR_PLATEAU_THRESHOLD" \
    subgraph_lr_plateau_metric="$SUBGRAPH_LR_PLATEAU_METRIC" \
    subgraph_early_stop_enabled="$SUBGRAPH_EARLY_STOP_ENABLED" \
    subgraph_early_stop_metric="$SUBGRAPH_EARLY_STOP_METRIC" \
    subgraph_early_stop_patience="$SUBGRAPH_EARLY_STOP_PATIENCE" \
    subgraph_early_stop_min_delta="$SUBGRAPH_EARLY_STOP_MIN_DELTA" \
    subgraph_early_stop_min_epochs="$SUBGRAPH_EARLY_STOP_MIN_EPOCHS" \
    subgraph_rearev_latent_reasoning_enabled="$SUBGRAPH_REAREV_LATENT_REASONING_ENABLED" \
    subgraph_rearev_latent_residual_alpha="$SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA" \
    subgraph_rearev_latent_update_mode="$SUBGRAPH_REAREV_LATENT_UPDATE_MODE" \
    subgraph_rearev_global_gate_enabled="$SUBGRAPH_REAREV_GLOBAL_GATE_ENABLED" \
    subgraph_rearev_logit_global_fusion_enabled="$SUBGRAPH_REAREV_LOGIT_GLOBAL_FUSION_ENABLED" \
    subgraph_rearev_dynamic_halting_enabled="$SUBGRAPH_REAREV_DYNAMIC_HALTING_ENABLED" \
    subgraph_rearev_dynamic_halting_min_steps="$SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS" \
    subgraph_rearev_dynamic_halting_threshold="$SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD" \
    subgraph_rearev_asymmetric_yz_enabled="$SUBGRAPH_REAREV_ASYMMETRIC_YZ_ENABLED" \
    subgraph_ddp_find_unused_parameters="$SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS"
