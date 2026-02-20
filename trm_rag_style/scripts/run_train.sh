#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2}
DATASET=${DATASET:-webqsp}
MODEL_IMPL=${MODEL_IMPL:-trm_hier6}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}
EMB_TAG=${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}
EMB_DIR=${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}
CKPT=${CKPT:-}
CKPT_DIR=${CKPT_DIR:-trm_rag_style/ckpt/${DATASET}_${MODEL_IMPL}}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-4}
MAX_STEPS=${MAX_STEPS:-4}
MAX_NEIGHBORS=${MAX_NEIGHBORS:-256}
PRUNE_KEEP=${PRUNE_KEEP:-64}
PRUNE_RAND=${PRUNE_RAND:-64}
TRAIN_STYLE=${TRAIN_STYLE:-gnn_rag}
# GNN-RAG-like phase1 defaults (relation CE objective + legacy-like eval)
if [[ "$TRAIN_STYLE" == "gnn_rag" || "$TRAIN_STYLE" == "gnn-rag" || "$TRAIN_STYLE" == "paperstyle" ]]; then
  DEF_BEAM=10
  DEF_EVAL_BEAM=10
  DEF_EVAL_PRED_TOPK=10
  DEF_EVAL_USE_HALT=false
  DEF_EVAL_NO_CYCLE=false
  DEF_EVAL_EVERY=1
  DEF_EVAL_START=1
  ENDPOINT_LOSS_MODE=${ENDPOINT_LOSS_MODE:-aux}
  RELATION_AUX_WEIGHT=${RELATION_AUX_WEIGHT:-1.0}
  ENDPOINT_AUX_WEIGHT=${ENDPOINT_AUX_WEIGHT:-0.0}
  METRIC_ALIGN_AUX_WEIGHT=${METRIC_ALIGN_AUX_WEIGHT:-0.0}
  HALT_AUX_WEIGHT=${HALT_AUX_WEIGHT:-0.0}
else
  DEF_BEAM=8
  DEF_EVAL_BEAM=8
  DEF_EVAL_PRED_TOPK=5
  DEF_EVAL_USE_HALT=true
  DEF_EVAL_NO_CYCLE=true
  DEF_EVAL_EVERY=2
  DEF_EVAL_START=2
  ENDPOINT_LOSS_MODE=${ENDPOINT_LOSS_MODE:-metric_align_main}
  RELATION_AUX_WEIGHT=${RELATION_AUX_WEIGHT:-0.2}
  ENDPOINT_AUX_WEIGHT=${ENDPOINT_AUX_WEIGHT:-0.2}
  METRIC_ALIGN_AUX_WEIGHT=${METRIC_ALIGN_AUX_WEIGHT:-0.0}
  HALT_AUX_WEIGHT=${HALT_AUX_WEIGHT:-0.1}
fi
BEAM=${BEAM:-$DEF_BEAM}
START_TOPK=${START_TOPK:-5}
EVAL_LIMIT=${EVAL_LIMIT:-200}
DEBUG_EVAL_N=${DEBUG_EVAL_N:-5}
EVAL_EVERY_EPOCHS=${EVAL_EVERY_EPOCHS:-$DEF_EVAL_EVERY}
EVAL_START_EPOCH=${EVAL_START_EPOCH:-$DEF_EVAL_START}
EVAL_NO_CYCLE=${EVAL_NO_CYCLE:-$DEF_EVAL_NO_CYCLE}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-4}
EVAL_MAX_NEIGHBORS=${EVAL_MAX_NEIGHBORS:-256}
EVAL_PRUNE_KEEP=${EVAL_PRUNE_KEEP:-64}
EVAL_BEAM=${EVAL_BEAM:-$DEF_EVAL_BEAM}
EVAL_START_TOPK=${EVAL_START_TOPK:-5}
EVAL_PRED_TOPK=${EVAL_PRED_TOPK:-$DEF_EVAL_PRED_TOPK}
EVAL_USE_HALT=${EVAL_USE_HALT:-$DEF_EVAL_USE_HALT}
EVAL_MIN_HOPS_BEFORE_STOP=${EVAL_MIN_HOPS_BEFORE_STOP:-2}
PHASE2_START_EPOCH=${PHASE2_START_EPOCH:-0}
PHASE2_ENDPOINT_LOSS_MODE=${PHASE2_ENDPOINT_LOSS_MODE:-}
PHASE2_RELATION_AUX_WEIGHT=${PHASE2_RELATION_AUX_WEIGHT:-}
PHASE2_ENDPOINT_AUX_WEIGHT=${PHASE2_ENDPOINT_AUX_WEIGHT:-}
PHASE2_METRIC_ALIGN_AUX_WEIGHT=${PHASE2_METRIC_ALIGN_AUX_WEIGHT:-}
PHASE2_HALT_AUX_WEIGHT=${PHASE2_HALT_AUX_WEIGHT:-}
PHASE2_AUTO_ENABLED=${PHASE2_AUTO_ENABLED:-false}
PHASE2_AUTO_METRIC=${PHASE2_AUTO_METRIC:-dev_f1}
PHASE2_AUTO_THRESHOLD=${PHASE2_AUTO_THRESHOLD:-}
PHASE2_AUTO_PATIENCE=${PHASE2_AUTO_PATIENCE:-0}
PHASE2_AUTO_MIN_EPOCH=${PHASE2_AUTO_MIN_EPOCH:-1}
PHASE2_AUTO_MIN_DELTA=${PHASE2_AUTO_MIN_DELTA:-0.001}
PHASE2_RL_REWARD_METRIC=${PHASE2_RL_REWARD_METRIC:-f1}
PHASE2_RL_ENTROPY_WEIGHT=${PHASE2_RL_ENTROPY_WEIGHT:-0.0}
PHASE2_RL_SAMPLE_TEMP=${PHASE2_RL_SAMPLE_TEMP:-1.0}
PHASE2_RL_USE_GREEDY_BASELINE=${PHASE2_RL_USE_GREEDY_BASELINE:-true}
PHASE2_RL_NO_CYCLE=${PHASE2_RL_NO_CYCLE:-true}
PHASE2_RL_ADV_CLIP=${PHASE2_RL_ADV_CLIP:-}
TRAIN_ACC_MODE=${TRAIN_ACC_MODE:-endpoint_proxy}
TRAIN_SANITY_EVAL_EVERY_PCT=${TRAIN_SANITY_EVAL_EVERY_PCT:-0}
TRAIN_SANITY_EVAL_LIMIT=${TRAIN_SANITY_EVAL_LIMIT:-5}
TRAIN_SANITY_EVAL_BEAM=${TRAIN_SANITY_EVAL_BEAM:-5}
TRAIN_SANITY_EVAL_START_TOPK=${TRAIN_SANITY_EVAL_START_TOPK:-5}
TRAIN_SANITY_EVAL_PRED_TOPK=${TRAIN_SANITY_EVAL_PRED_TOPK:-1}
TRAIN_SANITY_EVAL_NO_CYCLE=${TRAIN_SANITY_EVAL_NO_CYCLE:-$EVAL_NO_CYCLE}
TRAIN_SANITY_EVAL_USE_HALT=${TRAIN_SANITY_EVAL_USE_HALT:-false}
TRAIN_SANITY_EVAL_MAX_NEIGHBORS=${TRAIN_SANITY_EVAL_MAX_NEIGHBORS:-$EVAL_MAX_NEIGHBORS}
TRAIN_SANITY_EVAL_PRUNE_KEEP=${TRAIN_SANITY_EVAL_PRUNE_KEEP:-$EVAL_PRUNE_KEEP}
ORACLE_DIAG_ENABLED=${ORACLE_DIAG_ENABLED:-true}
ORACLE_DIAG_LIMIT=${ORACLE_DIAG_LIMIT:-500}
ORACLE_DIAG_FAIL_THRESHOLD=${ORACLE_DIAG_FAIL_THRESHOLD:--1}
ORACLE_DIAG_ONLY=${ORACLE_DIAG_ONLY:-false}
WANDB_MODE=${WANDB_MODE:-disabled}
WANDB_PROJECT=${WANDB_PROJECT:-graph-traverse}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
DDP_FIND_UNUSED=${DDP_FIND_UNUSED:-true}
FREEZE_LM_HEAD=${FREEZE_LM_HEAD:-true}
DDP_TIMEOUT_MINUTES=${DDP_TIMEOUT_MINUTES:-180}
export DDP_TIMEOUT_MINUTES

if [ ! -f "$EMB_DIR/entity_embeddings.npy" ] || [ ! -f "$EMB_DIR/relation_embeddings.npy" ] || [ ! -f "$EMB_DIR/query_train.npy" ]; then
  echo "[err] required embedding files are missing in $EMB_DIR"
  echo "      expected: entity_embeddings.npy, relation_embeddings.npy, query_train.npy"
  echo "      run: DATASET=$DATASET EMB_MODEL=$EMB_MODEL bash trm_rag_style/scripts/run_embed.sh"
  exit 2
fi

TORCHRUN=${TORCHRUN:-torchrun}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-3}
$TORCHRUN --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --embedding_model "$EMB_MODEL" \
  --stage train \
  --override \
    emb_tag="$EMB_TAG" \
    emb_dir="$EMB_DIR" \
    ckpt="$CKPT" \
    ckpt_dir="$CKPT_DIR" \
    epochs="$EPOCHS" \
    batch_size="$BATCH_SIZE" \
    lr="$LR" \
    max_steps="$MAX_STEPS" \
    max_neighbors="$MAX_NEIGHBORS" \
    prune_keep="$PRUNE_KEEP" \
    prune_rand="$PRUNE_RAND" \
    beam="$BEAM" \
    start_topk="$START_TOPK" \
    eval_limit="$EVAL_LIMIT" \
    debug_eval_n="$DEBUG_EVAL_N" \
    eval_every_epochs="$EVAL_EVERY_EPOCHS" \
    eval_start_epoch="$EVAL_START_EPOCH" \
    eval_no_cycle="$EVAL_NO_CYCLE" \
    eval_max_steps="$EVAL_MAX_STEPS" \
    eval_max_neighbors="$EVAL_MAX_NEIGHBORS" \
    eval_prune_keep="$EVAL_PRUNE_KEEP" \
    eval_beam="$EVAL_BEAM" \
    eval_start_topk="$EVAL_START_TOPK" \
    eval_pred_topk="$EVAL_PRED_TOPK" \
    eval_use_halt="$EVAL_USE_HALT" \
    eval_min_hops_before_stop="$EVAL_MIN_HOPS_BEFORE_STOP" \
    endpoint_loss_mode="$ENDPOINT_LOSS_MODE" \
    relation_aux_weight="$RELATION_AUX_WEIGHT" \
    endpoint_aux_weight="$ENDPOINT_AUX_WEIGHT" \
    metric_align_aux_weight="$METRIC_ALIGN_AUX_WEIGHT" \
    halt_aux_weight="$HALT_AUX_WEIGHT" \
    phase2_start_epoch="$PHASE2_START_EPOCH" \
    phase2_endpoint_loss_mode="$PHASE2_ENDPOINT_LOSS_MODE" \
    phase2_relation_aux_weight="$PHASE2_RELATION_AUX_WEIGHT" \
    phase2_endpoint_aux_weight="$PHASE2_ENDPOINT_AUX_WEIGHT" \
    phase2_metric_align_aux_weight="$PHASE2_METRIC_ALIGN_AUX_WEIGHT" \
    phase2_halt_aux_weight="$PHASE2_HALT_AUX_WEIGHT" \
    phase2_auto_enabled="$PHASE2_AUTO_ENABLED" \
    phase2_auto_metric="$PHASE2_AUTO_METRIC" \
    phase2_auto_threshold="$PHASE2_AUTO_THRESHOLD" \
    phase2_auto_patience="$PHASE2_AUTO_PATIENCE" \
    phase2_auto_min_epoch="$PHASE2_AUTO_MIN_EPOCH" \
    phase2_auto_min_delta="$PHASE2_AUTO_MIN_DELTA" \
    phase2_rl_reward_metric="$PHASE2_RL_REWARD_METRIC" \
    phase2_rl_entropy_weight="$PHASE2_RL_ENTROPY_WEIGHT" \
    phase2_rl_sample_temp="$PHASE2_RL_SAMPLE_TEMP" \
    phase2_rl_use_greedy_baseline="$PHASE2_RL_USE_GREEDY_BASELINE" \
    phase2_rl_no_cycle="$PHASE2_RL_NO_CYCLE" \
    phase2_rl_adv_clip="$PHASE2_RL_ADV_CLIP" \
    train_acc_mode="$TRAIN_ACC_MODE" \
    train_sanity_eval_every_pct="$TRAIN_SANITY_EVAL_EVERY_PCT" \
    train_sanity_eval_limit="$TRAIN_SANITY_EVAL_LIMIT" \
    train_sanity_eval_beam="$TRAIN_SANITY_EVAL_BEAM" \
    train_sanity_eval_start_topk="$TRAIN_SANITY_EVAL_START_TOPK" \
    train_sanity_eval_pred_topk="$TRAIN_SANITY_EVAL_PRED_TOPK" \
    train_sanity_eval_no_cycle="$TRAIN_SANITY_EVAL_NO_CYCLE" \
    train_sanity_eval_use_halt="$TRAIN_SANITY_EVAL_USE_HALT" \
    train_sanity_eval_max_neighbors="$TRAIN_SANITY_EVAL_MAX_NEIGHBORS" \
    train_sanity_eval_prune_keep="$TRAIN_SANITY_EVAL_PRUNE_KEEP" \
    oracle_diag_enabled="$ORACLE_DIAG_ENABLED" \
    oracle_diag_limit="$ORACLE_DIAG_LIMIT" \
    oracle_diag_fail_threshold="$ORACLE_DIAG_FAIL_THRESHOLD" \
    oracle_diag_only="$ORACLE_DIAG_ONLY" \
    ddp_find_unused="$DDP_FIND_UNUSED" \
    freeze_lm_head="$FREEZE_LM_HEAD" \
    wandb_mode="$WANDB_MODE" \
    wandb_project="$WANDB_PROJECT" \
    wandb_entity="$WANDB_ENTITY" \
    wandb_run_name="$WANDB_RUN_NAME"
