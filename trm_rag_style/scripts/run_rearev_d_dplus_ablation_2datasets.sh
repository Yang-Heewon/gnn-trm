#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# 2-dataset ablation launcher (8 settings per dataset/variant)
# - datasets: cwq,webqsp
# - recursion sweep: 0,3,6,9   (num_ins fixed)
# - num_ins sweep:   1,2,3,4   (recursion fixed)
# => 8 settings per dataset/variant
# - variants: d(rearev_bfs), dplus(rearev_dplus)
# => 32 runs total with 2 datasets; each run is two-phase (phase1+phase2) => 64 training stages.
#
# Usage:
#   cd <repo-root>
#   bash trm_rag_style/scripts/run_rearev_d_dplus_ablation_2datasets.sh
#
# Optional overrides:
#   DATASETS="cwq,webqsp"
#   RECURSION_LIST="0,3,6,9"
#   NUM_INS_LIST="1,2,3,4"
#   RUN_REC_SWEEP=true
#   RUN_INS_SWEEP=true
#   VARIANTS="d,dplus"
#   FIXED_RECURSION_FOR_NUM_INS=4
#   FIXED_NUM_INS_FOR_RECURSION=3
#   PHASE1_EPOCHS=25
#   PHASE2_EPOCHS=25
#   PHASE2_SUBGRAPH_EARLY_STOP_ENABLED=true
#   PHASE2_SUBGRAPH_EARLY_STOP_METRIC=dev_hit1
#   PHASE2_SUBGRAPH_EARLY_STOP_PATIENCE=4
#   PHASE2_SUBGRAPH_EARLY_STOP_MIN_DELTA=1e-3
#   PHASE2_SUBGRAPH_EARLY_STOP_MIN_EPOCHS=8
#   BATCH_SIZE=2
#   PHASE2_BATCH_SIZE=2
#   SUBGRAPH_GRAD_ACCUM_STEPS=4
#   PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS=4
#   BATCH1_RECURSIONS=3,6,9
#   WANDB_MODE=online
#   WANDB_PROJECT=graph-traverse
#   WANDB_NAME_PREFIX=ablation
#   WANDB_GROUP_PREFIX=ablation
#   RUN_PREFIX=abl
#   BASE_MASTER_PORT=30100
#   CONTINUE_ON_ERROR=true
#   PRIMARY_GPUS=0,1,2,3
#   PRIMARY_NPROC=4
#   FALLBACK_GPUS=0,1
#   FALLBACK_NPROC=2
#   PHASE2_GPUS=0,1,2,3
#   PHASE2_NPROC_PER_NODE=4
#   DDP_FIND_UNUSED_DEFAULT=false
#   DDP_FIND_UNUSED_FOR_REC0=true
#   OOM_SAFE_PROFILE=true
#   REC_HIGH_MEM_THRESHOLD=6
#   REC_VHIGH_MEM_THRESHOLD=9
#   PHASE1_MAX_NODES_BASE=4096
#   PHASE1_MAX_EDGES_BASE=16384
#   PHASE1_MAX_NODES_HIGH=2048
#   PHASE1_MAX_EDGES_HIGH=8192
#   PHASE1_MAX_NODES_VHIGH=1536
#   PHASE1_MAX_EDGES_VHIGH=6144
#   PHASE2_MAX_NODES_BASE=2048
#   PHASE2_MAX_EDGES_BASE=8192
#   PHASE2_MAX_NODES_HIGH=1536
#   PHASE2_MAX_EDGES_HIGH=6144
#   PHASE2_MAX_NODES_VHIGH=1024
#   PHASE2_MAX_EDGES_VHIGH=4096
#   ONLY_REMAINING=true

DATASETS="${DATASETS:-cwq,webqsp}"
RECURSION_LIST="${RECURSION_LIST:-0,3,6,9}"
NUM_INS_LIST="${NUM_INS_LIST:-1,2,3,4}"
RUN_REC_SWEEP="${RUN_REC_SWEEP:-true}"
RUN_INS_SWEEP="${RUN_INS_SWEEP:-true}"
VARIANTS="${VARIANTS:-d,dplus}"
FIXED_RECURSION_FOR_NUM_INS="${FIXED_RECURSION_FOR_NUM_INS:-4}"
FIXED_NUM_INS_FOR_RECURSION="${FIXED_NUM_INS_FOR_RECURSION:-3}"

PHASE1_EPOCHS="${PHASE1_EPOCHS:-25}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-25}"
PHASE2_SUBGRAPH_EARLY_STOP_ENABLED="${PHASE2_SUBGRAPH_EARLY_STOP_ENABLED:-true}"
PHASE2_SUBGRAPH_EARLY_STOP_METRIC="${PHASE2_SUBGRAPH_EARLY_STOP_METRIC:-dev_hit1}"
PHASE2_SUBGRAPH_EARLY_STOP_PATIENCE="${PHASE2_SUBGRAPH_EARLY_STOP_PATIENCE:-4}"
PHASE2_SUBGRAPH_EARLY_STOP_MIN_DELTA="${PHASE2_SUBGRAPH_EARLY_STOP_MIN_DELTA:-1e-3}"
PHASE2_SUBGRAPH_EARLY_STOP_MIN_EPOCHS="${PHASE2_SUBGRAPH_EARLY_STOP_MIN_EPOCHS:-8}"

BATCH_SIZE="${BATCH_SIZE:-2}"
PHASE2_BATCH_SIZE="${PHASE2_BATCH_SIZE:-2}"
SUBGRAPH_GRAD_ACCUM_STEPS="${SUBGRAPH_GRAD_ACCUM_STEPS:-4}"
PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS="${PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS:-4}"
BATCH1_RECURSIONS="${BATCH1_RECURSIONS:-3,6,9}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-ablation}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-ablation}"
RUN_PREFIX="${RUN_PREFIX:-abl}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-30100}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
DDP_FIND_UNUSED_DEFAULT="${DDP_FIND_UNUSED_DEFAULT:-false}"
DDP_FIND_UNUSED_FOR_REC0="${DDP_FIND_UNUSED_FOR_REC0:-true}"
ONLY_REMAINING="${ONLY_REMAINING:-false}"

PRIMARY_GPUS="${PRIMARY_GPUS:-0,1,2,3}"
PRIMARY_NPROC="${PRIMARY_NPROC:-4}"
FALLBACK_GPUS="${FALLBACK_GPUS:-0,1}"
FALLBACK_NPROC="${FALLBACK_NPROC:-2}"
PHASE2_GPUS="${PHASE2_GPUS:-0,1,2,3}"
PHASE2_NPROC_PER_NODE="${PHASE2_NPROC_PER_NODE:-4}"

OOM_SAFE_PROFILE="${OOM_SAFE_PROFILE:-true}"
REC_HIGH_MEM_THRESHOLD="${REC_HIGH_MEM_THRESHOLD:-6}"
REC_VHIGH_MEM_THRESHOLD="${REC_VHIGH_MEM_THRESHOLD:-9}"
PHASE1_MAX_NODES_BASE="${PHASE1_MAX_NODES_BASE:-4096}"
PHASE1_MAX_EDGES_BASE="${PHASE1_MAX_EDGES_BASE:-16384}"
PHASE1_MAX_NODES_HIGH="${PHASE1_MAX_NODES_HIGH:-2048}"
PHASE1_MAX_EDGES_HIGH="${PHASE1_MAX_EDGES_HIGH:-8192}"
PHASE1_MAX_NODES_VHIGH="${PHASE1_MAX_NODES_VHIGH:-1536}"
PHASE1_MAX_EDGES_VHIGH="${PHASE1_MAX_EDGES_VHIGH:-6144}"
PHASE2_MAX_NODES_BASE="${PHASE2_MAX_NODES_BASE:-2048}"
PHASE2_MAX_EDGES_BASE="${PHASE2_MAX_EDGES_BASE:-8192}"
PHASE2_MAX_NODES_HIGH="${PHASE2_MAX_NODES_HIGH:-1536}"
PHASE2_MAX_EDGES_HIGH="${PHASE2_MAX_EDGES_HIGH:-6144}"
PHASE2_MAX_NODES_VHIGH="${PHASE2_MAX_NODES_VHIGH:-1024}"
PHASE2_MAX_EDGES_VHIGH="${PHASE2_MAX_EDGES_VHIGH:-4096}"

PHASE1_SUBGRAPH_LR_SCHEDULER="${PHASE1_SUBGRAPH_LR_SCHEDULER:-cosine}"
PHASE2_SUBGRAPH_LR_SCHEDULER="${PHASE2_SUBGRAPH_LR_SCHEDULER:-plateau}"
PHASE1_SUBGRAPH_LR_MIN="${PHASE1_SUBGRAPH_LR_MIN:-1e-6}"
PHASE2_SUBGRAPH_LR_MIN="${PHASE2_SUBGRAPH_LR_MIN:-1e-6}"

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
IFS=',' read -r -a REC_ARR <<< "$RECURSION_LIST"
IFS=',' read -r -a INS_ARR <<< "$NUM_INS_LIST"
IFS=',' read -r -a VARIANT_ARR <<< "$VARIANTS"

num_settings=0
if [[ "$RUN_REC_SWEEP" == "true" ]]; then
  num_settings=$(( num_settings + ${#REC_ARR[@]} ))
fi
if [[ "$RUN_INS_SWEEP" == "true" ]]; then
  num_settings=$(( num_settings + ${#INS_ARR[@]} ))
fi
total_runs=$(( ${#DATASET_ARR[@]} * num_settings * ${#VARIANT_ARR[@]} ))
total_train_stages=$(( total_runs * 2 ))

echo "[plan] datasets=${DATASETS}"
echo "[plan] recursion settings=${#REC_ARR[@]} num_ins settings=${#INS_ARR[@]} variants=${VARIANTS}"
echo "[plan] total runs=${total_runs}, total train stages(phase1+phase2)=${total_train_stages}"
echo "[plan] phase1_epochs=${PHASE1_EPOCHS}, phase2_epochs=${PHASE2_EPOCHS}"
echo "[plan] using 4 GPUs for phase1/fallback/phase2"
echo "[plan] only_remaining=${ONLY_REMAINING}"
echo "[plan] run_rec_sweep=${RUN_REC_SWEEP} run_ins_sweep=${RUN_INS_SWEEP}"

run_idx=0
ok_runs=0
fail_runs=0
skip_runs=0

compute_mem_profile() {
  local rec="$1"
  local ins="$2"
  local p1n="$PHASE1_MAX_NODES_BASE"
  local p1e="$PHASE1_MAX_EDGES_BASE"
  local p2n="$PHASE2_MAX_NODES_BASE"
  local p2e="$PHASE2_MAX_EDGES_BASE"
  if [[ "$OOM_SAFE_PROFILE" == "true" ]]; then
    if (( rec >= REC_VHIGH_MEM_THRESHOLD )) || (( ins >= 4 )); then
      p1n="$PHASE1_MAX_NODES_VHIGH"; p1e="$PHASE1_MAX_EDGES_VHIGH"
      p2n="$PHASE2_MAX_NODES_VHIGH"; p2e="$PHASE2_MAX_EDGES_VHIGH"
    elif (( rec >= REC_HIGH_MEM_THRESHOLD )) || (( ins >= 3 )); then
      p1n="$PHASE1_MAX_NODES_HIGH"; p1e="$PHASE1_MAX_EDGES_HIGH"
      p2n="$PHASE2_MAX_NODES_HIGH"; p2e="$PHASE2_MAX_EDGES_HIGH"
    fi
  fi
  echo "$p1n $p1e $p2n $p2e"
}

use_batch1_for_rec() {
  local rec="$1"
  [[ ",${BATCH1_RECURSIONS}," == *",${rec},"* ]]
}

is_cycle_completed() {
  local dataset="$1"
  local cycle_id="$2"
  local pat="trm_agent/ckpt/${dataset}_${MODEL_IMPL}_rearev_D_phase2_${RUN_PREFIX}_${cycle_id}_*"
  local d
  shopt -s nullglob
  local dirs=( $pat )
  shopt -u nullglob
  for d in "${dirs[@]}"; do
    if compgen -G "${d}/best_*.txt" > /dev/null; then
      return 0
    fi
  done
  return 1
}

for dataset in "${DATASET_ARR[@]}"; do
  for variant in "${VARIANT_ARR[@]}"; do
    # 1) recursion sweep (num_ins fixed)
    if [[ "$RUN_REC_SWEEP" == "true" ]]; then
    for rec in "${REC_ARR[@]}"; do
      ins="${FIXED_NUM_INS_FOR_RECURSION}"
      if [[ "$ins" -lt 1 ]]; then
        echo "[warn] FIXED_NUM_INS_FOR_RECURSION=$ins is invalid. use 1."
        ins=1
      fi
      ddp_find_unused="$DDP_FIND_UNUSED_DEFAULT"
      if [[ "$rec" == "0" && "$DDP_FIND_UNUSED_FOR_REC0" == "true" ]]; then
        ddp_find_unused="true"
      fi
      read -r phase1_max_nodes phase1_max_edges phase2_max_nodes phase2_max_edges < <(compute_mem_profile "$rec" "$ins")
      run_batch_size="$BATCH_SIZE"
      run_phase2_batch_size="$PHASE2_BATCH_SIZE"
      if use_batch1_for_rec "$rec"; then
        run_batch_size="1"
        run_phase2_batch_size="1"
      fi

      if [[ "$variant" == "dplus" ]]; then
        gnn_variant="rearev_dplus"
        latent_update="attn"
      elif [[ "$variant" == "d" ]]; then
        gnn_variant="rearev_bfs"
        latent_update="gru"
      else
        echo "[warn] unknown variant=$variant; skip"
        continue
      fi

      cycle_id="${dataset}_${variant}_recSweep_r${rec}_i${ins}"
      if [[ "$ONLY_REMAINING" == "true" ]] && is_cycle_completed "$dataset" "$cycle_id"; then
        skip_runs=$((skip_runs + 1))
        echo "[skip] completed cycle: ${cycle_id}"
        continue
      fi
      run_idx=$((run_idx + 1))
      master_port=$((BASE_MASTER_PORT + run_idx * 10))
      fallback_port=$((master_port + 1))
      phase2_port=$((master_port + 2))
      run_tag="${RUN_PREFIX}_${cycle_id}_$(date +%Y%m%d_%H%M%S)"
      wandb_group="${WANDB_GROUP_PREFIX}_${cycle_id}"
      phase1_name="${WANDB_NAME_PREFIX}_${cycle_id}_phase1"
      phase2_name="${WANDB_NAME_PREFIX}_${cycle_id}_phase2"

      echo ""
      echo "[run ${run_idx}/${total_runs}] dataset=${dataset} variant=${variant} rec=${rec} ins=${ins} (recursion-sweep)"
      echo "  run_tag=${run_tag}"
      echo "  wandb_group=${wandb_group}"
      echo "  subgraph_ddp_find_unused_parameters=${ddp_find_unused}"
      echo "  batch_size(phase1/phase2)=${run_batch_size}/${run_phase2_batch_size}"
      echo "  phase1_max_nodes/edges=${phase1_max_nodes}/${phase1_max_edges} phase2_max_nodes/edges=${phase2_max_nodes}/${phase2_max_edges}"

      set +e
      RUN_TAG="$run_tag" \
      DATASET="$dataset" \
      PRIMARY_GPUS="$PRIMARY_GPUS" PRIMARY_NPROC="$PRIMARY_NPROC" \
      FALLBACK_GPUS="$FALLBACK_GPUS" FALLBACK_NPROC="$FALLBACK_NPROC" \
      PHASE2_GPUS="$PHASE2_GPUS" PHASE2_NPROC_PER_NODE="$PHASE2_NPROC_PER_NODE" \
      MASTER_PORT="$master_port" \
      FALLBACK_MASTER_PORT="$fallback_port" \
      PHASE2_MASTER_PORT="$phase2_port" \
      PHASE1_EPOCHS="$PHASE1_EPOCHS" \
      PHASE2_EPOCHS="$PHASE2_EPOCHS" \
      BATCH_SIZE="$run_batch_size" \
      PHASE2_BATCH_SIZE="$run_phase2_batch_size" \
      SUBGRAPH_GRAD_ACCUM_STEPS="$SUBGRAPH_GRAD_ACCUM_STEPS" \
      PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS="$PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS" \
      PHASE1_SUBGRAPH_RECURSION_STEPS="$rec" \
      PHASE2_SUBGRAPH_RECURSION_STEPS="$rec" \
      PHASE1_SUBGRAPH_REAREV_NUM_INS="$ins" \
      PHASE2_SUBGRAPH_REAREV_NUM_INS="$ins" \
      PHASE1_SUBGRAPH_MAX_NODES="$phase1_max_nodes" \
      PHASE1_SUBGRAPH_MAX_EDGES="$phase1_max_edges" \
      PHASE2_SUBGRAPH_MAX_NODES="$phase2_max_nodes" \
      PHASE2_SUBGRAPH_MAX_EDGES="$phase2_max_edges" \
      SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS="$ddp_find_unused" \
      PHASE2_SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS="$ddp_find_unused" \
      PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}" \
      SUBGRAPH_GNN_VARIANT="$gnn_variant" \
      PHASE1_SUBGRAPH_REAREV_LATENT_UPDATE_MODE="$latent_update" \
      PHASE2_SUBGRAPH_REAREV_LATENT_UPDATE_MODE="$latent_update" \
      PHASE1_SUBGRAPH_LR_SCHEDULER="$PHASE1_SUBGRAPH_LR_SCHEDULER" \
      PHASE2_SUBGRAPH_LR_SCHEDULER="$PHASE2_SUBGRAPH_LR_SCHEDULER" \
      PHASE1_SUBGRAPH_LR_MIN="$PHASE1_SUBGRAPH_LR_MIN" \
      PHASE2_SUBGRAPH_LR_MIN="$PHASE2_SUBGRAPH_LR_MIN" \
      PHASE2_SUBGRAPH_EARLY_STOP_ENABLED="$PHASE2_SUBGRAPH_EARLY_STOP_ENABLED" \
      PHASE2_SUBGRAPH_EARLY_STOP_METRIC="$PHASE2_SUBGRAPH_EARLY_STOP_METRIC" \
      PHASE2_SUBGRAPH_EARLY_STOP_PATIENCE="$PHASE2_SUBGRAPH_EARLY_STOP_PATIENCE" \
      PHASE2_SUBGRAPH_EARLY_STOP_MIN_DELTA="$PHASE2_SUBGRAPH_EARLY_STOP_MIN_DELTA" \
      PHASE2_SUBGRAPH_EARLY_STOP_MIN_EPOCHS="$PHASE2_SUBGRAPH_EARLY_STOP_MIN_EPOCHS" \
      WANDB_MODE="$WANDB_MODE" \
      WANDB_PROJECT="$WANDB_PROJECT" \
      WANDB_GROUP="$wandb_group" \
      PHASE1_WANDB_RUN_NAME="$phase1_name" \
      PHASE2_WANDB_RUN_NAME="$phase2_name" \
      PHASE1_WANDB_GROUP="$wandb_group" \
      PHASE2_WANDB_GROUP="$wandb_group" \
      bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
      rc=$?
      set -e

      if [[ $rc -eq 0 ]]; then
        ok_runs=$((ok_runs + 1))
      else
        fail_runs=$((fail_runs + 1))
        echo "[fail] run failed: ${cycle_id} (rc=$rc)"
        if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
          exit "$rc"
        fi
      fi
    done
    fi

    # 2) num_ins sweep (recursion fixed)
    if [[ "$RUN_INS_SWEEP" == "true" ]]; then
    for ins_raw in "${INS_ARR[@]}"; do
      rec="${FIXED_RECURSION_FOR_NUM_INS}"
      ins="$ins_raw"
      if [[ "$ins" -lt 1 ]]; then
        echo "[warn] num_ins=$ins is invalid. use 1."
        ins=1
      fi
      ddp_find_unused="$DDP_FIND_UNUSED_DEFAULT"
      if [[ "$rec" == "0" && "$DDP_FIND_UNUSED_FOR_REC0" == "true" ]]; then
        ddp_find_unused="true"
      fi
      read -r phase1_max_nodes phase1_max_edges phase2_max_nodes phase2_max_edges < <(compute_mem_profile "$rec" "$ins")
      run_batch_size="$BATCH_SIZE"
      run_phase2_batch_size="$PHASE2_BATCH_SIZE"
      if use_batch1_for_rec "$rec"; then
        run_batch_size="1"
        run_phase2_batch_size="1"
      fi

      if [[ "$variant" == "dplus" ]]; then
        gnn_variant="rearev_dplus"
        latent_update="attn"
      elif [[ "$variant" == "d" ]]; then
        gnn_variant="rearev_bfs"
        latent_update="gru"
      else
        echo "[warn] unknown variant=$variant; skip"
        continue
      fi

      cycle_id="${dataset}_${variant}_insSweep_r${rec}_i${ins}"
      if [[ "$ONLY_REMAINING" == "true" ]] && is_cycle_completed "$dataset" "$cycle_id"; then
        skip_runs=$((skip_runs + 1))
        echo "[skip] completed cycle: ${cycle_id}"
        continue
      fi
      run_idx=$((run_idx + 1))
      master_port=$((BASE_MASTER_PORT + run_idx * 10))
      fallback_port=$((master_port + 1))
      phase2_port=$((master_port + 2))
      run_tag="${RUN_PREFIX}_${cycle_id}_$(date +%Y%m%d_%H%M%S)"
      wandb_group="${WANDB_GROUP_PREFIX}_${cycle_id}"
      phase1_name="${WANDB_NAME_PREFIX}_${cycle_id}_phase1"
      phase2_name="${WANDB_NAME_PREFIX}_${cycle_id}_phase2"

      echo ""
      echo "[run ${run_idx}/${total_runs}] dataset=${dataset} variant=${variant} rec=${rec} ins=${ins} (num_ins-sweep)"
      echo "  run_tag=${run_tag}"
      echo "  wandb_group=${wandb_group}"
      echo "  subgraph_ddp_find_unused_parameters=${ddp_find_unused}"
      echo "  batch_size(phase1/phase2)=${run_batch_size}/${run_phase2_batch_size}"
      echo "  phase1_max_nodes/edges=${phase1_max_nodes}/${phase1_max_edges} phase2_max_nodes/edges=${phase2_max_nodes}/${phase2_max_edges}"

      set +e
      RUN_TAG="$run_tag" \
      DATASET="$dataset" \
      PRIMARY_GPUS="$PRIMARY_GPUS" PRIMARY_NPROC="$PRIMARY_NPROC" \
      FALLBACK_GPUS="$FALLBACK_GPUS" FALLBACK_NPROC="$FALLBACK_NPROC" \
      PHASE2_GPUS="$PHASE2_GPUS" PHASE2_NPROC_PER_NODE="$PHASE2_NPROC_PER_NODE" \
      MASTER_PORT="$master_port" \
      FALLBACK_MASTER_PORT="$fallback_port" \
      PHASE2_MASTER_PORT="$phase2_port" \
      PHASE1_EPOCHS="$PHASE1_EPOCHS" \
      PHASE2_EPOCHS="$PHASE2_EPOCHS" \
      BATCH_SIZE="$run_batch_size" \
      PHASE2_BATCH_SIZE="$run_phase2_batch_size" \
      SUBGRAPH_GRAD_ACCUM_STEPS="$SUBGRAPH_GRAD_ACCUM_STEPS" \
      PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS="$PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS" \
      PHASE1_SUBGRAPH_RECURSION_STEPS="$rec" \
      PHASE2_SUBGRAPH_RECURSION_STEPS="$rec" \
      PHASE1_SUBGRAPH_REAREV_NUM_INS="$ins" \
      PHASE2_SUBGRAPH_REAREV_NUM_INS="$ins" \
      PHASE1_SUBGRAPH_MAX_NODES="$phase1_max_nodes" \
      PHASE1_SUBGRAPH_MAX_EDGES="$phase1_max_edges" \
      PHASE2_SUBGRAPH_MAX_NODES="$phase2_max_nodes" \
      PHASE2_SUBGRAPH_MAX_EDGES="$phase2_max_edges" \
      SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS="$ddp_find_unused" \
      PHASE2_SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS="$ddp_find_unused" \
      PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}" \
      SUBGRAPH_GNN_VARIANT="$gnn_variant" \
      PHASE1_SUBGRAPH_REAREV_LATENT_UPDATE_MODE="$latent_update" \
      PHASE2_SUBGRAPH_REAREV_LATENT_UPDATE_MODE="$latent_update" \
      PHASE1_SUBGRAPH_LR_SCHEDULER="$PHASE1_SUBGRAPH_LR_SCHEDULER" \
      PHASE2_SUBGRAPH_LR_SCHEDULER="$PHASE2_SUBGRAPH_LR_SCHEDULER" \
      PHASE1_SUBGRAPH_LR_MIN="$PHASE1_SUBGRAPH_LR_MIN" \
      PHASE2_SUBGRAPH_LR_MIN="$PHASE2_SUBGRAPH_LR_MIN" \
      PHASE2_SUBGRAPH_EARLY_STOP_ENABLED="$PHASE2_SUBGRAPH_EARLY_STOP_ENABLED" \
      PHASE2_SUBGRAPH_EARLY_STOP_METRIC="$PHASE2_SUBGRAPH_EARLY_STOP_METRIC" \
      PHASE2_SUBGRAPH_EARLY_STOP_PATIENCE="$PHASE2_SUBGRAPH_EARLY_STOP_PATIENCE" \
      PHASE2_SUBGRAPH_EARLY_STOP_MIN_DELTA="$PHASE2_SUBGRAPH_EARLY_STOP_MIN_DELTA" \
      PHASE2_SUBGRAPH_EARLY_STOP_MIN_EPOCHS="$PHASE2_SUBGRAPH_EARLY_STOP_MIN_EPOCHS" \
      WANDB_MODE="$WANDB_MODE" \
      WANDB_PROJECT="$WANDB_PROJECT" \
      WANDB_GROUP="$wandb_group" \
      PHASE1_WANDB_RUN_NAME="$phase1_name" \
      PHASE2_WANDB_RUN_NAME="$phase2_name" \
      PHASE1_WANDB_GROUP="$wandb_group" \
      PHASE2_WANDB_GROUP="$wandb_group" \
      bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
      rc=$?
      set -e

      if [[ $rc -eq 0 ]]; then
        ok_runs=$((ok_runs + 1))
      else
        fail_runs=$((fail_runs + 1))
        echo "[fail] run failed: ${cycle_id} (rc=$rc)"
        if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
          exit "$rc"
        fi
      fi
    done
    fi
  done
done

echo ""
echo "[done] run summary: ok=${ok_runs} fail=${fail_runs} skip=${skip_runs} total_plan=${total_runs}"
if [[ "$fail_runs" -gt 0 ]]; then
  exit 1
fi
