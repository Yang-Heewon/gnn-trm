#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# 2-dataset paired ablation launcher
# - datasets: cwq,webqsp
# - paired configs: (recursion, num_ins) = (0,1), (3,2), (6,3), (9,4)
# - variants: d(rearev_bfs), dplus(rearev_dplus)
# => 16 runs total; each run is two-phase (phase1+phase2) => 32 training stages.
#
# Usage:
#   cd /data2/workspace/heewon/KGQA
#   bash trm_rag_style/scripts/run_rearev_d_dplus_ablation_2datasets.sh
#
# Optional overrides:
#   DATASETS="cwq,webqsp"
#   RECURSION_LIST="0,3,6,9"
#   NUM_INS_LIST="1,2,3,4"
#   VARIANTS="d,dplus"
#   PHASE1_EPOCHS=25
#   PHASE2_EPOCHS=25
#   BATCH_SIZE=2
#   PHASE2_BATCH_SIZE=2
#   SUBGRAPH_GRAD_ACCUM_STEPS=4
#   PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS=4
#   WANDB_MODE=online
#   WANDB_PROJECT=graph-traverse
#   WANDB_NAME_PREFIX=ablation
#   WANDB_GROUP_PREFIX=ablation
#   RUN_PREFIX=abl
#   BASE_MASTER_PORT=30100
#   CONTINUE_ON_ERROR=true

DATASETS="${DATASETS:-cwq,webqsp}"
RECURSION_LIST="${RECURSION_LIST:-0,3,6,9}"
NUM_INS_LIST="${NUM_INS_LIST:-1,2,3,4}"
VARIANTS="${VARIANTS:-d,dplus}"

PHASE1_EPOCHS="${PHASE1_EPOCHS:-25}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-25}"

BATCH_SIZE="${BATCH_SIZE:-2}"
PHASE2_BATCH_SIZE="${PHASE2_BATCH_SIZE:-2}"
SUBGRAPH_GRAD_ACCUM_STEPS="${SUBGRAPH_GRAD_ACCUM_STEPS:-4}"
PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS="${PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS:-4}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
WANDB_NAME_PREFIX="${WANDB_NAME_PREFIX:-ablation}"
WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-ablation}"
RUN_PREFIX="${RUN_PREFIX:-abl}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-30100}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

PHASE1_SUBGRAPH_LR_SCHEDULER="${PHASE1_SUBGRAPH_LR_SCHEDULER:-cosine}"
PHASE2_SUBGRAPH_LR_SCHEDULER="${PHASE2_SUBGRAPH_LR_SCHEDULER:-plateau}"
PHASE1_SUBGRAPH_LR_MIN="${PHASE1_SUBGRAPH_LR_MIN:-1e-6}"
PHASE2_SUBGRAPH_LR_MIN="${PHASE2_SUBGRAPH_LR_MIN:-1e-6}"

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
IFS=',' read -r -a REC_ARR <<< "$RECURSION_LIST"
IFS=',' read -r -a INS_ARR <<< "$NUM_INS_LIST"
IFS=',' read -r -a VARIANT_ARR <<< "$VARIANTS"

if [[ "${#REC_ARR[@]}" -ne "${#INS_ARR[@]}" ]]; then
  echo "[err] RECURSION_LIST and NUM_INS_LIST must have the same length for paired ablation." >&2
  echo "      got ${#REC_ARR[@]} vs ${#INS_ARR[@]}" >&2
  exit 2
fi

total_runs=$(( ${#DATASET_ARR[@]} * ${#REC_ARR[@]} * ${#VARIANT_ARR[@]} ))
total_train_stages=$(( total_runs * 2 ))

echo "[plan] datasets=${DATASETS}"
echo "[plan] paired configs=${#REC_ARR[@]} variants=${VARIANTS}"
echo "[plan] total runs=${total_runs}, total train stages(phase1+phase2)=${total_train_stages}"
echo "[plan] phase1_epochs=${PHASE1_EPOCHS}, phase2_epochs=${PHASE2_EPOCHS}"
echo "[plan] using 4 GPUs for phase1/fallback/phase2"

run_idx=0
ok_runs=0
fail_runs=0

for dataset in "${DATASET_ARR[@]}"; do
  for idx in "${!REC_ARR[@]}"; do
    rec="${REC_ARR[$idx]}"
    ins="${INS_ARR[$idx]}"
    for variant in "${VARIANT_ARR[@]}"; do
      run_idx=$((run_idx + 1))
      master_port=$((BASE_MASTER_PORT + run_idx * 10))
      fallback_port=$((master_port + 1))
      phase2_port=$((master_port + 2))

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

      cycle_id="${dataset}_${variant}_r${rec}_i${ins}"
      run_tag="${RUN_PREFIX}_${cycle_id}_$(date +%Y%m%d_%H%M%S)"
      wandb_group="${WANDB_GROUP_PREFIX}_${cycle_id}"
      phase1_name="${WANDB_NAME_PREFIX}_${cycle_id}_phase1"
      phase2_name="${WANDB_NAME_PREFIX}_${cycle_id}_phase2"

      echo ""
      echo "[run ${run_idx}/${total_runs}] dataset=${dataset} variant=${variant} rec=${rec} ins=${ins}"
      echo "  run_tag=${run_tag}"
      echo "  wandb_group=${wandb_group}"

      set +e
      RUN_TAG="$run_tag" \
      DATASET="$dataset" \
      PRIMARY_GPUS="0,1,2,3" PRIMARY_NPROC="4" \
      FALLBACK_GPUS="0,1,2,3" FALLBACK_NPROC="4" \
      PHASE2_GPUS="0,1,2,3" PHASE2_NPROC_PER_NODE="4" \
      MASTER_PORT="$master_port" \
      FALLBACK_MASTER_PORT="$fallback_port" \
      PHASE2_MASTER_PORT="$phase2_port" \
      PHASE1_EPOCHS="$PHASE1_EPOCHS" \
      PHASE2_EPOCHS="$PHASE2_EPOCHS" \
      BATCH_SIZE="$BATCH_SIZE" \
      PHASE2_BATCH_SIZE="$PHASE2_BATCH_SIZE" \
      SUBGRAPH_GRAD_ACCUM_STEPS="$SUBGRAPH_GRAD_ACCUM_STEPS" \
      PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS="$PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS" \
      PHASE1_SUBGRAPH_RECURSION_STEPS="$rec" \
      PHASE2_SUBGRAPH_RECURSION_STEPS="$rec" \
      PHASE1_SUBGRAPH_REAREV_NUM_INS="$ins" \
      PHASE2_SUBGRAPH_REAREV_NUM_INS="$ins" \
      SUBGRAPH_GNN_VARIANT="$gnn_variant" \
      PHASE1_SUBGRAPH_REAREV_LATENT_UPDATE_MODE="$latent_update" \
      PHASE2_SUBGRAPH_REAREV_LATENT_UPDATE_MODE="$latent_update" \
      PHASE1_SUBGRAPH_LR_SCHEDULER="$PHASE1_SUBGRAPH_LR_SCHEDULER" \
      PHASE2_SUBGRAPH_LR_SCHEDULER="$PHASE2_SUBGRAPH_LR_SCHEDULER" \
      PHASE1_SUBGRAPH_LR_MIN="$PHASE1_SUBGRAPH_LR_MIN" \
      PHASE2_SUBGRAPH_LR_MIN="$PHASE2_SUBGRAPH_LR_MIN" \
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
  done
done

echo ""
echo "[done] run summary: ok=${ok_runs} fail=${fail_runs} total=${total_runs}"
if [[ "$fail_runs" -gt 0 ]]; then
  exit 1
fi
