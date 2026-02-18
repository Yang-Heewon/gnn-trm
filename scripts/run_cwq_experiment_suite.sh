#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

MAX_STEPS="${MAX_STEPS:-4}"
MAX_PATHS="${MAX_PATHS:-4}"
MINE_MAX_NEIGHBORS="${MINE_MAX_NEIGHBORS:-128}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-24}"

EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
EMBED_GPUS="${EMBED_GPUS:-0,1,2}"
EMB_TAG="${EMB_TAG:-e5}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-6}"
LR="${LR:-0.0001}"
TRAIN_MAX_NEIGHBORS="${TRAIN_MAX_NEIGHBORS:-256}"
TRAIN_PRUNE_KEEP="${TRAIN_PRUNE_KEEP:-64}"
TRAIN_PRUNE_RAND="${TRAIN_PRUNE_RAND:-64}"

EVAL_LIMIT="${EVAL_LIMIT:-200}"
DEBUG_EVAL_N="${DEBUG_EVAL_N:-5}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-2}"
EVAL_START_EPOCH="${EVAL_START_EPOCH:-2}"
TEST_EVAL_LIMIT="${TEST_EVAL_LIMIT:--1}"

EVAL_MAX_NEIGHBORS="${EVAL_MAX_NEIGHBORS:-256}"
EVAL_PRUNE_KEEP="${EVAL_PRUNE_KEEP:-64}"
EVAL_START_TOPK="${EVAL_START_TOPK:-5}"

WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-cwq_suite}"
DDP_FIND_UNUSED="${DDP_FIND_UNUSED:-true}"

S2_MAX_STEPS_GRID="${S2_MAX_STEPS_GRID:-2,3,4}"
S2_BEAM_GRID="${S2_BEAM_GRID:-8,16,32}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_ROOT="${SUITE_ROOT:-$REPO_ROOT/trm_agent/experiments/${DATASET}_suite_${STAMP}}"
LOG_DIR="$SUITE_ROOT/logs"
RESULTS_CSV="$SUITE_ROOT/results.csv"

if [[ "$DATASET" != "cwq" ]]; then
  echo "[err] this suite script is designed for DATASET=cwq"
  exit 1
fi

mkdir -p "$SUITE_ROOT" "$LOG_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] configuration loaded."
  echo "[dry-run] this suite performs staged best-selection, so DRY_RUN exits before execution."
  echo "[dry-run] SUITE_ROOT=$SUITE_ROOT"
  exit 0
fi

count_gpus() {
  local gpus="$1"
  if [[ "$gpus" == *","* ]]; then
    awk -F, '{print NF}' <<<"$gpus"
  else
    echo 1
  fi
}
NPROC="$(count_gpus "$CUDA_VISIBLE_DEVICES")"

echo "[suite] root: $SUITE_ROOT"
echo "[suite] nproc: $NPROC (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

if [[ ! -f "$RESULTS_CSV" ]]; then
  cat > "$RESULTS_CSV" <<'CSV'
stage,exp_name,status,path_policy,shortest_k,eval_no_cycle,endpoint_aux_weight,eval_max_steps,eval_beam,eval_start_topk,epochs,batch_size,lr,dev_hit1,dev_f1,dev_skip,test_hit1,test_f1,test_skip,ckpt,processed_dir,emb_dir,ckpt_dir,train_log,test_log
CSV
fi

parse_metric_line() {
  local line="$1"
  local hit="NA"
  local f1="NA"
  local skip="NA"
  if [[ "$line" =~ Hit@1=([0-9.]+)[[:space:]]F1=([0-9.]+)[[:space:]]Skip=([0-9]+) ]]; then
    hit="${BASH_REMATCH[1]}"
    f1="${BASH_REMATCH[2]}"
    skip="${BASH_REMATCH[3]}"
  fi
  echo "$hit,$f1,$skip"
}

run_and_tee() {
  local log_file="$1"
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] $*" | tee -a "$log_file"
    return 0
  fi
  set +e
  "$@" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

ensure_variant() {
  local policy="$1"
  local shortest_k="$2"
  local variant_id="${policy}_k${shortest_k}"
  local processed_dir="$SUITE_ROOT/processed/$variant_id"
  local emb_dir="$SUITE_ROOT/emb/${variant_id}_${EMB_TAG}"
  local prep_log="$LOG_DIR/preprocess_${variant_id}.log"
  local embed_log="$LOG_DIR/embed_${variant_id}.log"

  if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -rf "$processed_dir" "$emb_dir"
  fi

  if [[ ! -f "$processed_dir/train.jsonl" ]]; then
    mkdir -p "$processed_dir"
    run_and_tee "$prep_log" "$PYTHON_BIN" -m trm_agent.run \
      --dataset "$DATASET" \
      --model_impl "$MODEL_IMPL" \
      --stage preprocess \
      --override \
        processed_dir="$processed_dir" \
        max_steps="$MAX_STEPS" \
        max_paths="$MAX_PATHS" \
        mine_max_neighbors="$MINE_MAX_NEIGHBORS" \
        preprocess_workers="$PREPROCESS_WORKERS" \
        train_path_policy="$policy" \
        train_shortest_k="$shortest_k"
  fi

  if [[ ! -f "$emb_dir/query_train.npy" || ! -f "$emb_dir/query_dev.npy" || ! -f "$emb_dir/entity_embeddings.npy" ]]; then
    mkdir -p "$emb_dir"
    run_and_tee "$embed_log" env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m trm_agent.run \
      --dataset "$DATASET" \
      --model_impl "$MODEL_IMPL" \
      --embedding_model "$EMB_MODEL" \
      --stage embed \
      --override \
        processed_dir="$processed_dir" \
        emb_dir="$emb_dir" \
        emb_tag="$EMB_TAG" \
        embed_device="$EMBED_DEVICE" \
        embed_gpus="$EMBED_GPUS"
  fi

  echo "$processed_dir|$emb_dir"
}

RUN_INDEX=0

run_experiment() {
  local stage="$1"
  local exp_name="$2"
  local path_policy="$3"
  local shortest_k="$4"
  local eval_no_cycle="$5"
  local endpoint_aux_weight="$6"
  local eval_max_steps="$7"
  local eval_beam="$8"
  local eval_start_topk="$9"

  RUN_INDEX=$((RUN_INDEX + 1))
  local master_port=$((MASTER_PORT_BASE + RUN_INDEX))
  local status="ok"
  local train_log="$LOG_DIR/${exp_name}_train.log"
  local test_log="$LOG_DIR/${exp_name}_test.log"
  local ckpt_dir="$SUITE_ROOT/ckpt/$exp_name"
  local ckpt=""
  local dev_line=""
  local test_line=""

  local variant_paths
  variant_paths="$(ensure_variant "$path_policy" "$shortest_k")"
  local processed_dir="${variant_paths%%|*}"
  local emb_dir="${variant_paths##*|}"

  mkdir -p "$ckpt_dir"
  if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -f "$ckpt_dir"/model_ep*.pt
  fi

  local wandb_run_name="${WANDB_RUN_PREFIX}_${exp_name}"
  if ! run_and_tee "$train_log" env \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    TORCH_NCCL_BLOCKING_WAIT=1 \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" --nproc_per_node="$NPROC" --master_port="$master_port" -m trm_agent.run \
      --dataset "$DATASET" \
      --model_impl "$MODEL_IMPL" \
      --embedding_model "$EMB_MODEL" \
      --stage train \
      --override \
        processed_dir="$processed_dir" \
        emb_dir="$emb_dir" \
        ckpt_dir="$ckpt_dir" \
        max_steps="$MAX_STEPS" \
        max_neighbors="$TRAIN_MAX_NEIGHBORS" \
        prune_keep="$TRAIN_PRUNE_KEEP" \
        prune_rand="$TRAIN_PRUNE_RAND" \
        epochs="$EPOCHS" \
        batch_size="$BATCH_SIZE" \
        lr="$LR" \
        ddp_find_unused="$DDP_FIND_UNUSED" \
        eval_limit="$EVAL_LIMIT" \
        debug_eval_n="$DEBUG_EVAL_N" \
        eval_every_epochs="$EVAL_EVERY_EPOCHS" \
        eval_start_epoch="$EVAL_START_EPOCH" \
        eval_no_cycle="$eval_no_cycle" \
        eval_max_steps="$eval_max_steps" \
        eval_max_neighbors="$EVAL_MAX_NEIGHBORS" \
        eval_prune_keep="$EVAL_PRUNE_KEEP" \
        eval_beam="$eval_beam" \
        eval_start_topk="$eval_start_topk" \
        endpoint_aux_weight="$endpoint_aux_weight" \
        wandb_mode="$WANDB_MODE" \
        wandb_project="$WANDB_PROJECT" \
        wandb_entity="$WANDB_ENTITY" \
        wandb_run_name="$wandb_run_name"; then
    status="train_fail"
  fi

  ckpt="$(ls -t "$ckpt_dir"/model_ep*.pt 2>/dev/null | head -n 1 || true)"
  if [[ -z "$ckpt" ]]; then
    status="no_ckpt"
  fi

  if [[ "$status" == "ok" ]]; then
    if ! run_and_tee "$test_log" "$PYTHON_BIN" -m trm_agent.run \
      --dataset "$DATASET" \
      --model_impl "$MODEL_IMPL" \
      --stage test \
      --ckpt "$ckpt" \
      --override \
        processed_dir="$processed_dir" \
        emb_dir="$emb_dir" \
        batch_size="$BATCH_SIZE" \
        eval_limit="$TEST_EVAL_LIMIT" \
        debug_eval_n="$DEBUG_EVAL_N" \
        eval_no_cycle="$eval_no_cycle" \
        eval_max_steps="$eval_max_steps" \
        eval_max_neighbors="$EVAL_MAX_NEIGHBORS" \
        eval_prune_keep="$EVAL_PRUNE_KEEP" \
        eval_beam="$eval_beam" \
        eval_start_topk="$eval_start_topk"; then
      status="test_fail"
    fi
  fi

  dev_line="$(grep -E '\[Dev\] Hit@1=' "$train_log" | tail -n 1 || true)"
  test_line="$(grep -E '\[Test\] Hit@1=' "$test_log" | tail -n 1 || true)"

  local dev_vals test_vals
  dev_vals="$(parse_metric_line "$dev_line")"
  test_vals="$(parse_metric_line "$test_line")"

  echo "$stage,$exp_name,$status,$path_policy,$shortest_k,$eval_no_cycle,$endpoint_aux_weight,$eval_max_steps,$eval_beam,$eval_start_topk,$EPOCHS,$BATCH_SIZE,$LR,${dev_vals},${test_vals},$ckpt,$processed_dir,$emb_dir,$ckpt_dir,$train_log,$test_log" >> "$RESULTS_CSV"
  echo "[done] $exp_name status=$status dev=(${dev_vals}) test=(${test_vals})"
}

choose_best_from_stage() {
  local stage="$1"
  "$PYTHON_BIN" - <<PY
import csv, sys
path = "$RESULTS_CSV"
stage = "$stage"
best = None
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("stage") != stage:
            continue
        if row.get("status") != "ok":
            continue
        try:
            f1 = float(row.get("dev_f1", "nan"))
        except Exception:
            continue
        if best is None or f1 > best[0]:
            best = (f1, row)
if best is None:
    print("")
    sys.exit(0)
row = best[1]
keys = [
    "exp_name",
    "path_policy",
    "shortest_k",
    "eval_no_cycle",
    "endpoint_aux_weight",
    "eval_max_steps",
    "eval_beam",
    "eval_start_topk",
]
print("|".join(row.get(k, "") for k in keys))
PY
}

echo "[stage1] core ablations"
run_experiment "stage1" "s1_baseline_all" "all" "1" "false" "0.0" "4" "8" "$EVAL_START_TOPK"
run_experiment "stage1" "s1_shortest_only" "shortest_only" "1" "false" "0.0" "4" "8" "$EVAL_START_TOPK"
run_experiment "stage1" "s1_shortest_no_cycle" "shortest_only" "1" "true" "0.0" "4" "8" "$EVAL_START_TOPK"
run_experiment "stage1" "s1_shortest_no_cycle_endpoint" "shortest_only" "1" "true" "0.2" "4" "8" "$EVAL_START_TOPK"

BEST1="$(choose_best_from_stage stage1)"
if [[ -z "$BEST1" ]]; then
  echo "[err] no successful stage1 run"
  exit 1
fi
IFS='|' read -r BEST1_NAME BEST1_POLICY BEST1_K BEST1_NOCYCLE BEST1_AUX BEST1_MS BEST1_BEAM BEST1_TOPK <<<"$BEST1"
echo "[stage1-best] $BEST1_NAME (policy=$BEST1_POLICY k=$BEST1_K no_cycle=$BEST1_NOCYCLE aux=$BEST1_AUX)"

echo "[stage2] search tuning"
IFS=',' read -r -a S2_STEPS <<< "$S2_MAX_STEPS_GRID"
IFS=',' read -r -a S2_BEAMS <<< "$S2_BEAM_GRID"
for ms in "${S2_STEPS[@]}"; do
  for bm in "${S2_BEAMS[@]}"; do
    exp="s2_${BEST1_NAME}_ms${ms}_b${bm}"
    run_experiment "stage2" "$exp" "$BEST1_POLICY" "$BEST1_K" "$BEST1_NOCYCLE" "$BEST1_AUX" "$ms" "$bm" "$EVAL_START_TOPK"
  done
done

BEST2="$(choose_best_from_stage stage2)"
if [[ -z "$BEST2" ]]; then
  echo "[err] no successful stage2 run"
  exit 1
fi
IFS='|' read -r BEST2_NAME BEST2_POLICY BEST2_K BEST2_NOCYCLE BEST2_AUX BEST2_MS BEST2_BEAM BEST2_TOPK <<<"$BEST2"
echo "[stage2-best] $BEST2_NAME (ms=$BEST2_MS beam=$BEST2_BEAM topk=$BEST2_TOPK)"

echo "[stage3] shortest-k comparison"
run_experiment "stage3" "s3_shortest_only" "shortest_only" "1" "$BEST2_NOCYCLE" "$BEST2_AUX" "$BEST2_MS" "$BEST2_BEAM" "$BEST2_TOPK"
run_experiment "stage3" "s3_shortest_k2" "shortest_k" "2" "$BEST2_NOCYCLE" "$BEST2_AUX" "$BEST2_MS" "$BEST2_BEAM" "$BEST2_TOPK"
run_experiment "stage3" "s3_shortest_k3" "shortest_k" "3" "$BEST2_NOCYCLE" "$BEST2_AUX" "$BEST2_MS" "$BEST2_BEAM" "$BEST2_TOPK"

echo "[summary] results csv: $RESULTS_CSV"
"$PYTHON_BIN" - <<PY
import csv
path = "$RESULTS_CSV"
rows = []
with open(path, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("status") != "ok":
            continue
        try:
            r["_dev_f1"] = float(r.get("dev_f1", "nan"))
            r["_test_f1"] = float(r.get("test_f1", "nan"))
        except Exception:
            continue
        rows.append(r)
rows.sort(key=lambda x: (x["_dev_f1"], x["_test_f1"]), reverse=True)
print("Top 10 by dev_f1:")
for r in rows[:10]:
    print(
        f'{r["stage"]} {r["exp_name"]} '
        f'dev(hit1={r["dev_hit1"]}, f1={r["dev_f1"]}) '
        f'test(hit1={r["test_hit1"]}, f1={r["test_f1"]})'
    )
PY
