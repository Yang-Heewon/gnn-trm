#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data2/workspace/heewon/KGQA"
PYTORCH_BIN="/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun"
CKPT_DIR="${REPO_ROOT}/trm_agent/ckpt/cwq_trm_hier6_rearev_main_gpu23_full"
EMB_DIR="${REPO_ROOT}/trm_agent/emb/cwq_e5_w4_g4"
TEST_JSON="${REPO_ROOT}/trm_agent/processed/cwq/test.jsonl"
QUERY_TEST_NPY="${EMB_DIR}/query_test.npy"
LOG_DIR="${REPO_ROOT}/logs/test_ep10_15_gpu01"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

for ep in 10 11 12 13 14 15; do
  ckpt="${CKPT_DIR}/model_ep${ep}.pt"
  while [[ ! -f "${ckpt}" ]]; do
    echo "[wait] missing ${ckpt}; sleep 60s"
    sleep 60
  done

  port=$((29710 + ep))
  log_file="${LOG_DIR}/ep${ep}.log"
  echo "[run] ep=${ep} ckpt=${ckpt} port=${port} log=${log_file}"

  CUDA_VISIBLE_DEVICES=0,1 \
  "${PYTORCH_BIN}" --nproc_per_node=2 --master_port="${port}" \
    -m trm_agent.run \
    --dataset cwq \
    --model_impl trm_hier6 \
    --stage test \
    --ckpt "${ckpt}" \
    --override \
      subgraph_reader_enabled=true \
      emb_tag=e5_w4_g4 \
      emb_dir=trm_agent/emb/cwq_e5_w4_g4 \
      eval_json=trm_agent/processed/cwq/test.jsonl \
      query_emb_eval_npy=trm_agent/emb/cwq_e5_w4_g4/query_test.npy \
      batch_size=4 \
      eval_limit=-1 \
      wandb_mode=offline \
    2>&1 | tee "${log_file}"
done

echo "[done] all tests completed (ep10~ep15)"
