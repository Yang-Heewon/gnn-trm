#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data2/workspace/heewon/KGQA"
PYTHON_BIN="/data2/workspace/heewon/anaconda3/envs/taiLab/bin/python"
EMB_DIR="${REPO_ROOT}/trm_agent/emb/cwq_e5_w4_g4"
LOG_DIR="${REPO_ROOT}/logs/test_ep10_16_gpu0_ab"

A_CKPT_DIR="${REPO_ROOT}/trm_agent/ckpt/cwq_trm_hier6_rearev_A_gpu01_fullDev"
B_CKPT_DIR="${REPO_ROOT}/trm_agent/ckpt/cwq_trm_hier6_rearev_B_gpu23_fullDev"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

run_one() {
  local tag="$1"
  local ckpt_dir="$2"
  for ep in 10 11 12 13 14 15 16; do
    local ckpt="${ckpt_dir}/model_ep${ep}.pt"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[skip] ${tag} ep${ep} missing: ${ckpt}"
      continue
    fi
    local log_file="${LOG_DIR}/${tag}_ep${ep}.log"
    echo "[run] ${tag} ep=${ep} ckpt=${ckpt}"
    CUDA_VISIBLE_DEVICES=0 \
    "${PYTHON_BIN}" -m trm_agent.run \
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
}

run_one "A" "${A_CKPT_DIR}"
run_one "B" "${B_CKPT_DIR}"

echo "[done] A/B test ep10~16 on GPU0 finished"
