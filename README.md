# GRAPH-TRAVERSE (Subgraph Reader Only)

CWQ 기준으로 **Subgraph Reader 2개 모드(v2 / v2-highdim)만** 재현하도록 정리한 레포입니다.

- `v2` (기본): v2 objective, 기본 hidden/recursion
- `v2-highdim`: v2 objective 유지 + hidden/dim, recursion 확장

## 1) 핵심 실행 파일

- `trm_rag_style/scripts/run_download.sh`
- `trm_rag_style/scripts/run_embed.sh`
- `trm_rag_style/scripts/run_test.sh`
- `trm_rag_style/scripts/run_train_subgraph_v2_resume.sh`
- `trm_rag_style/scripts/run_train_subgraph_v2_highdim.sh`
- `trm_rag_style/scripts/run_all_v2.sh`
- `trm_rag_style/scripts/run_all_v2_highdim.sh`

## 2) 환경 준비

```bash
cd /path/to/GRAPH-TRAVERSE
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

선택:

```bash
wandb login
```

## 3) 데이터 다운로드 (CWQ)

```bash
cd /path/to/GRAPH-TRAVERSE
DATASET=cwq bash trm_rag_style/scripts/run_download.sh
```

생성 확인:

- `data/CWQ/train_split.jsonl`
- `data/CWQ/dev_split.jsonl`
- `data/CWQ/test_split.jsonl`
- `data/CWQ/entities.txt`
- `data/CWQ/relations.txt`

## 4) 전처리 + 임베딩

```bash
cd /path/to/GRAPH-TRAVERSE
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
RUN_PREPROCESS=1 \
PREPROCESS_WORKERS=4 \
EMBED_GPUS=0,1,2,3 \
bash trm_rag_style/scripts/run_embed.sh
```

생성 확인:

- `trm_agent/processed/cwq/train.jsonl`
- `trm_agent/processed/cwq/dev.jsonl`
- `trm_agent/processed/cwq/test.jsonl`
- `trm_agent/emb/cwq_e5/entity_embeddings.npy`
- `trm_agent/emb/cwq_e5/relation_embeddings.npy`
- `trm_agent/emb/cwq_e5/query_train.npy`
- `trm_agent/emb/cwq_e5/query_dev.npy`
- `trm_agent/emb/cwq_e5/query_test.npy`

## 5) 학습 모드 A: v2 (기본)

기본값:

- `hidden_size=512`
- `subgraph_recursion_steps=12`
- `ranking/split_reverse/direction = false`

### 5-1) scratch

```bash
cd /path/to/GRAPH-TRAVERSE
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
MASTER_PORT=29606 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMB_DIR=trm_agent/emb/cwq_e5 \
CKPT= \
SUBGRAPH_RESUME_EPOCH=-1 \
EPOCHS=50 \
BATCH_SIZE=1 \
EVAL_LIMIT=-1 \
CKPT_DIR=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2 \
WANDB_MODE=online \
WANDB_RUN_NAME=cwq_v2_scratch \
bash trm_rag_style/scripts/run_train_subgraph_v2_resume.sh
```

### 5-2) resume

```bash
cd /path/to/GRAPH-TRAVERSE
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
MASTER_PORT=29606 \
CKPT=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2/model_ep30.pt \
SUBGRAPH_RESUME_EPOCH=30 \
EPOCHS=20 \
BATCH_SIZE=1 \
EVAL_LIMIT=-1 \
CKPT_DIR=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2 \
WANDB_MODE=online \
WANDB_RUN_NAME=cwq_v2_resume_ep30 \
bash trm_rag_style/scripts/run_train_subgraph_v2_resume.sh
```

## 6) 학습 모드 B: v2-highdim

기본값:

- `hidden_size=768`
- `subgraph_recursion_steps=16`
- `lr=1.5e-5`
- `subgraph_lr_scheduler=cosine`
- `v2 objective 유지` (ranking/split_reverse/direction=false)

```bash
cd /path/to/GRAPH-TRAVERSE
CUDA_VISIBLE_DEVICES=1,2,3 \
NPROC_PER_NODE=3 \
MASTER_PORT=29608 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMB_DIR=trm_agent/emb/cwq_e5 \
CKPT= \
SUBGRAPH_RESUME_EPOCH=-1 \
EPOCHS=50 \
BATCH_SIZE=1 \
EVAL_LIMIT=-1 \
CKPT_DIR=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2_highdim \
WANDB_MODE=online \
WANDB_RUN_NAME=cwq_v2_highdim_scratch \
bash trm_rag_style/scripts/run_train_subgraph_v2_highdim.sh
```

## 7) 단일 체크포인트 테스트

`run_test.sh` 기본은 v2 기준(`split_reverse=false`, `direction=false`)입니다.

```bash
cd /path/to/GRAPH-TRAVERSE
CUDA_VISIBLE_DEVICES=0 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMB_DIR=trm_agent/emb/cwq_e5 \
CKPT=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2/model_ep27.pt \
EVAL_LIMIT=-1 \
BATCH_SIZE=8 \
SUBGRAPH_RECURSION_STEPS=12 \
SUBGRAPH_MAX_NODES=2048 \
SUBGRAPH_MAX_EDGES=8192 \
bash trm_rag_style/scripts/run_test.sh
```

highdim ckpt 테스트 예시:

```bash
cd /path/to/GRAPH-TRAVERSE
CUDA_VISIBLE_DEVICES=0 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMB_DIR=trm_agent/emb/cwq_e5 \
CKPT=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2_highdim/model_ep20.pt \
EVAL_LIMIT=-1 \
BATCH_SIZE=8 \
SUBGRAPH_RECURSION_STEPS=16 \
SUBGRAPH_MAX_NODES=2048 \
SUBGRAPH_MAX_EDGES=8192 \
bash trm_rag_style/scripts/run_test.sh
```

## 8) epoch 20+ 체크포인트 일괄 테스트

```bash
cd /path/to/GRAPH-TRAVERSE
mkdir -p logs

for d in \
  trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2 \
  trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2_highdim
 do
  for p in "$d"/model_ep*.pt; do
    [ -e "$p" ] || continue
    ep="$(basename "$p" | sed -E 's/model_ep([0-9]+)\.pt/\1/')"
    [ "$ep" -ge 20 ] || continue
    name="$(basename "$d")_ep${ep}"
    echo "[RUN] $name"

    rec=12
    if echo "$d" | grep -q "highdim"; then
      rec=16
    fi

    CUDA_VISIBLE_DEVICES=0 \
    DATASET=cwq \
    EMB_MODEL=intfloat/multilingual-e5-large \
    EMB_TAG=e5 \
    EMB_DIR=trm_agent/emb/cwq_e5 \
    CKPT="$p" \
    EVAL_LIMIT=-1 \
    BATCH_SIZE=8 \
    SUBGRAPH_RECURSION_STEPS="$rec" \
    SUBGRAPH_MAX_NODES=2048 \
    SUBGRAPH_MAX_EDGES=8192 \
    bash trm_rag_style/scripts/run_test.sh \
      > "logs/test_${name}.log" 2>&1
  done
done

grep -nE "\[Test-Subgraph\]" logs/test_*.log
```

## 9) 올인원 실행

### 9-1) v2

```bash
cd /path/to/GRAPH-TRAVERSE
CUDA_VISIBLE_DEVICES=0,1,2 \
NPROC_PER_NODE=3 \
MASTER_PORT=29606 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMBED_GPUS=0,1,2,3 \
PREPROCESS_WORKERS=4 \
bash trm_rag_style/scripts/run_all_v2.sh
```

### 9-2) v2-highdim

```bash
cd /path/to/GRAPH-TRAVERSE
CUDA_VISIBLE_DEVICES=1,2,3 \
NPROC_PER_NODE=3 \
MASTER_PORT=29608 \
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMBED_GPUS=0,1,2,3 \
PREPROCESS_WORKERS=4 \
bash trm_rag_style/scripts/run_all_v2_highdim.sh
```

## 10) 메모

- 학습 후 자동 test는 `trm_rag_style/configs/base.json`의 `auto_test_after_train=true`로 동작합니다.
- tqdm 진행바는 TTY 여부에 따라 실시간/줄로그 모드가 자동 전환됩니다.
