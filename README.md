# KGQA ReaRev-Only Subgraph Reader

This repository currently runs **only the ReaRev-style subgraph reader path**.

- `baseline` subgraph recurrence is disabled in code.
- `subgraph_gnn_variant` is fixed to `rearev_bfs` during train/test.
- Recursion depth is user-controlled by `subgraph_recursion_steps`.

## 1) Repo Layout

- `trm_agent/`: processed jsonl, embeddings, checkpoints
- `trm_unified/`: core logic (`subgraph_reader.py`, `train_core.py`)
- `trm_rag_style/`: config and pipeline entrypoint

## 2) Environment

```bash
cd /data2/workspace/heewon/KGQA
pip install -r requirements.txt
```

## 3) Data and Embeddings

```bash
bash trm_rag_style/scripts/run_download.sh
```

```bash
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5_w4_g4 \
EMBED_GPUS=0,1,2,3 \
EMBED_BATCH_SIZE=512 \
bash trm_rag_style/scripts/run_embed.sh
```

## 4) Train (ReaRev Only)

```bash
python -m trm_agent.run \
  --dataset cwq \
  --model_impl trm_hier6 \
  --stage train \
  --override \
    subgraph_reader_enabled=true \
    emb_tag=e5_w4_g4 \
    emb_dir=trm_agent/emb/cwq_e5_w4_g4 \
    subgraph_recursion_steps=8 \
    subgraph_rearev_num_ins=3 \
    subgraph_rearev_adapt_stages=1
```

## 5) Test

```bash
python -m trm_agent.run \
  --dataset cwq \
  --model_impl trm_hier6 \
  --stage test \
  --ckpt trm_agent/ckpt/<your_run_dir>/model_epXX.pt \
  --override \
    subgraph_reader_enabled=true \
    emb_tag=e5_w4_g4 \
    emb_dir=trm_agent/emb/cwq_e5_w4_g4 \
    eval_json=trm_agent/processed/cwq/test.jsonl \
    query_emb_eval_npy=trm_agent/emb/cwq_e5_w4_g4/query_test.npy \
    batch_size=8
```

## 6) Main Knobs

- `subgraph_recursion_steps`: inner recurrent steps per stage
- `subgraph_rearev_num_ins`: number of instructions
- `subgraph_rearev_adapt_stages`: adaptive stage count
- `subgraph_outer_reasoning_enabled`: optional outer loop on top of inner ReaRev recurrence
- `subgraph_max_nodes`, `subgraph_max_edges`: subgraph budget and runtime/memory tradeoff

## 7) Notes

- Current training objective is node-level BCE-with-logits (with optional hard-negative/ranking losses).
- ReaRev logic is implemented in `trm_unified/subgraph_reader.py`.
- This is a ReaRev-style integration in this codebase, not a strict 1:1 reproduction of the original repository.
