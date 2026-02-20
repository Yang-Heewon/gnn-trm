# GRAPH-TRAVERSE

Minimal TRM KG traversal pipeline with only the active flow:

1. Download/convert HF RoG data
2. Preprocess + embed
3. Train (phase1 / optional phase2 RL)
4. Test evaluation

## Main Entrypoints

- `experiments/paperstyle_rl/run_pipeline.py` (stage runner)
- `experiments/paperstyle_rl/00_download.sh` .. `04_eval_phase2_test.sh`
- `trm_rag_style/scripts/run_download.sh`
- `trm_rag_style/scripts/run_embed.sh`
- `trm_rag_style/scripts/run_train.sh`
- `trm_rag_style/scripts/run_test.sh`

## One-Command Pipeline

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
python experiments/paperstyle_rl/run_pipeline.py --stage all
```

## Stage-by-Stage

```bash
python experiments/paperstyle_rl/run_pipeline.py --stage download
python experiments/paperstyle_rl/run_pipeline.py --stage embed
python experiments/paperstyle_rl/run_pipeline.py --stage phase1
python experiments/paperstyle_rl/run_pipeline.py --stage phase2
python experiments/paperstyle_rl/run_pipeline.py --stage test
```

## Download Only

Full CWQ from HF:

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
PYTHON_BIN=/data2/workspace/heewon/anaconda3/envs/taiLab/bin/python \
bash scripts/download_cwq_hf.sh
```

CWQ vocab only (`entities.txt`, `relations.txt`):

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
PYTHON_BIN=/data2/workspace/heewon/anaconda3/envs/taiLab/bin/python \
bash scripts/download_cwq_vocab_only.sh
```

## Embed Only

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
DATASET=all \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5 \
EMBED_BACKEND=transformers \
EMBED_GPUS=0,1,2 \
EMBED_BATCH_SIZE=512 \
bash trm_rag_style/scripts/run_embed.sh
```

`run_embed.sh` supports `DATASET=cwq|webqsp|all`.

## Supervised Residual Preset

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
DATASET=cwq EMB_MODEL=intfloat/multilingual-e5-large EMB_TAG=e5 \
bash trm_rag_style/scripts/run_train_supervised_residual_wandb.sh
```

## Important Notes

- Keep `EMB_MODEL`/`EMB_TAG` consistent across embed/train/test.
- Embeddings are resolved from `trm_agent/emb/${DATASET}_${EMB_TAG}`.
- See `experiments/paperstyle_rl/README.md` for full env var details.
