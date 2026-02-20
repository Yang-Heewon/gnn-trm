# Paperstyle -> RL Pipeline

This folder contains the active preset:

1. `download` (RoG HF download/convert)
2. `embed` (preprocess + embedding)
3. `phase1` supervised train
4. `phase2` RL fine-tune
5. `test` evaluation

## Entrypoint

Use `run_pipeline.py`:

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
python experiments/paperstyle_rl/run_pipeline.py --stage all
```

Run one stage only:

```bash
python experiments/paperstyle_rl/run_pipeline.py --stage download
python experiments/paperstyle_rl/run_pipeline.py --stage embed
python experiments/paperstyle_rl/run_pipeline.py --stage phase1
python experiments/paperstyle_rl/run_pipeline.py --stage phase2
python experiments/paperstyle_rl/run_pipeline.py --stage test
```

## Linux Bash Wrappers

Primary wrappers:

- `00_download.sh` (download + convert)
- `01_embed.sh` (preprocess + embed)
- `02_train_phase1.sh`
- `03_train_phase2_rl.sh`
- `04_eval_phase2_test.sh`
- `run_all.sh`
- `run_all_wandb.sh`

`run_all.sh` sequence is:
1. `00_download.sh`
2. `01_embed.sh`
3. `02_train_phase1.sh`
4. `03_train_phase2_rl.sh`
5. `04_eval_phase2_test.sh`

## Key Environment Variables

Main variables:

- `DATASET` (default: `cwq`)
- `EMB_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `EMB_TAG` (default: sanitized from `EMB_MODEL`)
- `EMBED_STYLE` (default: `gnn_rag_gnn_exact`)
- `EMBED_BACKEND` (default: `sentence_transformers`)
- `DATA_SOURCE` (default: `rog_hf`, used by `trm_rag_style/scripts/run_download.sh`)
- `RUN_PREPROCESS` (default: `1`, used by `trm_rag_style/scripts/run_embed.sh`)
- `EPOCHS_PHASE1` (default: `5`)
- `EPOCHS_PHASE2` (default: `20`)
- `BATCH_SIZE_PHASE1` (default: `6`)
- `BATCH_SIZE_PHASE2` (default: `2`)
- `LR` (default: `2e-4`)
- `NPROC_PER_NODE` (default: `3`)
- `NPROC_PER_NODE_PHASE2` (default: `1`)
- `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`
- `CKPT_DIR_PHASE1`, `CKPT_DIR_PHASE2`, `RESULTS_DIR`
- `QUERY_RESIDUAL_ENABLED`, `QUERY_RESIDUAL_ALPHA`, `QUERY_RESIDUAL_MODE`

Test-stage variables:

- `TEST_EVAL_LIMIT`, `TEST_DEBUG_EVAL_N`, `TEST_EVAL_PRED_TOPK`
- `TEST_EVAL_NO_CYCLE`, `TEST_EVAL_USE_HALT`
- `TEST_EVAL_MAX_NEIGHBORS`, `TEST_EVAL_PRUNE_KEEP`
- `TEST_EVAL_BEAM`, `TEST_EVAL_START_TOPK`

## Notes

- In `gnn_rag_gnn_exact` mode, query/passage prefixes are empty.
- Use the same `EMB_MODEL`/`EMB_TAG` for embed/train/test stages.
- `run_embed.sh` supports `DATASET=cwq|webqsp|all`.
- Phase2 automatically uses the latest phase1 checkpoint unless `PHASE1_CKPT` is set.
- Test stage uses the latest phase2 checkpoint unless `PHASE2_CKPT` is set.
- Test summaries/logs are under `experiments/paperstyle_rl/results/`.
