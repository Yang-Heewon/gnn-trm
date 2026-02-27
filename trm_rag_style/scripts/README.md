# Script Guide (ReaRev-Only Subgraph Reader)

This repository is now configured to run the **ReaRev-only** subgraph reader path.

- `baseline` recurrence is disabled in code.
- Subgraph variant is fixed to `rearev_bfs` during train/test.
- Recursion depth is controlled by `SUBGRAPH_RECURSION_STEPS`.

## Core Scripts

- `run_download.sh`
  - Download/prepare dataset files.

- `run_embed.sh`
  - Build entity/relation/query embeddings.

- `run_train_subgraph_v2_resume.sh`
  - ReaRev subgraph reader training (script name is legacy).
  - Supports resume via `CKPT` and `SUBGRAPH_RESUME_EPOCH`.

- `run_train_subgraph_outer_yz_resume.sh`
  - ReaRev subgraph reader with optional outer latent loop controls.
  - Supports:
    - `FROM_SCRATCH=true` (fresh training)
    - `FROM_SCRATCH=false` (resume fine-tune)
    - cosine scheduling + gradient accumulation.

- `run_test.sh`
  - Test entrypoint for ReaRev subgraph checkpoints.

## Utilities

- `run_select_subgraph_ckpt_devtest_gap.sh`
  - Evaluate epoch-range checkpoints and rank by dev/test consistency.

- `run_select_and_finetune_subgraph.sh`
  - Selection + fine-tuning helper.

## Important Env Vars

- Data/embedding:
  - `DATASET`, `EMB_MODEL`, `EMB_TAG`, `EMB_DIR`
- DDP/runtime:
  - `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `MASTER_PORT`
- ReaRev subgraph model:
  - `HIDDEN_SIZE`, `SUBGRAPH_RECURSION_STEPS`
  - `SUBGRAPH_REAREV_NUM_INS`, `SUBGRAPH_REAREV_ADAPT_STAGES`
  - `SUBGRAPH_MAX_NODES`, `SUBGRAPH_MAX_EDGES`
  - `SUBGRAPH_OUTER_REASONING_STEPS`
  - `SUBGRAPH_RANKING_ENABLED`, `SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED`
  - `SUBGRAPH_GRAD_ACCUM_STEPS`
- Optimization:
  - `LR`, `SUBGRAPH_LR_SCHEDULER`, `SUBGRAPH_LR_MIN`

## Typical Start

```bash
cd /data2/workspace/heewon/KGQA
bash trm_rag_style/scripts/run_train_subgraph_outer_yz_resume.sh
```

Then override env vars as needed.
