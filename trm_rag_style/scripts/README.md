# Script Guide (ReaRev-Only Subgraph Reader)

This repository is now configured to run the **ReaRev-only** subgraph reader path.

- `baseline` recurrence is disabled in code.
- Subgraph variant is fixed to `rearev_bfs` during train/test.
- Recursion depth is controlled by `SUBGRAPH_RECURSION_STEPS`.

## Core Scripts (D / D+latent)

- `run_download.sh`
  - Download/prepare dataset files.

- `run_embed.sh`
  - Build entity/relation/query embeddings.

- `run_rearev_d_gpu01_hit06.sh`
  - D phase1 training launcher (supports latent/gate/fusion/halt + KL deep supervision).

- `run_rearev_d_auto_fallback_4to2.sh`
  - Auto fallback wrapper: try 4-GPU run, fallback to 2-GPU on OOM.

- `run_rearev_d_phase2_resume.sh`
  - D phase2 resume launcher (BCE + ranking + hard negatives).

- `run_rearev_d_two_phase_auto.sh`
  - Two-phase D pipeline (phase1 select best checkpoint -> phase2 fine-tune).

- `run_test.sh`
  - Test entrypoint for ReaRev subgraph checkpoints.

## Important Env Vars

- Data/embedding:
  - `DATASET`, `EMB_MODEL`, `EMB_TAG`, `EMB_DIR`
- DDP/runtime:
  - `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `MASTER_PORT`
- ReaRev subgraph model:
  - `HIDDEN_SIZE`, `SUBGRAPH_RECURSION_STEPS`
  - `SUBGRAPH_REAREV_NUM_INS`, `SUBGRAPH_REAREV_ADAPT_STAGES`
  - `SUBGRAPH_MAX_NODES`, `SUBGRAPH_MAX_EDGES`
  - `SUBGRAPH_RANKING_ENABLED`, `SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED`
  - `SUBGRAPH_GRAD_ACCUM_STEPS`
  - `SUBGRAPH_DEEP_SUPERVISION_ENABLED`, `SUBGRAPH_DEEP_SUPERVISION_WEIGHT`
  - `SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT`, `SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT`
- Optimization:
  - `LR`, `SUBGRAPH_LR_SCHEDULER`, `SUBGRAPH_LR_MIN`

## Typical Start

```bash
cd /data2/workspace/heewon/KGQA
bash trm_rag_style/scripts/run_rearev_d_auto_fallback_4to2.sh
```

Then override env vars as needed.
