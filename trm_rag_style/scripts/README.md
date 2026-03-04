# Script Guide (ReaRev-Only Subgraph Reader)

This repository is now configured to run the **ReaRev-only** subgraph reader path.

- `baseline` recurrence is disabled in code.
- Subgraph variant is fixed to `rearev_bfs` during train/test.
- Recursion depth is controlled by `SUBGRAPH_RECURSION_STEPS`.

## Recursive ReaRev Summary

The core model is a recursive graph reasoner with optional latent/global controls:

- Step update uses relation-aware forward/inverse message passing.
- Node update can be gated by global latent state (`global_gate`).
- Node score can fuse local node state + global latent context (`logit_global_fusion`).
- Latent state is updated every recursion step (`GRU` over weighted node context).
- Dynamic halting predicts stop probability from latent state.

Phase objectives:

- Phase1 (`rearev_kl`): KL over node distribution, optional deep supervision
  - `KL + lambda_ds * (step_ce + halt_bce)`
- Phase2 (`bce`): BCE + ranking + hard negatives

## Core Scripts (D / D+latent)

- `run_download.sh`
  - Download/prepare dataset files.

- `run_embed.sh`
  - Build entity/relation/query embeddings.

- `run_rearev_d_gpu01_hit06.sh`
  - D phase1 training launcher (supports latent/gate/fusion/halt + KL deep supervision).

- `run_rearev_d_rearevcore_trm.sh`
  - ReaRev core fixed training + TRM-style supervision (`rearev_kl_trm`).
  - Encoder/query embeddings stay fixed; only subgraph reader parameters are trained.

- `run_rearev_d_auto_fallback_4to2.sh`
  - Auto fallback wrapper: try 4-GPU run, fallback to 2-GPU on OOM.

- `run_rearev_d_phase2_resume.sh`
  - D phase2 resume launcher (BCE + ranking + hard negatives).

- `run_rearev_d_two_phase_auto.sh`
  - Two-phase D pipeline (phase1 convergence/early-stop -> best checkpoint -> phase2 fine-tune).
  - Supports phase2 best-ckpt selection (`PHASE2_BEST_METRIC=dev_hit1|dev_f1`) and optional latent ablation.

- `run_rearev_d_latent_max_protocol.sh`
  - End-to-end latent-max protocol (phase1 latent-dominant + phase2 hit-focused) with multi-seed loop.

- `run_test.sh`
  - Test entrypoint for ReaRev subgraph checkpoints.
  - Optional hop-wise relation tracing:
    - `SUBGRAPH_TRACE_RELATION_TOPK_ENABLED=true`
    - `SUBGRAPH_TRACE_RELATION_TOPK=5`
    - `SUBGRAPH_TRACE_MAX_EXAMPLES=3`

- `run_rearev_d_latent_ablation_eval.sh`
  - Same-checkpoint latent ablation eval (`full / no_latent / no_global / no_ins_delta`).

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
  - `SUBGRAPH_EARLY_STOP_ENABLED`, `SUBGRAPH_EARLY_STOP_METRIC`, `SUBGRAPH_EARLY_STOP_PATIENCE`
  - `SUBGRAPH_EARLY_STOP_MIN_DELTA`, `SUBGRAPH_EARLY_STOP_MIN_EPOCHS`
  - `SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA`
  - `SUBGRAPH_KL_NO_POSITIVE_MODE` (`uniform` | `skip`)
- Optimization:
  - `LR`, `SUBGRAPH_LR_SCHEDULER`, `SUBGRAPH_LR_MIN`
- Test trace/debug:
  - `SUBGRAPH_TRACE_RELATION_TOPK_ENABLED`
  - `SUBGRAPH_TRACE_RELATION_TOPK`
  - `SUBGRAPH_TRACE_LOG_EXAMPLES` (default `5`)
  - `SUBGRAPH_TRACE_DUMP_MAX_EXAMPLES` (default `1000`)
  - `SUBGRAPH_TRACE_MAX_EXAMPLES` (legacy one-knob fallback)
  - `SUBGRAPH_TRACE_PATH_DUMP_JSONL`

## Typical Start

```bash
cd /data2/workspace/heewon/KGQA
bash trm_rag_style/scripts/run_rearev_d_auto_fallback_4to2.sh
```

Then override env vars as needed.

## Quick Compare (D vs D+latent)

### D baseline two-phase

```bash
cd /data2/workspace/heewon/KGQA
RUN_TAG=d_baseline_4gpu \
WANDB_MODE=online WANDB_PROJECT=graph-traverse \
PRIMARY_GPUS=0,1,2,3 PRIMARY_NPROC=4 \
FALLBACK_GPUS=0,1,2,3 FALLBACK_NPROC=4 \
PHASE2_GPUS=0,1,2,3 PHASE2_NPROC_PER_NODE=4 \
MASTER_PORT=29781 PHASE2_MASTER_PORT=29791 \
EPOCHS=16 PHASE2_EPOCHS=5 \
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
```

### D+latent phase1 (KL + deep supervision)

```bash
cd /data2/workspace/heewon/KGQA
RUN_TAG=d_latent_ds_4gpu \
WANDB_MODE=online WANDB_PROJECT=graph-traverse \
PRIMARY_GPUS=0,1,2,3 PRIMARY_NPROC=4 \
FALLBACK_GPUS=0,1,2,3 FALLBACK_NPROC=4 \
PHASE2_GPUS=0,1,2,3 PHASE2_NPROC_PER_NODE=4 \
MASTER_PORT=29811 PHASE2_MASTER_PORT=29821 \
PHASE1_SUBGRAPH_DEEP_SUPERVISION_ENABLED=true \
PHASE1_SUBGRAPH_DEEP_SUPERVISION_WEIGHT=0.03 \
PHASE1_SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT=1.0 \
PHASE1_SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT=0.3 \
PHASE1_SUBGRAPH_KL_NO_POSITIVE_MODE=skip \
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
```

ReaRev-core + TRM-style example:

```bash
cd /data2/workspace/heewon/KGQA
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
MASTER_PORT=29673 \
WANDB_MODE=online \
WANDB_PROJECT=graph-traverse \
WANDB_RUN_NAME="cwq-rearev-core-trm" \
SUBGRAPH_REAREV_TRM_WEIGHT=0.05 \
bash trm_rag_style/scripts/run_rearev_d_rearevcore_trm.sh
```
