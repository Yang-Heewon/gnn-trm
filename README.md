# KGQA Recursive ReaRev (D / D+latent)

This repository is configured for the **subgraph-reader ReaRev path only**.


- Subgraph variant is fixed to `rearev_bfs`.
- Core training target is the D pipeline:
  - Phase 1: ReaRev KL objective (optional deep supervision)
  - Phase 2: BCE + ranking + hard negatives

## 1) Repo Layout

- `trm_unified/subgraph_reader.py`: recursive ReaRev model + losses
- `trm_unified/train_core.py`: train/test entry; subgraph-only guard
- `trm_rag_style/scripts/`: runnable scripts for D phase1/phase2
- `trm_agent/`: processed data, embeddings, checkpoints

## 2) Recursive ReaRev Core

`RecursiveSubgraphReader` runs recurrent graph reasoning over each subgraph.
Main state variables are:

- `h_t`: node hidden states
- `p_t`: node distribution (`softmax` over node logits)
- `z_t`: latent reasoning state (when latent enabled)

Per recursion step, the model does:

1. Build step instruction vectors (optionally modulated by `z_t`).
2. Run relation-aware forward/inverse message passing on edges.
3. Update node states (`h_t -> h_{t+1}`), optionally with global gate:
   - gate uses `[h_t ; z_t]` to suppress noisy node updates.
4. Score nodes, optionally with global-local logit fusion:
   - node score uses both local node state and global latent context.
5. Convert scores to distribution `p_{t+1}`.
6. Update latent state with GRU:
   - `z_{t+1} = GRU(context_from(h_{t+1}, p_{t+1}), z_t)`.
7. Optional dynamic halting accumulates stop probability from latent state.

Key knobs:

- `subgraph_recursion_steps`
- `subgraph_rearev_num_ins`
- `subgraph_rearev_adapt_stages`
- `subgraph_rearev_latent_reasoning_enabled`
- `subgraph_rearev_global_gate_enabled`
- `subgraph_rearev_logit_global_fusion_enabled`
- `subgraph_rearev_dynamic_halting_enabled`

## 3) Loss Design (D Pipeline)

### Phase 1 (from scratch): `subgraph_loss_mode=rearev_kl`

Base objective:

- `L = KL(target_node_distribution || predicted_node_distribution)`

Optional deep supervision (this repo addition):

- `L = KL + lambda_ds * (w_ce * L_step_ce + w_halt * L_halt)`

where:

- `L_step_ce`: CE over step-level node logits
- `L_halt`: BCE over step-level halt logits
- `lambda_ds`: `subgraph_deep_supervision_weight`

Note: this is not a separate "latent-only loss". It is an auxiliary supervision term
that also updates latent-related parameters through the recurrent path.

### Phase 2 (resume): `subgraph_loss_mode=bce`

- `L = BCE_hardneg + ranking_weight * L_rank`

with:

- auto/fixed pos-weight options
- hard negative node filtering
- ranking margin loss over hard negatives

## 4) Environment

```bash
cd /data2/workspace/heewon/KGQA
pip install -r requirements.txt
```

## 5) Data and Embeddings

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

## 6) Train / Test Entry

Phase 1 (auto fallback 4->2 GPU on OOM):

```bash
bash trm_rag_style/scripts/run_rearev_d_auto_fallback_4to2.sh
```

Phase 2 (resume from selected phase1 checkpoint):

```bash
bash trm_rag_style/scripts/run_rearev_d_phase2_resume.sh
```

Two-phase automation:

```bash
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
```

Test:

```bash
bash trm_rag_style/scripts/run_test.sh
```

## 7) Deep Supervision Knobs (Phase 1)

- `SUBGRAPH_DEEP_SUPERVISION_ENABLED=true|false`
- `SUBGRAPH_DEEP_SUPERVISION_WEIGHT` (recommended start: `0.03~0.10`)
- `SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT` (default `1.0`)
- `SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT` (default `0.3~1.0`)
- `SUBGRAPH_REAREV_TRM_TMINUS1_NO_GRAD` (`false` for full-step gradient)
- `SUBGRAPH_REAREV_TRM_DETACH_CARRY` (`false` for full-step gradient)

## 8) Quick Recipes (D / D+latent)

### A) D Baseline (two-phase auto)

```bash
cd /data2/workspace/heewon/KGQA
RUN_TAG=d_baseline_4gpu \
WANDB_MODE=online \
WANDB_PROJECT=graph-traverse \
PRIMARY_GPUS=0,1,2,3 PRIMARY_NPROC=4 \
FALLBACK_GPUS=0,1,2,3 FALLBACK_NPROC=4 \
PHASE2_GPUS=0,1,2,3 PHASE2_NPROC_PER_NODE=4 \
MASTER_PORT=29781 PHASE2_MASTER_PORT=29791 \
EPOCHS=16 PHASE2_EPOCHS=5 \
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
```

### B) D+latent (KL + deep supervision in phase1)

```bash
cd /data2/workspace/heewon/KGQA
RUN_TAG=d_latent_ds_4gpu \
WANDB_MODE=online \
WANDB_PROJECT=graph-traverse \
PRIMARY_GPUS=0,1,2,3 PRIMARY_NPROC=4 \
FALLBACK_GPUS=0,1,2,3 FALLBACK_NPROC=4 \
PHASE2_GPUS=0,1,2,3 PHASE2_NPROC_PER_NODE=4 \
MASTER_PORT=29811 PHASE2_MASTER_PORT=29821 \
EPOCHS=16 PHASE2_EPOCHS=5 \
PHASE1_SUBGRAPH_LOSS_MODE=rearev_kl \
PHASE1_SUBGRAPH_DEEP_SUPERVISION_ENABLED=true \
PHASE1_SUBGRAPH_DEEP_SUPERVISION_WEIGHT=0.03 \
PHASE1_SUBGRAPH_DEEP_SUPERVISION_CE_WEIGHT=1.0 \
PHASE1_SUBGRAPH_DEEP_SUPERVISION_HALT_WEIGHT=0.3 \
PHASE1_SUBGRAPH_KL_NO_POSITIVE_MODE=skip \
PHASE1_SUBGRAPH_REAREV_LATENT_REASONING_ENABLED=true \
PHASE1_SUBGRAPH_REAREV_LATENT_RESIDUAL_ALPHA=0.25 \
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
```

### C) Batch=1 for both phases

Use these two variables together:

- `BATCH_SIZE=1`
- `PHASE2_BATCH_SIZE=1`

Optionally keep effective batch with accumulation:

- `SUBGRAPH_GRAD_ACCUM_STEPS=8`
- `PHASE2_SUBGRAPH_GRAD_ACCUM_STEPS=8`

## 9) Path Trace During Test

To print and dump multi-hop explicit paths:

```bash
cd /data2/workspace/heewon/KGQA
CKPT=/path/to/model_epXX.pt
CUDA_VISIBLE_DEVICES=0 \
python -m trm_agent.run \
  --dataset cwq \
  --model_impl trm_hier6 \
  --stage test \
  --ckpt "$CKPT" \
  --override \
    subgraph_reader_enabled=true \
    emb_tag=e5_w4_g4 \
    wandb_mode=offline \
    subgraph_trace_relation_topk_enabled=true \
    subgraph_trace_relation_topk=5 \
    subgraph_trace_log_examples=5 \
    subgraph_trace_dump_max_examples=1000 \
    subgraph_trace_path_dump_jsonl=logs/trace/test_paths.jsonl
```
