# KGQA Recursive ReaRev (D 2-Phase + Asymmetric y/z)

This repo now supports two runnable experiment tracks:

- Track A: **D baseline (2-phase)**
- Track B: **Asymmetric y/z (phase1-only, KL+Halt)**

You can run each track separately, or run both sequentially.

## 1) Main Scripts

- D baseline (2-phase):
  - `trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh`
- Asymmetric y/z phase1-only:
  - `trm_rag_style/scripts/run_rearev_d_phase1_asym_only.sh`
- Dual launcher (baseline/asym/both):
  - `trm_rag_style/scripts/run_rearev_d_dual_tracks.sh`

## 2) Quick Start

### A) D baseline only (2-phase)

```bash
cd /data2/workspace/heewon/KGQA
RUN_TAG=d_baseline_v1 \
WANDB_MODE=online WANDB_PROJECT=graph-traverse \
PRIMARY_GPUS=0,1,2,3 PRIMARY_NPROC=4 \
FALLBACK_GPUS=0,1 FALLBACK_NPROC=2 \
PHASE2_GPUS=0,1,2,3 PHASE2_NPROC_PER_NODE=4 \
MASTER_PORT=29781 FALLBACK_MASTER_PORT=29782 PHASE2_MASTER_PORT=29791 \
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
```

### B) Asymmetric y/z only (phase1-only)

```bash
cd /data2/workspace/heewon/KGQA
RUN_TAG=asym_phase1_v1 \
WANDB_MODE=online WANDB_PROJECT=graph-traverse \
PRIMARY_GPUS=0,1,2,3 PRIMARY_NPROC=4 \
FALLBACK_GPUS=0,1 FALLBACK_NPROC=2 \
MASTER_PORT=29801 FALLBACK_MASTER_PORT=29802 \
bash trm_rag_style/scripts/run_rearev_d_phase1_asym_only.sh
```

### C) Both tracks sequentially (baseline -> asym)

```bash
cd /data2/workspace/heewon/KGQA
RUN_TAG=dual_v1 \
MODE=both \
WANDB_MODE=online WANDB_PROJECT=graph-traverse \
PRIMARY_GPUS=0,1,2,3 PRIMARY_NPROC=4 \
FALLBACK_GPUS=0,1 FALLBACK_NPROC=2 \
bash trm_rag_style/scripts/run_rearev_d_dual_tracks.sh
```

Run only one mode through dual launcher:

- `MODE=baseline`
- `MODE=asym`
- `MODE=both`

## 3) Training Behavior by Track

### Track A: D baseline (2-phase)

- Phase1 objective: `rearev_kl` (coarse)
- Phase2 objective: `bce + ranking + hard negatives` (fine)
- Script: `run_rearev_d_two_phase_auto.sh`

### Track B: Asymmetric y/z (phase1-only)

- Objective: `rearev_kl_halt`
- KL supervision: `step_uniform`
- Asymmetric update: `subgraph_rearev_asymmetric_yz_enabled=true`
- No phase2 ranking/BCE-hardneg in default path
- Script: `run_rearev_d_phase1_asym_only.sh`

## 4) Asymmetric y/z Module Summary

For one sample with:

- `A = adapt_stages`
- `R = recursion_steps`
- `I = num_instructions`

Flow:

1. Project inputs (`x_node`, `x_rel`, `x_q`) -> (`h0`, `rel`, `q`).
2. Initialize latent `z0` and seed-based answer state `y0`.
3. For each stage:
   - run inner ReaRev recursion `R` times (relation-aware forward/inverse message passing)
   - update `h`, `z`
   - update `y` once at stage end (asymmetric)
4. Compute loss: `L_total = L_KL + w_halt * L_halt`
5. Single backward per mini-batch: `L_total.backward()`

Operation counts per sample:

- message-passing pair calls: `A * R * I`
- `z` updates: `A * R`
- `y` updates (asymmetric): `A`

## 5) Sanity Checks

On startup `[SubgraphReader]` line for asym track, verify:

- `loss_mode=rearev_kl_halt`
- `rearev_asym_yz=True`
- `kl_sup=step_uniform`

For D baseline phase2, verify script log includes phase2 launch and `subgraph_loss_mode=bce`.

## 6) Test Path Trace

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
