# KGQA Recursive ReaRev (Phase1 Asymmetric y/z)

This repository is documented around a single training path:

- `Phase1-only`
- `asymmetric y/z` recursion enabled
- loss = `KL + Halt BCE`
- no phase2 ranking/BCE-hardneg in this default path

Primary launcher:

- `trm_rag_style/scripts/run_rearev_d_phase1_asym_only.sh`

---

## 1) Scope

This README focuses on the isolated asymmetric latent reasoning module training.

- model core: `trm_unified/subgraph_reader.py`
- train entry: `trm_unified/train_core.py`
- pipeline wrapper: `trm_rag_style/trm_pipeline/train.py`
- phase1-only script: `trm_rag_style/scripts/run_rearev_d_phase1_asym_only.sh`

---

## 2) Model Structure (Asymmetric y/z)

For one sample, with:

- `A = adapt_stages`
- `R = recursion_steps`
- `I = num_instructions`

### Input

- `x_node`: node embeddings
- `x_rel`: relation/edge embeddings
- `x_q`: question embedding

### Projection

- `h0 = node_proj(x_node)`
- `rel = rel_proj(x_rel)`
- `q = q_proj(x_q)`

### State Variables

- `h`: node hidden state
- `z`: global latent recurrent state
- `y`: node-logit answer state

### Asymmetric Loop

1. Initialize `z0` and seed-based `y0`.
2. For each outer stage `s in [1..A]`:
   - Inner recursion (`t in [1..R]`):
     - relation-aware message passing (forward/inverse) with `I` instructions
     - update `h` and `z`
     - keep `y` fixed during inner loop
   - Stage end: update `y` once from `(y_prev, z_updated, h)`.
3. Collect supervised outputs per stage (`step_logits`, `step_halt_logits`).

### Operation Counts Per Sample

- message-passing pair calls: `A * R * I`
- latent (`z`) updates: `A * R`
- answer (`y`) updates in asymmetric mode: `A`

---

## 3) Loss Design (Current Default)

Default mode:

- `subgraph_loss_mode = rearev_kl_halt`
- `subgraph_kl_supervision_mode = step_uniform`

Total objective:

- `L_total = L_KL + w_halt * L_halt`

Where:

- `L_KL`: stage-wise KL supervision (uniform over supervised stages)
- `L_halt`: stage-wise halt BCE supervision
- `w_halt`: `subgraph_rearev_trm_halt_bce_weight`

Backward behavior:

- build one scalar `L_total`
- call `.backward()` once per mini-batch
- optimizer step follows `grad_accum_steps`

---

## 4) Default Training Configuration

The phase1-only launcher defaults to:

- `subgraph_rearev_asymmetric_yz_enabled=true`
- `subgraph_loss_mode=rearev_kl_halt`
- `subgraph_kl_no_positive_mode=skip`
- `subgraph_kl_supervision_mode=step_uniform`
- `subgraph_rearev_trm_style_enabled=true`
- `subgraph_rearev_trm_supervise_all_stages=true`
- `subgraph_ranking_enabled=false`
- `subgraph_bce_hard_negative_enabled=false`

Common structural defaults:

- `subgraph_rearev_adapt_stages=2`
- `subgraph_recursion_steps=4`
- `subgraph_rearev_num_ins=3`

---

## 5) How To Run (Phase1-only)

```bash
cd /data2/workspace/heewon/KGQA
RUN_TAG=phase1_asym_only_v1 \
WANDB_MODE=online \
WANDB_PROJECT=graph-traverse \
PRIMARY_GPUS=0,1,2,3 PRIMARY_NPROC=4 \
FALLBACK_GPUS=0,1,2,3 FALLBACK_NPROC=4 \
MASTER_PORT=29901 FALLBACK_MASTER_PORT=29902 \
bash trm_rag_style/scripts/run_rearev_d_phase1_asym_only.sh
```

Sanity check in startup log (`[SubgraphReader]` line):

- `loss_mode=rearev_kl_halt`
- `rearev_asym_yz=True`
- `kl_sup=step_uniform`

---

## 6) Useful Overrides

You can override at launch time, for example:

```bash
SUBGRAPH_REAREV_ADAPT_STAGES=3 \
SUBGRAPH_RECURSION_STEPS=5 \
SUBGRAPH_REAREV_TRM_HALT_BCE_WEIGHT=1.2 \
SUBGRAPH_GRAD_ACCUM_STEPS=8 \
bash trm_rag_style/scripts/run_rearev_d_phase1_asym_only.sh
```

---

## 7) Path Trace During Test

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

---

## 8) Notes

- This README intentionally does not describe D+ or two-phase recipes.
- If you still use legacy scripts, treat this README as the reference for the current asymmetric phase1 path.
