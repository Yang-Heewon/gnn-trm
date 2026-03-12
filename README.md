# KGQA (D/D+ Only)

This repo is maintained for two subgraph-reader variants:

- `D`: `rearev_bfs` + latent update `gru`
- `D+`: `rearev_dplus` + latent update `attn`

## Main scripts

- `trm_rag_style/scripts/run_download.sh`: dataset preparation
- `trm_rag_style/scripts/run_embed.sh`: embedding generation
- `trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh`: D two-phase training (phase1 -> auto best ckpt -> phase2)
- `trm_rag_style/scripts/run_rearev_dplus_two_phase_auto.sh`: D+ two-phase training
- `trm_rag_style/scripts/run_test.sh`: test/eval with a specific checkpoint
- `trm_rag_style/scripts/run_test_best_ckpt.sh`: auto-select best checkpoint from phase2 dir and run test
- `trm_rag_style/scripts/run_rearev_d_dplus_ablation_2datasets.sh`: D/D+ 2-dataset ablation
- `trm_rag_style/scripts/run_rearev_d_dplus_ablation_remaining.sh`: resume only unfinished ablations

## Quick Start

1. Prepare data.

```bash
cd <repo-root>
bash trm_rag_style/scripts/run_download.sh
```

2. Build embeddings.

```bash
cd <repo-root>
bash trm_rag_style/scripts/run_embed.sh
```

3. Train D (two-phase).

```bash
cd <repo-root>
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
```

4. Test the best phase2 checkpoint (`dev_f1` by default).

```bash
cd <repo-root>
CKPT_DIR=trm_agent/ckpt/<your_phase2_dir> bash trm_rag_style/scripts/run_test_best_ckpt.sh
```

`run_rearev_d_two_phase_auto.sh` writes:

- `best_dev_f1.txt`
- `best_dev_hit1.txt`

inside the phase2 checkpoint directory, and `run_test_best_ckpt.sh` uses these files.

## D+ Training

```bash
cd <repo-root>
bash trm_rag_style/scripts/run_rearev_dplus_two_phase_auto.sh
```

## Large Ablation

```bash
cd <repo-root>
bash trm_rag_style/scripts/run_rearev_d_dplus_ablation_2datasets.sh
```

Resume only unfinished cycles:

```bash
cd <repo-root>
bash trm_rag_style/scripts/run_rearev_d_dplus_ablation_remaining.sh
```
