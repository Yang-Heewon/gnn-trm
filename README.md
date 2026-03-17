# KGQA (D/D+ Only)

This repo is maintained for two subgraph-reader variants:

- `D`: `rearev_bfs` + latent update `gru`
- `D+`: `rearev_dplus` + latent update `attn`

## Main scripts

- `scripts/run_rearev_r6i5.py`: Python `r6i5` launcher for `cwq` / `webqsp` (`prepare`, two-phase train, `test-best`)
- `trm_rag_style/scripts/run_download.sh`: dataset preparation
- `trm_rag_style/scripts/run_embed.sh`: embedding generation
- `trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh`: D two-phase training (phase1 -> auto best ckpt -> phase2)
- `trm_rag_style/scripts/run_rearev_dplus_two_phase_auto.sh`: D+ two-phase training
- `trm_rag_style/scripts/run_test.sh`: test/eval with a specific checkpoint
- `trm_rag_style/scripts/run_test_best_ckpt.sh`: auto-select best checkpoint from phase2 dir and run test
- `trm_rag_style/scripts/run_rearev_d_dplus_ablation_2datasets.sh`: D/D+ 2-dataset ablation
- `trm_rag_style/scripts/run_rearev_d_dplus_ablation_remaining.sh`: resume only unfinished ablations

## Quick Start

### Python `r6i5` Launcher

For PowerShell / Windows-friendly runs, use the Python launcher instead of the Bash wrappers.

1. Prepare CWQ data + preprocess + embeddings.

```powershell
python scripts/run_rearev_r6i5.py --dataset cwq --mode prepare
```

If you want to stop before training, you can also run the stages separately:

```powershell
python scripts/run_rearev_r6i5.py --dataset cwq --mode preprocess
python scripts/run_rearev_r6i5.py --dataset cwq --mode embed
```

2. Train CWQ with the `r6i5` recipe (`recursion_steps=6`, `num_ins=5`).

```powershell
python scripts/run_rearev_r6i5.py --dataset cwq --mode train
```

3. Run the best phase2 checkpoint on test.

```powershell
python scripts/run_rearev_r6i5.py --dataset cwq --mode test-best --metric dev_f1
```

WebQSP uses the same entrypoint:

```powershell
python scripts/run_rearev_r6i5.py --dataset webqsp --mode all
```

Notes:

- Default `wandb_mode` is `online` in this launcher. Set `--wandb-mode offline` if you want local-only runs.
- Default W&B target is `heewon6205-chung-ang-university/paper_final`. Override with `--wandb-project` or `--wandb-entity` if needed.
- On native Windows the launcher defaults to single-process training (`nproc=1`) for better compatibility.
- Phase1 and Phase2 both include OOM-aware retry logic: after CUDA OOM, the launcher clears cache, waits briefly, retries with fallback `nproc`, and can shrink `max_nodes` / `max_edges` if needed.
- `preprocess_workers` is clamped conservatively so data preparation does not try to occupy every CPU core.
- Use `--dry-run` to print the exact commands without starting a long job.
- CWQ `r6i5` preset:
  `phase1`: 16 epochs, batch size 1, lr `2e-4`, loss `rearev_kl`, `max_nodes=2048`, `max_edges=8192`
  `phase2`: 16 epochs, batch size 1, lr `5e-5`, loss `bce`, `max_nodes=2048`, `max_edges=8192`
  Common: `recursion_steps=6`, `num_ins=5`, `adapt_stages=2`, `gnn_variant=rearev_bfs`, `wandb_mode=online`
- The launcher now keeps W&B runtime/cache files inside the repo `wandb/` directory to avoid Windows permission issues under `%LOCALAPPDATA%`.

## Medical / Custom KGQA

The repo can now ingest repo-ready custom datasets beyond `cwq` / `webqsp` as long as they are converted into the processed JSONL schema used by the subgraph reader.

1. Convert a medical KGQA dataset into repo format.

```powershell
python scripts/prepare_medical_kgqa.py --dataset biohopr --train-in <raw-train.jsonl> --dev-in <raw-dev.jsonl> --test-in <raw-test.jsonl>
```

For `primekgqa` and other KG-style datasets, use the same script with the generic field mapping knobs if your raw field names differ:

```powershell
python scripts/prepare_medical_kgqa.py --dataset primekgqa --train-in <train.jsonl> --seed-field question_entities --answer-field answers --triple-field subgraph.tuples
```

`medhopqa` usually needs entity-linked subgraphs first. Once you have `train/dev/test.jsonl` plus `entities.txt` and `relations.txt`, the repo can train on it through the same custom-dataset path.

2. Run preprocess/embed/train through the generic runner.

```powershell
python -m trm_agent.run --dataset biohopr --stage preprocess
python -m trm_agent.run --dataset biohopr --stage embed --embedding_model <medical-bert-model> --override embed_backend=transformers emb_tag=medicalbert
python -m trm_agent.run --dataset biohopr --stage train --model_impl trm_hier6 --override subgraph_reader_enabled=true
```

Any Hugging Face BERT-style encoder can be used for medical embeddings because the embedder already supports `AutoTokenizer` / `AutoModel` mean-pooling. In practice, medical-domain encoders work best with `embed_backend=transformers`.

The Python launcher can also run converted custom datasets directly:

```powershell
python scripts/run_rearev_r6i5.py --dataset biohopr --mode preprocess
python scripts/run_rearev_r6i5.py --dataset biohopr --mode embed
python scripts/run_rearev_r6i5.py --dataset biohopr --mode prepare --embedding-model <medical-bert-model> --embed-backend transformers
python scripts/run_rearev_r6i5.py --dataset biohopr --mode train --wandb-mode offline
python scripts/run_rearev_r6i5.py --dataset biohopr --mode test-best
```

The same applies to `primekgqa` and `medhopqa` after their converted files are placed under `data/<dataset>/`.

If you first download the raw medical datasets into `data/primekgqa_raw/` or `data/medhopqa_raw/`, the launcher now auto-converts them into `data/<dataset>/` during `--mode preprocess` or `--mode prepare`.

```powershell
python scripts/download_medical_kgqa_data.py --dataset primekgqa
python scripts/run_rearev_r6i5.py --dataset primekgqa --mode prepare --embed-auto-batch-vram-frac 0.6

python scripts/download_medical_kgqa_data.py --dataset medhopqa
python scripts/run_rearev_r6i5.py --dataset medhopqa --mode prepare --embed-auto-batch-vram-frac 0.6
```

The same Python commands work on Linux as well; only the shell prompt differs.

### Bash Pipeline

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
