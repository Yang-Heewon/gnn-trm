# Script Guide (D / D+ Only)

Only D and D+ pipelines are supported.

## Supported Variants

- `rearev_bfs` (D)
- `rearev_dplus` (D+)

## Core Scripts

- `../../scripts/run_rearev_r6i5.py`: Python `r6i5` launcher for `cwq` / `webqsp` (`prepare`, train, `test-best`)
- `run_download.sh`: dataset download/prepare
- `run_embed.sh`: embedding build
- `run_rearev_d_auto_fallback_4to2.sh`: phase1 fallback launcher (4GPU -> 2GPU on OOM)
- `run_rearev_d_gpu01_hit06.sh`: D phase1 launcher
- `run_rearev_d_phase2_resume.sh`: D phase2 launcher (resume from phase1 checkpoint)
- `run_rearev_d_two_phase_auto.sh`: D phase1 -> auto best ckpt -> phase2
- `run_rearev_dplus_two_phase_auto.sh`: D+ phase1 -> auto best ckpt -> phase2
- `run_rearev_d_webqsp_two_phase_match_cwq.sh`: WebQSP two-phase launcher
- `run_rearev_d_dplus_ablation_2datasets.sh`: D/D+ 2-dataset ablation launcher
- `run_rearev_d_dplus_ablation_remaining.sh`: run only unfinished ablation cycles
- `run_test.sh`: checkpoint test/eval
- `run_test_best_ckpt.sh`: read `best_dev_f1.txt` or `best_dev_hit1.txt` and run test

## Minimal Training

```bash
cd <repo-root>
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
```

D+:

```bash
cd <repo-root>
bash trm_rag_style/scripts/run_rearev_dplus_two_phase_auto.sh
```

## Test Best Checkpoint

Default metric is `dev_f1`.

```bash
cd <repo-root>
CKPT_DIR=trm_agent/ckpt/<phase2_dir> bash trm_rag_style/scripts/run_test_best_ckpt.sh
```

For `dev_hit1`:

```bash
cd <repo-root>
CKPT_DIR=trm_agent/ckpt/<phase2_dir> METRIC=dev_hit1 bash trm_rag_style/scripts/run_test_best_ckpt.sh
```

If `CKPT_DIR` is omitted, latest phase2 checkpoint directory is used automatically.
