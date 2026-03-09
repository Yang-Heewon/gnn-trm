# Script Guide (D / D+ Only)

Only D and D+ pipelines are supported.

## Supported variants

- `rearev_bfs` (D)
- `rearev_dplus` (D+)

## Core scripts

- `run_download.sh`: dataset download/prepare
- `run_embed.sh`: embedding build
- `run_rearev_d_auto_fallback_4to2.sh`: phase1 fallback launcher
- `run_rearev_d_gpu01_hit06.sh`: D phase1 launcher
- `run_rearev_d_phase2_resume.sh`: D phase2 launcher
- `run_rearev_d_two_phase_auto.sh`: D phase1->best ckpt->phase2
- `run_rearev_dplus_two_phase_auto.sh`: D+ phase1->best ckpt->phase2
- `run_rearev_d_webqsp_two_phase_match_cwq.sh`: WebQSP two-phase launcher
- `run_rearev_d_dplus_ablation_2datasets.sh`: 2-dataset paired ablation (D + D+, total 32 train stages)
- `run_test.sh`: checkpoint test/eval

## Minimal run

```bash
cd /data2/workspace/heewon/KGQA
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
```

D+:

```bash
cd /data2/workspace/heewon/KGQA
bash trm_rag_style/scripts/run_rearev_dplus_two_phase_auto.sh
```
