# KGQA (D/D+ Only)

This repository is now maintained for two subgraph-reader variants only:

- `D`: `rearev_bfs` + latent update `gru`
- `D+`: `rearev_dplus` + latent update `attn`

All E/official and non-D/D+ launch paths were removed from this repo workflow.

## Main run scripts

- `trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh`
- `trm_rag_style/scripts/run_rearev_dplus_two_phase_auto.sh`
- `trm_rag_style/scripts/run_rearev_d_webqsp_two_phase_match_cwq.sh`
- `trm_rag_style/scripts/run_rearev_d_dplus_ablation_2datasets.sh`

## Quick start

```bash
cd /data2/workspace/heewon/KGQA
bash trm_rag_style/scripts/run_rearev_d_two_phase_auto.sh
```

D+ run:

```bash
cd /data2/workspace/heewon/KGQA
bash trm_rag_style/scripts/run_rearev_dplus_two_phase_auto.sh
```

32-stage ablation run (D and D+, cwq and webqsp, paired configs):

```bash
cd /data2/workspace/heewon/KGQA
bash trm_rag_style/scripts/run_rearev_d_dplus_ablation_2datasets.sh
```
