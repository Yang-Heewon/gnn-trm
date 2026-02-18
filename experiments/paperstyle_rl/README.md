# Paperstyle -> RL Preset

이 폴더는 다음 실험 흐름만 분리해서 실행하기 위한 전용 preset입니다.

1. `논문작업` 스타일 전처리/학습 재현 (phase1, relation CE 중심)
2. phase1 체크포인트에서 RL phase2 추가 파인튜닝

## 빠른 실행

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
bash experiments/paperstyle_rl/run_all.sh
```

## 단계별 실행

```bash
bash experiments/paperstyle_rl/00_preprocess.sh
bash experiments/paperstyle_rl/01_embed.sh
bash experiments/paperstyle_rl/02_train_phase1.sh
bash experiments/paperstyle_rl/03_train_phase2_rl.sh
```

## 주요 환경변수

공통 변수는 `experiments/paperstyle_rl/env.sh`에 있습니다.

- `DATASET` (default: `cwq`)
- `EPOCHS_PHASE1` (default: `30`)
- `EPOCHS_PHASE2` (default: `20`)
- `BATCH_SIZE` (default: `6`)
- `LR` (default: `1e-4`)
- `WANDB_ENTITY` (default: `heewon6205-chung-ang-university`)
- `CKPT_DIR_PHASE1`, `CKPT_DIR_PHASE2`

예시:

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
EPOCHS_PHASE1=20 EPOCHS_PHASE2=10 WANDB_ENTITY=<your_entity> \
bash experiments/paperstyle_rl/run_all.sh
```

## 참고

- phase2 스크립트는 기본으로 phase1의 마지막 체크포인트를 자동 탐색합니다.
- 자동 탐색이 안되면 `PHASE1_CKPT=/path/to/model_epX.pt`를 지정하세요.
