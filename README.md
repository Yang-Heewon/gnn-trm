# GRAPH-TRAVERSE

TRM 기반 그래프 경로 추론 파이프라인입니다.

용어 정리(이름을 이해하기 쉽게):
- `trm_agent`: TRM-agent 실행/오케스트레이션 레이어(권장 이름)
- `trm_agent_pipeline`: 전처리 중심 alias 엔트리포인트
- `trm_unified`: 실제 전처리/임베딩/학습/평가 코어 엔진
- 기존 경로(`trm_rag_style`, `graph_pipeline`)도 레거시 호환으로 유지

## 처음부터 시작하기

1. 저장소 클론
```bash
git clone https://github.com/Yang-Heewon/GRAPH-TRAVERSE.git
cd GRAPH-TRAVERSE
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

3. 데이터 다운로드 + 경로 정리 + 전처리
```bash
bash scripts/setup_and_preprocess.sh
```
- 기본값은 `DATASET=all`이며 `CWQ -> WebQSP` 순서로 전처리합니다.
- `DATASET=cwq` 또는 `DATASET=webqsp`로 단일 데이터셋만 처리할 수 있습니다.
- 기본 Google Drive 파일 URL은 스크립트에 내장되어 있습니다.
- 기본 학습 경로 정책은 `train_path_policy=shortest_only` 입니다(질문당 최단 경로 1개).
- 필요 시 최단 k개 경로:
```bash
TRAIN_PATH_POLICY=shortest_k TRAIN_SHORTEST_K=3 bash scripts/setup_and_preprocess.sh
```

4. 임베딩 생성
```bash
DATASET=webqsp bash trm_agent/scripts/run_embed.sh
```

5. 학습
```bash
DATASET=webqsp MODEL_IMPL=trm_hier6 bash trm_agent/scripts/run_train.sh
```
- `MODEL_IMPL=trm` 또는 `MODEL_IMPL=trm_hier6`
- 기본으로 `eval_no_cycle=true`(재방문 금지)와 `endpoint_aux_weight=0.2`가 적용됩니다.
- 기본 dev 평가는 `200개`, `2epoch마다`, `ep2부터` 수행됩니다.
- 학습 시작 전에 `OracleDiag`가 출력됩니다.
  - `answer_in_subgraph`: 정답 엔티티가 서브그래프에 존재하는 비율
  - `reachable@K`: 현재 `eval_max_steps=K`에서 시작 엔티티로 정답 도달 가능한 비율
  - `reachable_any`: hop 제한 없이 도달 가능한 비율
- 탐색 튜닝 예시:
```bash
DATASET=cwq MODEL_IMPL=trm_hier6 \
EVAL_NO_CYCLE=true EVAL_MAX_STEPS=3 EVAL_BEAM=16 EVAL_START_TOPK=5 \
bash trm_agent/scripts/run_train.sh
```
- OracleDiag 임계치 미만이면 학습을 강제 중단하려면:
```bash
DATASET=cwq MODEL_IMPL=trm_hier6 \
ORACLE_DIAG_FAIL_THRESHOLD=0.30 \
bash trm_agent/scripts/run_train.sh
```
- OracleDiag만 먼저 보고 싶으면:
```bash
DATASET=cwq MODEL_IMPL=trm_hier6 \
ORACLE_DIAG_ONLY=true ORACLE_DIAG_LIMIT=-1 \
bash trm_agent/scripts/run_train.sh
```

- 논문작업 스타일 초기학습(phase1, relation CE 중심)으로 시작:
```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
DATASET=cwq TRAIN_PATH_POLICY=all TRAIN_SHORTEST_K=1 \
bash trm_rag_style/scripts/run_preprocess.sh

DATASET=cwq WANDB_MODE=online WANDB_ENTITY=heewon6205-chung-ang-university \
bash trm_rag_style/scripts/run_train_phase1_paperstyle_wandb.sh
```

6. 테스트
```bash
DATASET=webqsp MODEL_IMPL=trm_hier6 CKPT=/path/to/model_ep1.pt bash trm_agent/scripts/run_test.sh
```
- 전체 test 1회 + no-cycle 평가 예시:
```bash
DATASET=cwq MODEL_IMPL=trm_hier6 CKPT=/path/to/model_ep11.pt \
EVAL_LIMIT=-1 EVAL_NO_CYCLE=true EVAL_MAX_STEPS=3 EVAL_BEAM=16 \
bash trm_agent/scripts/run_test.sh
```

## 실험 자동화
추천 ablation/tuning 순서(4개 core ablation -> stage2 탐색 -> shortest-k 비교)를 한 번에 실행:
```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
CUDA_VISIBLE_DEVICES=0,1,2 \
WANDB_MODE=online WANDB_PROJECT=graph-traverse WANDB_ENTITY=heewon6205-chung-ang-university \
bash scripts/run_cwq_experiment_suite.sh
```

- 결과: `trm_agent/experiments/cwq_suite_*/results.csv`
- 기본 stage2 grid: `eval_max_steps in {2,3,4}`, `eval_beam in {8,16,32}`
- 커스텀 grid 예시:
```bash
S2_MAX_STEPS_GRID=2,3 S2_BEAM_GRID=8,16,24 EPOCHS=8 BATCH_SIZE=6 \
bash scripts/run_cwq_experiment_suite.sh
```

## 전용 프리셋 폴더
`논문작업` 스타일 재현 후 RL phase2를 이어서 돌리는 전용 실행 폴더:
```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
bash experiments/paperstyle_rl/run_all.sh
```

## 한 줄 실행 예시
```bash
python -m trm_agent.run --dataset webqsp --stage all --model_impl trm_hier6
```

같은 명령의 쉬운 이름 버전:
```bash
python -m trm_agent_pipeline.run --dataset webqsp --stage all --model_impl trm_hier6
```

## Repo-Only 로컬 동작 확인(외부 모델 다운로드 없이)
아래 명령은 저장소 내부 코드/데이터만으로 `preprocess -> embed -> train -> test`를 끝까지 실행합니다.

```bash
bash scripts/run_local_demo.sh
```

- `scripts/make_demo_webqsp.py`가 장난감 WebQSP 형식 데이터를 생성합니다.
- 임베딩/토크나이저는 `local-hash` 모드를 사용합니다(원격 Hugging Face 다운로드 없음).

## Notes
- TRM 모듈은 저장소의 `TinyRecursiveModels/`(로컬 사용본)를 사용합니다.
- 데이터 자동 설정 상세: `data/README.md`
- 필요 시 `TRM_ROOT` 환경변수로 TRM 모듈 경로를 재지정할 수 있습니다.
- 전처리 BFS 깊이/경로 수 제어 예시:
```bash
python -m trm_agent.run \
  --dataset cwq \
  --stage preprocess \
  --override max_steps=6 max_paths=8 mine_max_neighbors=256
```
- 전처리 split 정책:
`train`은 BFS로 relation-path를 채굴해 학습에 사용하고, `dev/test`는 endpoint 도달 평가(`Hit@1`, `F1`)용으로 path가 없어도 유지합니다.
- CWQ 먼저, 이후 WebQSP를 같은 설정으로 전처리:
```bash
MAX_STEPS=6 MAX_PATHS=8 MINE_MAX_NEIGHBORS=256 bash scripts/preprocess_cwq_then_webqsp.sh
```
- CPU 사용률을 높이고 tqdm 진행률로 확인:
```bash
MAX_STEPS=4 PREPROCESS_WORKERS=24 bash scripts/setup_and_preprocess.sh
```
- 서브그래프 인덱스 매핑 점검(엔티티/릴레이션/이름):
```bash
python scripts/inspect_subgraph_mapping.py \
  --input data/CWQ/train_split.jsonl \
  --entities_txt data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt \
  --relations_txt data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt \
  --entity_names_json data/data/entities_names.json \
  --index 0 --show_tuples 10
```
