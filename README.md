# GRAPH-TRAVERSE

TRM 기반 그래프 경로 추론 파이프라인입니다.

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

4. 임베딩 생성
```bash
DATASET=webqsp bash trm_rag_style/scripts/run_embed.sh
```

5. 학습
```bash
DATASET=webqsp MODEL_IMPL=trm_hier6 bash trm_rag_style/scripts/run_train.sh
```
- `MODEL_IMPL=trm` 또는 `MODEL_IMPL=trm_hier6`

6. 테스트
```bash
DATASET=webqsp MODEL_IMPL=trm_hier6 CKPT=/path/to/model_ep1.pt bash trm_rag_style/scripts/run_test.sh
```

## 한 줄 실행 예시
```bash
python -m trm_rag_style.run --dataset webqsp --stage all --model_impl trm_hier6
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
python -m trm_rag_style.run \
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
- 서브그래프 인덱스 매핑 점검(엔티티/릴레이션/이름):
```bash
python scripts/inspect_subgraph_mapping.py \
  --input data/CWQ/train_split.jsonl \
  --entities_txt data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt \
  --relations_txt data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt \
  --entity_names_json data/data/entities_names.json \
  --index 0 --show_tuples 10
```
