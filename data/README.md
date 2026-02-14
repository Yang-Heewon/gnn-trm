# Data Setup (Google Drive)

아래 스크립트는 제공된 Google Drive 파일에서 데이터를 내려받고, 저장소 표준 경로로 자동 정리합니다.

- 기본 파일 URL:
  - `https://drive.google.com/file/d/13DduGb1C-O6udi744WxNVnOVDJ6122JG/view?usp=sharing`

## 1) 다운로드 + 경로 정리
```bash
bash scripts/download_data.sh
```

필수:
- `gdown` 설치 필요 (`pip install gdown`)

옵션:
- 다른 파일 URL 사용
```bash
GDRIVE_FILE_URL="https://drive.google.com/file/d/.../view?usp=sharing" bash scripts/download_data.sh
```
- 기존 폴더 URL을 계속 쓰고 싶다면
```bash
GDRIVE_FILE_URL="" GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/...." bash scripts/download_data.sh
```
- Google Drive 다운로드를 건너뛰고 직접 URL만 사용
```bash
SKIP_GDRIVE=1 WEBQSP_URL="https://..." CWQ_URL="https://..." bash scripts/download_data.sh
```

## 2) 전처리까지 한 번에
```bash
DATASET=webqsp bash scripts/setup_and_preprocess.sh
```

- `DATASET=cwq` 또는 `DATASET=all` 가능
- `DATASET=all`은 `CWQ -> WebQSP` 순서로 전처리합니다.
- BFS 깊이/경로 개수 조절은 `--override`로 가능:
```bash
python -m trm_rag_style.run --dataset cwq --stage preprocess --override max_steps=6 max_paths=8 mine_max_neighbors=256
```

## 전처리 정책 (요청 반영)
- `train`: 질문 + 시작 엔티티(`entities`) + 서브그래프(`subgraph.tuples`)에서 BFS로 `valid_paths`를 채굴하고, relation 경로(`relation_paths`)를 학습 신호로 사용합니다.
- `dev/test`: 시작 엔티티에서 정답 엔티티(`answers_cid`) 도달 과제를 위해 샘플을 유지합니다. 이 split은 path가 없어도 drop하지 않으며, 평가는 endpoint 도달 기준으로 `Hit@1`, `F1`을 사용합니다.
- BFS depth는 `max_steps`, 경로 수는 `max_paths`, 이웃 제한은 `mine_max_neighbors`로 제어합니다.

## 튜플 숫자 매핑 규칙
- 예시 튜플: `[303793, 2327, 303791]`
- 첫 번째/세 번째 숫자(주어/목적어): `entities.txt`의 줄 인덱스입니다. 해당 줄의 엔티티 ID를 키로 `data/data/entities_names.json`에서 이름으로 매핑할 수 있습니다.
- 두 번째 숫자(관계): `relations.txt`의 줄 인덱스이며, 해당 줄 텍스트가 relation 문자열입니다.

검증 도구:
```bash
python scripts/inspect_subgraph_mapping.py \
  --input data/CWQ/train_split.jsonl \
  --entities_txt data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt \
  --relations_txt data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt \
  --entity_names_json data/data/entities_names.json \
  --index 0 --show_tuples 10
```

## 기대 경로
WebQSP:
- `data/webqsp/train.json`
- `data/webqsp/dev.json`
- `data/webqsp/entities.txt`
- `data/webqsp/relations.txt`

CWQ:
- `data/CWQ/train_split.jsonl`
- `data/CWQ/dev_split.jsonl`
- `data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt`
- `data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt`
