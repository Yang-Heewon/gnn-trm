# Data Setup (Google Drive)

아래 스크립트는 제공된 Google Drive 폴더에서 데이터를 내려받고, 저장소 표준 경로로 자동 정리합니다.

- 기본 폴더 URL:
  - `https://drive.google.com/drive/folders/1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp?usp=sharing`

## 1) 다운로드 + 경로 정리
```bash
bash scripts/download_data.sh
```

필수:
- `gdown` 설치 필요 (`pip install gdown`)

옵션:
- 다른 폴더 URL 사용
```bash
GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/...." bash scripts/download_data.sh
```
- Google Drive 다운로드 건너뛰고 직접 URL만 사용
```bash
SKIP_GDRIVE=1 WEBQSP_URL="https://..." CWQ_URL="https://..." bash scripts/download_data.sh
```

## 2) 전처리까지 한 번에
```bash
DATASET=webqsp bash scripts/setup_and_preprocess.sh
```

- `DATASET=cwq` 또는 `DATASET=all` 가능

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
