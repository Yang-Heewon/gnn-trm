# trm_rag_style

`TRM-RAG` 저장소 사용감에 맞춘 TRM 파이프라인입니다.

- 구조 분리: `trm_pipeline/`(전처리/임베딩/학습/테스트), `llm/`(확장용)
- 설정 분리: `configs/*.json`
- 실행 오케스트레이터: `python -m trm_rag_style.run`

## 폴더 구조
- `configs/`
  - `base.json`: 공통 하이퍼파라미터
  - `cwq.json`, `webqsp.json`: 데이터셋 경로
  - `train_trm.json`, `train_trm_hier6.json`: 모델별 기본값
- `trm_pipeline/`
  - `preprocess.py`, `embed.py`, `train.py`, `test.py`
- `llm/`
  - LLM 확장 placeholder (`prepare_stub.py`)
- `scripts/`
  - `run_preprocess.sh`, `run_embed.sh`, `run_train.sh`, `run_test.sh`, `run_all.sh`

## 빠른 실행
### 1) 전처리 (CWQ/WebQSP)
```bash
DATASET=webqsp bash trm_rag_style/scripts/run_preprocess.sh
DATASET=cwq bash trm_rag_style/scripts/run_preprocess.sh
```

### 2) 임베딩 모델 교체 가능
```bash
DATASET=webqsp EMB_MODEL=intfloat/multilingual-e5-large bash trm_rag_style/scripts/run_embed.sh
DATASET=webqsp EMB_MODEL=BAAI/bge-base-en-v1.5 bash trm_rag_style/scripts/run_embed.sh
```

### 3) 학습 (TRM / TRM6)
```bash
DATASET=webqsp MODEL_IMPL=trm_hier6 bash trm_rag_style/scripts/run_train.sh
DATASET=cwq MODEL_IMPL=trm bash trm_rag_style/scripts/run_train.sh
```

### 4) 체크포인트 테스트
```bash
DATASET=webqsp MODEL_IMPL=trm_hier6 CKPT=/path/to/model_ep1.pt bash trm_rag_style/scripts/run_test.sh
```

## 단일 명령 오케스트레이션
```bash
python -m trm_rag_style.run --dataset webqsp --stage all --model_impl trm_hier6
```

## override 예시
공통 config를 실행 시점에 덮어쓸 수 있습니다.
```bash
python -m trm_rag_style.run \
  --dataset webqsp \
  --stage train \
  --model_impl trm_hier6 \
  --override hidden_size=768 num_heads=12 epochs=10 lr=5e-5
```

## 참고
- 내부 코어 로직은 `trm_unified` 모듈을 재사용합니다.
- 기존 `trm_gnnrag_style` 경로는 호환용 alias로 유지했습니다.
