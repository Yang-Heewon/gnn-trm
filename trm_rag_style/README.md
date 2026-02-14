# trm_rag_style

TRM-RAG 실행 오케스트레이터입니다.

## 전제
- 저장소 루트에서 실행
- 기본 TRM 모듈은 저장소의 `TinyRecursiveModels`
- 데이터 파일이 `data/` 아래 존재

## 데이터 준비 + 전처리
```bash
DATASET=webqsp bash scripts/setup_and_preprocess.sh
```

## 기본 실행
```bash
DATASET=webqsp bash trm_rag_style/scripts/run_embed.sh
DATASET=webqsp MODEL_IMPL=trm_hier6 bash trm_rag_style/scripts/run_train.sh
DATASET=webqsp MODEL_IMPL=trm_hier6 CKPT=/path/to/model_ep1.pt bash trm_rag_style/scripts/run_test.sh
```

## 단일 엔트리포인트
```bash
python -m trm_rag_style.run --dataset webqsp --stage all --model_impl trm_hier6
```
