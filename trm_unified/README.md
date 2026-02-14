# trm_unified

CWQ/WebQSP를 같은 파이프라인으로 처리하는 독립 폴더입니다. 기존 스크립트는 건드리지 않고 별도 실행합니다.

## 구성
- `pipeline.py`: 단일 엔트리포인트 (`preprocess`, `embed`, `train`, `test`)
- `data.py`: 데이터 표준화 + WebQSP BFS path mining
- `embedder.py`: entity/relation/query 임베딩 생성 (HF 모델 선택)
- `train_core.py`: TRM 학습/체크포인트/테스트 로더

## 핵심 기능
- CWQ/WebQSP 전처리 분리
- WebQSP는 `valid_paths`가 없으므로 `start entities -> answer entities` BFS로 path 감독 신호 생성
- 임베딩 모델 교체 가능 (`--model_name`)
- 모델 구현 선택 가능 (`--model_impl trm_hier6|trm`)
- resume 가능 (`train --ckpt ...`)
- 테스트 로더 제공 (`test --ckpt ...`)

## 1) 전처리
### CWQ
```bash
python -m trm_unified.pipeline preprocess \
  --dataset cwq \
  --train_in /data2/workspace/heewon/논문작업/data/CWQ/train_split.jsonl \
  --dev_in   /data2/workspace/heewon/논문작업/data/CWQ/dev_split.jsonl \
  --entities_txt /data2/workspace/heewon/논문작업/data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt \
  --out_dir /data2/workspace/heewon/논문작업/trm_unified/processed/cwq
```

### WebQSP
```bash
python -m trm_unified.pipeline preprocess \
  --dataset webqsp \
  --train_in /data2/workspace/heewon/논문작업/data/webqsp/train.json \
  --dev_in   /data2/workspace/heewon/논문작업/data/webqsp/dev.json \
  --entities_txt /data2/workspace/heewon/논문작업/data/webqsp/entities.txt \
  --out_dir /data2/workspace/heewon/논문작업/trm_unified/processed/webqsp \
  --max_steps 4 --max_paths 4 --mine_max_neighbors 128
```

## 2) 임베딩 생성
예: E5-large
```bash
python -m trm_unified.pipeline embed \
  --model_name intfloat/multilingual-e5-large \
  --entities_txt /data2/workspace/heewon/논문작업/data/webqsp/entities.txt \
  --relations_txt /data2/workspace/heewon/논문작업/data/webqsp/relations.txt \
  --train_jsonl /data2/workspace/heewon/논문작업/trm_unified/processed/webqsp/train.jsonl \
  --dev_jsonl   /data2/workspace/heewon/논문작업/trm_unified/processed/webqsp/dev.jsonl \
  --out_dir /data2/workspace/heewon/논문작업/trm_unified/emb/webqsp_e5
```

다른 모델도 `--model_name`만 바꾸면 됩니다 (예: `BAAI/bge-base-en-v1.5`).

## 3) 학습
TRM6:
```bash
torchrun --nproc_per_node=3 --master_port=29500 -m trm_unified.pipeline train \
  --model_impl trm_hier6 \
  --train_json /data2/workspace/heewon/논문작업/trm_unified/processed/webqsp/train.jsonl \
  --entities_txt /data2/workspace/heewon/논문작업/data/webqsp/entities.txt \
  --relations_txt /data2/workspace/heewon/논문작업/data/webqsp/relations.txt \
  --entity_emb_npy /data2/workspace/heewon/논문작업/trm_unified/emb/webqsp_e5/entity_embeddings.npy \
  --relation_emb_npy /data2/workspace/heewon/논문작업/trm_unified/emb/webqsp_e5/relation_embeddings.npy \
  --query_emb_train_npy /data2/workspace/heewon/논문작업/trm_unified/emb/webqsp_e5/query_train.npy \
  --out_dir /data2/workspace/heewon/논문작업/trm_unified/ckpt/webqsp_trm6
```

TRM:
```bash
torchrun --nproc_per_node=3 --master_port=29501 -m trm_unified.pipeline train \
  --model_impl trm \
  --train_json /data2/workspace/heewon/논문작업/trm_unified/processed/cwq/train.jsonl \
  --entities_txt /data2/workspace/heewon/논문작업/data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt \
  --relations_txt /data2/workspace/heewon/논문작업/data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt \
  --entity_emb_npy /data2/workspace/heewon/논문작업/data/CWQ/embeddings_output/CWQ/e5/entity_embeddings.npy \
  --relation_emb_npy /data2/workspace/heewon/논문작업/data/CWQ/embeddings_output/CWQ/e5/relation_embeddings.npy \
  --query_emb_train_npy /data2/workspace/heewon/논문작업/data/CWQ/embeddings_output/CWQ/e5/query_train.npy \
  --out_dir /data2/workspace/heewon/논문작업/trm_unified/ckpt/cwq_trm
```

resume:
```bash
... -m trm_unified.pipeline train ... --ckpt /path/to/model_ep3.pt
```

## 4) 테스트(체크포인트 로딩 검증)
```bash
python -m trm_unified.pipeline test \
  --model_impl trm_hier6 \
  --ckpt /data2/workspace/heewon/논문작업/trm_unified/ckpt/webqsp_trm6/model_ep1.pt \
  --entities_txt /data2/workspace/heewon/논문작업/data/webqsp/entities.txt \
  --relations_txt /data2/workspace/heewon/논문작업/data/webqsp/relations.txt \
  --entity_emb_npy /data2/workspace/heewon/논문작업/trm_unified/emb/webqsp_e5/entity_embeddings.npy \
  --relation_emb_npy /data2/workspace/heewon/논문작업/trm_unified/emb/webqsp_e5/relation_embeddings.npy
```

## 주의
- HF 모델 다운로드가 필요할 수 있습니다.
- TinyRecursiveModels 경로는 기본값 `/data2/workspace/heewon/논문작업/TinyRecursiveModels`입니다.
- hidden size/head/seq_len 등은 `train` 인자로 재현 가능하게 전부 노출했습니다.
