# GRAPH-TRAVERSE

TRM 기반 그래프 경로 추론 파이프라인입니다.

## Quick Start

1. 의존성 설치
```bash
pip install -r requirements.txt
```

2. 데이터 다운로드 + 전처리
```bash
DATASET=webqsp bash scripts/setup_and_preprocess.sh
```

3. 학습
```bash
DATASET=webqsp MODEL_IMPL=trm_hier6 bash trm_rag_style/scripts/run_train.sh
```

## Notes
- TRM 모듈은 저장소의 `TinyRecursiveModels/`(로컬 사용본)를 사용합니다.
- 데이터 자동 설정 상세: `data/README.md`
- 필요 시 `TRM_ROOT` 환경변수로 TRM 모듈 경로를 재지정할 수 있습니다.
