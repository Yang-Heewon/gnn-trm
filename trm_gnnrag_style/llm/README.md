# llm module

GNN-RAG 구조를 맞추기 위해 분리한 placeholder입니다.

현재는 TRM 기반 GNN 학습/평가 파이프라인만 구현되어 있고,
향후 아래를 추가하면 됩니다.

- `retrieve_to_prompt.py`: GNN 출력 경로를 LLM 프롬프트 포맷으로 변환
- `generate.py`: 선택한 LLM으로 답변 생성
- `evaluate.py`: QA 메트릭(EM/F1) 계산
