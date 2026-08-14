# 아동학대 사전 예측 솔루션

아동센터 선생님이 아이의 ASD 관련 행동 특성과 생활환경 위험신호를 함께 확인하고, 추가 관찰의 우선순위를 정할 수 있도록 만든 Streamlit 프로젝트입니다.

이 프로젝트는 아동학대 여부나 ASD를 직접 진단하는 시스템이 아닙니다. 서로 다른 조사 대상에서 수집된 두 데이터를 각각 분석하고, 웹 화면에서 두 관찰점수를 함께 제공합니다.

## 분석 구성

- UCI 아동 ASD 행동 선별 데이터: 292명 중 중복 2명을 제외한 290명 사용
- NSCH 2024: 4~11세 중 ACE 9개 문항에 유효하게 응답한 19,304명 사용
- 행동 설문: A1~A10과 로지스틱 회귀계수 기반 가중점수
- 생활환경 설문: ACE 0~3개 집단과 4개 이상 집단을 비교해 선정한 8문항
- 생활환경 점수: 랜덤 포레스트 문항 중요도와 응답 차이를 반영한 0~100점
- 종합 결과: 두 점수의 평균과 각 영역의 높고 낮음을 조합한 2×2 관찰 유형

## 실행 방법

Python 3.12 환경을 권장합니다.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 주요 파일

- `app.py`: Streamlit 대시보드와 체크리스트
- `analysis_pipeline.py`: UCI 데이터 분석 및 모델링
- `scripts/nsch/modeling_nsch_ace4.py`: NSCH ACE 4개 이상 위험신호 분석
- `model_artifacts/asd/`: 행동 설문 모델과 분석 결과
- `model_artifacts/nsch_ace4_8q/`: 최종 생활환경 8문항 모델과 분석 결과
- `data/`: 분석 재현용 원자료

## 최종 모델

- 행동 특성: Logistic Regression
- 생활환경 위험신호: Random Forest

저장된 scikit-learn 모델과의 버전 충돌을 방지하기 위해 `requirements.txt`에서 scikit-learn 1.9.0을 고정했습니다.
