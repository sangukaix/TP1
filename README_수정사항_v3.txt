TP2 연관성 보정 v3 수정사항
============================

1) 3번 전체 요인 연관성 분석
- ethnicity, country_of_res는 Selection Train에서 10건 미만 범주를 'Other (rare)'로 통합했습니다.
- 범주형 변수의 연관성 크기는 일반 Cramér's V 대신 Bias-corrected Cramér's V를 사용했습니다.
- age는 Welch t-test와 절대값 point-biserial r을 사용합니다.
- 보정 후 p<0.05 요인은 A1~A10 행동 문항 10개이며, 개인·배경 요인은 없습니다.
- 거주 국가가 범주 수 때문에 과도하게 커 보이던 문제를 줄였습니다.

2) 4번 머신러닝 모델 비교
- 3번 보정 분석 결과를 그대로 이어 받아 p<0.05 요인만 입력 후보로 사용합니다.
- 최종 입력은 A1~A10 10개입니다.
- Logistic Regression / KNN / Decision Tree / Random Forest를 같은 Validation 조건에서 비교합니다.
- 현재 Validation에서는 Logistic Regression이 가장 높게 나옵니다.
- A1~A10이 Class/ASD 생성 규칙에 직접 포함되므로 높은 성능은 임상 진단 정확도가 아니라 선별 규칙 재현의 영향이 큽니다.

3) 7번 결론 및 활용
- 기존 '관찰·연계 매뉴얼'을 'ASD 선별 예측 참고 지표' 중심으로 변경했습니다.
- 머신러닝 중요도 상위 5개 행동 문항과 관찰 가능한 특징을 표시합니다.
- 행동 설명은 Autism Research Centre AQ-10 Child 문항을 한국어로 요약했습니다.
- Validation에서 YES 비율 90% 이상을 처음 충족한 '상위 5개 중 4개 이상' 기준을 탐색적 참고 규칙으로 선택했습니다.
- Final Test에서는 이 조건에 해당한 26명 중 24명(92.3%)이 Class/ASD YES였습니다.
- 이 92.3%는 자폐증 진단 확률이 아니라 현재 데이터의 Class/ASD 선별 레이블 비율입니다.
- 현재 데이터에는 아동학대 Target이 없으므로 '4개 이상이면 아동학대 가능성'이라고 해석하지 않습니다. 학대 여부는 별도의 학대 징후와 기관 절차로 판단해야 합니다.

4) 재현 파일
- analysis_pipeline.py: 전체 분석 재실행
- app.py: Streamlit 대시보드
- model_artifacts/: 분석 결과 파일
- 03_모델링_평가.ipynb에서 analysis_pipeline.py 전체 실행 가능
