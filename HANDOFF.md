# 프로젝트 인수인계 문서

## 1. 현재 프로젝트 목적

이 프로젝트는 두 분석 축을 가진 아동 관련 데이터 분석 대시보드이다.

- **ASD 분석(기존 완료 영역)**: Autism Child 데이터의 A1~A10 행동 문항을 바탕으로 ASD 선별 결과를 분석하고, 가중점수 기반 관찰 체크리스트를 제공한다.
- **NSCH 2024 분석(신규 진행 영역)**: National Survey of Children's Health 2024 자료에서 ACE(Adverse Childhood Experiences) 누적 경험을 바탕으로 `ACE_HIGH`를 분석할 예정이다.

현재 Streamlit 앱은 ASD 분석 화면만 제공한다. NSCH는 1단계(데이터 구조 확인·전처리 기준·후보 변수 선정)까지만 완료된 상태다.

## 최신 NSCH 분석 상태 (2026-08-11)

기존 ACE 분석은 `model_artifacts/nsch/`와 `scripts/nsch/analysis_pipeline_nsch.py`에 **보존**한다. 현재 앱의 7·8번 화면은 별도 분석인 `model_artifacts/nsch_asd/`를 사용한다.

- 목표: NSCH 2024에서 **현재 ASD 있음/없음**과 함께 나타나는 가족·경제·학교·생활 특성을 분석한다.
- 대상: 원본 51,375명 중 현재 ASD 여부를 정확히 정할 수 있는 **51,111명**(있음 2,230명 / 없음 48,881명). 264명은 무응답·논리적 건너뜀으로 제외했다.
- 후보: 진단·치료·심각도·ID·표본설계 정보는 제외하고 환경·생활 항목 **17개**를 선정했다.
- 방법: 학습/검증/최종평가를 60/20/20으로 층화 분할하고, 학습 자료 기준 결측 처리·카이제곱/Cramér's V·Spearman·원 변수 단위 랜덤 포레스트 중요도를 적용했다.
- 최종 입력 10개: 친구관계 어려움, 화면 사용 시간, 학교에서 잘하고 싶은 마음, 학교가 가정에 연락한 횟수, 가구 빈곤수준, 보호자 1 고용상태, 보호자 1 최종 학력, 가족 형태, 최근 12개월 가구 식품 상황, 아동의 동네 안전 인식.
- 비교 모델: 로지스틱 회귀, K-최근접 이웃, 의사결정나무, 랜덤 포레스트. 검증 ROC-AUC 기준으로 **로지스틱 회귀**를 선택했다.
- 최종평가: Accuracy 0.9461, Precision 0.3871, Recall 0.4036, F1 0.3952, ROC-AUC 0.8461.
- 재생성: `python scripts/nsch/modeling_nsch_asd.py` → `model_artifacts/nsch_asd/`에 모델·CSV를 생성한다.

## 2. 현재 폴더 구조

```text
TP1/
├─ app.py                              # 기존 ASD Streamlit 대시보드
├─ analysis_pipeline.py                # 기존 ASD 분석 파이프라인
├─ requirements.txt
├─ packages.txt                         # Streamlit Cloud의 한글 Matplotlib 폰트용 fonts-nanum
├─ HANDOFF.md
│
├─ data/
│  ├─ asd/
│  │  └─ Autism-Child-Data.csv
│  └─ nsch/
│     └─ nsch_2024e_topical.dta
│
├─ model_artifacts/
│  ├─ asd/                              # 앱이 읽는 기존 ASD 결과
│  │  └─ archive/                       # 화면 미사용·재현용 ASD 결과
│  └─ nsch/                             # NSCH 1단계 산출물
│
├─ notebooks/
│  ├─ asd/                              # 기존 ASD 분석 노트북 4개
│  └─ nsch/
│     └─ data_check.ipynb
│
├─ scripts/
│  └─ nsch/
│     ├─ analysis_pipeline_nsch.py      # NSCH 전용 1단계 분석 코드
│     └─ nsch_2024_topical.do           # NSCH 원본 Stata 코드북/라벨 참고 파일
│
├─ docs/asd/                            # 기존 교사용 문서와 최종 수정 기록
└─ archive/legacy_backups/               # 기존 ZIP 백업
```

## 3. 기존 ASD 분석에서 이미 완료된 내용

- Autism Child 데이터의 중복·결측 처리 및 EDA
- A1~A10 행동 문항과 ASD 선별 결과의 연관성 확인
- 기존 후보 모델 비교 및 최종 ASD 분석 결과 저장
- A1~A10 중요도를 합계 100점의 가중점수로 환산
- 가중점수 구간별 관찰·상담·전문 선별 연계 안내
- Streamlit 1~7 페이지의 기존 ASD 대시보드 구성
- 배포 환경의 Matplotlib 한글 폰트 대응(`packages.txt`의 `fonts-nanum`)

ASD 앱이 읽는 결과물은 `model_artifacts/asd/`에 있다. 화면에서 직접 쓰지 않는 재현·검증용 결과는 `model_artifacts/asd/archive/`에 보관한다.

## 4. 기존 ASD 로직에서 절대 변경하면 안 되는 부분

다음 항목은 NSCH 작업과 분리하여 **절대 변경하지 않는다**.

1. ASD의 **A1~A10 분석 로직**
2. 기존 ASD의 **가중점수 계산 방식** 및 점수 구간 기준
3. `app.py` 메인 헤딩 제목
   - 현재 제목: **`아동학대 의심 예측 설문조사`**
4. 기존 ASD 분석 결과 파일의 의미와 앱 화면의 기존 동작

NSCH 코드는 반드시 `scripts/nsch/analysis_pipeline_nsch.py`에서 별도로 관리한다. 기존 `analysis_pipeline.py`에 NSCH 로직을 섞지 않는다.

> 아래 5~10절의 ACE 관련 내용은 보존된 이전 분석 기록이다. 현재 앱 화면과 신규 분석에는 사용하지 않는다.

## 5. NSCH 2024 분석에서 현재까지 완료한 내용

대상 데이터: `data/nsch/nsch_2024e_topical.dta`

- 원본 행/열 수: **51,375행 / 457열**
- 중복 행 수: **0행**
- 9개 ACE 문항이 모두 정상 응답인 `ACE_HIGH` 생성 가능 행: **48,042행**
- `ACE_HIGH = 0`: **39,975행**
- `ACE_HIGH = 1`: **8,067행**
- 의미 기준으로 선정한 1차 후보 컬럼: **27개**
- 후보 컬럼의 최대 결측률: 약 **29.26%**
- 머신러닝 학습은 아직 수행하지 않음

후보 변수 영역:

- 아동 기본정보
- ASD, ADHD, 불안, 우울, 행동문제
- 가족/보호자 환경
- 경제 상황
- 주거 환경, 동네 안전, 이사 경험
- 수면, 스크린타임
- 학교/사회생활

주의: 초기에는 ‘현재 진단 상태’ 후속 문항을 후보로 보았으나, 논리적 건너뜀으로 결측이 85~95%여서 제외했다. 현재 후보에는 전체 응답자가 답하는 ‘진단 경험 여부’ 문항(`k2q35a`, `k2q31a`, `k2q33a`, `k2q32a`, `k2q34a`)을 사용한다.

## 6. ACE_HIGH 생성 기준

대상 ACE 열:

```text
ace1, ace3, ace4, ace5, ace6, ace7, ace8, ace9, ace10
```

정상 응답과 이진화 기준:

- `ace1` (생활 기본비용을 감당하기 어려웠던 빈도)
  - 정상 응답: 1=Never, 2=Rarely, 3=Somewhat often, 4=Very often
  - ACE 경험=1: **3 또는 4**
  - ACE 경험=0: 1 또는 2
- `ace3~ace10`
  - 정상 응답: 1=Yes, 2=No
  - ACE 경험=1: **1(Yes)**
  - ACE 경험=0: 2(No)

9개 ACE 문항이 모두 정상 응답인 행에서만 `ACE_COUNT`를 계산한다.

```text
ACE_COUNT 0~1개  → ACE_HIGH = 0
ACE_COUNT 2개 이상 → ACE_HIGH = 1
```

`ace1`, `ace3~ace10`, `ACE_COUNT`, `ACE_HIGH`는 Target을 직접 구성하므로 **X에 절대 포함하지 않는다**. 상세 기준은 `analysis_pipeline_nsch.py`의 주석과 `model_artifacts/nsch/leakage_features.csv`에 있다.

## 7. 생성된 NSCH 산출물과 용도

모든 파일은 `model_artifacts/nsch/`에 있다.

| 파일 | 용도 |
| --- | --- |
| `data_summary.csv` | 원본 행·열 수, 중복 수, ACE 완전응답 행 수, 후보 수 요약 |
| `missing_summary.csv` | 457개 전체 컬럼의 dtype, 결측 개수·비율 |
| `candidate_features.csv` | 27개 후보의 컬럼명, 한국어 설명, 영역, 사용 여부, 선정 이유, 결측 정보 |
| `excluded_features.csv` | 후보에서 제외한 430개 컬럼과 제외 분류·이유 |
| `target_summary.csv` | ACE 문항별 값 분포, ACE 이진화 기준, ACE_HIGH 분포 |
| `leakage_features.csv` | X 사용 금지인 ACE 원문항·합계·Target 목록 |
| `.gitkeep` | NSCH 산출물 폴더 유지용 파일 |

## 8. 아직 하지 않은 작업

- NSCH 후보 변수의 결측치 처리 기준 확정 및 적용
- ACE_HIGH와 후보 변수의 통계 검정
- 통계 결과를 근거로 한 최종 변수 선정
- Logistic Regression / KNN / Decision Tree / Random Forest 비교
- 최종 NSCH 모델 선정과 성능 평가
- Streamlit 대시보드에 NSCH 분석 과정 추가
- 생활환경 설문 추가
- 기존 ASD 점수와 NSCH 점수를 종합한 최종 판정 설계

## 9. 다음 작업 순서

다음 순서를 지킨다.

1. **NSCH 후보 변수 결측치 처리**
2. **통계 검정**
3. **변수 선정**
4. **Logistic/KNN/Decision Tree/Random Forest 비교**
5. **최종 NSCH 모델 선정**
6. **대시보드에 NSCH 분석 과정 추가**
7. **생활환경 설문 추가**
8. **기존 ASD 점수와 NSCH 점수를 종합판정**

NSCH 모델링을 시작할 때도 Target leakage 목록의 열은 절대 X에 넣지 않는다. 후보 변수의 결측은 무조건 0으로 채우지 말고, 값 의미·논리적 건너뜀·연령 적용 범위를 먼저 검토한다.

## 10. 코드 실행 경로와 주의사항

프로젝트 루트에서 실행한다.

```powershell
cd C:\Users\Admin\mbca\TP1F1\TP1
```

### ASD 앱

```powershell
streamlit run app.py
```

`app.py`가 읽는 경로:

- 원본 ASD 데이터: `data/asd/Autism-Child-Data.csv`
- ASD 산출물: `model_artifacts/asd/`

### ASD 파이프라인 재생성

```powershell
python analysis_pipeline.py
```

`analysis_pipeline.py`는 `data/asd/Autism-Child-Data.csv`를 읽고 `model_artifacts/asd/`에 결과를 생성한다.

### NSCH 1단계 산출물 재생성

```powershell
python scripts/nsch/analysis_pipeline_nsch.py
```

NSCH 스크립트가 읽는 경로:

- 원본: `data/nsch/nsch_2024e_topical.dta`
- 코드북 참고: `scripts/nsch/nsch_2024_topical.do`
- 출력: `model_artifacts/nsch/`

### 의존성 및 배포 주의사항

- Python 의존성은 `requirements.txt`에 있다.
- Streamlit Cloud 배포 시 `packages.txt`의 `fonts-nanum`을 유지해야 Matplotlib 그래프 한글이 깨지지 않는다.
- Git에 올릴 때 대용량 원본 데이터와 생성 산출물의 포함 여부는 팀의 저장소 정책을 확인한다.
- 기존 ASD 파일·제목·계산 로직을 NSCH 구현 편의를 위해 변경하지 않는다.
