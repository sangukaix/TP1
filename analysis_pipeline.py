from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "asd" / "Autism-Child-Data.csv"
ART = ROOT / "model_artifacts" / "asd"
ART.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "Class/ASD"
BEHAVIOR = [f"A{i}_Score" for i in range(1, 11)]
BACKGROUND = [
    "age",
    "gender",
    "ethnicity",
    "jaundice",
    "family_asd",
    "country_of_res",
    "used_app_before",
    "relation",
]
ALL_CANDIDATES = BEHAVIOR + BACKGROUND

# 범주 수가 많은 변수는 작은 표본 범주를 Other로 묶어 희소 범주 문제를 줄인다.
RARE_GROUP_FEATURES = ["ethnicity", "country_of_res"]
RARE_MIN_COUNT = 10
RARE_LABEL = "Other (rare)"

# AQ-10 Child 문항을 교사가 관찰하기 쉬운 ASD 특성 방향으로 요약한 표현
AQ10_OBSERVATION = {
    "A1_Score": "다른 사람이 잘 알아차리지 못하는 작은 소리를 자주 알아차림",
    "A2_Score": "전체보다 작은 세부사항에 더 집중하는 경향",
    "A3_Score": "여러 사람의 대화를 동시에 따라가기 어려움",
    "A4_Score": "한 활동에서 다른 활동으로 전환하기 어려움",
    "A5_Score": "또래와 대화를 계속 이어가는 방법을 어려워함",
    "A6_Score": "가벼운 사회적 대화(잡담)를 어려워함",
    "A7_Score": "이야기 속 인물의 의도나 감정을 파악하기 어려움",
    "A8_Score": "또래와 가상놀이·역할놀이를 즐기는 경향이 적음",
    "A9_Score": "표정만 보고 상대의 생각이나 감정을 파악하기 어려움",
    "A10_Score": "새 친구를 사귀기 어려움",
}


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    # [역할] 원본 CSV를 모델이 사용할 수 있는 정리된 DataFrame으로 바꾼다.
    # [입력] 아직 문자열·결측·오탈자 컬럼명이 섞여 있는 원본 데이터
    # [처리] 문자열 정리 → 컬럼명 통일 → 나이 숫자 변환 → 중복 제거 → 결측값 대체 → Target 생성
    # [출력] target(0/1)과 분석에 필요한 컬럼을 포함한 정제 데이터
    # [이유] 같은 의미의 값이 서로 다른 형태로 남아 있으면 통계검정과 모델 학습이 일관되지 않기 때문이다.
    # 원본 raw를 직접 수정하지 않고 복사본 df에서 작업한다.
    df = raw.copy()
    # object 자료형 컬럼만 골라 문자열 전처리를 적용한다.
    for c in df.select_dtypes(include=["object"]).columns:
        # 앞뒤 공백과 따옴표를 제거해 같은 응답이 다른 값으로 인식되지 않게 한다.
        df[c] = df[c].astype("string").str.strip().str.strip("'").str.strip('"')
        # 데이터에 들어 있는 ?를 pandas 결측값으로 바꾼다.
        df[c] = df[c].replace("?", pd.NA)

    # 원자료의 오탈자 컬럼명을 코드 전체에서 사용할 표준 이름으로 통일한다.
    df = df.rename(
        columns={
            "jundice": "jaundice",
            "austim": "family_asd",
            "contry_of_res": "country_of_res",
        }
    )
    # age가 문자열로 들어와도 숫자로 바꾸고, 변환할 수 없는 값은 결측으로 처리한다.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    # 완전히 같은 행은 하나만 남겨 중복 응답이 분석에 여러 번 반영되지 않게 한다.
    df = df.drop_duplicates().reset_index(drop=True)

    # 합의한 결측값 처리
    # age 결측값을 전체 age의 중앙값으로 채우기 위한 기준값을 계산한다.
    age_median = float(df["age"].median())
    # 평균보다 중앙값을 사용하면 극단적으로 큰/작은 나이의 영향을 덜 받는다.
    df["age"] = df["age"].fillna(age_median)
    # 범주형 결측은 별도 범주 Unknown으로 남겨 결측 자체의 의미를 보존한다.
    df["ethnicity"] = df["ethnicity"].fillna("Unknown")
    df["relation"] = df["relation"].fillna("Unknown")

    # 원래 Target의 문자값을 모델이 학습할 수 있는 숫자 라벨로 변환한다.
    # NO는 0, YES는 1로 정해 이후 분류·평가의 기준이 된다.
    df["target"] = df[TARGET].map({"NO": 0, "YES": 1}).astype(int)
    return df


def make_models():
    # [역할] 이번 프로젝트에서 비교할 네 가지 머신러닝 모델을 준비한다.
    # [모델] 로지스틱 회귀는 선형 관계, KNN은 가까운 관측값, 의사결정나무는 규칙,
    #        랜덤 포레스트는 여러 나무의 결합을 이용해 YES/NO를 분류한다.
    # [이유] 한 가지 모델만 임의로 선택하지 않고 같은 문제를 여러 방법으로 풀어 성능을 비교하기 위해서다.
    # max_iter는 로지스틱 회귀가 충분히 반복해 수렴하도록 설정한다.
    # random_state를 고정하면 실행할 때마다 같은 조건으로 결과를 재현할 수 있다.
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }


def build_preprocessor(columns):
    # [역할] 선택된 변수의 자료형에 맞춰 모델 입력 전처리기를 만든다.
    # [행동 문항] 이미 0/1로 되어 있으므로 원래 숫자를 그대로 통과시킨다.
    # [나이] 값의 크기 차이가 모델에 과도한 영향을 주지 않도록 StandardScaler로 표준화한다.
    # [범주형] 성별·인종·가족력처럼 선택지인 값은 OneHotEncoder로 별도 열로 변환한다.
    # [이유] 서로 다른 자료형을 같은 모델에 넣으려면 입력 표현을 통일해야 하기 때문이다.
    # 선택된 컬럼을 행동 문항·수치형·범주형 세 묶음으로 나눈다.
    behavior_cols = [c for c in columns if c in BEHAVIOR]
    numeric_cols = [c for c in columns if c == "age"]
    categorical_cols = [c for c in columns if c in BACKGROUND and c != "age"]

    # ColumnTransformer에 넣을 변환 규칙 목록이다.
    transformers = []
    if behavior_cols:
        # 행동 문항은 이미 0/1이므로 별도 변환 없이 통과시킨다.
        transformers.append(("behavior", "passthrough", behavior_cols))
    if numeric_cols:
        # age는 평균 0, 표준편차 1에 가깝게 조정한다.
        transformers.append(("numeric", StandardScaler(), numeric_cols))
    if categorical_cols:
        # 범주형 응답을 0/1 더미컬럼으로 바꾼다.
        # handle_unknown은 Validation/Test에 학습 때 없던 범주가 나와도 오류를 내지 않게 한다.
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )
    # 지정한 컬럼만 변환하고 나머지 컬럼은 모델 입력에서 버린다.
    return ColumnTransformer(transformers, remainder="drop")


def metrics(y_true, pred, proba):
    # [역할] 실제 정답과 모델의 결과를 비교해 성능표 한 줄을 만든다.
    # [입력] y_true=실제 정답, pred=YES/NO 예측, proba=YES일 확률
    # [출력] 정확도·정밀도·재현율·F1·ROC-AUC 다섯 지표
    # [이유] ASD YES/NO 자료처럼 한쪽 그룹이 적을 때 정확도 하나만 보면 성능을 잘못 판단할 수 있기 때문이다.
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def bias_corrected_cramers_v(table):
    # [역할] 두 범주형 변수의 관계 강도를 Cramér's V로 계산한다.
    # [처리] 먼저 카이제곱 통계량을 구한 뒤, 표본 수와 범주 수 때문에 관계가 커 보이는 편향을 보정한다.
    # [출력] 0에 가까우면 관계가 약하고 1에 가까우면 관계가 강한 효과크기 값
    # [이유] p-value는 표본 크기에 민감하므로 관계의 실제 강도도 함께 확인하기 위해 사용한다.
    chi2 = chi2_contingency(table)[0]
    n = float(table.to_numpy().sum())
    if n <= 1:
        return 0.0
    r, k = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def fit_rare_category_maps(train_df, min_count=RARE_MIN_COUNT):
    # [역할] 범주 수가 너무 적은 응답을 Other로 묶기 위한 기준표를 학습한다.
    # [입력] 반드시 Train/Selection 자료만 사용하며 Validation/Test는 보지 않는다.
    # [출력] 각 변수에서 유지할 범주 목록(maps)
    # [이유] 드문 범주 하나가 모델을 불안정하게 만들거나 우연한 패턴을 만들 수 있기 때문이다.
    maps = {}
    for feature in RARE_GROUP_FEATURES:
        counts = train_df[feature].value_counts(dropna=False)
        keep = counts[counts >= min_count].index.tolist()
        maps[feature] = keep
    return maps


def apply_rare_category_maps(df, maps):
    # [역할] 학습자료에서 만든 희소 범주 기준을 현재 DataFrame에 적용한다.
    # [처리] 학습 때 유지하지 않기로 한 범주는 모두 RARE_LABEL(Other)로 바꾼다.
    # [이유] Validation/Test에서 새 기준을 다시 만들면 평가자료 정보가 학습에 섞이는 데이터 누수가 발생하기 때문이다.
    out = df.copy()
    for feature, keep in maps.items():
        out[feature] = out[feature].where(out[feature].isin(keep), RARE_LABEL)
    return out


def grouping_summary(before_df, after_df, maps, stage):
    # [역할] 희소 범주 통합 전과 후가 어떻게 달라졌는지 요약표를 만든다.
    # [기록] 원래 범주 수, 유지 범주 수, Other로 묶인 행 수, 유지한 범주 목록
    # [이유] 전처리 과정이 실제로 어떤 범주를 합쳤는지 CSV로 검증할 수 있게 하기 위해서다.
    rows = []
    for feature in RARE_GROUP_FEATURES:
        rows.append({
            "stage": stage,
            "feature": feature,
            "min_count": RARE_MIN_COUNT,
            "original_category_count": int(before_df[feature].nunique(dropna=False)),
            "retained_category_count": int(len(maps.get(feature, []))),
            "after_grouping_category_count": int(after_df[feature].nunique(dropna=False)),
            "grouped_to_other_rows": int((after_df[feature] == RARE_LABEL).sum()),
            "retained_categories": " | ".join(map(str, maps.get(feature, []))),
        })
    return pd.DataFrame(rows)


def association_table(selection_train):
    # [역할] 각 후보 독립변수와 현재 ASD 여부(Target)의 통계적 관계를 한 번씩 검정한다.
    # [숫자형 age] Welch t-test로 두 그룹 평균 차이를 보고, point-biserial 상관계수로 관계 크기를 계산한다.
    # [범주형 변수] 카이제곱 검정으로 관계 유무를 보고, 보정 Cramér's V로 관계 강도를 계산한다.
    # [출력] 검정통계량, p-value, 효과크기, 표본 수, 유의 여부를 포함한 변수별 결과표
    # [이유] 모델에 넣을 변수를 데이터 근거로 좁히고, p-value와 관계 강도를 함께 확인하기 위해서다.
    rows = []
    for feature in ALL_CANDIDATES:
        # 한 번에 한 후보 변수와 Target만 뽑아 결측 행을 제외한다.
        if feature == "age":
            temp = selection_train[[feature, "target"]].dropna()
            no_vals = temp.loc[temp["target"].eq(0), feature].astype(float)
            yes_vals = temp.loc[temp["target"].eq(1), feature].astype(float)
            # NO 그룹과 YES 그룹의 나이 평균 차이를 Welch t-test로 검정한다.
            stat, p_value = ttest_ind(no_vals, yes_vals, equal_var=False)
            # 이진 Target과 연속형 age의 관계 방향·크기를 point-biserial 상관으로 계산한다.
            r_value, _ = pointbiserialr(temp["target"], temp[feature].astype(float))
            effect = abs(float(r_value))
            sparse_ratio = np.nan
            caution = ""
            test = "Welch t-test"
        else:
            temp = selection_train[[feature, "target"]].dropna()
            # 행은 후보 변수의 범주, 열은 Target 0/1인 교차표를 만든다.
            table = pd.crosstab(temp[feature], temp["target"])
            if table.shape[0] < 2 or table.shape[1] < 2:
                stat, p_value = 0.0, 1.0
                expected = np.ones_like(table, dtype=float)
                effect = 0.0
            else:
                # 실제 관측 교차표와 독립이라고 가정한 기대값을 비교해 카이제곱 검정을 수행한다.
                stat, p_value, _, expected = chi2_contingency(table)
                effect = bias_corrected_cramers_v(table)
            # 기대빈도가 5보다 작은 칸의 비율을 계산해 희소 범주 주의 여부를 표시한다.
            sparse_ratio = float((np.asarray(expected) < 5).mean()) if np.asarray(expected).size else np.nan
            caution = "희소 범주 주의" if sparse_ratio > 0.20 else ""
            test = "Chi-square"

        # 변수 하나의 검정 결과를 딕셔너리 한 행으로 저장한다.
        rows.append(
            {
                "feature": feature,
                "group": "behavior" if feature in BEHAVIOR else "background",
                "test": test,
                "statistic": float(stat),
                "p_value": float(p_value),
                "effect_size": float(effect),
                "effect_method": "|Point-biserial r|" if feature == "age" else "Bias-corrected Cramer's V",
                "significant_0_05": bool(p_value < 0.05),
                "n_used": int(len(temp)),
                "category_count": int(temp[feature].nunique()) if feature != "age" else np.nan,
                "expected_lt5_ratio": sparse_ratio,
                "caution": caution,
            }
        )

    # 모든 후보의 결과를 DataFrame으로 만들고 효과크기→p-value 순으로 정렬한다.
    out = pd.DataFrame(rows).sort_values(
        ["effect_size", "p_value"], ascending=[False, True]
    ).reset_index(drop=True)
    out["association_rank"] = np.arange(1, len(out) + 1)
    return out


def evaluate_models(features, train_df, val_df):
    # [역할] 동일한 변수 목록과 동일한 Train/Validation 자료로 네 모델을 공정하게 비교한다.
    # [처리] 각 모델마다 전처리기와 모델을 Pipeline으로 묶어 학습 → 예측 → 성능 계산을 반복한다.
    # [출력] 모델별 혼동행렬 값과 Accuracy·Precision·Recall·F1·ROC-AUC 표
    # [이유] 학습자료 성능이 아니라 처음 보는 Validation 자료 성능으로 최종 모델을 선택하기 위해서다.
    rows = []
    fitted = {}
    for model_name, estimator in make_models().items():
        # 전처리와 모델을 하나의 Pipeline으로 묶어 학습·예측 때 같은 변환을 보장한다.
        pipe = Pipeline(
            [
                ("preprocessor", build_preprocessor(features)),
                ("model", estimator),
            ]
        )
        # Train 자료의 정답을 이용해 모델의 패턴을 학습한다.
        pipe.fit(train_df[features], train_df["target"])
        # Validation 자료는 학습에 사용하지 않고 성능 확인에만 사용한다.
        pred = pipe.predict(val_df[features])
        # ROC-AUC 계산을 위해 YES일 확률도 함께 얻는다.
        proba = pipe.predict_proba(val_df[features])[:, 1]
        # TN/FP/FN/TP를 펼쳐 혼동행렬의 네 값을 기록한다.
        tn, fp, fn, tp = confusion_matrix(val_df["target"], pred).ravel()
        row = {
            "model": model_name,
            "configuration": "Association-selected features",
            "feature_set": ", ".join(features),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        # 위에서 계산한 다섯 가지 성능지표를 현재 모델 행에 추가한다.
        row.update(metrics(val_df["target"], pred, proba))
        rows.append(row)
        fitted[model_name] = pipe

    # F1을 우선하고 재현율·ROC-AUC를 보조 기준으로 모델 순위를 정한다.
    out = pd.DataFrame(rows).sort_values(
        ["f1", "recall", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1)
    return out, fitted


def raw_feature_from_transformed(name):
    # [역할] OneHotEncoder가 만든 가상 컬럼명을 원래 설문 변수명으로 연결한다.
    # [예시] gender_M, gender_F 같은 여러 더미를 다시 gender 하나로 묶는다.
    # [이유] 모델은 더미를 사용하지만 화면과 변수 선정은 원래 설문 항목 단위로 설명해야 하기 때문이다.
    if name.startswith("behavior__") or name.startswith("numeric__"):
        return name.split("__", 1)[1]
    text = name.split("__", 1)[-1]
    for feature in sorted([x for x in BACKGROUND if x != "age"], key=len, reverse=True):
        if text == feature or text.startswith(feature + "_"):
            return feature
    return text


def selected_feature_importance(features, train_df):
    # [역할] 선택된 변수들이 모델 분류에 얼마나 중요하게 사용됐는지 계산한다.
    # [로지스틱 회귀] 계수 절댓값을 사용해 각 입력이 분류 경계에 기여한 크기를 본다.
    # [랜덤 포레스트] 여러 결정나무가 분할에 사용한 변수 중요도를 본다.
    # [처리] 원-핫 인코딩된 더미들의 중요도를 원래 변수 단위로 합산한 뒤 두 모델 결과를 결합한다.
    # [이유] 통계적 유의성만으로 고르지 않고 서로 다른 모델 관점에서도 중요한 항목인지 확인하기 위해서다.
    # 같은 선택 변수로 중요도 비교용 로지스틱 회귀 Pipeline을 만든다.
    lr = Pipeline(
        [
            ("preprocessor", build_preprocessor(features)),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    # 같은 선택 변수로 비선형 패턴 확인용 랜덤 포레스트 Pipeline을 만든다.
    rf = Pipeline(
        [
            ("preprocessor", build_preprocessor(features)),
            ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ]
    )
    # 두 모델 모두 Selection Train에서만 학습한다.
    lr.fit(train_df[features], train_df["target"])
    rf.fit(train_df[features], train_df["target"])

    # 전처리 후 실제 모델에 들어간 컬럼명(원-핫 더미 포함)을 가져온다.
    feature_names = lr.named_steps["preprocessor"].get_feature_names_out()
    # 로지스틱 회귀에서는 계수의 부호보다 절댓값 크기를 중요도로 사용한다.
    lr_abs = np.abs(lr.named_steps["model"].coef_[0])
    # 랜덤 포레스트는 각 나무의 분할에 사용된 중요도를 가져온다.
    rf_imp = rf.named_steps["model"].feature_importances_

    rows = []
    for feature in features:
        # 현재 원래 변수에 해당하는 더미 컬럼들의 위치를 찾는다.
        idx = [i for i, n in enumerate(feature_names) if raw_feature_from_transformed(n) == feature]
        # 로지스틱 회귀 더미 계수는 대표값(RMS), RF 중요도는 합계로 원 변수에 통합한다.
        lr_group = float(np.sqrt(np.mean(lr_abs[idx] ** 2))) if idx else 0.0
        rf_group = float(np.sum(rf_imp[idx])) if idx else 0.0
        rows.append(
            {
                "feature": feature,
                "group": "behavior" if feature in BEHAVIOR else "background",
                "logistic_group_importance": lr_group,
                "random_forest_group_importance": rf_group,
                "encoded_column_count": len(idx),
            }
        )
    # 변수별 중요도를 표로 만들고 모델별 합이 1이 되도록 정규화한다.
    out = pd.DataFrame(rows)
    out["logistic_normalized"] = out["logistic_group_importance"] / out["logistic_group_importance"].sum()
    out["random_forest_normalized"] = out["random_forest_group_importance"] / out["random_forest_group_importance"].sum()
    # 두 모델 중요도를 같은 비중으로 평균해 통합 중요도를 만든다.
    out["combined_importance"] = (out["logistic_normalized"] + out["random_forest_normalized"]) / 2
    out = out.sort_values("combined_importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def integer_weight_points(importance_df, selected_model_name, total_points=100):
    # [역할] 행동 문항 중요도를 교사용 체크리스트의 정수 가중점수로 바꾼다.
    # [처리] 중요도 비율을 100점에 배분하고, 소수점 버림으로 생긴 남은 점수는 fractional remainder가 큰 문항에 배분한다.
    # [출력] 문항별 순위·원시 중요도·정규화 가중치·최종 점수
    # [이유] 화면에서 문항별 점수를 더해 0~100점 행동 관찰점수를 만들 수 있게 하기 위해서다.
    # 전체 변수 중 ASD 행동 문항만 체크리스트 가중치 계산 대상으로 남긴다.
    behavior_imp = importance_df[importance_df["group"].eq("behavior")].copy()
    if behavior_imp.empty:
        return pd.DataFrame(columns=["rank", "feature", "weight_source", "raw_importance", "normalized_weight", "points"])

    if selected_model_name == "Logistic Regression" and "logistic_group_importance" in behavior_imp.columns:
        source_col = "logistic_group_importance"
        source_label = "Logistic Regression coefficient magnitude"
    else:
        source_col = "combined_importance"
        source_label = "Combined ML importance"

    values = behavior_imp[source_col].astype(float).clip(lower=0)
    if float(values.sum()) <= 0:
        values = pd.Series(np.ones(len(behavior_imp)), index=behavior_imp.index)
    # 각 문항 중요도를 전체 중요도 합으로 나누어 비율로 만든다.
    normalized = values / values.sum()
    # 비율에 100을 곱해 100점 체크리스트의 소수점 원점수를 만든다.
    raw_points = normalized * total_points
    # 우선 소수점 아래를 버리고 정수 부분만 배정한다.
    base_points = np.floor(raw_points).astype(int)
    # 버림 때문에 남은 점수를 계산한다.
    remainder = int(total_points - base_points.sum())
    # 소수점이 큰 문항부터 남은 점수를 하나씩 추가한다.
    fractional = (raw_points - base_points).sort_values(ascending=False)
    points = base_points.copy()
    if remainder > 0:
        for idx in fractional.index[:remainder]:
            points.loc[idx] += 1

    out = behavior_imp[["feature", source_col]].copy()
    out["weight_source"] = source_label
    out["raw_importance"] = values.values
    out["normalized_weight"] = normalized.values
    out["points"] = points.values
    out = out.sort_values(["points", "normalized_weight"], ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["observation"] = out["feature"].map(AQ10_OBSERVATION)
    return out[["rank", "feature", "observation", "weight_source", "raw_importance", "normalized_weight", "points"]]


def weighted_score_series(df, weights_df):
    # [역할] 각 아동의 A1~A10 응답에 문항별 가중치를 곱해 총점을 계산한다.
    # [계산] 응답값(0/1) × 문항점수를 문항별로 더해 0~100점 점수를 만든다.
    # [출력] DataFrame 각 행에 대응하는 가중 관찰점수 Series
    # [이유] Streamlit 체크리스트의 선택 결과와 분석자료의 점수 구간을 같은 방식으로 계산하기 위해서다.
    score = pd.Series(0, index=df.index, dtype=int)
    for _, row in weights_df.iterrows():
        # 가중치 표에서 현재 문항명과 점수를 읽는다.
        feature = row["feature"]
        if feature in df.columns:
            # 해당 문항 응답이 1이면 점수를 더하고 0이면 더하지 않는다.
            score = score + df[feature].fillna(0).astype(int) * int(row["points"])
    return score.astype(int)


def weighted_threshold_table(df, weights_df, thresholds=range(0, 101, 5)):
    # [역할] 0점부터 100점까지 여러 기준점을 시험해 점수 해석표를 만든다.
    # [각 기준] 해당 점수 이상인 사람 수, 실제 YES 수, 정밀도, 재현율을 계산한다.
    # [이유] 어떤 점수부터 관심 관찰 대상으로 볼지 여러 기준을 비교할 수 있게 하기 위해서다.
    scores = weighted_score_series(df, weights_df)
    # 이 자료에서 실제 YES가 몇 명인지 기준으로 Recall의 분모를 고정한다.
    total_yes = int(df["target"].sum())
    rows = []
    for threshold in thresholds:
        # 현재 기준점 이상인 사람을 관심 관찰 대상으로 표시한다.
        flag = scores >= int(threshold)
        flagged = int(flag.sum())
        # 표시된 사람 중 실제 YES인 수가 True Positive다.
        tp = int(df.loc[flag, "target"].sum()) if flagged else 0
        precision = tp / flagged if flagged else np.nan
        recall = tp / total_yes if total_yes else np.nan
        rows.append({
            "threshold": int(threshold),
            "flagged_count": flagged,
            "yes_count": tp,
            "yes_rate_precision": precision,
            "yes_recall": recall,
        })
    return pd.DataFrame(rows)


def weighted_band_summary(df, weights_df, observe_cutoff, high_cutoff, very_high_cutoff):
    # [역할] 최종 점수를 일반·추가·고관찰·매우 고관찰 구간으로 나눈다.
    # [출력] 각 구간의 인원 수, 실제 YES 수, YES 비율, 평균 점수
    # [이유] 숫자 하나만 보여주는 대신 점수 구간별 분포와 Target 비율을 설명하기 위해서다.
    # 먼저 모든 행의 최종 가중점수를 계산한다.
    scores = weighted_score_series(df, weights_df)
    # 점수 경계값을 기준으로 각 행을 네 구간 중 하나에 배정한다.
    bands = pd.cut(
        scores,
        [-1, observe_cutoff - 1, high_cutoff - 1, very_high_cutoff - 1, 100],
        labels=[
            f"0-{observe_cutoff-1}",
            f"{observe_cutoff}-{high_cutoff-1}",
            f"{high_cutoff}-{very_high_cutoff-1}",
            f"{very_high_cutoff}-100",
        ],
    )
    temp = pd.DataFrame({"band": bands, "target": df["target"].to_numpy(), "score": scores.to_numpy()})
    out = temp.groupby("band", observed=False).agg(
        n=("target", "size"),
        yes_count=("target", "sum"),
        yes_rate=("target", "mean"),
        mean_score=("score", "mean"),
    ).reset_index()
    return out


def main():
    # 전체 분석을 실행하는 시작점이다.
    # 원자료 정리 → 데이터 분할 → 연관성 분석 → 모델 비교 → 가중점수 산출 순서로 진행한다.
    # 1) 원본 CSV를 읽고 컬럼명·결측값·Target을 분석 가능한 형태로 정리한다.
    # 원본 데이터 파일을 읽는다. 여기서부터 모든 분석의 출발점이 된다.
    # pd.read_csv는 CSV의 각 행을 관측치(DataFrame의 행)로, 각 열을 변수로 읽는다.
    # 이 시점에는 아직 결측값 처리나 Target 숫자 변환을 하지 않은 원본 상태다.
    raw = pd.read_csv(DATA_PATH)
    # 컬럼명 오탈자, 결측값, 중복, YES/NO Target을 정리해 이후 단계의 공통 입력으로 만든다.
    # clean_data의 반환값 df를 이후 모든 함수가 공통으로 사용한다.
    # 즉, 이후 단계는 원본 raw가 아니라 정리된 df를 기준으로 수행된다.
    df = clean_data(raw)

    # 2) Train/Validation/Test를 분리한다.
    # Train은 패턴 학습, Validation은 모델·기준 선택, Test는 마지막 성능 확인에만 사용한다.
    # Final Test는 모델 선택과 기준 설정이 끝날 때까지 보지 않아 최종 평가의 독립성을 유지한다.
    # 먼저 전체 데이터의 20%를 최종 시험용 자료로 떼어 둔다.
    # stratify는 target의 YES/NO 비율이 두 묶음에서 비슷하게 유지되도록 한다.
    dev, final_test = train_test_split(
        df, test_size=0.20, stratify=df["target"], random_state=RANDOM_STATE
    )
    # 분할 전 원래 행 번호는 분석에 필요 없으므로 0부터 다시 번호를 매긴다.
    dev = dev.reset_index(drop=True)
    final_test = final_test.reset_index(drop=True)
    # Development 80% 안에서 다시 75:25로 나누어 Selection Train과 Validation을 만든다.
    # 결과적으로 전체 기준 약 60% 학습, 20% 검증, 20% 최종평가가 된다.
    selection_train, validation = train_test_split(
        dev, test_size=0.25, stratify=dev["target"], random_state=RANDOM_STATE
    )
    selection_train = selection_train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    # 3) 희소 범주를 학습자료 기준으로 통합한다.
    # 범주가 너무 작은 응답을 Other로 묶어 통계검정과 원-핫 인코딩이 불안정해지는 것을 줄인다.
    # 범주가 많은 ethnicity/country는 Selection Train에서 10건 미만 범주를 Other로 통합한다.
    # 이 규칙은 Validation에 그대로 적용하여 Validation 정보를 미리 보지 않는다.
    # 범주 통합 기준은 Selection Train의 빈도만 보고 학습한다.
    # Validation/Test의 분포를 미리 이용하지 않기 위한 데이터 누수 방지 단계다.
    selection_rare_maps = fit_rare_category_maps(selection_train)
    # 학습자료와 검증자료에 똑같은 치환표를 적용한다.
    # 따라서 두 자료에서 같은 원래 범주가 같은 방식으로 처리된다.
    selection_train_model = apply_rare_category_maps(selection_train, selection_rare_maps)
    validation_model = apply_rare_category_maps(validation, selection_rare_maps)
    grouping_summary(selection_train, selection_train_model, selection_rare_maps, "selection_train").to_csv(
        ART / "category_grouping_summary.csv", index=False, encoding="utf-8-sig"
    )

    # 4) 후보 변수와 ASD Target의 통계적 연관성을 계산하고 결과 CSV를 저장한다.
    # 이 단계는 ‘무엇이 원인인가’를 증명하는 단계가 아니라, 어떤 항목이 Target과 함께 나타나는지 확인하는 단계다.
    # 3번 페이지에서 보여주는 카이제곱·Welch t-test·효과크기 결과의 원천이다.
    # 후보 변수 하나씩 Target과의 통계적 관계를 계산한다.
    # 범주형 변수는 카이제곱 검정과 Cramer's V, age는 point-biserial을 사용한다.
    assoc = association_table(selection_train_model)
    assoc.to_csv(ART / "all_feature_association.csv", index=False, encoding="utf-8-sig")
    assoc[assoc["group"].eq("background")].to_csv(
        ART / "background_association.csv", index=False, encoding="utf-8-sig"
    )

    # 5) p<.05인 항목만 모델 입력 후보로 남긴 뒤 네 가지 ML 모델을 Validation에서 비교한다.
    # p-value만으로 모델을 결정하지 않고, 동일한 후보 묶음을 네 모델에 넣어 Validation 성능을 비교한다.
    # significant_0_05가 True인 변수명만 빠르게 찾기 위해 set으로 만든다.
    selected_set = set(assoc.loc[assoc["significant_0_05"], "feature"].tolist())
    # 후보 목록의 원래 순서를 유지하면서 유의한 변수만 선택한다.
    selected_features = [f for f in ALL_CANDIDATES if f in selected_set]
    selected_behavior = [x for x in selected_features if x in BEHAVIOR]
    selected_background = [x for x in selected_features if x in BACKGROUND]

    # 같은 입력 변수와 같은 검증자료를 네 모델에 공통으로 사용한다.
    # 그래야 모델별 차이가 데이터 차이가 아니라 알고리즘 차이로 비교된다.
    model_compare, _ = evaluate_models(selected_features, selection_train_model, validation_model)
    model_compare.to_csv(ART / "model_validation_comparison.csv", index=False, encoding="utf-8-sig")
    # evaluate_models는 Validation ROC-AUC 등 기준으로 정렬된 결과를 반환한다.
    # 첫 행의 모델명을 최종 선택 모델로 기록한다.
    selected_model_name = str(model_compare.iloc[0]["model"])

    # 6) 선택 변수의 모델 기반 중요도를 계산한다.
    # 로지스틱 회귀와 랜덤 포레스트에서 모두 중요하게 사용된 항목을 확인해 결과 설명에 활용한다.
    # 로지스틱 회귀 계수와 랜덤 포레스트 중요도를 함께 저장해 가중치·설명에 사용한다.
    imp = selected_feature_importance(selected_features, selection_train_model)
    imp.to_csv(ART / "selected_feature_importance.csv", index=False, encoding="utf-8-sig")
    # 기존 파일명과의 호환성을 위해 같은 내용 저장
    imp.to_csv(ART / "all_feature_importance.csv", index=False, encoding="utf-8-sig")

    # 7) 선생님용 행동 체크리스트의 문항별 중요도를 총 100점 정수점수로 환산한다.
    # 이 점수는 원자료의 Class/ASD 규칙을 이해하기 위한 프로젝트용 관찰점수이며 새 모델의 확률을 뜻하지 않는다.
    # 점수 가중치는 Selection Train에서 학습된 중요도로 만들고, 구간 기준은 Development의 ASD 선별 YES/NO 가중점수 분포에서 정한 뒤 Final Test로 확인한다.
    # 모델 중요도를 100점 만점의 정수 문항 가중치로 변환한다.
    # 실제 예측 확률을 그대로 점수로 쓰는 것이 아니라, 설명 가능한 체크리스트 점수로 바꾸는 단계다.
    checklist_weights = integer_weight_points(imp, selected_model_name, total_points=100)
    checklist_weights.to_csv(ART / "weighted_checklist_weights.csv", index=False, encoding="utf-8-sig")

    # Validation에서 0, 5, 10 ... 100점 기준을 모두 시험한다.
    # 각 기준에서 몇 명이 표시되고 실제 YES가 얼마나 포함되는지 확인한다.
    validation_weighted = weighted_threshold_table(validation_model, checklist_weights)

    # 8) Development 자료에서 행동점수 구간을 정한다.
    # Test를 보지 않고 Development 안의 NO/YES 점수 분포만 이용해 일반·고관찰 기준을 정한다.
    # 가중점수 구간은 원래 Class/ASD 선별 레이블의 경계를 최대한 보존하도록 설정한다.
    # Final Test는 사용하지 않고 Development에서 NO의 최고점 바로 다음 점수를 고관찰 시작점으로 잡는다.
    # Development 각 행의 100점 행동점수를 계산한다.
    # 점수 계산에는 해당 행의 체크리스트 응답과 문항별 points가 사용된다.
    dev_weighted_scores = weighted_score_series(dev, checklist_weights)
    dev_no_max = int(dev_weighted_scores[dev["target"].eq(0)].max())
    dev_yes_min = int(dev_weighted_scores[dev["target"].eq(1)].min())
    # Development에서 실제 NO가 받은 가장 높은 점수보다 1점 높은 곳을 고관찰 시작점으로 둔다.
    checklist_high_cutoff = int(dev_no_max + 1)

    # 추가 관찰 구간은 고관찰 시작점 아래 15점 범위를 프로젝트 운영용 완충구간으로 둔다.
    # 고관찰 기준보다 15점 낮은 지점부터 추가 관찰 구간으로 둔다.
    # max(0, ...)는 기준이 음수가 되는 것을 막는다.
    checklist_observe_cutoff = max(0, checklist_high_cutoff - 15)

    # 고관찰 구간은 Development의 YES 가중점수 중앙값을 기준으로 둘로 나눈다.
    # 실제 YES 집단의 중앙 점수를 구해 높은 관찰 구간을 나누는 기준으로 사용한다.
    dev_yes_median = float(dev_weighted_scores[dev["target"].eq(1)].median())
    checklist_very_high_cutoff = int(round(dev_yes_median / 5) * 5)
    checklist_very_high_cutoff = max(checklist_high_cutoff + 5, min(checklist_very_high_cutoff, 95))

    validation_weighted.to_csv(ART / "weighted_checklist_threshold_validation.csv", index=False, encoding="utf-8-sig")
    weighted_band_summary(
        validation_model, checklist_weights, checklist_observe_cutoff, checklist_high_cutoff, checklist_very_high_cutoff
    ).to_csv(ART / "weighted_checklist_band_validation.csv", index=False, encoding="utf-8-sig")

    # 9) 선택된 모델을 Development 전체로 다시 학습하고 Final Test에서 한 번 평가한다.
    # 모델 선택이 끝난 뒤에만 학습자료를 늘려 최종 모델을 만들고, Test는 이때 처음 사용한다.
    # Development 전체로 재학습할 때는 희소 범주 규칙도 Development에서 다시 적합한다.
    # 모델 선택이 끝났으므로 이제 Development 전체를 최종 학습자료로 사용한다.
    # 이때도 희소 범주 규칙을 Development에서 새로 학습하고 Test에는 그대로 적용한다.
    final_rare_maps = fit_rare_category_maps(dev)
    dev_model = apply_rare_category_maps(dev, final_rare_maps)
    final_test_model = apply_rare_category_maps(final_test, final_rare_maps)
    # Test에서는 기준을 새로 고르지 않고, 앞에서 정한 기준을 적용해 분포만 확인한다.
    final_weighted = weighted_threshold_table(final_test_model, checklist_weights)
    final_weighted.to_csv(ART / "weighted_checklist_threshold_final_test.csv", index=False, encoding="utf-8-sig")
    weighted_band_summary(
        final_test_model, checklist_weights, checklist_observe_cutoff, checklist_high_cutoff, checklist_very_high_cutoff
    ).to_csv(ART / "weighted_checklist_band_final_test.csv", index=False, encoding="utf-8-sig")

    # Test 원자료에 점수와 실제 Target의 표시용 문자값을 붙여 결과 파일로 저장한다.
    final_scored = final_test_model.copy()
    final_scored["weighted_checklist_score"] = weighted_score_series(final_test_model, checklist_weights)
    final_scored["actual_label"] = np.where(final_scored["target"].eq(1), "YES", "NO")
    final_scored.to_csv(ART / "weighted_checklist_final_test_scored.csv", index=False, encoding="utf-8-sig")

    # 10) 최종 모델의 예측값·확률·혼동행렬·일반화 차이를 산출한다.
    # Train 정확도와 Test 정확도의 차이도 저장해 과적합 여부를 함께 확인한다.
    # 전처리와 모델을 하나의 Pipeline으로 묶으면 예측 때도 학습과 같은 변환이 자동 적용된다.
    final_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(selected_features)),
            ("model", make_models()[selected_model_name]),
        ]
    )
    # Development 전체로 최종 모델을 학습한다. Test 행은 fit에 절대 사용하지 않는다.
    final_pipeline.fit(dev_model[selected_features], dev_model["target"])
    # 학습자료 예측은 과적합 확인용, Test 예측은 처음 보는 자료의 일반화 성능 확인용이다.
    train_pred = final_pipeline.predict(dev_model[selected_features])
    test_pred = final_pipeline.predict(final_test_model[selected_features])
    test_proba = final_pipeline.predict_proba(final_test_model[selected_features])[:, 1]
    # Test의 실제 Target과 예측값을 비교해 Accuracy, Precision, Recall, F1, ROC-AUC를 계산한다.
    final_metrics = metrics(final_test["target"], test_pred, test_proba)
    final_metrics["train_accuracy"] = accuracy_score(dev["target"], train_pred)
    final_metrics["generalization_gap"] = final_metrics["train_accuracy"] - final_metrics["accuracy"]
    tn, fp, fn, tp = confusion_matrix(final_test["target"], test_pred).ravel()
    final_metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    # 선택 모델, 선택 변수, 최종 성능을 한 행에 모아 대시보드가 읽기 쉬운 표로 만든다.
    final_candidate = pd.DataFrame(
        [
            {
                "selected_model": selected_model_name,
                "configuration": "Association-selected features",
                "selected_item_count": len(selected_features),
                "selected_behavior_features": ", ".join(selected_behavior),
                "selected_background_features": ", ".join(selected_background),
                "final_features": ", ".join(selected_features),
                "threshold": 0.5,
                **final_metrics,
            }
        ]
    )
    final_candidate.to_csv(ART / "final_candidate_results.csv", index=False, encoding="utf-8-sig")

    # 11) ROC Curve와 개별 예측 결과를 대시보드용 CSV로 저장한다.
    # ROC 곡선은 분류 임계값을 바꿀 때의 FPR/TPR 변화를 저장한 것이다.
    fpr, tpr, thresholds = roc_curve(final_test["target"], test_proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        ART / "final_roc_curve.csv", index=False, encoding="utf-8-sig"
    )

    # 각 Test 행별 실제값, 예측값, YES 확률, 정답 여부를 함께 저장한다.
    pred_out = final_test.copy()
    pred_out["actual_label"] = np.where(final_test["target"].eq(1), "YES", "NO")
    pred_out["predicted_label"] = np.where(test_pred == 1, "YES", "NO")
    pred_out["model_score_yes"] = test_proba
    pred_out["correct"] = pred_out["actual_label"].eq(pred_out["predicted_label"])
    pred_out.to_csv(ART / "final_test_predictions.csv", index=False, encoding="utf-8-sig")

    # 12) 전처리와 최종 모델을 하나의 Bundle로 저장해 같은 방식으로 재현할 수 있게 한다.
    # joblib은 학습된 전처리기와 모델을 함께 직렬화한다.
    # 나중에 새 입력을 넣을 때도 동일한 전처리 규칙을 재사용할 수 있다.
    joblib.dump(
        {
            "model": final_pipeline,
            "selected_model_name": selected_model_name,
            "final_features": selected_features,
            "threshold": 0.5,
            "target_mapping": {"NO": 0, "YES": 1},
            "rare_category_maps": final_rare_maps,
            "rare_min_count": RARE_MIN_COUNT,
            "rare_label": RARE_LABEL,
        },
        ART / "final_model_bundle.joblib",
    )

    # 13) 데이터셋의 합계 규칙과 Target 규칙이 코드북/원자료와 일치하는지 검증한다.
    sum_match = bool((df[BEHAVIOR].sum(axis=1).to_numpy() == df["result"].to_numpy()).all())
    rule_match = bool((((df["result"] >= 7).astype(int)).to_numpy() == df["target"].to_numpy()).all())

    # 14) 분석 기준·분할 수·선택 변수·가중치·제한사항을 메타데이터로 남긴다.
    metadata = {
        "project_title": "아동학대 의심 예측 설문조사",
        "project_subtitle": "ASD 선별 관련 행동·개인·배경 특성 분석을 통한 관심 필요 아동 탐색",
        "actual_dataset_target": TARGET,
        "target_meaning": "A1~A10 점수 합계 규칙에 기반한 ASD 선별 레이블",
        "abuse_target_in_dataset": False,
        "random_state": RANDOM_STATE,
        "data_rows_raw": len(raw),
        "data_rows_after_dedup": len(df),
        "development_rows": len(dev),
        "final_test_rows": len(final_test),
        "selection_train_rows": len(selection_train),
        "validation_rows": len(validation),
        "missing_value_handling": {
            "age": f"median={float(df['age'].median()):.0f}",
            "ethnicity": "Unknown",
            "relation": "Unknown",
        },
        "behavior_candidates": BEHAVIOR,
        "background_candidates": BACKGROUND,
        "all_candidate_features": ALL_CANDIDATES,
        "excluded_features": {
            "result": "A1~A10 합계이며 Target 생성 규칙을 직접 포함하므로 독립 요인/모델 입력에서 제외",
            "age_desc": "모든 행에서 동일한 4-11 years 설명값이어서 제외",
            TARGET: "Target 자체이므로 입력 변수 아님",
        },
        "feature_selection_rule": "Selection Train에서 희소 범주 통합 후 p<0.05인 요인을 ML 입력 후보로 유지",
        "association_effect_size": {
            "categorical": "Bias-corrected Cramer's V",
            "age": "absolute point-biserial correlation",
        },
        "rare_category_grouping": {
            "features": RARE_GROUP_FEATURES,
            "min_count": RARE_MIN_COUNT,
            "label": RARE_LABEL,
            "selection_maps": selection_rare_maps,
            "final_maps": final_rare_maps,
        },
        "weighted_checklist": {
            "score_range": "0-100",
            "weight_source": checklist_weights.iloc[0]["weight_source"] if not checklist_weights.empty else "",
            "observe_cutoff": int(checklist_observe_cutoff),
            "high_cutoff": int(checklist_high_cutoff),
            "very_high_cutoff": int(checklist_very_high_cutoff),
            "cutoff_basis": {
                "development_no_max": int(dev_no_max),
                "development_yes_min": int(dev_yes_min),
                "development_yes_median": float(dev_yes_median),
            },
            "points": {str(r["feature"]): int(r["points"]) for _, r in checklist_weights.iterrows()},
            "purpose": "수업 프로젝트용 ASD 선별 관련 행동 관찰 참고점수",
            "not_for": "자폐증 진단 또는 아동학대 위험 확률 산출",
        },
        "selected_behavior_features": selected_behavior,
        "selected_background_features": selected_background,
        "final_features": selected_features,
        "selected_model": selected_model_name,
        "threshold": 0.5,
        "rule_checks": {
            "result_equals_A1_to_A10_sum": sum_match,
            "class_yes_equals_result_ge_7": rule_match,
        },
        "interpretation_warning": "A1~A10이 Target 생성에 직접 사용되므로 높은 모델 성능과 가중 체크리스트 점수는 독립적 임상 예측보다 선별 규칙 재현에 가깝다. 가중 체크리스트는 공식 AQ-10 채점법을 대체하지 않는다.",
    }
    with open(ART / "final_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 15) 결과를 해석할 때 주의해야 할 한계를 별도 CSV로 저장한다.
    limitations = pd.DataFrame(
        [
            {
                "limitation": "직접 목표값",
                "detail": "Target은 아동학대 여부가 아니라 Class/ASD 선별 결과임. 아동학대는 프로젝트 배경과 활용 맥락으로만 연결함.",
            },
            {
                "limitation": "Target 구조",
                "detail": "A1~A10 합계가 result이고 result>=7이 Class/ASD와 일치하므로 높은 성능은 선별 규칙 재현의 영향이 큼.",
            },
            {
                "limitation": "표본 크기",
                "detail": "중복 제거 후 290행이며 외부 검증 데이터가 없음.",
            },
            {
                "limitation": "희소 범주",
                "detail": "ethnicity와 country_of_res는 Selection/Development 내부에서 10건 미만 범주를 Other로 통합하고 보정된 Cramer's V를 사용했지만, 표본이 작아 세부 범주 차이를 일반화하면 안 됨.",
            },
            {
                "limitation": "가중 체크리스트",
                "detail": "100점 가중 체크리스트는 선택 모델의 중요도를 수업 프로젝트용으로 환산한 탐색적 지표이며 공식 AQ-10 채점법이나 임상 진단도구가 아님. 아동학대 여부도 이 점수로 판정할 수 없음.",
            },
            {
                "limitation": "연관성",
                "detail": "p-value, 연관성 크기, Feature Importance는 인과관계를 뜻하지 않음.",
            },
        ]
    )
    limitations.to_csv(ART / "project_limitations.csv", index=False, encoding="utf-8-sig")

    # 16) Streamlit이 빠르게 읽을 수 있도록 핵심 결과를 하나의 JSON으로 요약한다.
    summary = {
        "selected_features": selected_features,
        "selected_model": selected_model_name,
        "validation": model_compare.to_dict(orient="records"),
        "final_test": final_metrics,
        "weighted_checklist": {
            "weights": checklist_weights.to_dict(orient="records"),
            "observe_cutoff": int(checklist_observe_cutoff),
            "high_cutoff": int(checklist_high_cutoff),
            "very_high_cutoff": int(checklist_very_high_cutoff),
            "cutoff_basis": {
                "development_no_max": int(dev_no_max),
                "development_yes_min": int(dev_yes_min),
                "development_yes_median": float(dev_yes_median),
            },
            "validation": validation_weighted.to_dict(orient="records"),
            "final_test": final_weighted.to_dict(orient="records"),
        },
    }
    with open(ART / "dashboard_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("완료")
    print("선택 변수:", selected_features)
    print("선택 모델:", selected_model_name)
    print("Final Test:", final_metrics)


if __name__ == "__main__":
    main()
