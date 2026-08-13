"""NSCH 2024 전수 후보 탐색: ACE 4개 이상과 관련된 항목을 넓게 확인한다.

기존 14개 설문 후보를 바꾸기 전에, 457개 컬럼에서 기술·식별·정답 누수 항목만
제외하고 통계적 관계 크기와 Random Forest 중요도를 계산한다. 이 파일은 탐색용이며
기존 최종 모델이나 앱 산출물을 덮어쓰지 않는다.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from modeling_nsch_ace4 import DATA_PATH, ACE_COLUMNS, ace4_target, cramers_v, split_severity, severity_weight


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "model_artifacts" / "nsch_ace4_exploration"
SEED = 42

# 응답자 식별·표본설계·가중치·파생 플래그는 생활환경 요인이 아니므로 제외한다.
# ACE 문항과 ACE11은 이번 분석의 결과값과 직접 겹치므로 절대 입력 후보에 넣지 않는다.
EXACT_EXCLUDE = set(ACE_COLUMNS + [
    "ace11", "height", "hhid", "stratum", "fipsst", "formtype", "year", "weight",
    "tenure_if", "fpl_if", "a1_grade_if", "hhcount_if", "sc_race_r_if",
    "sc_hispanic_r_if", "sc_sex_if", "house_gen",
])
PREFIX_EXCLUDE = (
    "tot", "sc_", "birth_", "fwc", "cbsa", "metro", "mpc", "agepos",
)
SUFFIX_EXCLUDE = ("_if", "_desc")


def is_eligible(column, series, n_rows):
    """기술 항목과 결측·희소 범주가 심한 항목을 전수 탐색 후보에서 제외한다."""
    observed = int(series.notna().sum())
    categories = int(series.nunique(dropna=True))
    technical = (
        column in EXACT_EXCLUDE
        or column.startswith(PREFIX_EXCLUDE)
        or column.endswith(SUFFIX_EXCLUDE)
    )
    return not technical and observed >= n_rows * .65 and 2 <= categories <= 20


def rough_domain(column):
    """결과를 읽기 쉽게 넓은 자료 영역만 붙인다. 최종 설문 선정 기준은 아니다."""
    if column.startswith(("a1_", "a2_", "family", "hh", "fam", "tenure")):
        return "가족·가구"
    if column.startswith(("k10q", "missmortgage", "homeevic", "everhomeless", "moves")):
        return "주거·동네·경제"
    if column.startswith(("k7q", "k8q", "grades", "makefriend", "bull", "startschool")):
        return "학교·사회생활"
    if column in {"foodsit", "ebtcards", "ssi", "s9q34"}:
        return "경제·지원"
    if column in {"hoursleep", "screentime", "physactiv", "bedtime", "sugardrink", "vegetables", "fruit"}:
        return "생활습관"
    if column.startswith(("k2q", "k4q", "k5q", "k6q", "k11q", "k12q", "dental", "allerg", "heart", "diabetes", "autism", "fasd")):
        return "건강·의료 관련"
    return "기타 설문 항목"


def main():
    ART.mkdir(parents=True, exist_ok=True)
    df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    age_eligible, valid, ace_count, y_all = ace4_target(df)
    source = df.loc[valid].copy()
    y = y_all.loc[valid].astype(int)
    counts = ace_count.loc[valid].astype(int)

    eligible = [c for c in source if is_eligible(c, source[c], len(source))]
    severity = pd.Series(np.select([counts.le(3), counts.le(5)], [0, 1], default=2), index=counts.index)
    split = split_severity(severity)
    train = source.loc[split.eq("train"), eligible]
    y_train = y.loc[train.index]
    weights = severity_weight(counts.loc[train.index])

    # 1) 각 후보와 결과값의 관계 크기: 결측은 해당 항목의 실제 응답만 이용한다.
    rows = []
    for column in eligible:
        paired = pd.DataFrame({"x": train[column], "y": y_train}).dropna()
        table = pd.crosstab(paired["x"], paired["y"])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        statistic, p_value, effect_size = cramers_v(table)
        rows.append({
            "column": column,
            "domain": rough_domain(column),
            "nonmissing_count": len(paired),
            "missing_rate": 1 - len(paired) / len(train),
            "category_count": int(train[column].nunique(dropna=True)),
            "test_method": "카이제곱 검정 / Cramér's V",
            "chi_square": statistic,
            "p_value": p_value,
            "effect_size": effect_size,
        })
    screen = pd.DataFrame(rows)

    # 2) 모든 후보를 동시에 원-핫 변환한 Random Forest로 분류 중요도를 계산한다.
    preprocess = ColumnTransformer([(
        "all", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), eligible
    )])
    encoded = preprocess.fit_transform(train)
    forest = RandomForestClassifier(
        n_estimators=500, random_state=SEED, n_jobs=-1, class_weight="balanced_subsample",
        min_samples_leaf=5,
    )
    forest.fit(encoded, y_train, sample_weight=weights)
    importances, offset = {}, 0
    categories = preprocess.named_transformers_["all"].named_steps["onehot"].categories_
    for column, cats in zip(eligible, categories):
        width = len(cats)
        importances[column] = float(forest.feature_importances_[offset:offset + width].sum())
        offset += width

    screen["rf_importance"] = screen["column"].map(importances)
    screen["effect_rank"] = screen["effect_size"].rank(ascending=False, method="min")
    screen["rf_rank"] = screen["rf_importance"].rank(ascending=False, method="min")
    screen["combined_rank"] = screen["effect_rank"] + screen["rf_rank"]
    screen = screen.sort_values(["combined_rank", "rf_importance"], ascending=[True, False]).reset_index(drop=True)
    screen.insert(0, "screen_rank", np.arange(1, len(screen) + 1))
    screen["significant_0_05"] = screen["p_value"] < .05
    screen.to_csv(ART / "full_candidate_screen.csv", index=False, encoding="utf-8-sig")

    summary = {
        "raw_columns": int(df.shape[1]),
        "age_4_11_count": int(age_eligible.sum()),
        "ace_valid_4_11_count": int(valid.sum()),
        "ace4_high_count": int(y.sum()),
        "technical_or_unusable_excluded": int(df.shape[1] - len(eligible)),
        "screened_candidate_count": int(len(eligible)),
        "significant_candidate_count": int(screen["significant_0_05"].sum()),
    }
    (ART / "screen_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # Windows 콘솔의 기본 cp949 인코딩에서 Cramér's V의 특수문자가 깨질 수 있어,
    # 상세 순위는 저장한 UTF-8 CSV로 확인한다.
    print("상위 30개 항목은 full_candidate_screen.csv에 저장했습니다.")


if __name__ == "__main__":
    main()
