"""NSCH 2024 1단계: 데이터 구조, ACE_HIGH, 후보 변수 목록을 생성한다.

이 스크립트는 탐색·전처리 기준 수립만 수행하며 머신러닝 학습은 하지 않는다.
출처: 2024 NSCH Topical Variable List (U.S. Census Bureau, 2025-10-06).
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "nsch" / "nsch_2024e_topical.dta"
CODEBOOK_PATH = ROOT / "scripts" / "nsch" / "nsch_2024_topical.do"
ART = ROOT / "model_artifacts" / "nsch"

ACE_COLUMNS = ["ace1", "ace3", "ace4", "ace5", "ace6", "ace7", "ace8", "ace9", "ace10"]

# 2024 NSCH codebook: ACE1 is not a Yes/No question.
# 1=Never, 2=Rarely, 3=Somewhat often, 4=Very often.
# To represent economic hardship as an ACE, only 3/4 are counted as experienced.
# ACE3~ACE10 use 1=Yes and 2=No; only 1 is counted as experienced.
ACE_RULES = {
    "ace1": "3=Somewhat often 또는 4=Very often을 ACE 경험(1)으로 계산; 1=Never·2=Rarely는 0",
    "ace3": "1=Yes를 ACE 경험(1)으로 계산; 2=No는 0",
    "ace4": "1=Yes를 ACE 경험(1)으로 계산; 2=No는 0",
    "ace5": "1=Yes를 ACE 경험(1)으로 계산; 2=No는 0",
    "ace6": "1=Yes를 ACE 경험(1)으로 계산; 2=No는 0",
    "ace7": "1=Yes를 ACE 경험(1)으로 계산; 2=No는 0",
    "ace8": "1=Yes를 ACE 경험(1)으로 계산; 2=No는 0",
    "ace9": "1=Yes를 ACE 경험(1)으로 계산; 2=No는 0",
    "ace10": "1=Yes를 ACE 경험(1)으로 계산; 2=No는 0",
}

# 후보 변수는 Target을 직접 구성한 ACE 문항을 제외하고, 사전 측정된 것으로 해석 가능한 영역만 선정한다.
CANDIDATES = {
    "sc_age_years": ("아동 연령(세)", "아동 기본정보", "연령에 따른 ACE 경험 차이를 통제"),
    "sc_sex": ("아동 성별", "아동 기본정보", "기본 인구학적 특성"),
    "sc_race_r": ("아동 인종/민족", "아동 기본정보", "사회적 맥락의 인구학적 특성"),
    "k2q35a": ("ASD 진단 경험 여부", "ASD", "전체 응답자 대상 자폐 스펙트럼 장애 진단 경험"),
    "k2q31a": ("ADHD 진단 경험 여부", "ADHD", "전체 응답자 대상 ADD/ADHD 진단 경험"),
    "k2q33a": ("불안 진단 경험 여부", "불안", "전체 응답자 대상 불안 진단 경험"),
    "k2q32a": ("우울 진단 경험 여부", "우울", "전체 응답자 대상 우울 진단 경험"),
    "k2q34a": ("행동문제 진단 경험 여부", "행동문제", "전체 응답자 대상 행동문제 진단 경험"),
    "a1_menthealth": ("보호자 1 정신·정서 건강", "가족/보호자 환경", "주 보호자 건강 환경"),
    "a2_menthealth": ("보호자 2 정신·정서 건강", "가족/보호자 환경", "두 번째 보호자 건강 환경"),
    "a1_employed_r": ("보호자 1 고용 상태", "가족/보호자 환경", "가구 고용 환경"),
    "a2_employed_r": ("보호자 2 고용 상태", "가족/보호자 환경", "가구 고용 환경"),
    "a1_grade": ("보호자 1 최종 학력", "가족/보호자 환경", "가구 교육 자원"),
    "family_r": ("가족 구조", "가족/보호자 환경", "가구 구성 맥락"),
    "fpl_i1": ("가구 빈곤비율(FPL, 1차 대치값)", "경제 상황", "경제적 자원 수준"),
    "foodsit": ("최근 12개월 가구 식품 상황", "경제 상황", "식품 접근성"),
    "missmortgage": ("주택담보·임대료 미납 여부", "경제 상황", "주거비 부담"),
    "k10q40_r": ("아동의 동네 안전 인식", "동네 안전", "거주 지역 안전"),
    "k10q22": ("동네 노후·불량 주택", "주거 환경", "주거 주변 환경"),
    "k10q23": ("동네 기물파손", "동네 안전", "거주 지역 위험 신호"),
    "moves": ("아동 이사 횟수", "이사 경험", "주거 안정성"),
    "hoursleep": ("최근 1주 평균 수면시간", "수면", "건강 행동 지표"),
    "screentime": ("TV·휴대폰·컴퓨터 사용시간", "스크린타임", "일상 행동 지표"),
    "k10q41_r": ("아동의 학교 안전 인식", "학교/사회생활", "학교 환경 안전"),
    "k7q04r_r": ("학교가 가정에 문제를 연락한 횟수", "학교/사회생활", "학교 적응 신호"),
    "k7q82_r": ("학교에서 잘하려는 태도", "학교/사회생활", "학교 참여·태도"),
    "makefriend": ("친구를 사귀거나 유지하는 어려움", "학교/사회생활", "또래 관계"),
}

ID_OR_SURVEY_COLUMNS = {"fipsst", "stratum", "hhid", "formtype", "year", "fwc"}


def load_codebook_labels():
    """Stata .do 파일의 label var 문에서 영문 설명을 읽는다."""
    text = CODEBOOK_PATH.read_text(encoding="utf-8", errors="replace")
    return dict(re.findall(r'^label var\s+(\w+)\s+"([^"]*)"', text, flags=re.MULTILINE))


def create_target(df):
    """유효한 9개 ACE 응답이 있는 행에만 ACE_COUNT와 ACE_HIGH를 만든다."""
    valid = df["ace1"].isin([1.0, 2.0, 3.0, 4.0])
    for column in ACE_COLUMNS[1:]:
        valid &= df[column].isin([1.0, 2.0])

    ace_count = pd.Series(np.nan, index=df.index, dtype="float64")
    experienced = df.loc[valid, "ace1"].isin([3.0, 4.0]).astype(int)
    for column in ACE_COLUMNS[1:]:
        experienced += df.loc[valid, column].eq(1.0).astype(int)
    ace_count.loc[valid] = experienced

    ace_high = pd.Series(pd.NA, index=df.index, dtype="Int64")
    ace_high.loc[valid] = (ace_count.loc[valid] >= 2).astype(int)
    return valid, ace_count, ace_high


def classify_exclusion(column, missing_pct):
    if column in ACE_COLUMNS or column == "ACE_HIGH" or column == "ACE_COUNT":
        return "Target leakage", "ACE_HIGH를 직접 계산하는 변수이므로 X 사용 금지"
    if column in ID_OR_SURVEY_COLUMNS:
        return "ID/조사 관리용 변수", "식별·표본설계·조사연도·가중치 변수"
    if column.endswith("_if") or column.endswith("_f"):
        return "조사 관리용 변수", "대치·품질 플래그로 분석 변수에서 제외"
    if column.startswith("fpl_i") and column != "fpl_i1":
        return "중복 정보", "다중대치 FPL의 추가 대치본; 1차 대치값만 후보로 유지"
    if missing_pct >= 50:
        return "결측치 과다", "결측 비율이 50% 이상"
    return "현재 프로젝트 목적과 관련 없음", "1단계의 사전 정의 영역 후보에 포함하지 않음"


def stratified_split(target, seed=42):
    """Target을 유지한 60/20/20 분할. Test는 통계검정에 사용하지 않는다."""
    rng = np.random.default_rng(seed)
    split = pd.Series(index=target.index, dtype="object")
    for label in [0, 1]:
        indices = target.index[target.eq(label)].to_numpy().copy()
        rng.shuffle(indices)
        n_train = int(len(indices) * 0.60)
        n_val = int(len(indices) * 0.20)
        split.loc[indices[:n_train]] = "train"
        split.loc[indices[n_train:n_train + n_val]] = "validation"
        split.loc[indices[n_train + n_val:]] = "test"
    return split


def impute_candidates(train, validation, test):
    """Train 통계량만 사용해 후보 변수를 대치한다."""
    train_out, validation_out, test_out = train.copy(), validation.copy(), test.copy()
    rows = []
    for column in CANDIDATES:
        train_missing = int(train[column].isna().sum())
        validation_missing = int(validation[column].isna().sum())
        test_missing = int(test[column].isna().sum())
        unique_count = int(train[column].dropna().nunique())
        if pd.api.types.is_numeric_dtype(train[column]) and unique_count > 10:
            method = "train median"
            fill_value = float(train[column].median())
            reason = "연속형·수치형 변수의 극단값 영향을 줄이고 Train 정보만 사용"
        else:
            method = "train mode"
            mode = train[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else 0
            reason = "이진·범주형 응답에서 가장 일반적인 Train 응답으로 대치"
        train_out[column] = train[column].fillna(fill_value)
        validation_out[column] = validation[column].fillna(fill_value)
        test_out[column] = test[column].fillna(fill_value)
        rows.append({
            "column": column,
            "method": method,
            "fill_value_fitted_on_train": fill_value,
            "reason": reason,
            "train_missing_before": train_missing,
            "validation_missing_before": validation_missing,
            "test_missing_before": test_missing,
            "train_missing_after": int(train_out[column].isna().sum()),
            "validation_missing_after": int(validation_out[column].isna().sum()),
            "test_missing_after": int(test_out[column].isna().sum()),
        })
    return train_out, validation_out, test_out, pd.DataFrame(rows)


def run_statistical_tests(train, target):
    """Train만 사용해 Pearson 또는 Spearman 상관과 p-value를 계산한다."""
    rows = []
    for column in CANDIDATES:
        x = pd.to_numeric(train[column], errors="coerce")
        y = pd.to_numeric(target, errors="coerce")
        unique_count = int(x.nunique())
        if unique_count <= 1:
            statistic, p_value, method = np.nan, np.nan, "not testable (constant)"
        elif unique_count == 2:
            statistic, p_value = pearsonr(x, y)
            method = "Pearson (binary/point-biserial)"
        else:
            statistic, p_value = spearmanr(x, y)
            method = "Spearman (ordinal/numeric)"
        strength = abs(float(statistic)) if pd.notna(statistic) else np.nan
        strength_label = (
            "strong" if pd.notna(strength) and strength >= 0.30 else
            "moderate" if pd.notna(strength) and strength >= 0.10 else
            "weak" if pd.notna(strength) else "not testable"
        )
        rows.append({
            "column": column,
            "test_set": "train only",
            "test_method": method,
            "n": len(train),
            "statistic": statistic,
            "p_value": p_value,
            "absolute_relationship_strength": strength,
            "relationship_strength_label": strength_label,
            "significant_at_0_05": bool(pd.notna(p_value) and p_value < 0.05),
        })
    return pd.DataFrame(rows).sort_values(["p_value", "column"], na_position="last")


def main():
    ART.mkdir(parents=True, exist_ok=True)
    df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    labels = load_codebook_labels()
    valid_target, ace_count, ace_high = create_target(df)

    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(4)
    missing_summary = pd.DataFrame({
        "column": df.columns,
        "english_description": [labels.get(column, column) for column in df.columns],
        "dtype": [str(df[column].dtype) for column in df.columns],
        "missing_count": [int(missing_count[column]) for column in df.columns],
        "missing_pct": [float(missing_pct[column]) for column in df.columns],
    })
    missing_summary.to_csv(ART / "missing_summary.csv", index=False, encoding="utf-8-sig")

    data_summary = pd.DataFrame([
        ("raw_row_count", len(df)),
        ("raw_column_count", len(df.columns)),
        ("duplicate_row_count", int(df.duplicated().sum())),
        ("duplicate_row_pct", round(df.duplicated().mean() * 100, 4)),
        ("ace_complete_case_count", int(valid_target.sum())),
        ("ace_complete_case_pct", round(valid_target.mean() * 100, 4)),
        ("candidate_feature_count", len(CANDIDATES)),
    ], columns=["metric", "value"])
    data_summary.to_csv(ART / "data_summary.csv", index=False, encoding="utf-8-sig")

    candidate_rows = []
    for column, (korean, domain, reason) in CANDIDATES.items():
        candidate_rows.append({
            "column": column,
            "korean_description": korean,
            "domain": domain,
            "use_for_ml_candidate": "Y",
            "reason": reason,
            "english_description": labels.get(column, column),
            "dtype": str(df[column].dtype),
            "missing_count": int(missing_count[column]),
            "missing_pct": float(missing_pct[column]),
        })
    pd.DataFrame(candidate_rows).to_csv(ART / "candidate_features.csv", index=False, encoding="utf-8-sig")

    excluded_rows = []
    for column in df.columns:
        if column in CANDIDATES:
            continue
        category, reason = classify_exclusion(column, float(missing_pct[column]))
        excluded_rows.append({
            "column": column,
            "english_description": labels.get(column, column),
            "exclusion_category": category,
            "reason": reason,
            "dtype": str(df[column].dtype),
            "missing_count": int(missing_count[column]),
            "missing_pct": float(missing_pct[column]),
        })
    pd.DataFrame(excluded_rows).to_csv(ART / "excluded_features.csv", index=False, encoding="utf-8-sig")

    leakage_rows = pd.DataFrame([
        {"column": column, "reason": "ACE_HIGH 계산 직접 입력값. X 사용 금지."}
        for column in ACE_COLUMNS
    ] + [{"column": "ACE_COUNT", "reason": "ACE 문항 합계. X 사용 금지."},
         {"column": "ACE_HIGH", "reason": "분석 Target. X 사용 금지."}])
    leakage_rows.to_csv(ART / "leakage_features.csv", index=False, encoding="utf-8-sig")

    ace_rows = []
    for column in ACE_COLUMNS:
        for value, count in df[column].value_counts(dropna=False).sort_index().items():
            ace_rows.append({
                "section": "ACE response distribution",
                "column": column,
                "rule": ACE_RULES[column],
                "value": "missing" if pd.isna(value) else int(value),
                "count": int(count),
                "pct": round(float(count) / len(df) * 100, 4),
            })
    target_rows = pd.DataFrame(ace_rows + [
        {"section": "ACE_HIGH", "column": "ACE_HIGH", "rule": "ACE_COUNT 0~1 = 0; 2 이상 = 1", "value": "0", "count": int((ace_high == 0).sum()), "pct": round(float((ace_high == 0).mean()) * 100, 4)},
        {"section": "ACE_HIGH", "column": "ACE_HIGH", "rule": "ACE_COUNT 0~1 = 0; 2 이상 = 1", "value": "1", "count": int((ace_high == 1).sum()), "pct": round(float((ace_high == 1).mean()) * 100, 4)},
        {"section": "ACE_HIGH", "column": "ACE_HIGH", "rule": "9개 ACE 문항 모두 정상 응답", "value": "usable rows", "count": int(valid_target.sum()), "pct": round(float(valid_target.mean()) * 100, 4)},
    ])
    target_rows.to_csv(ART / "target_summary.csv", index=False, encoding="utf-8-sig")

    # 2단계: ACE_HIGH 완전응답 행만 60/20/20으로 분할한다.
    # 통계검정·결측 대치 기준은 Train에만 적합하며, Test는 여기서 사용하지 않는다.
    usable = df.loc[valid_target, list(CANDIDATES)].copy()
    usable_target = ace_high.loc[valid_target].astype(int)
    split = stratified_split(usable_target)
    split_summary = (
        pd.DataFrame({"split": split, "ACE_HIGH": usable_target})
        .groupby(["split", "ACE_HIGH"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    split_summary["pct_of_usable"] = (split_summary["count"] / len(usable) * 100).round(4)
    split_summary.to_csv(ART / "split_summary.csv", index=False, encoding="utf-8-sig")

    train = usable.loc[split.eq("train")]
    validation = usable.loc[split.eq("validation")]
    test = usable.loc[split.eq("test")]
    train_target = usable_target.loc[split.eq("train")]
    validation_target = usable_target.loc[split.eq("validation")]
    test_target = usable_target.loc[split.eq("test")]
    train, validation, test, imputation_summary = impute_candidates(train, validation, test)
    imputation_summary.to_csv(ART / "missing_treatment.csv", index=False, encoding="utf-8-sig")

    # 변수선정용 통계는 Train에서만 계산한다. Validation/Test는 p-value 계산에 사용하지 않는다.
    statistical_tests = run_statistical_tests(train, train_target)
    statistical_tests.to_csv(ART / "statistical_tests.csv", index=False, encoding="utf-8-sig")

    print(f"원본 행/열: {df.shape[0]:,} / {df.shape[1]:,}")
    print(f"중복 행: {int(df.duplicated().sum()):,}")
    print(f"ACE_HIGH 사용 가능 행: {int(valid_target.sum()):,}")
    print(f"후보 컬럼 수: {len(CANDIDATES)}")
    print(f"Train/Validation/Test: {len(train):,}/{len(validation):,}/{len(test):,}")
    print(f"Train 유의 변수(p<0.05): {int(statistical_tests['significant_at_0_05'].sum())}")
    print("머신러닝 학습은 수행하지 않았습니다.")


if __name__ == "__main__":
    main()
