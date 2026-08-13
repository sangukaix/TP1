"""NSCH 2024: 4~11세 아동의 ACE 4개 이상 위험신호 분류 모델.

ACE 문항은 결과값(정답)을 만들 때만 사용하며, 모델 입력에는 넣지 않는다.
모든 변수선정·결측 처리·모델 선택은 Train/Validation 자료에서만 수행한다.
"""
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "nsch" / "nsch_2024e_topical.dta"
ART = ROOT / "model_artifacts" / "nsch_ace4_8q"
SEED = 42
FINAL_COUNT = 8
ACE_COLUMNS = ["ace1", "ace3", "ace4", "ace5", "ace6", "ace7", "ace8", "ace9", "ace10"]
# 선택지 순서 자체가 생활환경 강도를 뜻하는 문항은, 설문 점수에서 응답 강도가
# 커질수록 반드시 더 큰 점수가 반영되도록 코드북 순서에 맞춰 고정한다.
ORDERED_RESPONSE_RATIOS = {
    "foodsit": {1: 0.0, 2: 1 / 3, 3: 2 / 3, 4: 1.0},
    "homeevic": {5: 0.0, 4: .25, 3: .50, 2: .75, 1: 1.0},
    "k7q04r_r": {1: 0.0, 2: .50, 3: 1.0},
    "bullied_r": {1: 0.0, 2: .25, 3: .50, 4: .75, 5: 1.0},
}

# 457개 전수 탐색(206개) 뒤, 코드북 기준으로 가족·경제·주거·동네·학교·생활 영역의
# 설문 가능 항목을 넓게 모은 2차 후보군이다. ACE 문항·진단·ID·조사관리 항목은 제외한다.
CANDIDATES = {
    "tenure": ("현재 주거 형태", "주거 안정", "현재 거주지의 소유·임대 형태 확인"),
    "hhlanguage": ("가정에서 주로 사용하는 언어", "가족 환경", "가정 내 의사소통 환경 확인"),
    "family_r": ("가족 형태", "가족 환경", "가구 구성의 생활환경 차이 확인"),
    "hhcount": ("함께 사는 가구원 수", "가족 환경", "가구 규모 확인"),
    "famcount": ("가족 구성원 수", "가족 환경", "가족 구성 규모 확인"),
    "a1_employed_r": ("주 보호자의 현재 고용 상태", "가족·경제", "가구의 일상 경제활동 상황 확인"),
    "a1_grade": ("주 보호자의 최종 학력", "가족·경제", "가구의 교육 자원 관련 특성 확인"),
    "a1_marital": ("주 보호자의 혼인 상태", "가족 환경", "보호자 생활 여건 확인"),
    "a1_relation": ("주 보호자와 아이의 관계", "가족 환경", "주 보호자 관계 확인"),
    "a2_employed_r": ("두 번째 보호자의 고용 상태", "가족·경제", "가구의 경제활동 상황 확인"),
    "a2_grade": ("두 번째 보호자의 최종 학력", "가족·경제", "가구의 교육 자원 관련 특성 확인"),
    "decisions_r": ("가족의 의사결정 참여", "가족 환경", "가정 내 의사결정 경험 확인"),
    "ebtcards": ("식품 지원 카드 사용", "경제", "식품 지원 이용 여부 확인"),
    "s9q34": ("현금 지원 프로그램 이용", "경제", "공적 지원 이용 여부 확인"),
    "ssi": ("보충소득 지원 이용", "경제", "공적 지원 이용 여부 확인"),
    "foodsit": ("최근 12개월 가구 식품 상황", "경제", "식생활 안정성 확인"),
    "missmortgage": ("주거비·임대료 납부 곤란", "주거·경제", "주거비 부담 여부 확인"),
    "everhomeless": ("거주지 없이 지낸 경험", "주거 안정", "주거 불안정 경험 확인"),
    "homeevic": ("주거 퇴거 걱정", "주거 안정", "주거 유지 불안 확인"),
    "k10q40_r": ("아이가 느끼는 동네 안전", "동네 환경", "주거 지역의 안전 인식 확인"),
    "k10q11": ("동네에 보도·놀이 공간이 있음", "동네 환경", "생활 주변 환경 확인"),
    "k10q12": ("동네에 도서관·공원 등 시설이 있음", "동네 환경", "생활 주변 자원 확인"),
    "k10q13": ("동네에서 도움을 받을 이웃이 있음", "동네 환경", "이웃 지원 환경 확인"),
    "k10q14": ("동네가 아동에게 안전함", "동네 환경", "지역 안전 환경 확인"),
    "k10q20": ("동네의 쓰레기·오염 문제", "동네 환경", "주변 환경 문제 확인"),
    "k10q22": ("동네의 노후·불량 주택", "동네 환경", "주변 주거환경 확인"),
    "k10q23": ("동네의 기물파손", "동네 환경", "주변 환경의 훼손 징후 확인"),
    "moves": ("이사 횟수", "주거 안정", "거주지 이동 안정성 확인"),
    "hoursleep": ("평균 수면 시간", "생활습관", "일상 생활리듬 확인"),
    "bedtime": ("평일 취침 시간", "생활습관", "일상 수면 리듬 확인"),
    "physactiv": ("신체활동 시간", "생활습관", "일상 활동 특성 확인"),
    "screentime": ("화면 사용 시간", "생활습관", "일상 미디어 사용 특성 확인"),
    "k10q41_r": ("아이가 느끼는 학교 안전", "학교 환경", "학교 내 안전 인식 확인"),
    "k7q04r_r": ("학교가 가정에 문제로 연락한 횟수", "학교 생활", "학교 적응 관련 신호 확인"),
    "k7q82_r": ("학교에서 잘하고 싶어 하는 마음", "학교 생활", "학교 참여·동기 특성 확인"),
    "k7q83_r": ("학교에서 과제를 끝까지 해냄", "학교 생활", "학교 참여 특성 확인"),
    "bullied_r": ("최근 따돌림·괴롭힘 경험", "학교·사회생활", "또래 환경 확인"),
    "makefriend": ("친구를 사귀거나 유지하는 어려움", "학교·사회생활", "또래 관계 특성 확인"),
    "grades": ("학교 성적", "학교 생활", "학교 적응 특성 확인"),
}
CATEGORICAL = list(CANDIDATES)
CHI_SQUARE = set(CANDIDATES)


def ace4_target(df):
    """9개 ACE 문항이 모두 유효한 4~11세 아동에서 ACE 개수와 4개 이상 결과값을 만든다."""
    age_eligible = df["sc_age_years"].between(4, 11)
    valid = age_eligible & df["ace1"].isin([1.0, 2.0, 3.0, 4.0])
    for col in ACE_COLUMNS[1:]:
        valid &= df[col].isin([1.0, 2.0])
    ace_count = pd.Series(np.nan, index=df.index, dtype="float64")
    experienced = df.loc[valid, "ace1"].isin([3.0, 4.0]).astype(int)
    for col in ACE_COLUMNS[1:]:
        experienced += df.loc[valid, col].eq(1.0).astype(int)
    ace_count.loc[valid] = experienced
    target = pd.Series(pd.NA, index=df.index, dtype="Int64")
    target.loc[valid] = (ace_count.loc[valid] >= 4).astype(int)
    return age_eligible, valid, ace_count, target


def split_target(y):
    """희귀한 ACE 4개 이상 집단의 비율을 유지해 60/20/20으로 나눈다."""
    rng = np.random.default_rng(SEED)
    split = pd.Series(index=y.index, dtype="object")
    for label in [0, 1]:
        idx = y.index[y.eq(label)].to_numpy().copy()
        rng.shuffle(idx)
        train_end = int(len(idx) * .60)
        val_end = train_end + int(len(idx) * .20)
        split.loc[idx[:train_end]] = "train"
        split.loc[idx[train_end:val_end]] = "validation"
        split.loc[idx[val_end:]] = "test"
    return split


def split_severity(severity):
    """0~3 / 4~5 / 6개 이상을 층으로 사용해 희귀한 고누적 사례도 각 자료에 나눈다."""
    rng = np.random.default_rng(SEED)
    split = pd.Series(index=severity.index, dtype="object")
    for label in [0, 1, 2]:
        idx = severity.index[severity.eq(label)].to_numpy().copy()
        rng.shuffle(idx)
        train_end = int(len(idx) * .60)
        val_end = train_end + int(len(idx) * .20)
        split.loc[idx[:train_end]] = "train"
        split.loc[idx[train_end:val_end]] = "validation"
        split.loc[idx[val_end:]] = "test"
    return split


def severity_weight(ace_count):
    """ACE 4개 이상 사례 안에서는 누적 개수에 비례해 4→1.0, 5→1.25, …, 8→2.0으로 반영한다."""
    count = pd.to_numeric(ace_count).astype(float)
    return pd.Series(np.where(count >= 4, count / 4.0, 1.0), index=count.index)


def cramers_v(table):
    chi2, p_value, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    return chi2, p_value, np.sqrt(phi2 / min(k - 1, r - 1))


def association_tests(train, y):
    """Train 자료에서만 후보별 관계의 통계적 유의성과 크기를 계산한다."""
    rows = []
    for col, (label, domain, reason) in CANDIDATES.items():
        if col in CHI_SQUARE:
            statistic, p_value, effect = cramers_v(pd.crosstab(train[col], y))
            method, effect_name = "카이제곱 검정 / Cramér's V", "Cramér's V"
        else:
            statistic, p_value = spearmanr(pd.to_numeric(train[col]), y)
            effect = abs(float(statistic))
            method, effect_name = "Spearman 순위상관분석", "|Spearman ρ|"
        rows.append({"column": col, "korean_name": label, "domain": domain, "reason": reason,
                     "test_method": method, "statistic": statistic, "p_value": p_value,
                     "effect_size": effect, "effect_size_name": effect_name,
                     "significant_0_05": bool(p_value < .05)})
    return pd.DataFrame(rows)


def rf_importance(train, y, stats, sample_weight):
    """범주형은 원-핫 인코딩 후 RF를 학습하고, 더미 중요도를 원래 항목 단위로 합산한다."""
    encoder = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    encoded = encoder.fit_transform(train[list(CANDIDATES)])
    rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, class_weight="balanced")
    rf.fit(encoded, y, sample_weight=sample_weight)
    values, offset = {}, 0
    for col, categories in zip(CATEGORICAL, encoder.named_transformers_["cat"].categories_):
        width = len(categories)
        values[col] = float(rf.feature_importances_[offset: offset + width].sum())
        offset += width
    out = stats.copy()
    out["rf_importance"] = out["column"].map(values)
    out["effect_rank"] = out["effect_size"].rank(ascending=False, method="min")
    out["rf_rank"] = out["rf_importance"].rank(ascending=False, method="min")
    out["selection_rank"] = out["effect_rank"] + out["rf_rank"]
    out = out.sort_values(["selection_rank", "rf_importance"], ascending=[True, False]).reset_index(drop=True)
    out["selected_final"] = False
    out.loc[: FINAL_COUNT - 1, "selected_final"] = True
    out["selection_reason"] = np.where(out["selected_final"], "관계 크기와 랜덤 포레스트 중요도를 함께 고려해 선정", f"최종 {FINAL_COUNT}문항 범위 밖")
    return out


def model_pipeline(selected):
    preprocess = ColumnTransformer([(
        "cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), selected
    )])
    def pipe(model):
        return Pipeline([("preprocess", preprocess), ("model", model)])
    return {
        "LogisticRegression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        "KNN": pipe(Pipeline([("scale", StandardScaler(with_mean=False)), ("model", KNeighborsClassifier(n_neighbors=15))])),
        "DecisionTree": pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=SEED)),
        "RandomForest": pipe(RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=-1)),
    }


def metrics(y, proba, threshold=.5, ace_count=None):
    pred = (proba >= threshold).astype(int)
    result = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, proba)}
    if ace_count is not None:
        weights = severity_weight(ace_count)
        result["severity_weighted_f1"] = f1_score(y, pred, sample_weight=weights, zero_division=0)
        severe = pd.to_numeric(ace_count).ge(6).to_numpy()
        result["ace6plus_recall"] = float(pred[severe].mean()) if severe.any() else np.nan
    return result


def weighted_survey_scores(frame, response_weights):
    """문항별 응답 반영점수를 더해 0~100 생활환경 설문 점수를 계산한다."""
    score = pd.Series(0.0, index=frame.index)
    for column in frame.columns:
        values = response_weights.loc[response_weights["column"].eq(column)].set_index("response_code")["max_contribution"]
        score += pd.to_numeric(frame[column]).map(values).fillna(0.0)
    return score.clip(0, 100)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    age_eligible, valid, ace_count, y_all = ace4_target(df)
    X, y = df.loc[valid, list(CANDIDATES)].copy(), y_all.loc[valid].astype(int)
    counts = ace_count.loc[valid].astype(int)
    severity = pd.Series(np.select([counts.le(3), counts.le(5)], [0, 1], default=2), index=counts.index)
    split = split_severity(severity)
    train, val, test = X.loc[split.eq("train")], X.loc[split.eq("validation")], X.loc[split.eq("test")]
    y_train, y_val, y_test = y.loc[train.index], y.loc[val.index], y.loc[test.index]
    count_train, count_val, count_test = counts.loc[train.index], counts.loc[val.index], counts.loc[test.index]
    weight_train, weight_val = severity_weight(count_train), severity_weight(count_val)

    # 변수선정은 Train만 사용한다.
    stats = association_tests(train, y_train)
    importance = rf_importance(train, y_train, stats, weight_train)
    selected = importance.loc[importance["selected_final"], "column"].tolist()

    models = model_pipeline(selected)
    rows, fitted = [], {}
    for name, model in models.items():
        fit_args = {"model__sample_weight": weight_train} if name != "KNN" else {}
        model.fit(train[selected], y_train, **fit_args)
        proba = model.predict_proba(val[selected])[:, 1]
        fitted[name] = model
        rows.append({"model": name, "split": "validation", **metrics(y_val, proba, ace_count=count_val)})
    comparison = pd.DataFrame(rows).sort_values(["severity_weighted_f1", "roc_auc", "ace6plus_recall"], ascending=False)
    winner = comparison.iloc[0]["model"]
    validation_proba = fitted[winner].predict_proba(val[selected])[:, 1]
    thresholds = np.arange(.05, .96, .01)
    threshold = float(max(thresholds, key=lambda t: f1_score(y_val, validation_proba >= t, sample_weight=weight_val, zero_division=0)))

    # 2차 모델: 1차 위험군(ACE 4개 이상) 안에서 4~5개와 6개 이상 패턴을 구분한다.
    high_train = count_train.ge(4)
    high_val = count_val.ge(4)
    high_test = count_test.ge(4)
    y2_train = count_train.loc[high_train].ge(6).astype(int)
    y2_val = count_val.loc[high_val].ge(6).astype(int)
    y2_test = count_test.loc[high_test].ge(6).astype(int)
    stage2_models = model_pipeline(selected)
    stage2_rows, stage2_fitted = [], {}
    for name, model in stage2_models.items():
        model.fit(train.loc[high_train, selected], y2_train)
        proba = model.predict_proba(val.loc[high_val, selected])[:, 1]
        stage2_fitted[name] = model
        stage2_rows.append({"model": name, "split": "validation", **metrics(y2_val, proba)})
    stage2_comparison = pd.DataFrame(stage2_rows).sort_values(["f1", "roc_auc", "recall"], ascending=False)
    stage2_winner = stage2_comparison.iloc[0]["model"]
    stage2_val_proba = stage2_fitted[stage2_winner].predict_proba(val.loc[high_val, selected])[:, 1]
    stage2_threshold = float(max(thresholds, key=lambda t: f1_score(y2_val, stage2_val_proba >= t, zero_division=0)))

    # 확률이 낮은 구간에 몰리는 불균형 자료 특성상, 화면의 0~100 점수는
    # Validation 예측분포 안에서의 상대 위치(백분위)로 환산한다.
    # 따라서 네 구간은 임의 절대점수가 아니라 같은 검증집단 내 상대적 위치다.
    validation_percentile = pd.Series(validation_proba).rank(method="average", pct=True).to_numpy() * 100
    band_edges = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    band_names = ["낮은 위험신호", "주의 위험신호", "높은 위험신호", "매우 높은 위험신호"]
    validation_bands = pd.cut(validation_percentile, bins=band_edges, labels=band_names, include_lowest=True, duplicates="drop")
    score_bands = (pd.DataFrame({"score_band": validation_bands, "ace4_high": y_val.to_numpy(), "ace_count": count_val.to_numpy(), "score": validation_percentile})
                   .groupby("score_band", observed=True).agg(validation_count=("ace4_high", "size"),
                       observed_ace4_rate=("ace4_high", "mean"), observed_mean_ace_count=("ace_count", "mean"), mean_score=("score", "mean"))
                   .reindex(band_names).reset_index())
    score_bands.insert(1, "lower_score", band_edges[:-1])
    score_bands.insert(2, "upper_score", band_edges[1:])
    calibration = pd.DataFrame({"raw_probability": validation_proba}).sort_values("raw_probability")

    # 선정 모델은 Train+Validation으로 재학습한 뒤, 끝까지 보지 않은 Test로 한 번만 평가한다.
    final_model = models[winner]
    final_X = pd.concat([train[selected], val[selected]])
    final_y = pd.concat([y_train, y_val])
    final_weight = pd.concat([weight_train, weight_val])
    final_args = {"model__sample_weight": final_weight} if winner != "KNN" else {}
    final_model.fit(final_X, final_y, **final_args)
    test_proba = final_model.predict_proba(test[selected])[:, 1]
    test_metrics = {"model": winner, "split": "test", "threshold": threshold, **metrics(y_test, test_proba, threshold, count_test)}

    final_stage2_model = stage2_models[stage2_winner]
    final_stage2_X = pd.concat([train.loc[high_train, selected], val.loc[high_val, selected]])
    final_stage2_y = pd.concat([y2_train, y2_val])
    final_stage2_model.fit(final_stage2_X, final_stage2_y)
    stage2_test_proba = final_stage2_model.predict_proba(test.loc[high_test, selected])[:, 1]
    stage2_test_metrics = {"model": stage2_winner, "split": "test", "threshold": stage2_threshold,
                           **metrics(y2_test, stage2_test_proba, stage2_threshold)}

    target_summary = pd.DataFrame([
        ["raw_rows", len(df)], ["raw_columns", len(df.columns)], ["age_4_11_count", int(age_eligible.sum())],
        ["ace_complete_4_11_count", int(valid.sum())], ["excluded_ace_invalid_4_11", int((age_eligible & ~valid).sum())],
        ["ace4_high_count", int(y.eq(1).sum())], ["ace4_low_count", int(y.eq(0).sum())],
    ], columns=["metric", "value"])
    split_summary = pd.DataFrame({"split": split, "ace4_high": y}).groupby(["split", "ace4_high"]).size().reset_index(name="count")
    severity_summary = (pd.DataFrame({"split": split, "severity_group": severity, "ace_count": counts})
                        .groupby(["split", "severity_group"]).size().reset_index(name="count"))
    severity_summary["severity_group"] = severity_summary["severity_group"].map({0: "ACE 0~3개", 1: "ACE 4~5개", 2: "ACE 6개 이상"})
    candidate_df = pd.DataFrame([{"column": c, "korean_name": n, "domain": d, "reason": r} for c, (n, d, r) in CANDIDATES.items()])
    # 화면에서 결측값 처리 근거를 보여 주기 위한 요약표.
    # 실제 대치는 Pipeline의 SimpleImputer가 Train 최빈값만으로 수행한다.
    imputation_rows = []
    for column in selected:
        train_mode = train[column].mode(dropna=True)
        analysis_missing = int(X[column].isna().sum())
        train_missing = int(train[column].isna().sum())
        if analysis_missing:
            imputation_rows.append({
                "column": column,
                "korean_name": CANDIDATES[column][0],
                "analysis_missing_count": analysis_missing,
                "train_missing_count": train_missing,
                "method": "Train 자료의 최빈 응답으로 대체",
                "train_mode_code": None if train_mode.empty else train_mode.iloc[0],
            })
    imputation_summary = pd.DataFrame(imputation_rows)
    # 화면의 최종 질문 표에서 두 그룹의 실제 응답 차이를 보여 주기 위한 설명용 분포다.
    # 이 표는 모델 선택에 사용하지 않으며, 변수선정·모델학습은 위의 Train 자료만 사용한다.
    group_rows = []
    for column in selected:
        for code, group in pd.DataFrame({"response_code": X[column], "ace4_high": y}).dropna().groupby("response_code"):
            low_count = int(group["ace4_high"].eq(0).sum())
            high_count = int(group["ace4_high"].eq(1).sum())
            low_total = int(y.eq(0).sum())
            high_total = int(y.eq(1).sum())
            group_rows.append({
                "column": column,
                "korean_name": CANDIDATES[column][0],
                "domain": CANDIDATES[column][1],
                "response_code": code,
                "low_count": low_count,
                "high_count": high_count,
                "low_rate": low_count / low_total,
                "high_rate": high_count / high_total,
                "difference_pp": (high_count / high_total - low_count / low_total) * 100,
            })
    group_response_comparison = pd.DataFrame(group_rows)

    # 설문 화면에서 각 응답이 어느 정도 반영되는지 설명할 수 있도록,
    # Train 자료의 위험군 응답 차이와 RF 원래 항목 중요도를 결합한 가중표를 만든다.
    # 이 표는 RF 확률 점수를 대체하지 않고, 응답별 반영 정도를 투명하게 보여 주는 용도다.
    selected_importance = importance.loc[importance["selected_final"], ["column", "rf_importance"]].copy()
    selected_importance["question_weight"] = selected_importance["rf_importance"] / selected_importance["rf_importance"].sum() * 100
    response_weight_rows = []
    low_total = int(y_train.eq(0).sum())
    high_total = int(y_train.eq(1).sum())
    for column in selected:
        feature_weight = float(selected_importance.loc[selected_importance["column"].eq(column), "question_weight"].iloc[0])
        train_responses = pd.DataFrame({"response_code": train[column], "ace4_high": y_train}).dropna()
        grouped = train_responses.groupby("response_code")["ace4_high"]
        response_counts = grouped.agg(high_count="sum", total_count="count").reset_index()
        response_counts["low_count"] = response_counts["total_count"] - response_counts["high_count"]
        response_counts["low_rate"] = response_counts["low_count"] / low_total
        response_counts["high_rate"] = response_counts["high_count"] / high_total
        response_counts["difference_pp"] = (response_counts["high_rate"] - response_counts["low_rate"]) * 100
        max_positive = max(float(response_counts["difference_pp"].clip(lower=0).max()), 1e-9)
        for _, row in response_counts.iterrows():
            observed_ratio = max(float(row["difference_pp"]), 0.0) / max_positive
            ordered_ratio = ORDERED_RESPONSE_RATIOS.get(column, {}).get(int(row["response_code"]), None)
            response_ratio = observed_ratio if ordered_ratio is None else float(ordered_ratio)
            response_weight_rows.append({
                "column": column,
                "response_code": row["response_code"],
                "question_weight": feature_weight,
                "response_difference_pp": float(row["difference_pp"]),
                "response_ratio": response_ratio,
                "score_method": "코드북 응답 순서" if ordered_ratio is not None else "위험군 응답 차이",
                "max_contribution": feature_weight * response_ratio,
            })
    response_weight_summary = pd.DataFrame(response_weight_rows)
    # 설문에서 쓰는 가중합 점수도 Validation으로 임계값과 성능을 확인한다.
    weighted_val_score = weighted_survey_scores(val[selected], response_weight_summary)
    weighted_test_score = weighted_survey_scores(test[selected], response_weight_summary)
    weighted_threshold = float(max(
        thresholds,
        key=lambda t: f1_score(y_val, weighted_val_score.div(100).ge(t), sample_weight=weight_val, zero_division=0),
    ))
    weighted_validation_metrics = {"model": "WeightedSurveyScore", "split": "validation", "threshold": weighted_threshold,
                                   **metrics(y_val, weighted_val_score.div(100), weighted_threshold, count_val)}
    weighted_test_metrics = {"model": "WeightedSurveyScore", "split": "test", "threshold": weighted_threshold,
                             **metrics(y_test, weighted_test_score.div(100), weighted_threshold, count_test)}
    # 화면의 구간·종합판정·검증 통계가 같은 실제 가중점수를 사용하도록 통일한다.
    # 위험신호가 높아지는 경계는 Validation에서 F1이 가장 높았던 임계값을 그대로 쓴다.
    high_start = weighted_threshold * 100
    high_scores = weighted_val_score[weighted_val_score.ge(high_start)]
    very_high_start = float(high_scores.median()) if not high_scores.empty else 75.0
    band_edges = np.array([0.0, 25.0, high_start, very_high_start, 100.0])
    cut_edges = band_edges.copy()
    cut_edges[-1] += 1e-9  # 정확히 100점인 응답도 마지막 구간에 포함한다.
    validation_bands = pd.cut(weighted_val_score, bins=cut_edges, labels=band_names, include_lowest=True, right=False)
    score_bands = (pd.DataFrame({"score_band": validation_bands, "ace4_high": y_val.to_numpy(), "ace_count": count_val.to_numpy(), "score": weighted_val_score.to_numpy()})
                   .groupby("score_band", observed=True).agg(validation_count=("ace4_high", "size"),
                       observed_ace4_rate=("ace4_high", "mean"), observed_mean_ace_count=("ace_count", "mean"), mean_score=("score", "mean"))
                   .reindex(band_names).reset_index())
    score_bands.insert(1, "lower_score", band_edges[:-1])
    score_bands.insert(2, "upper_score", band_edges[1:])
    calibration = pd.DataFrame({"survey_score": weighted_val_score}).sort_values("survey_score")
    target_summary.to_csv(ART / "target_summary.csv", index=False, encoding="utf-8-sig")
    split_summary.to_csv(ART / "split_summary.csv", index=False, encoding="utf-8-sig")
    severity_summary.to_csv(ART / "severity_summary.csv", index=False, encoding="utf-8-sig")
    candidate_df.to_csv(ART / "candidate_features.csv", index=False, encoding="utf-8-sig")
    imputation_summary.to_csv(ART / "imputation_summary.csv", index=False, encoding="utf-8-sig")
    group_response_comparison.to_csv(ART / "group_response_comparison.csv", index=False, encoding="utf-8-sig")
    response_weight_summary.to_csv(ART / "response_weight_summary.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(ART / "statistical_tests.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(ART / "rf_variable_importance.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(ART / "final_selection.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(ART / "model_validation_comparison.csv", index=False, encoding="utf-8-sig")
    stage2_comparison.to_csv(ART / "severity_validation_comparison.csv", index=False, encoding="utf-8-sig")
    score_bands.to_csv(ART / "environment_score_bands.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(ART / "environment_score_calibration.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([weighted_validation_metrics]).to_csv(ART / "weighted_score_validation_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([weighted_test_metrics]).to_csv(ART / "weighted_score_test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([test_metrics]).to_csv(ART / "final_test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([stage2_test_metrics]).to_csv(ART / "severity_test_metrics.csv", index=False, encoding="utf-8-sig")
    with (ART / "final_model_nsch_ace4.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)
    with (ART / "final_model_nsch_severity.pkl").open("wb") as handle:
        pickle.dump(final_stage2_model, handle)
    metadata = {"analysis_question": "4~11세 아동의 ACE 4개 이상 경험 위험신호와 환경·생활 특성",
                "target": {"definition": "ACE 9개 중 4개 이상 = 1, 0~3개 = 0", "age_range": "4~11세", "ace_columns": ACE_COLUMNS},
                "candidate_count": len(CANDIDATES), "final_features": selected, "final_feature_count": len(selected),
                "model": winner, "validation_threshold": threshold,
                "severity_weighting": {"rule": "ACE 0~3개=1.0, ACE 4개 이상은 ACE_COUNT/4", "purpose": "ACE 누적 개수가 많은 사례를 학습과 임계값 선택에 더 크게 반영"},
                "severity_model": {"target": "ACE 4개 이상 집단 안에서 6개 이상=1, 4~5개=0", "model": stage2_winner,
                                   "validation_threshold": stage2_threshold, "use": "1차 위험신호 점수에 누적 단계 패턴을 보조 표시"},
                "three_class_review": {"groups": ["ACE 0~3개", "ACE 4~5개", "ACE 6개 이상"], "decision": "불균형을 줄이기 위해 한 번에 3분류하지 않고 1차 4개 이상 분류 + 2차 6개 이상 분류의 계층형 구조 사용"},
                "environment_score_bands": {"method": "validation weighted-score fixed bands", "source": "validation only", "thresholds": [float(x) for x in band_edges]},
                "survey_score": {"method": "RF 문항 가중치 × 응답별 반영점수의 합", "validation_threshold": weighted_threshold,
                                 "validation_metrics": weighted_validation_metrics, "test_metrics": weighted_test_metrics}}
    (ART / "final_model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"4~11세 ACE 유효 자료: {len(y):,}, ACE 4개 이상: {int(y.sum()):,}, 최종 문항: {selected}")
    print(pd.DataFrame([test_metrics]).to_string(index=False))
    print("2차 누적단계 모델")
    print(pd.DataFrame([stage2_test_metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
