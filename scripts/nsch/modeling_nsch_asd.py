"""NSCH 2024: current ASD status and environmental/lifestyle characteristics.

Keeps the earlier ACE analysis untouched.  All feature selection uses train only.
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
ART = ROOT / "model_artifacts" / "nsch_asd"

# Environment/lifestyle candidates only. ASD diagnosis, severity, treatment, other diagnoses,
# IDs, survey design fields, and imputation flags are deliberately excluded.
CANDIDATES = {
    "family_r": ("가족 형태", "가족·보호자 환경", "가구 구성 맥락"),
    "a1_employed_r": ("보호자 1의 고용 상태", "가족·보호자 환경", "가구 고용 환경"),
    "a2_employed_r": ("보호자 2의 고용 상태", "가족·보호자 환경", "가구 고용 환경"),
    "a1_grade": ("보호자 1의 최종 학력", "가족·보호자 환경", "가구 교육 자원"),
    "fpl_i1": ("가구 빈곤수준", "경제 상황", "경제적 자원 수준"),
    "foodsit": ("최근 12개월 가구 식품 상황", "경제 상황", "식품 안정성"),
    "missmortgage": ("주거비·임대료 납부 곤란", "경제 상황", "주거비 부담"),
    "k10q40_r": ("아동의 동네 안전 인식", "주거·동네 환경", "거주 지역 안전"),
    "k10q22": ("동네의 노후·불량 주택", "주거·동네 환경", "주거 주변 환경"),
    "k10q23": ("동네의 기물파손", "주거·동네 환경", "거주 지역 위험 신호"),
    "moves": ("아동의 이사 횟수", "주거 안정성", "주거 안정성"),
    "hoursleep": ("최근 1주 평균 수면시간", "생활습관", "수면 습관"),
    "screentime": ("TV·휴대폰·컴퓨터 사용시간", "생활습관", "화면 사용 습관"),
    "k10q41_r": ("학교가 안전하다는 인식", "학교·사회생활", "학교 환경 안전"),
    "k7q04r_r": ("학교가 문제로 가정에 연락한 횟수", "학교·사회생활", "학교 적응 신호"),
    "k7q82_r": ("학교에서 잘하고 싶은 마음", "학교·사회생활", "학교 참여·태도"),
    "makefriend": ("친구를 사귀거나 유지하는 어려움", "학교·사회생활", "또래 관계"),
}

# Values represent categories rather than magnitudes in the model.
CATEGORICAL = [c for c in CANDIDATES if c != "fpl_i1"]
NUMERIC = ["fpl_i1"]
CHI_SQUARE = {"family_r", "a1_employed_r", "a2_employed_r", "foodsit", "missmortgage", "k10q40_r", "k10q22", "k10q23"}
FINAL_COUNT = 10
SEED = 42


def current_asd_target(df):
    """Return current ASD: yes=k2q35a 1 and k2q35b 1; no=ever no or current no.

    k2q35b is logically skipped for children whose k2q35a is No.
    """
    ever = df["k2q35a"]
    current = df["k2q35b"]
    yes = ever.eq(1) & current.eq(1)
    no = ever.eq(2) | (ever.eq(1) & current.eq(2))
    target = pd.Series(pd.NA, index=df.index, dtype="Int64")
    target.loc[yes] = 1
    target.loc[no] = 0
    return target


def split_target(y):
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


def train_impute(train, *others):
    fills, rows = {}, []
    for col in CANDIDATES:
        if col in NUMERIC:
            value, method = float(train[col].median()), "train median"
        else:
            mode = train[col].mode(dropna=True)
            value, method = (mode.iloc[0] if not mode.empty else 0), "train mode"
        fills[col] = value
        rows.append({"column": col, "method": method, "fill_value": value, "train_missing": int(train[col].isna().sum())})
    outputs = [frame.fillna(fills) for frame in (train, *others)]
    return (*outputs, pd.DataFrame(rows))


def cramers_v(table):
    chi2, p_value, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    return chi2, p_value, np.sqrt(phi2 / min(k - 1, r - 1))


def association_tests(train, y):
    rows = []
    for col, (label, domain, _) in CANDIDATES.items():
        if col in CHI_SQUARE:
            table = pd.crosstab(train[col], y)
            statistic, p_value, effect = cramers_v(table)
            method = "Chi-square / Cramér's V"
            effect_name = "Cramér's V"
        else:
            statistic, p_value = spearmanr(pd.to_numeric(train[col]), y)
            effect = abs(float(statistic))
            method = "Spearman rank correlation"
            effect_name = "|Spearman rho|"
        rows.append({
            "column": col, "korean_name": label, "domain": domain, "test_method": method,
            "statistic": statistic, "p_value": p_value, "effect_size": effect,
            "effect_size_name": effect_name, "significant_0_05": bool(p_value < .05),
        })
    return pd.DataFrame(rows).sort_values(["p_value", "effect_size"], ascending=[True, False])


def rf_importance(train, y, stats):
    encoder = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    encoded = encoder.fit_transform(train[list(CANDIDATES)])
    rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, class_weight="balanced")
    rf.fit(encoded, y)
    values, offset = {}, 0
    for col, categories in zip(CATEGORICAL, encoder.named_transformers_["cat"].categories_):
        width = len(categories)
        values[col] = float(rf.feature_importances_[offset:offset + width].sum())
        offset += width
    for col in NUMERIC:
        values[col] = float(rf.feature_importances_[offset])
        offset += 1
    out = stats.merge(pd.DataFrame({"column": list(CANDIDATES), "rf_importance": [values[c] for c in CANDIDATES]}), on="column")
    out["effect_rank"] = out["effect_size"].rank(ascending=False, method="min")
    out["rf_rank"] = out["rf_importance"].rank(ascending=False, method="min")
    out["selection_rank"] = out["effect_rank"] + out["rf_rank"]
    out = out.sort_values(["selection_rank", "rf_importance"], ascending=[True, False])
    out["selected_final"] = False
    out.loc[out.index[:FINAL_COUNT], "selected_final"] = True
    out["selection_reason"] = np.where(out["selected_final"], "통계적 관계·관계 크기·RF 중요도·설문 활용 가능성을 함께 고려", "최종 설문 길이 제한 또는 상대적 중요도·관계 크기 낮음")
    return out


def model_pipeline(selected):
    categorical = [c for c in selected if c in CATEGORICAL]
    numeric = [c for c in selected if c in NUMERIC]
    preprocess = ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ])
    def pipe(model):
        return Pipeline([("preprocess", preprocess), ("model", model)])
    return {
        "LogisticRegression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        "KNN": pipe(KNeighborsClassifier(n_neighbors=15)),
        "DecisionTree": pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=SEED)),
        "RandomForest": pipe(RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=-1)),
    }


def metrics(y, proba, threshold=.5):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def main():
    ART.mkdir(parents=True, exist_ok=True)
    df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    y = current_asd_target(df)
    usable = y.notna()
    X, y = df.loc[usable, list(CANDIDATES)].copy(), y.loc[usable].astype(int)
    split = split_target(y)
    train, val, test = X.loc[split.eq("train")], X.loc[split.eq("validation")], X.loc[split.eq("test")]
    y_train, y_val, y_test = y.loc[train.index], y.loc[val.index], y.loc[test.index]
    train_imp, val_imp, test_imp, impute_summary = train_impute(train, val, test)
    stats = association_tests(train_imp, y_train)
    importance = rf_importance(train_imp, y_train, stats)
    selected = importance.loc[importance["selected_final"], "column"].tolist()

    models = model_pipeline(selected)
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(train, y_train)
        proba = model.predict_proba(val)[:, 1]
        fitted[name] = model
        rows.append({"model": name, "split": "validation", **metrics(y_val, proba)})
    comparison = pd.DataFrame(rows).sort_values(["roc_auc", "f1", "recall"], ascending=False)
    winner = comparison.iloc[0]["model"]
    validation_proba = fitted[winner].predict_proba(val)[:, 1]
    thresholds = np.arange(.05, .96, .01)
    threshold = float(max(thresholds, key=lambda t: f1_score(y_val, validation_proba >= t, zero_division=0)))
    # Four descriptive score bands are set from validation-score quartiles only.
    # They describe relative score position; the binary 2×2 high/low rule remains
    # the validation F1-optimal threshold above.
    quartiles = np.quantile(validation_proba, [.25, .50, .75])
    band_edges = np.r_[0.0, quartiles, 1.0]
    band_names = ["낮은 점수 구간", "중간-낮은 점수 구간", "중간-높은 점수 구간", "높은 점수 구간"]
    validation_bands = pd.cut(validation_proba, bins=band_edges, include_lowest=True, labels=band_names)
    score_bands = (
        pd.DataFrame({"score_band": validation_bands, "current_asd": y_val.to_numpy(), "score": validation_proba})
        .groupby("score_band", observed=True)
        .agg(validation_count=("current_asd", "size"), observed_asd_rate=("current_asd", "mean"), mean_score=("score", "mean"))
        .reindex(band_names)
        .reset_index()
    )
    score_bands.insert(1, "lower_score", band_edges[:-1])
    score_bands.insert(2, "upper_score", band_edges[1:])

    final_model = models[winner]
    final_model.fit(pd.concat([train, val]), pd.concat([y_train, y_val]))
    test_proba = final_model.predict_proba(test)[:, 1]
    test_metrics = {"model": winner, "split": "test", "threshold": threshold, **metrics(y_test, test_proba, threshold)}

    target_summary = pd.DataFrame([
        ["raw_rows", len(df)], ["raw_columns", len(df.columns)], ["usable_current_asd", int(usable.sum())],
        ["excluded_target_invalid", int((~usable).sum())], ["current_asd_yes", int(y.eq(1).sum())], ["current_asd_no", int(y.eq(0).sum())],
    ], columns=["metric", "value"])
    split_summary = pd.DataFrame({"split": split, "current_asd": y}).groupby(["split", "current_asd"]).size().reset_index(name="count")
    candidate_df = pd.DataFrame([{"column": c, "korean_name": n, "domain": d, "reason": r} for c, (n, d, r) in CANDIDATES.items()])
    final_selection = importance[["column", "korean_name", "domain", "test_method", "statistic", "p_value", "effect_size", "rf_importance", "rf_rank", "selected_final", "selection_reason"]]

    target_summary.to_csv(ART / "target_summary.csv", index=False, encoding="utf-8-sig")
    split_summary.to_csv(ART / "split_summary.csv", index=False, encoding="utf-8-sig")
    candidate_df.to_csv(ART / "candidate_features.csv", index=False, encoding="utf-8-sig")
    impute_summary.to_csv(ART / "imputation_summary.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(ART / "statistical_tests.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(ART / "rf_variable_importance.csv", index=False, encoding="utf-8-sig")
    final_selection.to_csv(ART / "final_selection.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(ART / "model_validation_comparison.csv", index=False, encoding="utf-8-sig")
    score_bands.to_csv(ART / "environment_score_bands.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([test_metrics]).to_csv(ART / "final_test_metrics.csv", index=False, encoding="utf-8-sig")
    with (ART / "final_model_nsch_asd.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)
    metadata = {
        "analysis_question": "Current ASD status and environmental/lifestyle characteristics",
        "target": {"ever_asd_column": "k2q35a", "current_asd_column": "k2q35b", "yes_rule": "k2q35a=1 and k2q35b=1", "no_rule": "k2q35a=2 or (k2q35a=1 and k2q35b=2)", "excluded": "missing, logical skip, not in universe, or invalid current status"},
        "candidate_count": len(CANDIDATES), "final_features": selected, "final_feature_count": len(selected),
        "model": winner, "validation_threshold": threshold,
        "environment_score_bands": {"method": "validation prediction-score quartiles", "source": "validation only", "thresholds": [float(x) for x in band_edges]},
        "categorical_features": [c for c in selected if c in CATEGORICAL], "numeric_features": [c for c in selected if c in NUMERIC],
    }
    (ART / "final_model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
