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
    """수업 범위 안에서 데이터를 정리한다."""
    df = raw.copy()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("string").str.strip().str.strip("'").str.strip('"')
        df[c] = df[c].replace("?", pd.NA)

    df = df.rename(
        columns={
            "jundice": "jaundice",
            "austim": "family_asd",
            "contry_of_res": "country_of_res",
        }
    )
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)

    # 합의한 결측값 처리
    age_median = float(df["age"].median())
    df["age"] = df["age"].fillna(age_median)
    df["ethnicity"] = df["ethnicity"].fillna("Unknown")
    df["relation"] = df["relation"].fillna("Unknown")

    df["target"] = df[TARGET].map({"NO": 0, "YES": 1}).astype(int)
    return df


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }


def build_preprocessor(columns):
    behavior_cols = [c for c in columns if c in BEHAVIOR]
    numeric_cols = [c for c in columns if c == "age"]
    categorical_cols = [c for c in columns if c in BACKGROUND and c != "age"]

    transformers = []
    if behavior_cols:
        transformers.append(("behavior", "passthrough", behavior_cols))
    if numeric_cols:
        transformers.append(("numeric", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def metrics(y_true, pred, proba):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def bias_corrected_cramers_v(table):
    """범주 수가 많을 때 커질 수 있는 Cramer's V 편향을 보정한다."""
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
    """Selection/Development의 빈도만 보고 유지할 범주를 정한다."""
    maps = {}
    for feature in RARE_GROUP_FEATURES:
        counts = train_df[feature].value_counts(dropna=False)
        keep = counts[counts >= min_count].index.tolist()
        maps[feature] = keep
    return maps


def apply_rare_category_maps(df, maps):
    out = df.copy()
    for feature, keep in maps.items():
        out[feature] = out[feature].where(out[feature].isin(keep), RARE_LABEL)
    return out


def grouping_summary(before_df, after_df, maps, stage):
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
    """모델 입력 변수를 고르기 위한 18개 요인 연관성 분석."""
    rows = []
    for feature in ALL_CANDIDATES:
        if feature == "age":
            temp = selection_train[[feature, "target"]].dropna()
            no_vals = temp.loc[temp["target"].eq(0), feature].astype(float)
            yes_vals = temp.loc[temp["target"].eq(1), feature].astype(float)
            stat, p_value = ttest_ind(no_vals, yes_vals, equal_var=False)
            r_value, _ = pointbiserialr(temp["target"], temp[feature].astype(float))
            effect = abs(float(r_value))
            sparse_ratio = np.nan
            caution = ""
            test = "Welch t-test"
        else:
            temp = selection_train[[feature, "target"]].dropna()
            table = pd.crosstab(temp[feature], temp["target"])
            if table.shape[0] < 2 or table.shape[1] < 2:
                stat, p_value = 0.0, 1.0
                expected = np.ones_like(table, dtype=float)
                effect = 0.0
            else:
                stat, p_value, _, expected = chi2_contingency(table)
                effect = bias_corrected_cramers_v(table)
            sparse_ratio = float((np.asarray(expected) < 5).mean()) if np.asarray(expected).size else np.nan
            caution = "희소 범주 주의" if sparse_ratio > 0.20 else ""
            test = "Chi-square"

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

    out = pd.DataFrame(rows).sort_values(
        ["effect_size", "p_value"], ascending=[False, True]
    ).reset_index(drop=True)
    out["association_rank"] = np.arange(1, len(out) + 1)
    return out


def evaluate_models(features, train_df, val_df):
    rows = []
    fitted = {}
    for model_name, estimator in make_models().items():
        pipe = Pipeline(
            [
                ("preprocessor", build_preprocessor(features)),
                ("model", estimator),
            ]
        )
        pipe.fit(train_df[features], train_df["target"])
        pred = pipe.predict(val_df[features])
        proba = pipe.predict_proba(val_df[features])[:, 1]
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
        row.update(metrics(val_df["target"], pred, proba))
        rows.append(row)
        fitted[model_name] = pipe

    out = pd.DataFrame(rows).sort_values(
        ["f1", "recall", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    out["selection_rank"] = np.arange(1, len(out) + 1)
    return out, fitted


def raw_feature_from_transformed(name):
    if name.startswith("behavior__") or name.startswith("numeric__"):
        return name.split("__", 1)[1]
    text = name.split("__", 1)[-1]
    for feature in sorted([x for x in BACKGROUND if x != "age"], key=len, reverse=True):
        if text == feature or text.startswith(feature + "_"):
            return feature
    return text


def selected_feature_importance(features, train_df):
    lr = Pipeline(
        [
            ("preprocessor", build_preprocessor(features)),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    rf = Pipeline(
        [
            ("preprocessor", build_preprocessor(features)),
            ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ]
    )
    lr.fit(train_df[features], train_df["target"])
    rf.fit(train_df[features], train_df["target"])

    feature_names = lr.named_steps["preprocessor"].get_feature_names_out()
    lr_abs = np.abs(lr.named_steps["model"].coef_[0])
    rf_imp = rf.named_steps["model"].feature_importances_

    rows = []
    for feature in features:
        idx = [i for i, n in enumerate(feature_names) if raw_feature_from_transformed(n) == feature]
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
    out = pd.DataFrame(rows)
    out["logistic_normalized"] = out["logistic_group_importance"] / out["logistic_group_importance"].sum()
    out["random_forest_normalized"] = out["random_forest_group_importance"] / out["random_forest_group_importance"].sum()
    out["combined_importance"] = (out["logistic_normalized"] + out["random_forest_normalized"]) / 2
    out = out.sort_values("combined_importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def integer_weight_points(importance_df, selected_model_name, total_points=100):
    """선택 모델의 중요도를 정수 점수로 바꾸되 전체 합을 100점으로 맞춘다."""
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
    normalized = values / values.sum()
    raw_points = normalized * total_points
    base_points = np.floor(raw_points).astype(int)
    remainder = int(total_points - base_points.sum())
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
    score = pd.Series(0, index=df.index, dtype=int)
    for _, row in weights_df.iterrows():
        feature = row["feature"]
        if feature in df.columns:
            score = score + df[feature].fillna(0).astype(int) * int(row["points"])
    return score.astype(int)


def weighted_threshold_table(df, weights_df, thresholds=range(0, 101, 5)):
    """Validation에서 5점 단위의 해석하기 쉬운 컷을 탐색한다."""
    scores = weighted_score_series(df, weights_df)
    total_yes = int(df["target"].sum())
    rows = []
    for threshold in thresholds:
        flag = scores >= int(threshold)
        flagged = int(flag.sum())
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
    scores = weighted_score_series(df, weights_df)
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
    raw = pd.read_csv(DATA_PATH)
    df = clean_data(raw)

    # Final Test는 마지막까지 보지 않는다.
    dev, final_test = train_test_split(
        df, test_size=0.20, stratify=df["target"], random_state=RANDOM_STATE
    )
    dev = dev.reset_index(drop=True)
    final_test = final_test.reset_index(drop=True)
    selection_train, validation = train_test_split(
        dev, test_size=0.25, stratify=dev["target"], random_state=RANDOM_STATE
    )
    selection_train = selection_train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    # 범주가 많은 ethnicity/country는 Selection Train에서 10건 미만 범주를 Other로 통합한다.
    # 이 규칙은 Validation에 그대로 적용하여 Validation 정보를 미리 보지 않는다.
    selection_rare_maps = fit_rare_category_maps(selection_train)
    selection_train_model = apply_rare_category_maps(selection_train, selection_rare_maps)
    validation_model = apply_rare_category_maps(validation, selection_rare_maps)
    grouping_summary(selection_train, selection_train_model, selection_rare_maps, "selection_train").to_csv(
        ART / "category_grouping_summary.csv", index=False, encoding="utf-8-sig"
    )

    # 3번 페이지: 18개 요인 연관성 확인
    assoc = association_table(selection_train_model)
    assoc.to_csv(ART / "all_feature_association.csv", index=False, encoding="utf-8-sig")
    assoc[assoc["group"].eq("background")].to_csv(
        ART / "background_association.csv", index=False, encoding="utf-8-sig"
    )

    # 4번 페이지 입력 변수: 희소 범주 통합 + 보정된 연관성 분석 후 p<.05인 요인만 후보로 유지.
    selected_set = set(assoc.loc[assoc["significant_0_05"], "feature"].tolist())
    selected_features = [f for f in ALL_CANDIDATES if f in selected_set]
    selected_behavior = [x for x in selected_features if x in BEHAVIOR]
    selected_background = [x for x in selected_features if x in BACKGROUND]

    model_compare, _ = evaluate_models(selected_features, selection_train_model, validation_model)
    model_compare.to_csv(ART / "model_validation_comparison.csv", index=False, encoding="utf-8-sig")
    selected_model_name = str(model_compare.iloc[0]["model"])

    # 선택 변수의 모델 기반 중요도
    imp = selected_feature_importance(selected_features, selection_train_model)
    imp.to_csv(ART / "selected_feature_importance.csv", index=False, encoding="utf-8-sig")
    # 기존 파일명과의 호환성을 위해 같은 내용 저장
    imp.to_csv(ART / "all_feature_importance.csv", index=False, encoding="utf-8-sig")

    # 선생님용 가중 행동 체크리스트: 선택 모델의 행동 중요도를 100점으로 환산한다.
    # 점수 가중치는 Selection Train에서 학습된 중요도로 만들고, 구간 기준은 Development의 ASD 선별 YES/NO 가중점수 분포에서 정한 뒤 Final Test로 확인한다.
    checklist_weights = integer_weight_points(imp, selected_model_name, total_points=100)
    checklist_weights.to_csv(ART / "weighted_checklist_weights.csv", index=False, encoding="utf-8-sig")

    validation_weighted = weighted_threshold_table(validation_model, checklist_weights)

    # 가중점수 구간은 원래 Class/ASD 선별 레이블의 경계를 최대한 보존하도록 설정한다.
    # Final Test는 사용하지 않고 Development에서 NO의 최고점 바로 다음 점수를 고관찰 시작점으로 잡는다.
    dev_weighted_scores = weighted_score_series(dev, checklist_weights)
    dev_no_max = int(dev_weighted_scores[dev["target"].eq(0)].max())
    dev_yes_min = int(dev_weighted_scores[dev["target"].eq(1)].min())
    checklist_high_cutoff = int(dev_no_max + 1)

    # 추가 관찰 구간은 고관찰 시작점 아래 15점 범위를 프로젝트 운영용 완충구간으로 둔다.
    checklist_observe_cutoff = max(0, checklist_high_cutoff - 15)

    # 고관찰 구간은 Development의 YES 가중점수 중앙값을 기준으로 둘로 나눈다.
    dev_yes_median = float(dev_weighted_scores[dev["target"].eq(1)].median())
    checklist_very_high_cutoff = int(round(dev_yes_median / 5) * 5)
    checklist_very_high_cutoff = max(checklist_high_cutoff + 5, min(checklist_very_high_cutoff, 95))

    validation_weighted.to_csv(ART / "weighted_checklist_threshold_validation.csv", index=False, encoding="utf-8-sig")
    weighted_band_summary(
        validation_model, checklist_weights, checklist_observe_cutoff, checklist_high_cutoff, checklist_very_high_cutoff
    ).to_csv(ART / "weighted_checklist_band_validation.csv", index=False, encoding="utf-8-sig")

    # Development 전체로 재학습할 때는 희소 범주 규칙도 Development에서 다시 적합한다.
    final_rare_maps = fit_rare_category_maps(dev)
    dev_model = apply_rare_category_maps(dev, final_rare_maps)
    final_test_model = apply_rare_category_maps(final_test, final_rare_maps)
    final_weighted = weighted_threshold_table(final_test_model, checklist_weights)
    final_weighted.to_csv(ART / "weighted_checklist_threshold_final_test.csv", index=False, encoding="utf-8-sig")
    weighted_band_summary(
        final_test_model, checklist_weights, checklist_observe_cutoff, checklist_high_cutoff, checklist_very_high_cutoff
    ).to_csv(ART / "weighted_checklist_band_final_test.csv", index=False, encoding="utf-8-sig")

    final_scored = final_test_model.copy()
    final_scored["weighted_checklist_score"] = weighted_score_series(final_test_model, checklist_weights)
    final_scored["actual_label"] = np.where(final_scored["target"].eq(1), "YES", "NO")
    final_scored.to_csv(ART / "weighted_checklist_final_test_scored.csv", index=False, encoding="utf-8-sig")

    # Development 전체로 재학습 후 Final Test 1회 평가
    final_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(selected_features)),
            ("model", make_models()[selected_model_name]),
        ]
    )
    final_pipeline.fit(dev_model[selected_features], dev_model["target"])
    train_pred = final_pipeline.predict(dev_model[selected_features])
    test_pred = final_pipeline.predict(final_test_model[selected_features])
    test_proba = final_pipeline.predict_proba(final_test_model[selected_features])[:, 1]
    final_metrics = metrics(final_test["target"], test_pred, test_proba)
    final_metrics["train_accuracy"] = accuracy_score(dev["target"], train_pred)
    final_metrics["generalization_gap"] = final_metrics["train_accuracy"] - final_metrics["accuracy"]
    tn, fp, fn, tp = confusion_matrix(final_test["target"], test_pred).ravel()
    final_metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

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

    fpr, tpr, thresholds = roc_curve(final_test["target"], test_proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        ART / "final_roc_curve.csv", index=False, encoding="utf-8-sig"
    )

    pred_out = final_test.copy()
    pred_out["actual_label"] = np.where(final_test["target"].eq(1), "YES", "NO")
    pred_out["predicted_label"] = np.where(test_pred == 1, "YES", "NO")
    pred_out["model_score_yes"] = test_proba
    pred_out["correct"] = pred_out["actual_label"].eq(pred_out["predicted_label"])
    pred_out.to_csv(ART / "final_test_predictions.csv", index=False, encoding="utf-8-sig")

    # 모델 저장은 재현성을 위해 유지하되 대시보드에서는 직접 예측 기능으로 사용하지 않는다.
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

    sum_match = bool((df[BEHAVIOR].sum(axis=1).to_numpy() == df["result"].to_numpy()).all())
    rule_match = bool((((df["result"] >= 7).astype(int)).to_numpy() == df["target"].to_numpy()).all())

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

    # 대시보드 요약
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
