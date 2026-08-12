from pathlib import Path
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_pipeline_nsch as prep

ART = prep.ART
EXCLUDED_ASSOCIATION = {"k2q35a", "k2q31a", "k2q33a", "k2q32a", "k2q34a", "a1_menthealth", "a2_menthealth"}

def main():
    df = pd.read_stata(prep.DATA_PATH, convert_categoricals=False)
    valid, _, target = prep.create_target(df)
    usable = df.loc[valid, list(prep.CANDIDATES)].copy()
    y = target.loc[valid].astype(int).copy()
    split = prep.stratified_split(y)
    train = usable.loc[split.eq("train")]
    val = usable.loc[split.eq("validation")]
    test = usable.loc[split.eq("test")]
    y_train, y_val, y_test = y.loc[train.index], y.loc[val.index], y.loc[test.index]
    train, val, test, _ = prep.impute_candidates(train, val, test)

    eligible = [c for c in prep.CANDIDATES if c not in EXCLUDED_ASSOCIATION]
    categorical_all = ["family_r", "foodsit", "missmortgage", "a1_grade", "k7q04r_r", "k7q82_r", "k10q41_r", "makefriend"]
    numeric_all = [c for c in eligible if c not in categorical_all]
    rf_encoder = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical_all), ("num", "passthrough", numeric_all)])
    encoded_train = rf_encoder.fit_transform(train[eligible])
    rf_all = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")
    rf_all.fit(encoded_train, y_train)
    imp_values = dict.fromkeys(eligible, 0.0)
    offset = 0
    for feature, categories in zip(categorical_all, rf_encoder.named_transformers_["cat"].categories_):
        width = len(categories)
        imp_values[feature] = float(rf_all.feature_importances_[offset:offset + width].sum())
        offset += width
    for feature in numeric_all:
        imp_values[feature] = float(rf_all.feature_importances_[offset])
        offset += 1
    imp = pd.DataFrame({"column": eligible, "rf_importance": [imp_values[c] for c in eligible]})
    stats = pd.read_csv(ART / "statistical_tests.csv")[["column", "p_value", "absolute_relationship_strength"]]
    imp = imp.merge(stats, on="column", how="left")
    imp["rf_rank"] = imp["rf_importance"].rank(ascending=False, method="min")
    imp["stat_rank"] = imp["absolute_relationship_strength"].rank(ascending=False, method="min")
    imp["combined_rank"] = imp["rf_rank"] + imp["stat_rank"]
    imp = imp.sort_values(["combined_rank", "rf_importance"], ascending=[True, False])
    selected = imp.head(12)["column"].tolist()
    imp["selected_final"] = imp["column"].isin(selected)
    imp.to_csv(ART / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"column": selected, "selection_order": range(1, len(selected)+1)}).to_csv(ART / "selected_features.csv", index=False, encoding="utf-8-sig")

    categorical = ["family_r", "foodsit", "missmortgage", "a1_grade", "k7q04r_r", "k7q82_r", "k10q41_r", "makefriend"]
    numeric = [c for c in selected if c not in categorical]
    prep_model = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical), ("num", StandardScaler(), numeric)])
    def pipe(estimator): return Pipeline([("preprocess", prep_model), ("model", estimator)])
    models = {
        "LogisticRegression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        "KNN": pipe(KNeighborsClassifier(n_neighbors=15)),
        "DecisionTree": pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42)),
        "RandomForest": pipe(RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)),
    }
    def metrics(model, X, yy, split_name):
        pred = model.predict(X)
        score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
        return {"split": split_name, "accuracy": accuracy_score(yy, pred), "precision": precision_score(yy, pred, zero_division=0), "recall": recall_score(yy, pred, zero_division=0), "f1": f1_score(yy, pred, zero_division=0), "roc_auc": roc_auc_score(yy, score)}
    rows = []
    for name, model in models.items():
        model.fit(train[selected], y_train)
        rows.append({"model": name, **metrics(model, val[selected], y_val, "validation")})
    comparison = pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False)
    comparison.to_csv(ART / "model_validation_comparison.csv", index=False, encoding="utf-8-sig")
    winner = comparison.iloc[0]["model"]
    final_model = models[winner]
    final_model.fit(pd.concat([train[selected], val[selected]]), pd.concat([y_train, y_val]))
    test_result = pd.DataFrame([{"model": winner, **metrics(final_model, test[selected], y_test, "test")}])
    test_result.to_csv(ART / "final_test_metrics.csv", index=False, encoding="utf-8-sig")
    with (ART / "final_model_nsch.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)
    (ART / "final_model_metadata.json").write_text(json.dumps({"model": winner, "features": selected, "selection_rule": "top 12 by combined RF importance and train statistical relationship rank", "association_excluded": sorted(EXCLUDED_ASSOCIATION)}, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
