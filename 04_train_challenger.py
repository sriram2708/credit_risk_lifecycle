"""Step 4. Build a machine learning challenger and compare it with the scorecard.

The challenger is a gradient boosted tree model. Four features are forced to
move in the direction that makes business sense, so the model is easier to trust.

Run:  python 04_train_challenger.py
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (
    CAT_FEATURES, FEATURES, MODELS, NUM_FEATURES, OUTPUTS, TARGET, gini, ks,
    load_model_table,
)

# +1 means risk can only rise as the feature rises. 1 with a minus sign means
# risk can only fall. 0 means no rule.
DIRECTION = {"fico_range_low": -1, "annual_inc": -1, "dti": 1, "inq_last_6mths": 1}


def prepare(df, categories):
    X = df[FEATURES].copy()
    for col in CAT_FEATURES:
        # Values not seen in training become missing.
        known = X[col].where(X[col].isin(categories[col]))
        X[col] = pd.Categorical(known, categories=categories[col])
    return X


def main():
    df = load_model_table()
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    categories = {c: sorted(train[c].unique().tolist()) for c in CAT_FEATURES}
    X_train, X_test = prepare(train, categories), prepare(test, categories)

    # Hold back one fifth of the training loans to decide when to stop adding trees.
    fit_rows = np.random.default_rng(42).random(len(X_train)) < 0.8
    y_train = train[TARGET].to_numpy()

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        monotone_constraints=[DIRECTION.get(f, 0) for f in FEATURES],
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train[fit_rows], y_train[fit_rows],
        eval_set=[(X_train[~fit_rows], y_train[~fit_rows])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    print("Trees used:", model.best_iteration_)

    metrics = {}
    for name, X, part in (("train", X_train, train), ("test", X_test, test)):
        prob = model.predict_proba(X)[:, 1]
        metrics[name] = {
            "gini": float(gini(part[TARGET], prob)),
            "ks": float(ks(part[TARGET], prob)),
        }
        print(name, metrics[name])

    # Contribution of each feature to each prediction (SHAP values from LightGBM).
    sample = X_test.sample(min(20000, len(X_test)), random_state=42)
    contrib = model.predict(sample, pred_contrib=True)[:, :-1]
    importance = (
        pd.DataFrame({"feature": FEATURES, "mean_abs_contribution": np.abs(contrib).mean(axis=0)})
        .sort_values("mean_abs_contribution", ascending=False)
    )
    importance.to_csv(OUTPUTS / "challenger_importance.csv", index=False)

    joblib.dump(model, MODELS / "challenger.pkl")
    (MODELS / "challenger_categories.json").write_text(json.dumps(categories, indent=2))
    (OUTPUTS / "challenger_metrics.json").write_text(json.dumps(metrics, indent=2))

    score_metrics = json.loads((OUTPUTS / "scorecard_metrics.json").read_text())
    compare = pd.DataFrame(
        {
            "model": ["Scorecard", "Challenger"],
            "test_gini": [score_metrics["test"]["gini"], metrics["test"]["gini"]],
            "test_ks": [score_metrics["test"]["ks"], metrics["test"]["ks"]],
        }
    )
    compare.to_csv(OUTPUTS / "model_comparison.csv", index=False)
    print(compare)


if __name__ == "__main__":
    main()
