"""Step 3. Build the credit scorecard.

Each feature is grouped into bands, each band gets a weight of evidence value,
and a logistic regression turns those into points. A higher score means lower risk.

Run:  python 03_train_scorecard.py
"""
import json

from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

from common import (
    CAT_FEATURES, FEATURES, MODELS, OUTPUTS, TARGET, gini, ks, load_model_table,
)


def main():
    df = load_model_table()
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    binning = BinningProcess(
        variable_names=FEATURES,
        categorical_variables=CAT_FEATURES,
        selection_criteria={"iv": {"min": 0.02}},
    )
    scorecard = Scorecard(
        binning_process=binning,
        estimator=LogisticRegression(max_iter=1000),
        scaling_method="pdo_odds",
        scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600},
    )
    scorecard.fit(train[FEATURES], train[TARGET])

    metrics = {}
    for name, part in (("train", train), ("test", test)):
        prob = scorecard.predict_proba(part[FEATURES])[:, 1]
        metrics[name] = {
            "loans": int(len(part)),
            "bad_rate": float(part[TARGET].mean()),
            "gini": float(gini(part[TARGET], prob)),
            "ks": float(ks(part[TARGET], prob)),
        }
        print(name, metrics[name])

    scorecard.save(str(MODELS / "scorecard.pkl"))
    scorecard.table(style="summary").to_csv(OUTPUTS / "scorecard_table.csv", index=False)

    # Strength of each feature, measured by information value.
    iv = scorecard.binning_process_.summary()[["name", "iv", "selected"]].sort_values("iv", ascending=False)
    iv.to_csv(OUTPUTS / "feature_strength.csv", index=False)

    (OUTPUTS / "scorecard_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved the scorecard, the points table and the metrics")


if __name__ == "__main__":
    main()
