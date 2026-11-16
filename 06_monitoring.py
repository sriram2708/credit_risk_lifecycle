"""Step 6. Monitor the scorecard on loans issued after the training period.

For each quarter this checks three things.
1. Stability: has the mix of scores moved away from the training mix (PSI).
2. Ranking power: is Gini holding up.
3. Accuracy: does the predicted bad rate match the actual bad rate.

Run:  python 06_monitoring.py
"""
import numpy as np
import pandas as pd
from optbinning import Scorecard

from common import (
    DATABASE, FEATURES, MODELS, OUTPUTS, TARGET, get_connection, gini, load_model_table,
    psi, use_snowflake,
)

PSI_AMBER = 0.10
PSI_RED = 0.25
GINI_DROP_AMBER = 0.03
GINI_DROP_RED = 0.06
# Actual bad rate divided by predicted bad rate. 1.0 means the model is accurate.
ACCURACY_AMBER = 1.10
ACCURACY_RED = 1.25


def main():
    df = load_model_table()
    scorecard = Scorecard.load(str(MODELS / "scorecard.pkl"))
    df["score"] = scorecard.score(df[FEATURES])
    df["pd"] = scorecard.predict_proba(df[FEATURES])[:, 1]

    train = df[df["split"] == "train"]
    train_gini = gini(train[TARGET], train["pd"])

    # Ten score bands with equal numbers of training loans.
    edges = np.unique(np.quantile(train["score"], np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf

    later = df[df["split"] != "train"].copy()
    later["quarter"] = later["issue_date"].dt.to_period("Q").astype(str)

    rows = []
    for quarter, part in later.groupby("quarter"):
        stability = psi(train["score"], part["score"], edges)
        quarter_gini = gini(part[TARGET], part["pd"])
        drop = train_gini - quarter_gini
        ratio = float(part[TARGET].mean() / part["pd"].mean())
        gap = max(ratio, 1 / ratio)
        if stability > PSI_RED or drop > GINI_DROP_RED or gap > ACCURACY_RED:
            status = "Red"
        elif stability > PSI_AMBER or drop > GINI_DROP_AMBER or gap > ACCURACY_AMBER:
            status = "Amber"
        else:
            status = "Green"
        rows.append(
            {
                "quarter": quarter,
                "split": part["split"].iloc[0],
                "loans": int(len(part)),
                "average_score": float(part["score"].mean()),
                "psi": stability,
                "gini": float(quarter_gini),
                "gini_drop_from_train": float(drop),
                "predicted_bad_rate": float(part["pd"].mean()),
                "actual_bad_rate": float(part[TARGET].mean()),
                "actual_over_predicted": ratio,
                "status": status,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUTS / "monitoring.csv", index=False)
    print(out.to_string(index=False))

    # Score mix by band, training against the latest quarter, for the app chart.
    latest = later[later["quarter"] == later["quarter"].max()]
    labels = [f"Band {i + 1}" for i in range(len(edges) - 1)]
    mix = pd.DataFrame(
        {
            "band": labels,
            "training_share": np.histogram(train["score"], bins=edges)[0] / len(train),
            "latest_quarter_share": np.histogram(latest["score"], bins=edges)[0] / len(latest),
        }
    )
    mix.to_csv(OUTPUTS / "score_mix.csv", index=False)

    if use_snowflake():
        from snowflake.connector.pandas_tools import write_pandas

        conn = get_connection()
        upload = out.copy()
        upload.columns = [c.upper() for c in upload.columns]
        write_pandas(
            conn, upload, "MONITORING", database=DATABASE, schema="MART",
            auto_create_table=True, overwrite=True,
        )
        conn.close()
        print("Wrote the results to", f"{DATABASE}.MART.MONITORING")


if __name__ == "__main__":
    main()
