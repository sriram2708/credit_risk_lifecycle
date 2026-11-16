"""Step 5. Estimate expected credit loss on the 2015 loan book.

Expected loss = chance of default x share lost when a loan defaults x amount lent.
Three economic cases are shown. The stress sizes are examples, not forecasts.

Run:  python 05_expected_loss.py
"""
import numpy as np
import pandas as pd
from optbinning import Scorecard

from common import FEATURES, MODELS, OUTPUTS, TARGET, load_model_table

# Each case multiplies the odds of default.
SCENARIOS = {"Base": 1.0, "Downside": 1.3, "Severe": 1.7}


def loss_share(df):
    """Share of the amount lent that was never recovered, on defaulted loans."""
    bad = df[df[TARGET] == 1]
    recovered = (bad["total_rec_prncp"].fillna(0) + bad["recoveries"].fillna(0)) / bad["funded_amnt"]
    return float((1 - recovered).clip(0, 1).mean())


def main():
    df = load_model_table()
    train = df[df["split"] == "train"]
    book = df[df["split"] == "monitor"].copy()

    scorecard = Scorecard.load(str(MODELS / "scorecard.pkl"))
    pd_base = scorecard.predict_proba(book[FEATURES])[:, 1]
    lgd = loss_share(train)
    ead = book["funded_amnt"].to_numpy()

    rows = []
    odds = pd_base / (1 - pd_base)
    for name, factor in SCENARIOS.items():
        pd_case = factor * odds / (1 + factor * odds)
        loss = float(np.sum(pd_case * lgd * ead))
        rows.append(
            {
                "scenario": name,
                "odds_multiplier": factor,
                "average_default_chance": float(pd_case.mean()),
                "loss_given_default": lgd,
                "amount_lent": float(ead.sum()),
                "expected_loss": loss,
                "loss_rate": loss / float(ead.sum()),
            }
        )

    # What was really lost on this book, for comparison with the base case.
    bad = book[book[TARGET] == 1]
    actual = float(
        (bad["funded_amnt"] - bad["total_rec_prncp"].fillna(0) - bad["recoveries"].fillna(0))
        .clip(lower=0)
        .sum()
    )
    rows.append(
        {
            "scenario": "Actual result",
            "odds_multiplier": np.nan,
            "average_default_chance": float(book[TARGET].mean()),
            "loss_given_default": loss_share(book),
            "amount_lent": float(ead.sum()),
            "expected_loss": actual,
            "loss_rate": actual / float(ead.sum()),
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUTS / "expected_loss.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
