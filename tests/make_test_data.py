"""Makes a small made up file in the Lending Club layout.

Use it only to check that your setup works before the large download.
The numbers are random and mean nothing. Do not publish results from it.

Run:  python tests/make_test_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 60000

issue = pd.to_datetime("2011-06-01") + pd.to_timedelta(rng.integers(0, 365 * 6, n), unit="D")
fico = np.clip(rng.normal(695, 30, n), 660, 845).round()
dti = np.clip(rng.normal(18, 8, n), 0, 45).round(2)
income = np.exp(rng.normal(11.1, 0.5, n)).round()
inq = rng.poisson(0.8, n)
util = np.clip(rng.normal(55, 24, n), 0, 120).round(1)
term = rng.choice([" 36 months", " 60 months"], n, p=[0.75, 0.25])
purpose = rng.choice(
    ["debt_consolidation", "credit_card", "home_improvement", "small_business", "other"],
    n, p=[0.55, 0.22, 0.08, 0.03, 0.12],
)
home = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.42, 0.48, 0.10])

drift = (issue.year - 2012) * 0.06
z = (
    -1.9 - 0.012 * (fico - 695) + 0.03 * (dti - 18) - 0.4 * np.log(income / 65000)
    + 0.15 * inq + 0.006 * (util - 55) + 0.5 * (purpose == "small_business")
    + 0.15 * (home == "RENT") + drift
)
bad = rng.random(n) < 1 / (1 + np.exp(-z))
finished = issue < pd.Timestamp("2016-01-01")
status = np.where(~finished & (rng.random(n) < 0.6), "Current", np.where(bad, "Charged Off", "Fully Paid"))

amount = (rng.integers(40, 1400, n) * 25).astype(float)
paid_share = np.where(status == "Charged Off", rng.uniform(0.05, 0.7, n), 1.0)
emp = rng.choice(
    ["< 1 year", "1 year", "2 years", "3 years", "5 years", "8 years", "10+ years", None],
    n, p=[0.09, 0.07, 0.10, 0.09, 0.14, 0.12, 0.33, 0.06],
)
history_years = rng.uniform(3, 35, n)
first_credit = issue - pd.to_timedelta((history_years * 365.25).astype(int), unit="D")

df = pd.DataFrame(
    {
        "id": np.arange(1, n + 1),
        "issue_d": issue.strftime("%b-%Y"),
        "loan_status": status,
        "term": term,
        "loan_amnt": amount,
        "funded_amnt": amount,
        "annual_inc": income,
        "dti": dti,
        "emp_length": emp,
        "home_ownership": home,
        "purpose": purpose,
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], n),
        "fico_range_low": fico,
        "delinq_2yrs": rng.poisson(0.3, n),
        "inq_last_6mths": inq,
        "open_acc": rng.integers(2, 25, n),
        "pub_rec": rng.poisson(0.15, n),
        "revol_bal": np.exp(rng.normal(9.2, 0.9, n)).round(),
        "revol_util": util,
        "total_acc": rng.integers(4, 60, n),
        "earliest_cr_line": first_credit.strftime("%b-%Y"),
        "total_rec_prncp": (amount * paid_share).round(2),
        "recoveries": np.where(status == "Charged Off", amount * rng.uniform(0, 0.1, n), 0).round(2),
    }
)
df.loc[rng.random(n) < 0.01, "revol_util"] = np.nan

out = Path(__file__).parent / "test_loans.csv"
df.to_csv(out, index=False)
print("Wrote", out)
