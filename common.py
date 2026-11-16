"""Shared settings and helper functions used by every step."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
for folder in (DATA, MODELS, OUTPUTS):
    folder.mkdir(exist_ok=True)

DATABASE = "CREDIT_RISK"

# Only fields known at the time of the loan application are used as inputs.
# Grade, interest rate and all payment fields are left out to avoid leakage.
NUM_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "emp_length_years",
    "fico_range_low",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "credit_history_years",
]
CAT_FEATURES = ["home_ownership", "purpose", "verification_status"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "bad"

# Plain names shown in the app and in reports.
FEATURE_LABELS = {
    "loan_amnt": "Loan amount",
    "annual_inc": "Annual income",
    "dti": "Debt to income ratio",
    "emp_length_years": "Years in current job",
    "fico_range_low": "Credit bureau score",
    "delinq_2yrs": "Late payments in last 2 years",
    "inq_last_6mths": "Credit enquiries in last 6 months",
    "open_acc": "Open credit accounts",
    "pub_rec": "Public records",
    "revol_bal": "Revolving balance",
    "revol_util": "Revolving credit used (percent)",
    "total_acc": "Total credit accounts",
    "credit_history_years": "Years of credit history",
    "home_ownership": "Home ownership",
    "purpose": "Loan purpose",
    "verification_status": "Income verification",
}


def use_snowflake():
    """Snowflake is used when the account name is set as an environment variable."""
    return bool(os.getenv("SNOWFLAKE_ACCOUNT"))


def get_connection():
    import snowflake.connector

    settings = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    }
    if os.getenv("SNOWFLAKE_AUTHENTICATOR"):
        settings["authenticator"] = os.environ["SNOWFLAKE_AUTHENTICATOR"]
    return snowflake.connector.connect(**settings)


def load_model_table():
    """Read the modelling table that step 02 saved."""
    df = pd.read_parquet(DATA / "loans_model.parquet")
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    for col in NUM_FEATURES + ["funded_amnt", "total_rec_prncp", "recoveries"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in CAT_FEATURES:
        df[col] = df[col].astype(object).where(df[col].notna(), "MISSING").astype(str)
    df[TARGET] = df[TARGET].astype(int)
    return df


def gini(y_true, y_prob):
    return 2 * roc_auc_score(y_true, y_prob) - 1


def ks(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def psi(expected, actual, edges):
    """Population stability index of a score, using fixed band edges."""
    exp_share = np.histogram(expected, bins=edges)[0] / len(expected)
    act_share = np.histogram(actual, bins=edges)[0] / len(actual)
    exp_share = np.clip(exp_share, 1e-4, None)
    act_share = np.clip(act_share, 1e-4, None)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def points_breakdown(scorecard, X):
    """Points given to one applicant for each feature, next to the average
    points the training population received for that feature."""
    table = scorecard.table(style="detailed")
    bp = scorecard.binning_process_
    rows = []
    for name in bp.get_support(names=True):
        idx = int(bp.get_binned_variable(name).transform(X[name], metric="indices")[0])
        part = table[table["Variable"] == name].reset_index(drop=True)
        counted = part[part["Count"] > 0]
        average = float((counted["Points"] * counted["Count"]).sum() / counted["Count"].sum())
        rows.append(
            {
                "feature": FEATURE_LABELS.get(name, name),
                "band": str(part.loc[idx, "Bin"]),
                "points": float(part.loc[idx, "Points"]),
                "average_points": average,
            }
        )
    out = pd.DataFrame(rows)
    out["difference"] = out["points"] - out["average_points"]
    return out.sort_values("difference").reset_index(drop=True)
