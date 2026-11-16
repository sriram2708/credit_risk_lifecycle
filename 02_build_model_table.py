"""Step 2. Build the modelling table.

With Snowflake: runs sql/build_model_table.sql and pulls the result down.
Without Snowflake: applies the same rules in pandas so the project still runs.

Run:  python 02_build_model_table.py
"""
import numpy as np
import pandas as pd

from common import DATA, DATABASE, ROOT, get_connection, use_snowflake


def build_in_snowflake():
    conn = get_connection()
    sql = (ROOT / "sql" / "build_model_table.sql").read_text()
    conn.execute_string(sql)
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {DATABASE}.MART.LOANS_MODEL")
    df = cur.fetch_pandas_all()
    cur.execute(f"SELECT * FROM {DATABASE}.MART.BAD_RATE_BY_QUARTER")
    print(cur.fetch_pandas_all().to_string(index=False))
    conn.close()
    df.columns = [c.lower() for c in df.columns]
    return df


def build_locally():
    raw = pd.read_parquet(DATA / "loans_raw.parquet")
    issue = pd.to_datetime(raw["issue_date"])
    keep = (
        (raw["term_months"] == 36)
        & raw["loan_status"].isin(["Fully Paid", "Charged Off"])
        & (issue >= "2012-01-01")
        & (issue <= "2015-12-31")
    )
    df = raw[keep].copy()
    df["issue_date"] = issue[keep]
    df["bad"] = (df["loan_status"] == "Charged Off").astype(int)
    year = df["issue_date"].dt.year
    df["split"] = np.where(year <= 2013, "train", np.where(year == 2014, "test", "monitor"))
    return df.drop(columns=["loan_status", "term_months"])


def main():
    df = build_in_snowflake() if use_snowflake() else build_locally()
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    df.to_parquet(DATA / "loans_model.parquet", index=False)
    summary = df.groupby("split")["bad"].agg(loans="count", bad_rate="mean")
    print(summary)
    print("Saved data/loans_model.parquet")


if __name__ == "__main__":
    main()
