"""Step 1. Read the Lending Club file, tidy the fields, and load them into Snowflake.

Run:  python 01_prepare_and_load.py path/to/accepted_2007_to_2018Q4.csv
"""
import sys

import numpy as np
import pandas as pd

from common import DATA, DATABASE, get_connection, use_snowflake

RAW_COLUMNS = [
    "id", "issue_d", "loan_status", "term", "loan_amnt", "funded_amnt",
    "annual_inc", "dti", "emp_length", "home_ownership", "purpose",
    "verification_status", "fico_range_low", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "earliest_cr_line", "total_rec_prncp", "recoveries",
]


def to_number(series):
    """Turn text such as '45.2%' into a number. Leaves numbers as they are."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def job_years(series):
    """'10+ years' becomes 10, '< 1 year' becomes 0, blank stays missing."""
    text = series.astype(str)
    years = pd.to_numeric(text.str.extract(r"(\d+)")[0], errors="coerce")
    years = years.where(~text.str.contains("<", regex=False), 0)
    return years.astype(float)


def main(csv_path):
    print("Reading", csv_path)
    df = pd.read_csv(csv_path, usecols=RAW_COLUMNS, low_memory=False)

    # The file has a few footer rows with no loan details. Drop them.
    df = df[df["loan_amnt"].notna() & df["issue_d"].notna()].copy()

    issue = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    first_credit = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")

    out = pd.DataFrame(
        {
            "id": df["id"].astype(str),
            "issue_date": issue.dt.strftime("%Y-%m-%d"),
            "loan_status": df["loan_status"].astype(str),
            "term_months": pd.to_numeric(
                df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            ),
            "loan_amnt": to_number(df["loan_amnt"]),
            "funded_amnt": to_number(df["funded_amnt"]),
            "annual_inc": to_number(df["annual_inc"]),
            "dti": to_number(df["dti"]),
            "emp_length_years": job_years(df["emp_length"]),
            "home_ownership": df["home_ownership"].astype(str),
            "purpose": df["purpose"].astype(str),
            "verification_status": df["verification_status"].astype(str),
            "fico_range_low": to_number(df["fico_range_low"]),
            "delinq_2yrs": to_number(df["delinq_2yrs"]),
            "inq_last_6mths": to_number(df["inq_last_6mths"]),
            "open_acc": to_number(df["open_acc"]),
            "pub_rec": to_number(df["pub_rec"]),
            "revol_bal": to_number(df["revol_bal"]),
            "revol_util": to_number(df["revol_util"]),
            "total_acc": to_number(df["total_acc"]),
            "credit_history_years": ((issue - first_credit).dt.days / 365.25).round(1),
            "total_rec_prncp": to_number(df["total_rec_prncp"]),
            "recoveries": to_number(df["recoveries"]),
        }
    )
    out = out[out["issue_date"].notna()].reset_index(drop=True)
    out.to_parquet(DATA / "loans_raw.parquet", index=False)
    print("Saved", len(out), "rows to data/loans_raw.parquet")

    if not use_snowflake():
        print("Snowflake settings not found. Skipping the load. The next step will run locally.")
        return

    from snowflake.connector.pandas_tools import write_pandas

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DATABASE}.RAW")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DATABASE}.MART")

    upload = out.copy()
    upload.columns = [c.upper() for c in upload.columns]
    write_pandas(
        conn, upload, "LOANS", database=DATABASE, schema="RAW",
        auto_create_table=True, overwrite=True,
    )
    print("Loaded", len(upload), "rows into", f"{DATABASE}.RAW.LOANS")
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python 01_prepare_and_load.py path/to/accepted_2007_to_2018Q4.csv")
    main(sys.argv[1])
