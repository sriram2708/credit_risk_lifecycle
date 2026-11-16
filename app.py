"""Credit risk app. Run with:  streamlit run app.py

The app reads the saved models and result files in this repo.
It does not need a database connection, so it stays live at no cost.
"""
import json

import joblib
import pandas as pd
import streamlit as st
from optbinning import Scorecard

from common import CAT_FEATURES, FEATURES, FEATURE_LABELS, MODELS, OUTPUTS, points_breakdown

st.set_page_config(page_title="Credit Risk Model Lifecycle", layout="wide")


@st.cache_resource
def load_models():
    scorecard = Scorecard.load(str(MODELS / "scorecard.pkl"))
    challenger = joblib.load(MODELS / "challenger.pkl")
    categories = json.loads((MODELS / "challenger_categories.json").read_text())
    return scorecard, challenger, categories


@st.cache_data
def load_csv(name):
    return pd.read_csv(OUTPUTS / name)


scorecard, challenger, categories = load_models()

page = st.sidebar.radio(
    "Page", ["Score an applicant", "Model results", "Expected loss", "Monitoring"]
)
st.sidebar.caption(
    "Built on public Lending Club loans (United States, 36 month loans, 2012 to 2015). "
    "For learning only. Not for real lending decisions."
)

if page == "Score an applicant":
    st.title("Score an applicant")
    left, mid, right = st.columns(3)
    with left:
        loan_amnt = st.number_input("Loan amount", 500, 40000, 12000, step=500)
        annual_inc = st.number_input("Annual income", 5000, 500000, 65000, step=1000)
        dti = st.slider("Debt to income ratio", 0.0, 45.0, 18.0)
        emp_length_years = st.slider("Years in current job", 0, 10, 5)
        fico = st.slider("Credit bureau score", 660, 850, 700)
    with mid:
        delinq_2yrs = st.number_input("Late payments in last 2 years", 0, 20, 0)
        inq_last_6mths = st.number_input("Credit enquiries in last 6 months", 0, 10, 1)
        open_acc = st.number_input("Open credit accounts", 0, 60, 10)
        pub_rec = st.number_input("Public records", 0, 10, 0)
        total_acc = st.number_input("Total credit accounts", 1, 120, 24)
    with right:
        revol_bal = st.number_input("Revolving balance", 0, 500000, 12000, step=500)
        revol_util = st.slider("Revolving credit used (percent)", 0.0, 120.0, 50.0)
        credit_history_years = st.slider("Years of credit history", 0.0, 50.0, 15.0)
        chosen = {c: st.selectbox(FEATURE_LABELS[c], categories[c]) for c in CAT_FEATURES}

    applicant = pd.DataFrame(
        [
            {
                "loan_amnt": float(loan_amnt),
                "annual_inc": float(annual_inc),
                "dti": float(dti),
                "emp_length_years": float(emp_length_years),
                "fico_range_low": float(fico),
                "delinq_2yrs": float(delinq_2yrs),
                "inq_last_6mths": float(inq_last_6mths),
                "open_acc": float(open_acc),
                "pub_rec": float(pub_rec),
                "revol_bal": float(revol_bal),
                "revol_util": float(revol_util),
                "total_acc": float(total_acc),
                "credit_history_years": float(credit_history_years),
                **chosen,
            }
        ]
    )[FEATURES]

    score = float(scorecard.score(applicant)[0])
    chance = float(scorecard.predict_proba(applicant)[0, 1])

    challenger_input = applicant.copy()
    for col in CAT_FEATURES:
        challenger_input[col] = pd.Categorical(challenger_input[col], categories=categories[col])
    challenger_chance = float(challenger.predict_proba(challenger_input)[0, 1])

    a, b, c = st.columns(3)
    a.metric("Scorecard score", f"{score:.0f}")
    b.metric("Chance of default (scorecard)", f"{chance:.1%}")
    c.metric("Chance of default (challenger)", f"{challenger_chance:.1%}")

    st.subheader("Why this score")
    st.write(
        "Each row shows the points this applicant received for one feature, next to the "
        "average for past borrowers. Rows at the top pull the score down the most."
    )
    reasons = points_breakdown(scorecard, applicant).round(1)
    reasons.columns = ["Feature", "Applicant band", "Points", "Average points", "Difference"]
    st.dataframe(reasons, hide_index=True)

elif page == "Model results":
    st.title("Model results")
    metrics = json.loads((OUTPUTS / "scorecard_metrics.json").read_text())
    a, b, c, d = st.columns(4)
    a.metric("Training loans", f"{metrics['train']['loans']:,}")
    b.metric("Test loans (2014)", f"{metrics['test']['loans']:,}")
    c.metric("Test Gini", f"{metrics['test']['gini']:.3f}")
    d.metric("Test KS", f"{metrics['test']['ks']:.3f}")

    st.subheader("Scorecard against the machine learning challenger")
    st.dataframe(load_csv("model_comparison.csv").round(3), hide_index=True)

    st.subheader("Feature strength (information value)")
    strength = load_csv("feature_strength.csv")
    strength["name"] = strength["name"].map(FEATURE_LABELS).fillna(strength["name"])
    st.bar_chart(strength.set_index("name")["iv"])

    st.subheader("What drives the challenger")
    drivers = load_csv("challenger_importance.csv")
    drivers["feature"] = drivers["feature"].map(FEATURE_LABELS).fillna(drivers["feature"])
    st.bar_chart(drivers.set_index("feature")["mean_abs_contribution"])

    st.subheader("Scorecard points table")
    st.dataframe(load_csv("scorecard_table.csv").round(2), hide_index=True)

elif page == "Expected loss":
    st.title("Expected credit loss on the 2015 loan book")
    st.write(
        "Expected loss is the chance of default multiplied by the share lost on default "
        "and by the amount lent. The stress cases multiply the odds of default by a fixed "
        "number. They are examples, not forecasts."
    )
    loss = load_csv("expected_loss.csv")
    st.bar_chart(loss.set_index("scenario")["expected_loss"])
    shown = loss.copy()
    for col in ("average_default_chance", "loss_given_default", "loss_rate"):
        shown[col] = (shown[col] * 100).round(2)
    for col in ("amount_lent", "expected_loss"):
        shown[col] = shown[col].round(0)
    shown.columns = [
        "Case", "Odds multiplier", "Default chance (%)", "Loss on default (%)",
        "Amount lent", "Loss", "Loss rate (%)",
    ]
    st.dataframe(shown, hide_index=True)

else:
    st.title("Model monitoring")
    monitor = load_csv("monitoring.csv")
    latest = monitor.iloc[-1]
    a, b, c, d = st.columns(4)
    a.metric("Latest quarter", latest["quarter"])
    b.metric("Status", latest["status"])
    c.metric("Stability index (PSI)", f"{latest['psi']:.3f}")
    d.metric("Gini", f"{latest['gini']:.3f}")

    st.subheader("Predicted against actual bad rate")
    st.line_chart(monitor.set_index("quarter")[["predicted_bad_rate", "actual_bad_rate"]])

    st.subheader("Gini by quarter")
    st.line_chart(monitor.set_index("quarter")["gini"])

    st.subheader("Stability index by quarter")
    st.write("Below 0.10 is stable. Above 0.25 means the borrower mix has changed a lot.")
    st.line_chart(monitor.set_index("quarter")["psi"])

    st.subheader("Score mix: training against the latest quarter")
    st.bar_chart(load_csv("score_mix.csv").set_index("band"), stack=False)

    st.subheader("All quarters")
    st.dataframe(monitor.round(4), hide_index=True)
