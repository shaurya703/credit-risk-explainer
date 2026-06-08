"""Streamlit dashboard for the Credit Risk Explainer.

The dashboard is a thin client: it never loads the model. All predictions and
explanations come from the FastAPI service (configured via API_URL), so there is a
single source of model truth. It only imports `src` for shared labels and the
paths to the pre-rendered evaluation figures.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Streamlit puts this file's directory on sys.path, not the repo root, so make the
# shared `src` package importable (mirrors the notebook setup).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import requests
import streamlit as st

from src import config

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk Explainer", page_icon="💳", layout="wide")

RED = "#c0392b"    # increases default risk
GREEN = "#1b7837"  # decreases default risk


@st.cache_data(ttl=60)
def get_model_info() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def predict(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Could not reach the prediction API at {API_URL}: {e}")
        return None


def contribution_chart(explanation: list[dict], base_value: float, top_n: int = 10):
    """Horizontal bar of the strongest SHAP contributions for one prediction."""
    items = sorted(explanation, key=lambda c: abs(c["shap_value"]), reverse=True)[:top_n]
    items = items[::-1]  # largest at top
    labels = [f"{c['label']}  =  {c['value']:.2f}" for c in items]
    vals = [c["shap_value"] for c in items]
    colors = [RED if v > 0 else GREEN for v in vals]

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(items) + 1))
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("← lowers risk      SHAP contribution (log-odds)      raises risk →")
    ax.set_title("Why this decision")
    fig.tight_layout()
    return fig


def risk_color(band: str) -> str:
    return {"Low": GREEN, "Moderate": "#d4ac0d", "Elevated": "#e67e22", "High": RED}.get(band, "#333")


# --------------------------------------------------------------------------- #
st.title("💳 Credit Risk Explainer")
st.caption("Predict loan-default risk — and see *why*. Powered by an XGBoost model "
           "with SHAP explanations, served from a FastAPI backend.")

info = get_model_info()
if info is None:
    st.warning(f"The prediction API at **{API_URL}** is not reachable yet. "
               "Start it (or `docker compose up`) and reload.")

tab_assess, tab_insights = st.tabs(["🔎 Risk Assessment", "📊 Model Insights"])

# ---- Tab 1: per-applicant assessment ----
with tab_assess:
    stats = (info or {}).get("feature_stats", {})

    def default(col, fallback):
        return float(stats.get(col, {}).get("median", fallback))

    st.subheader("Applicant details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 100, int(default("age", 45)))
        monthly_income = st.number_input("Monthly income", 0, 100000,
                                         int(default("MonthlyIncome", 5400)), step=100)
        num_dependents = st.slider("Number of dependents", 0, 10,
                                   int(default("NumberOfDependents", 0)))
    with c2:
        revolving_utilization = st.slider("Credit utilization ratio", 0.0, 1.5,
                                          round(default("RevolvingUtilizationOfUnsecuredLines", 0.3), 2),
                                          step=0.01,
                                          help="Card/line balances divided by total limit")
        debt_ratio = st.slider("Debt ratio", 0.0, 2.0,
                               round(default("DebtRatio", 0.35), 2), step=0.01,
                               help="Monthly debt payments / monthly income")
        num_open_credit_lines = st.slider("Open credit lines & loans", 0, 40,
                                          int(default("NumberOfOpenCreditLinesAndLoans", 8)))
    with c3:
        num_real_estate_loans = st.slider("Real-estate loans / lines", 0, 15,
                                          int(default("NumberRealEstateLoansOrLines", 1)))
        num_30_59 = st.slider("Times 30-59 days late", 0, 20, 0)
        num_60_89 = st.slider("Times 60-89 days late", 0, 20, 0)
        num_90 = st.slider("Times 90+ days late", 0, 20, 0)

    if st.button("Assess risk", type="primary", use_container_width=True):
        payload = {
            "revolving_utilization": revolving_utilization,
            "age": age,
            "num_30_59_days_past_due": num_30_59,
            "debt_ratio": debt_ratio,
            "monthly_income": monthly_income,
            "num_open_credit_lines": num_open_credit_lines,
            "num_90_days_late": num_90,
            "num_real_estate_loans": num_real_estate_loans,
            "num_60_89_days_past_due": num_60_89,
            "num_dependents": num_dependents,
        }
        result = predict(payload)
        if result:
            left, right = st.columns([1, 2])
            with left:
                st.metric("Default probability", f"{result['probability']*100:.1f}%")
                band = result["risk_band"]
                st.markdown(f"### :{'red' if band in ('High','Elevated') else 'green'}[{band} risk]")
                decision = result["decision"].upper()
                st.markdown(f"**Decision @ {result['threshold']:.2f} threshold: "
                            f":{'red' if decision=='DECLINE' else 'green'}[{decision}]**")
                st.caption("Decision threshold is set from a 5:1 false-negative:false-"
                           "positive cost assumption — see Model Insights.")
            with right:
                st.pyplot(contribution_chart(result["explanation"], result["base_value"]))
            st.info("Red bars pushed this applicant **toward** default; green bars pushed "
                    "**away**. Bar length is each feature's contribution in log-odds.")

# ---- Tab 2: global model insights ----
with tab_insights:
    if info is None:
        st.stop()
    st.subheader("How good is the model? (held-out test set)")
    tm = info.get("test_metrics", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.3f}")
    m2.metric("PR-AUC", f"{tm.get('pr_auc', 0):.3f}")
    m3.metric("Recall", f"{tm.get('recall', 0):.2f}")
    m4.metric("Precision", f"{tm.get('precision', 0):.2f}")
    st.caption(f"Best model: **{info.get('best_model')}** · decision threshold "
               f"**{info.get('threshold'):.2f}** · default rate "
               f"{info.get('dataset', {}).get('default_rate', 0)*100:.1f}%")

    st.subheader("Global feature importance")
    imp = info.get("global_importance", [])[:12]
    if imp:
        feats = [config.FEATURE_LABELS.get(r["feature"], r["feature"]) for r in imp][::-1]
        vals = [r["mean_abs_shap"] for r in imp][::-1]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.barh(feats, vals, color="#2c3e50")
        ax.set_xlabel("Mean |SHAP| (average impact on risk)")
        ax.set_title("What drives the model overall")
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Performance & explanation charts")
    figs = [
        ("roc_curves.png", "ROC curves"),
        ("pr_curves.png", "Precision-Recall curves"),
        ("calibration.png", "Calibration"),
        ("confusion_matrix.png", "Confusion matrix"),
        ("shap_global_beeswarm.png", "SHAP summary (beeswarm)"),
    ]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(figs):
        path = config.FIGURES / fname
        if path.exists():
            cols[i % 2].image(str(path), caption=caption, use_column_width=True)
