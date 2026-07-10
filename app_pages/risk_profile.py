"""
app_pages/risk_profile.py -- Tab 3: Risk Profile

Per spec: replaces the four separate outcome tabs. User picks one of 4
outcome cards; population SHAP chart, patient LIME chart, odds ratio,
auto-generated insight text, and high-risk list all update to match.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from core.theme import page_header, risk_badge_html
from core.model_registry import OUTCOME_LABELS, OUTCOME_PHRASES, load_registry
from core.explainers import shap_population_drivers, lime_patient_factors
from core.data_io import get_patient_row, get_patient_feature_row

OUTCOME_KEYS = ["aki", "sepsis", "mortality", "readmission"]


def render(enriched_df, feature_df, pid):
    page_header("Risk Profile", "Population and patient-level view for one outcome at a time")

    if "selected_outcome" not in st.session_state:
        st.session_state["selected_outcome"] = "mortality"

    # ------------------------------------------------------------
    # Outcome selector — 4 cards
    # ------------------------------------------------------------
    cols = st.columns(4)
    for i, outcome in enumerate(OUTCOME_KEYS):
        active = st.session_state["selected_outcome"] == outcome
        clicked = cols[i].button(
            OUTCOME_LABELS[outcome],
            key=f"outcome_card_{outcome}",
            use_container_width=True,
            type="primary" if active else "secondary",
        )
        if clicked:
            st.session_state["selected_outcome"] = outcome
            st.rerun()

    outcome = st.session_state["selected_outcome"]
    st.markdown(f"### {OUTCOME_LABELS[outcome]} - Population View")

    # ------------------------------------------------------------
    # Patient selector + Odds Ratio summary
    # ------------------------------------------------------------
    top_row = st.columns([1, 2])
    with top_row[0]:
        patient_ids = enriched_df["patient_id"].tolist()
        idx = patient_ids.index(pid) if pid in patient_ids else 0
        selected_pid = st.selectbox("Select Patient", patient_ids, index=idx, key="risk_profile_patient")
        st.session_state["selected_patient"] = selected_pid

    p = get_patient_row(enriched_df, selected_pid)
    risk_col = f"{outcome}_risk"
    patient_prob = p[risk_col]
    cohort_mean_prob = enriched_df[risk_col].mean()

    def to_odds(prob):
        prob = min(max(prob, 1e-6), 1 - 1e-6)
        return prob / (1 - prob)

    odds_ratio = to_odds(patient_prob) / to_odds(cohort_mean_prob)

    with top_row[1]:
        st.markdown(
            f"""
            <div class="optoc-card">
            <b>{selected_pid}</b> is <b>{odds_ratio:.1f}x</b> more likely to
            {OUTCOME_PHRASES[outcome]} than the average patient in this ICU cohort.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ------------------------------------------------------------
    # Key Risk Factors (All Patients) — SHAP
    # ------------------------------------------------------------
    left, right = st.columns(2)
    with left:
        st.markdown("**Key Risk Factors (All Patients)**")
        pop_df = shap_population_drivers(outcome, feature_df, top_n=10)
        fig = go.Figure(go.Bar(x=pop_df["impact"], y=pop_df["feature"], orientation="h",
                                 marker_color="#2563EB"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Mean |impact| across cohort")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Contributing Factors (This Patient)**")
        patient_feat_row = get_patient_feature_row(feature_df, enriched_df, selected_pid)
        lime_df = lime_patient_factors(outcome, patient_feat_row, top_n=10)
        colors = ["#C00000" if d == "Increases risk" else "#2563EB" for d in lime_df["direction"]]
        fig = go.Figure(go.Bar(x=lime_df["weight"], y=lime_df["feature"], orientation="h",
                                 marker_color=colors))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # Auto-generated key clinical insight
    # ------------------------------------------------------------
    top3 = pop_df.head(3)["feature"].tolist()
    insight = (
        f"Across today's cohort, {', '.join(top3[:-1])} and {top3[-1]} are the strongest drivers of "
        f"{OUTCOME_LABELS[outcome].lower()}. Pharmacists reviewing high-risk patients for this outcome "
        f"should prioritize checking these factors first."
    )
    st.markdown(f'<div class="optoc-card">{insight}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # High-risk patient list
    # ------------------------------------------------------------
    st.markdown(f"**High-Risk Patients ({OUTCOME_LABELS[outcome]} > 0.70)**")
    high_risk = enriched_df[enriched_df[risk_col] > 0.70].copy()
    if len(high_risk) == 0:
        st.info("No patients currently above the 0.70 threshold for this outcome.")
    else:
        high_risk["Risk %"] = (high_risk[risk_col] * 100).round(1).astype(str) + "%"

        registry = load_registry()
        entry = registry[outcome]
        model = entry["model"]
        explainer = entry["shap_explainer"]
        feature_names = entry["feature_names"]
        preprocessor = model.named_steps["prep"]

        high_risk_feat = feature_df.loc[high_risk.index]
        X_t = preprocessor.transform(high_risk_feat)
        shap_vals = np.array(explainer.shap_values(X_t))
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        from core.explainers import _clean_feature_name
        top_factors = [
            _clean_feature_name(feature_names[np.argmax(np.abs(row))])
            for row in shap_vals
        ]
        high_risk["Top Risk Factor"] = top_factors

        st.dataframe(
            high_risk[["patient_id", "Risk %", "Top Risk Factor"]].rename(columns={"patient_id": "Patient ID"}),
            use_container_width=True, hide_index=True,
        )
