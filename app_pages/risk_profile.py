"""
app_pages/risk_profile.py -- Tab 3: Clinical Insights

Merges the former Risk Profile tab (patient-specific explainability --
outcome selector, population SHAP, patient LIME, odds ratio, high-risk
list) with the former Domain Specific Risk tab (population-level
7-domain heatmap) into one "Clinical Insights" page, per the decision
to stop splitting closely-related explanatory views across tabs.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from core.theme import page_header, risk_badge_html, risk_cell_style, RISK_COLORS
from core.model_registry import OUTCOME_LABELS, OUTCOME_PHRASES, load_registry
from core.explainers import shap_population_drivers, lime_patient_factors
from core.data_io import get_patient_row, get_patient_feature_row
from core.scoring import DOMAIN_FUNCTIONS, DOMAIN_LABELS, DOMAIN_DATA_COMPLETENESS

OUTCOME_KEYS = ["aki", "sepsis", "mortality", "readmission"]

# Raw feature columns (see models/required_feature_columns.txt) a
# pharmacist can directly act on -- medication/regimen choices they
# control. Everything else (labs, vitals, comorbidity burden, SOFA,
# age, etc.) reflects disease severity or demographics and isn't
# something a pharmacist can change directly, so it's "clinical context."
ACTIONABLE_FEATURES = {
    "nephrotoxin_count", "total_discharge_meds", "mrci_simplified",
    "flag_tpn", "flag_hit_risk_proxy", "flag_qtc_prolonging_med_exposure",
    "flag_anticholinergic_exposure", "flag_cefepime_exposure",
    "flag_linezolid_exposure", "flag_steroid_induced_hyperglycemia",
}


def _split_by_actionability(df, n=5):
    """Splits a SHAP/LIME results DataFrame (must have a "raw" column)
    into (top-n actionable, top-n non-actionable), both already sorted
    by whatever order they arrived in (importance, highest first)."""
    actionable = df[df["raw"].isin(ACTIONABLE_FEATURES)].head(n)
    non_actionable = df[~df["raw"].isin(ACTIONABLE_FEATURES)].head(n)
    return actionable, non_actionable

DOMAIN_INTRO = (
    "The 7-Domain Clinical Risk Assessment scores each ICU patient across seven distinct organ "
    "systems — Neurological &amp; Sedation, Pulmonary, Cardiovascular, Renal, GI/Metabolic, "
    "Haematology/Coagulation, and Infectious Disease — using bedside-available laboratory "
    "values, vital signs, medication exposures, and medical history. Each domain produces an "
    "independent score (0 to 10+) colour-coded as LOW, MODERATE, or HIGH, allowing the "
    "pharmacist to immediately identify which organ system is most at risk and target "
    "interventions accordingly."
)


def render(enriched_df, feature_df, display_df, pid):
    page_header("Clinical Insights",
                "Patient-specific explainability and population-level context, one outcome or domain at a time")

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
    top_row = st.columns([1, 2], vertical_alignment="center")
    with top_row[0]:
        patient_ids = enriched_df["patient_id"].tolist()
        idx = patient_ids.index(pid) if pid in patient_ids else 0
        selected_pid = st.selectbox("Select Patient", patient_ids, index=idx, key="risk_profile_patient")
        if selected_pid != pid:
            st.session_state["selected_patient"] = selected_pid
            st.rerun()

    p = get_patient_row(enriched_df, selected_pid)
    risk_col = f"{outcome}_risk"
    patient_prob = p[risk_col]
    cohort_mean_prob = enriched_df[risk_col].mean()

    def to_odds(prob):
        prob = min(max(prob, 1e-6), 1 - 1e-6)
        return prob / (1 - prob)

    odds_ratio = to_odds(patient_prob) / to_odds(cohort_mean_prob)

    with top_row[1]:
        # odds_ratio < 1 reads as "less likely" (inverted), matching
        # validate_dashboard.py's odds-ratio text logic -- the previous
        # version always said "more likely" even when the ratio was below 1.
        if odds_ratio >= 1:
            comparison = f"{odds_ratio:.1f}x more likely"
        else:
            comparison = f"{1 / odds_ratio:.1f}x less likely"
        st.markdown(
            f"""
            <div class="optoc-card">
            Patient <b>{selected_pid}</b> is <b>{comparison}</b> to
            {OUTCOME_PHRASES[outcome]} than the average patient.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ------------------------------------------------------------
    # Plain-language summary, visible by default -- computed from the
    # same LIME factors used in the technical chart below, just surfaced
    # first and translated to a sentence instead of a bar chart. The
    # SHAP/LIME visualizations remain available in "Advanced
    # Explainability" for anyone who wants to look deeper.
    # ------------------------------------------------------------
    patient_feat_row = get_patient_feature_row(feature_df, enriched_df, selected_pid)
    lime_df = lime_patient_factors(outcome, patient_feat_row, top_n=10)
    increasing = lime_df[lime_df["direction"] == "Increases risk"].head(4)["feature"].tolist()

    if increasing:
        joined = (increasing[0] if len(increasing) == 1
                  else ", ".join(increasing[:-1]) + f", and {increasing[-1]}")
        plain_summary = f"{OUTCOME_LABELS[outcome]} is elevated for {selected_pid} primarily because of {joined.lower()}."
    else:
        plain_summary = (
            f"No single factor stands out as driving {selected_pid}'s {OUTCOME_LABELS[outcome].lower()} "
            f"-- risk appears diffusely spread across several smaller contributors (see Advanced "
            f"Explainability below)."
        )
    st.markdown(f'<div class="optoc-card" style="font-size:15.5px;">{plain_summary}</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # Advanced Explainability -- the technical SHAP/LIME layer, collapsed
    # by default so it doesn't compete with the plain-language summary
    # above for a clinician who just wants the answer.
    # ------------------------------------------------------------
    with st.expander("Advanced Explainability (SHAP / LIME)", expanded=False):
        pop_df = shap_population_drivers(outcome, feature_df, top_n=10)

        def _factor_bar(df, x_col, colors):
            if df.empty:
                st.caption("None of the top factors fall in this category.")
                return
            fig = go.Figure(go.Bar(x=df[x_col], y=df["feature"], orientation="h", marker_color=colors))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Key Risk Factors (All Patients)**")
        shap_actionable, shap_clinical = _split_by_actionability(pop_df, n=5)
        shap_left, shap_right = st.columns(2)
        with shap_left:
            st.markdown("##### Pharmacist-Actionable Factors")
            st.caption("Variables the pharmacist team can directly intervene on")
            _factor_bar(shap_actionable, "impact", "#2ecc71")
        with shap_right:
            st.markdown("##### Clinical Context Factors")
            st.caption("Disease severity and demographics -- inform risk but aren't directly modifiable")
            _factor_bar(shap_clinical, "impact", "#78909C")

        st.markdown("---")

        st.markdown("**Contributing Factors (This Patient)**")
        lime_actionable, lime_clinical = _split_by_actionability(lime_df, n=5)
        lime_left, lime_right = st.columns(2)
        with lime_left:
            st.markdown("##### Pharmacist-Actionable Factors")
            st.caption("What's driving this patient's risk that can be changed")
            colors = ["#C00000" if d == "Increases risk" else "#2563EB" for d in lime_actionable["direction"]]
            _factor_bar(lime_actionable, "weight", colors)
        with lime_right:
            st.markdown("##### Clinical Context Factors")
            st.caption("Background clinical factors for this patient")
            colors = ["#C00000" if d == "Increases risk" else "#2563EB" for d in lime_clinical["direction"]]
            _factor_bar(lime_clinical, "weight", colors)

        st.markdown("---")

        top3 = pop_df.head(3)["feature"].tolist()
        insight_bullets = "".join(
            f"<li>{feat} is among the strongest drivers of {OUTCOME_LABELS[outcome].lower()}.</li>"
            for feat in top3
        )
        st.markdown(
            f'<div class="optoc-card"><ul style="margin:0; padding-left:20px;">{insight_bullets}'
            f'<li>Pharmacists reviewing high-risk patients for this outcome should prioritize '
            f'checking these factors first.</li></ul></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ------------------------------------------------------------
    # High-risk patient list
    # ------------------------------------------------------------
    st.markdown(f"**High-Risk Patients ({OUTCOME_LABELS[outcome]} > 0.70)**")
    high_risk = enriched_df[enriched_df[risk_col] > 0.70].copy()

    if selected_pid not in high_risk["patient_id"].values:
        st.info(f"{selected_pid} is not at High Risk of {OUTCOME_LABELS[outcome]}.")

    if len(high_risk) == 0:
        st.caption("No patients in this cohort are currently above the 0.70 threshold for this outcome.")
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

    st.markdown("---")

    # ------------------------------------------------------------
    # Population-level context: 7-Domain Heatmap (moved here from the
    # former Domain Specific Risk tab, and before that briefly on the
    # Homepage -- this is where population-level explainability belongs,
    # alongside the patient-specific explainability above).
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">Population Context: 7-Domain Risk</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="optoc-card">{DOMAIN_INTRO}</div>', unsafe_allow_html=True)

    domain_names = list(DOMAIN_FUNCTIONS.keys())

    if "domain_card_filter" not in st.session_state:
        st.session_state["domain_card_filter"] = None

    st.markdown('<div class="optoc-section-title">Domain Summary — Click a Domain to Filter</div>',
                unsafe_allow_html=True)
    st.caption(
        "Each card shows how many patients are currently HIGH risk in that domain. Click "
        "\"Filter\" under a domain to narrow the heatmap below to only those patients."
    )

    card_cols = st.columns(7)
    for i, domain in enumerate(domain_names):
        level_col = f"domain_{domain}_level"
        high_count = (display_df[level_col] == "HIGH").sum()
        with card_cols[i]:
            st.markdown(f"**{DOMAIN_LABELS[domain]}**")
            st.metric("HIGH", high_count, label_visibility="collapsed")
            avail, total = DOMAIN_DATA_COMPLETENESS[domain]
            completeness_color = RISK_COLORS["LOW"]["text"] if avail == total else RISK_COLORS["MEDIUM"]["text"]
            st.markdown(
                f'<span style="font-size:10.5px; color:{completeness_color};">'
                f'{avail}/{total} criteria scorable on this data</span>',
                unsafe_allow_html=True,
            )
            if st.button("Filter", key=f"domain_card_{domain}"):
                st.session_state["domain_card_filter"] = domain
                st.rerun()

    st.caption(
        "\"Criteria scorable\" = how many of that domain's scoring rules have the data they need "
        "in this dataset. A domain below 100% can still show LOW even when several of its "
        "criteria structurally can't fire here (the field they check isn't collected) -- treat a "
        "LOW score on a low-completeness domain as \"not enough data to flag,\" not \"verified low.\""
    )

    if st.session_state["domain_card_filter"]:
        st.info(f"Showing only patients at HIGH risk in "
                f"{DOMAIN_LABELS[st.session_state['domain_card_filter']]}.")
        if st.button("Clear domain filter"):
            st.session_state["domain_card_filter"] = None
            st.rerun()

    f1, f2 = st.columns(2)
    domain_units = ["All"] + sorted(display_df["icu_unit"].dropna().unique().tolist())
    domain_unit_filter = f1.selectbox("ICU Unit", domain_units, key="domain_unit_filter")
    sort_domain = f2.selectbox("Sort by domain", domain_names, format_func=lambda d: DOMAIN_LABELS[d])

    domain_filtered = display_df.copy()
    if domain_unit_filter != "All":
        domain_filtered = domain_filtered[domain_filtered["icu_unit"] == domain_unit_filter]
    if st.session_state["domain_card_filter"]:
        d = st.session_state["domain_card_filter"]
        domain_filtered = domain_filtered[domain_filtered[f"domain_{d}_level"] == "HIGH"]

    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    domain_filtered = domain_filtered.sort_values(
        by=f"domain_{sort_domain}_level", key=lambda s: s.map(order)
    )

    st.markdown("#### Domain Heatmap")

    heatmap_cols = ["patient_id"] + [f"domain_{d}_level" for d in domain_names]
    heatmap_display = domain_filtered[heatmap_cols].rename(
        columns={f"domain_{d}_level": DOMAIN_LABELS[d] for d in domain_names}
    ).rename(columns={"patient_id": "Patient ID"})

    def _domain_style(val):
        return risk_cell_style(val) if val in RISK_COLORS else ""

    # Switched from st.table to st.dataframe with row-click navigation --
    # st.table can't be clicked at all, and this is the specific
    # improvement requested (click a heatmap row to jump to that
    # patient). Trade-off: st.dataframe's grid renderer ignores
    # Styler.set_table_styles, so the column HEADERS are no longer bold
    # (cell coloring and the bold Patient ID column still work fine via
    # Styler.map, which st.dataframe does respect).
    heatmap_styled = (
        heatmap_display.style
        .map(_domain_style, subset=[DOMAIN_LABELS[d] for d in domain_names])
        .map(lambda _: "font-weight:700; color:#000000;", subset=["Patient ID"])
    )
    heatmap_event = st.dataframe(
        heatmap_styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="domain_heatmap_table",
    )

    heatmap_selected = heatmap_event.selection.rows if heatmap_event and heatmap_event.selection else []
    if heatmap_selected:
        clicked_pid = heatmap_display.iloc[heatmap_selected[0]]["Patient ID"]
        st.session_state["selected_patient"] = clicked_pid
        st.session_state["active_tab"] = "Individual Patient"
        st.rerun()

    st.caption(
        f"Click any row above to open that patient's Individual Patient profile. Highlighted "
        f"patient: {pid}."
    )
