"""
app_pages/individual_patient.py -- Tab 2: Individual Patient Profile

Per spec: deep dive on one patient. 5 gauges (AKI/Sepsis/Mortality/
Readmission/Composite), composite tier badge, 7-domain panel, outcome
selector (Mortality/Readmission only) driving two independent
explanation charts (SHAP = Contributing Factors, LIME = Personalized
Risk Summary), vitals snapshot, medication profile + alerts,
time-series trends, pharmacist note, print summary.
"""

import streamlit as st
import plotly.graph_objects as go

from core.theme import page_header, risk_badge_html, risk_tier, RISK_COLORS
from core.data_io import get_patient_row, get_patient_feature_row, get_patient_timeseries
from core.model_registry import load_registry, OUTCOME_LABELS
from core.explainers import shap_patient_factors, lime_patient_factors
from core.scoring import DOMAIN_FUNCTIONS, DOMAIN_LABELS
from core.med_alerts import get_medication_alerts, SEVERITY_STYLE_CLASS

OUTCOME_KEYS = ["aki", "sepsis", "mortality", "readmission"]


def _gauge(value_pct, title, tier, clickable_note=""):
    color = RISK_COLORS[tier]["border"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": RISK_COLORS["LOW"]["bg"]},
                {"range": [40, 70], "color": RISK_COLORS["MEDIUM"]["bg"]},
                {"range": [70, 100], "color": RISK_COLORS["HIGH"]["bg"]},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def render(enriched_df, feature_df, timeseries_df, pid):
    registry = load_registry()
    p = get_patient_row(enriched_df, pid)
    if p is None:
        st.warning("No patient selected.")
        return

    st.markdown('<div class="optoc-section-title">Individual Patient Profile</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="background-color:#0F172A; color:white; padding:14px 20px;
                     border-radius:10px; margin-bottom:14px;">
            <b>Patient ID:</b> {p['patient_id']} &nbsp;|&nbsp;
            <b>Age:</b> {p['age']} &nbsp;|&nbsp; <b>Sex:</b> {p['sex']} &nbsp;|&nbsp;
            <b>ICU Unit:</b> {p['icu_unit']} &nbsp;|&nbsp;
            <b>Days in ICU:</b> {p['days_in_icu']} &nbsp;|&nbsp;
            <b>Admitted:</b> {p['admission_date']} &nbsp;|&nbsp;
            <b>Discharge Due:</b> {p['discharge_due_date']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # 5 gauges
    # ------------------------------------------------------------
    gauge_cols = st.columns(5)
    for i, outcome in enumerate(OUTCOME_KEYS):
        risk_pct = p[f"{outcome}_risk"] * 100
        tier = p[f"{outcome}_tier"]
        with gauge_cols[i]:
            st.plotly_chart(_gauge(risk_pct, OUTCOME_LABELS[outcome], tier), use_container_width=True)
            if st.button(f"View {OUTCOME_LABELS[outcome]} details →", key=f"gauge_{outcome}"):
                st.session_state["active_tab"] = "Risk Profile"
                st.session_state["selected_outcome"] = outcome
                st.session_state["selected_patient"] = pid
                st.rerun()

    with gauge_cols[4]:
        st.plotly_chart(_gauge(p["composite_score"], "Composite Score", p["composite_tier"]),
                         use_container_width=True)

    # ------------------------------------------------------------
    # Composite tier badge
    # ------------------------------------------------------------
    st.markdown(
        f"""
        <div style="text-align:center; margin: 10px 0 20px 0;">
            {risk_badge_html(p['composite_tier'], f"{p['composite_tier']} RISK")}
            <div style="margin-top:6px;">This patient is classified as
            <b>{p['composite_tier']}</b> risk based on a composite score of
            <b>{p['composite_score']:.0f}%</b>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # 7-Domain Risk Panel
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">7-Domain Risk Panel</div>', unsafe_allow_html=True)
    domain_cols = st.columns(7)
    for i, (domain, fn) in enumerate(DOMAIN_FUNCTIONS.items()):
        score, level, reasons = fn(p)
        with domain_cols[i]:
            st.markdown(f"**{DOMAIN_LABELS[domain]}**")
            st.markdown(risk_badge_html(level), unsafe_allow_html=True)
            st.caption(f"Score: {score}")
            with st.expander("Drivers"):
                if reasons:
                    for text, pts in reasons:
                        st.write(f"- {text} (+{pts})")
                else:
                    st.write("No contributing factors flagged.")
            if st.button("View in heatmap →", key=f"domain_{domain}"):
                st.session_state["active_tab"] = "Domain Specific Risk"
                st.session_state["selected_patient"] = pid
                st.rerun()

    st.markdown("---")

    # ------------------------------------------------------------
    # Outcome selector (Mortality / Readmission only) + two explanation charts
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">Why This Patient\'s Risk Is Elevated</div>',
                unsafe_allow_html=True)
    selected_outcome = st.radio("Explain outcome:", ["mortality", "readmission"],
                                 format_func=lambda o: OUTCOME_LABELS[o], horizontal=True,
                                 key="tab2_outcome_selector")

    patient_feat_row = get_patient_feature_row(feature_df, enriched_df, pid)

    left, right = st.columns([55, 45])
    with left:
        st.markdown("**Contributing Factors**")
        try:
            shap_df = shap_patient_factors(selected_outcome, patient_feat_row, top_n=10)
            colors = ["#C00000" if v > 0 else "#2563EB" for v in shap_df["impact"]]
            fig = go.Figure(go.Bar(
                x=shap_df["impact"], y=shap_df["feature"], orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="Impact on risk (red = increases, blue = decreases)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Contributing factors unavailable: {e}")

    with right:
        outcome_risk_pct = p[f"{selected_outcome}_risk"] * 100
        st.markdown(f"**Why this patient's {OUTCOME_LABELS[selected_outcome]} is {outcome_risk_pct:.0f}%**")
        try:
            lime_df = lime_patient_factors(selected_outcome, patient_feat_row, top_n=10)
            colors = ["#C00000" if d == "Increases risk" else "#2563EB" for d in lime_df["direction"]]
            fig = go.Figure(go.Bar(
                x=lime_df["weight"], y=lime_df["feature"], orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Personalized risk summary unavailable: {e}")

    st.markdown("---")

    # ------------------------------------------------------------
    # Vitals & labs snapshot
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">Current Vitals & Laboratory Results</div>',
                unsafe_allow_html=True)
    vitals = [
        ("MAP", p.get("map_min"), "mmHg"), ("HR", p.get("heart_rate"), "bpm"),
        ("RR", p.get("respiratory_rate"), "br/min"), ("SpO2", p.get("spo2"), "%"),
        ("Lactate", p.get("lactate_peak"), "mmol/L"), ("Creatinine", p.get("creatinine_current"), "mg/dL"),
        ("WBC", p.get("wbc_first"), "K/uL"), ("Fibrinogen", p.get("fibrinogen_first"), "mg/dL"),
    ]
    vcols = st.columns(4)
    for i, (label, val, unit) in enumerate(vitals):
        vcols[i % 4].metric(label, f"{val} {unit}" if val is not None else "N/A")

    st.markdown("---")

    # ------------------------------------------------------------
    # Medication profile + alerts
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">Medication Profile</div>', unsafe_allow_html=True)
    med_flags = {
        "Vasopressor": p.get("on_vasopressor"), "Mechanical Ventilation": p.get("on_mechanical_ventilation"),
        "RRT/Dialysis": p.get("on_rrt"), "High-Flow O2/NIV": p.get("on_hfnc_niv"),
        "Heparin": p.get("on_heparin"), "TPN": p.get("on_tpn"), "Steroid": p.get("on_steroid"),
        "Cefepime": p.get("on_cefepime"), "Linezolid": p.get("on_linezolid"),
        "QTc-prolonging med": p.get("on_qtc_med"), "Anticholinergic": p.get("on_anticholinergic"),
    }
    active_meds = [name for name, flag in med_flags.items() if flag]
    st.write(", ".join(active_meds) if active_meds else "No active medication flags recorded.")
    st.caption(f"Nephrotoxic drug count: {p.get('nephrotoxin_count', 'N/A')} · "
               f"Total discharge medications: {p.get('total_discharge_meds', 'N/A')}")

    st.markdown("**⚠️ Medication Alerts**")
    alerts = get_medication_alerts(p, aki_risk=p["aki_risk"])
    if not alerts:
        st.success("No medication alerts triggered for this patient's current profile.")
    else:
        for a in alerts:
            css_class = SEVERITY_STYLE_CLASS.get(a["severity"], "optoc-alert-info")
            st.markdown(f'<div class="{css_class}">{a["text"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # Time-series trends
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">Time-Series Trends</div>', unsafe_allow_html=True)
    ts = get_patient_timeseries(timeseries_df, pid)
    if len(ts) == 0:
        st.info("No time-series data available for this patient.")
    else:
        plot_vars = [("map", "MAP (mmHg)"), ("hr", "Heart Rate (bpm)"), ("rr", "Respiratory Rate"),
                     ("spo2", "SpO2 (%)"), ("lactate", "Lactate (mmol/L)"), ("creatinine", "Creatinine (mg/dL)")]
        tcols = st.columns(3)
        for i, (col, label) in enumerate(plot_vars):
            if col in ts.columns:
                fig = go.Figure(go.Scatter(x=ts["hours_since_admission"], y=ts[col], mode="lines+markers"))
                fig.update_layout(title=label, height=220, margin=dict(t=30, b=10, l=10, r=10),
                                    xaxis_title="Hours since admission")
                tcols[i % 3].plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # Pharmacist note + print summary
    # ------------------------------------------------------------
    st.text_area("Pharmacist note for this patient (not saved to database, for session reference only)")
    st.caption("🖨️ Print summary: export this patient's profile as PDF is not wired up in this build "
               "(no PDF-rendering library was in scope for this pass) -- flagged as an open item.")
