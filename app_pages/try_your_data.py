"""
app_pages/try_your_data.py -- Tab 5: Try Your Data!

Per spec: upload a CSV (or manually enter) raw Tier-1 clinical fields,
validate columns, run the same pre-trained pkl models, show a results
table in the same format as the Homepage table, allow CSV download.

The manual-entry form below covers the core Tier-1 fields needed to
run the models. It does not yet implement every optional Tier-2
time-series add-row table or the full comorbidity checklist UI from
the spec (28 total checkboxes) -- flagged as a simplification given
scope/time; the CSV upload path supports the full raw schema already
(same columns as optoc_sample_patients.csv), so it's the more complete
option today.
"""

import streamlit as st
import pandas as pd

from core.theme import page_header, risk_cell_style, RISK_COLORS
from core.features import build_features_df
from core.model_registry import predict_all, load_registry
from core.theme import risk_tier

RAW_REQUIRED_COLUMNS = [
    "patient_id", "sex", "icu_unit", "age", "icu_los_days",
    "heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate", "spo2",
    "gcs_total", "rass_min", "urine_output_24h", "fluid_balance_24h",
    "sofa_total", "creatinine_baseline", "creatinine_current", "lactate_peak",
    "wbc_first", "magnesium_first", "phosphate_first", "troponin_t_peak",
    "fibrinogen_first", "haemoglobin_latest", "glucose_current",
]


def render():
    page_header("Try Your Data!", "Upload a patient CSV or manually enter data to run the pre-trained models")

    mode = st.radio("Input method:", ["Upload CSV", "Enter Manually"], horizontal=True)

    if mode == "Upload CSV":
        _render_csv_upload()
    else:
        _render_manual_entry()


def _render_csv_upload():
    with st.expander("Required columns"):
        st.write(
            "Your CSV should follow the same schema as `optoc_sample_patients.csv` "
            "(raw Tier-1 clinical fields, not the pre-engineered model features)."
        )
        st.code(", ".join(RAW_REQUIRED_COLUMNS) + ", ... (comorbidity_*, on_* medication flags)",
                 language="text")
        st.caption("A downloadable template isn't wired up in this build — use "
                     "data/optoc_sample_patients.csv in the project folder as a template.")

    uploaded = st.file_uploader("Upload patient data CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to run risk predictions.")
        return

    df_input = pd.read_csv(uploaded)
    st.subheader("Data Preview")
    st.dataframe(df_input.head(), use_container_width=True)

    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df_input.columns]
    st.subheader("Column Validation")
    for col in RAW_REQUIRED_COLUMNS:
        if col in df_input.columns:
            st.write(f"✅ {col}")
        else:
            st.write(f"❌ {col} (missing)")

    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    if st.button("🚀 Run Risk Predictions"):
        with st.spinner("Running ML models... this may take a few seconds."):
            feature_df, patient_ids = build_features_df(df_input)
            registry = load_registry()
            preds = predict_all(feature_df)

            result = df_input.copy()
            for outcome, probs in preds.items():
                result[f"{outcome}_risk"] = probs
                threshold = registry[outcome]["threshold"]
                result[f"{outcome}_tier"] = [risk_tier(p, threshold) for p in probs]
            result["composite_score"] = (
                result[["aki_risk", "sepsis_risk", "mortality_risk", "readmission_risk"]].mean(axis=1) * 100
            ).round(1)

        st.success(f"Risk predictions complete for {len(result)} patient(s).")
        _show_results_table(result)


def _render_manual_entry():
    st.caption("Core Tier-1 fields. (Full 28-checkbox comorbidity list and Tier-2 time-series "
                 "add-row tables are simplified in this build — see module docstring.)")

    with st.form("manual_entry_form"):
        c1, c2, c3 = st.columns(3)
        patient_id = c1.text_input("Patient ID", "P-NEW-01")
        sex = c2.selectbox("Sex", ["M", "F"])
        icu_unit = c3.selectbox("ICU Unit", ["MICU", "SICU", "CCU", "CVICU", "TSICU", "OTHER"])

        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 120, 65)
        icu_los_days = c2.number_input("Days in ICU so far", 0.0, 365.0, 3.0)
        sofa_total = c3.number_input("SOFA Score (Day 1 total)", 0, 24, 6)

        st.markdown("**Vital Signs**")
        c1, c2, c3, c4 = st.columns(4)
        heart_rate = c1.number_input("Heart Rate (bpm)", 0, 250, 88)
        systolic_bp = c2.number_input("Systolic BP (mmHg)", 0, 300, 110)
        diastolic_bp = c3.number_input("Diastolic BP (mmHg)", 0, 200, 70)
        respiratory_rate = c4.number_input("Respiratory Rate", 0, 60, 18)

        c1, c2, c3, c4 = st.columns(4)
        spo2 = c1.number_input("SpO2 (%)", 0, 100, 95)
        gcs_total = c2.number_input("GCS Total", 3, 15, 14)
        rass_min = c3.number_input("RASS Score", -5, 4, -1)
        urine_output_24h = c4.number_input("Urine Output 24h (mL)", 0, 10000, 900)

        fluid_balance_24h = st.number_input("Fluid Balance 24h (mL)", -5000, 10000, 500)

        st.markdown("**Labs**")
        c1, c2, c3, c4 = st.columns(4)
        creatinine_baseline = c1.number_input("Creatinine baseline (mg/dL)", 0.0, 15.0, 1.0)
        creatinine_current = c2.number_input("Creatinine current (mg/dL)", 0.0, 15.0, 1.1)
        lactate_peak = c3.number_input("Lactate peak (mmol/L)", 0.0, 20.0, 1.5)
        wbc_first = c4.number_input("WBC first (K/uL)", 0.0, 50.0, 10.0)

        c1, c2, c3, c4 = st.columns(4)
        magnesium_first = c1.number_input("Magnesium (mEq/L)", 0.0, 5.0, 2.0)
        phosphate_first = c2.number_input("Phosphate (mg/dL)", 0.0, 10.0, 3.5)
        troponin_t_peak = c3.number_input("Troponin T peak (ng/mL)", 0.0, 10.0, 0.02)
        fibrinogen_first = c4.number_input("Fibrinogen (mg/dL)", 0.0, 1000.0, 300.0)

        c1, c2 = st.columns(2)
        haemoglobin_latest = c1.number_input("Hemoglobin (g/dL)", 0.0, 20.0, 10.0)
        glucose_current = c2.number_input("Glucose current (mg/dL)", 0.0, 800.0, 130.0)

        st.markdown("**Active Support / Medications**")
        c1, c2, c3, c4 = st.columns(4)
        on_vasopressor = c1.checkbox("On Vasopressor?")
        on_mechanical_ventilation = c2.checkbox("On Mechanical Ventilation?")
        on_rrt = c3.checkbox("On RRT/Dialysis?")
        on_hfnc_niv = c4.checkbox("On HFNC/BiPAP/CPAP?")

        c1, c2, c3, c4 = st.columns(4)
        on_heparin = c1.checkbox("On Heparin?")
        on_tpn = c2.checkbox("On TPN?")
        on_steroid = c3.checkbox("On Corticosteroids?")
        on_cefepime = c4.checkbox("On Cefepime?")

        c1, c2, c3 = st.columns(3)
        on_linezolid = c1.checkbox("On Linezolid?")
        on_qtc_med = c2.checkbox("On QTc-prolonging drug?")
        on_anticholinergic = c3.checkbox("On anticholinergic drug?")

        c1, c2 = st.columns(2)
        nephrotoxin_count = c1.number_input("Number of nephrotoxic drugs active", 0, 10, 0)
        total_discharge_meds = c2.number_input("Total medications at discharge", 0, 50, 8)

        st.markdown("**Comorbidities**")
        c1, c2, c3, c4, c5 = st.columns(5)
        comorb_flags = {
            "comorbidity_MI": c1.checkbox("MI"), "comorbidity_CHF": c2.checkbox("CHF"),
            "comorbidity_stroke": c3.checkbox("Stroke"), "comorbidity_dementia": c4.checkbox("Dementia"),
            "comorbidity_COPD": c5.checkbox("COPD"),
        }
        c1, c2, c3, c4, c5 = st.columns(5)
        comorb_flags.update({
            "comorbidity_diabetes_uncomplicated": c1.checkbox("Diabetes (uncomp.)"),
            "comorbidity_diabetes_complicated": c2.checkbox("Diabetes (comp.)"),
            "comorbidity_CKD": c3.checkbox("CKD"), "comorbidity_cancer": c4.checkbox("Cancer"),
            "comorbidity_metastatic_cancer": c5.checkbox("Metastatic Cancer"),
        })

        submitted = st.form_submit_button("🚀 Run Risk Predictions")

    if submitted:
        row = {
            "patient_id": patient_id, "sex": sex, "icu_unit": icu_unit, "age": age,
            "icu_los_days": icu_los_days, "sofa_total": sofa_total,
            "heart_rate": heart_rate, "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
            "respiratory_rate": respiratory_rate, "spo2": spo2, "fio2": None,
            "gcs_total": gcs_total, "rass_min": rass_min, "urine_output_24h": urine_output_24h,
            "fluid_balance_24h": fluid_balance_24h, "map_min": None,
            "creatinine_baseline": creatinine_baseline, "creatinine_current": creatinine_current,
            "lactate_peak": lactate_peak, "wbc_first": wbc_first, "magnesium_first": magnesium_first,
            "phosphate_first": phosphate_first, "troponin_t_peak": troponin_t_peak,
            "fibrinogen_first": fibrinogen_first, "haemoglobin_latest": haemoglobin_latest,
            "glucose_current": glucose_current,
            "on_vasopressor": int(on_vasopressor), "on_mechanical_ventilation": int(on_mechanical_ventilation),
            "on_rrt": int(on_rrt), "on_hfnc_niv": int(on_hfnc_niv), "on_heparin": int(on_heparin),
            "on_tpn": int(on_tpn), "on_steroid": int(on_steroid), "on_cefepime": int(on_cefepime),
            "on_linezolid": int(on_linezolid), "on_qtc_med": int(on_qtc_med),
            "on_anticholinergic": int(on_anticholinergic),
            "nephrotoxin_count": nephrotoxin_count, "total_discharge_meds": total_discharge_meds,
        }
        row.update({k: int(v) for k, v in comorb_flags.items()})

        df_input = pd.DataFrame([row])
        with st.spinner("Running ML models... this may take a few seconds."):
            feature_df, patient_ids = build_features_df(df_input)
            registry = load_registry()
            preds = predict_all(feature_df)
            result = df_input.copy()
            for outcome, probs in preds.items():
                result[f"{outcome}_risk"] = probs
                threshold = registry[outcome]["threshold"]
                result[f"{outcome}_tier"] = [risk_tier(p, threshold) for p in probs]
            result["composite_score"] = (
                result[["aki_risk", "sepsis_risk", "mortality_risk", "readmission_risk"]].mean(axis=1) * 100
            ).round(1)

        st.success("Risk prediction complete.")
        _show_results_table(result)


def _show_results_table(result):
    st.markdown("### Results")
    display_cols = ["patient_id", "aki_risk", "aki_tier", "sepsis_risk", "sepsis_tier",
                     "mortality_risk", "mortality_tier", "readmission_risk", "readmission_tier",
                     "composite_score"]
    display_cols = [c for c in display_cols if c in result.columns]
    display = result[display_cols].copy()
    for outcome in ["aki", "sepsis", "mortality", "readmission"]:
        if f"{outcome}_risk" in display.columns:
            display[f"{outcome}_risk"] = (display[f"{outcome}_risk"] * 100).round(1).astype(str) + "%"

    def _style(val):
        return risk_cell_style(val) if val in RISK_COLORS else ""

    tier_cols = [c for c in display.columns if c.endswith("_tier")]
    styled = display.style.map(_style, subset=tier_cols)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download results as CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="optoc_try_your_data_results.csv",
        mime="text/csv",
    )
