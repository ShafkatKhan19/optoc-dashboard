"""
app_pages/individual_patient.py -- Tab 2: Individual Patient Profile

Restructured around the clinical workflow rather than the AI workflow:
opens with an auto-generated clinical summary + prioritized pharmacist
actions, then a compact per-outcome risk summary (replacing 5 full
gauge charts), 7-domain panel, vitals/labs, comorbidities, medication
alerts, vital-sign trends, pharmacist note, and print summary. The
SHAP/LIME explainability deep-dive now lives only in the Clinical
Insights tab (reached via "More Details") -- keeping it here too was
pure duplication of the same charts.
"""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from core.theme import page_header, risk_badge_html, RISK_COLORS
from core.data_io import get_patient_row, get_patient_timeseries
from core.model_registry import OUTCOME_LABELS
from core.scoring import DOMAIN_FUNCTIONS, DOMAIN_LABELS, DOMAIN_DATA_COMPLETENESS
from core.med_alerts import get_medication_alerts

OUTCOME_KEYS = ["aki", "sepsis", "mortality", "readmission"]

# Generic recommended action per domain, used to build the priority
# actions list when a domain is HIGH and isn't already covered by a
# more specific medication alert recommendation.
DOMAIN_ACTIONS = {
    "Neuro": "Reassess sedation depth and delirium risk (review sedating/anticholinergic medications)",
    "Pulmonary": "Reassess respiratory support and ventilator weaning readiness",
    "Cardio": "Reassess hemodynamic support (fluids/vasopressors) and monitor perfusion (lactate, MAP)",
    "Renal": "Review nephrotoxic medications and adjust renal dosing",
    "GI": "Monitor electrolytes and glucose closely",
    "Heme": "Monitor coagulation status and platelet trend",
    "ID": "Reassess antimicrobial regimen and infection source control",
}


def _clinical_summary(p, domain_data, alerts):
    """Rule-based synthesis (not free-text AI generation) of this
    patient's top clinical drivers and a short prioritized action list,
    built from the same 7-domain scoring rules and medication alert
    logic already used elsewhere on this page -- not new/separate logic,
    just surfaced first instead of last."""
    ranked = sorted(
        [(d, score, level, reasons) for d, (score, level, reasons) in domain_data.items()
         if level in ("HIGH", "MODERATE")],
        key=lambda x: x[1], reverse=True,
    )

    driver_phrases = []
    for domain, score, level, reasons in ranked[:4]:
        if reasons:
            top_reason = max(reasons, key=lambda r: r[1])[0]
            driver_phrases.append(f"{DOMAIN_LABELS[domain].lower()} ({top_reason.lower()})")
        else:
            driver_phrases.append(DOMAIN_LABELS[domain].lower())

    if driver_phrases:
        joined = (driver_phrases[0] if len(driver_phrases) == 1
                  else ", ".join(driver_phrases[:-1]) + f", and {driver_phrases[-1]}")
        summary = f"This patient's elevated risk appears driven primarily by {joined}."
    else:
        summary = (
            f"No single dominant clinical driver was identified from the current 7-domain "
            f"assessment -- the {p['composite_tier'].lower()} composite risk reflects the "
            f"underlying outcome models rather than one flagged domain."
        )

    actionable_alerts = [a for a in alerts if a["severity"] in ("CRITICAL", "ACTION NEEDED", "WARNING")]
    interventions = []
    for a in actionable_alerts:
        clause = a["text"].split(": ", 1)[-1] if ": " in a["text"] else a["text"]
        interventions.append(clause)
    for domain, score, level, reasons in ranked:
        if level == "HIGH" and domain in DOMAIN_ACTIONS:
            interventions.append(DOMAIN_ACTIONS[domain])

    seen, deduped = set(), []
    for item in interventions:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return summary, deduped[:5]


def _fmt_submetric(val):
    if val is None:
        return "N/A"
    if isinstance(val, float) and val != val:  # NaN
        return "N/A"
    return val


def _submetric_badge(label, val, unit=""):
    display = _fmt_submetric(val)
    if display != "N/A" and unit:
        display = f"{display} {unit}"
    st.markdown(
        f"""
        <div style="background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;
                     padding:5px 10px; text-align:center; margin:2px 0 10px 0;">
            <span style="color:#64748B; font-size:11px;">{label}:</span>
            <span style="font-weight:700; font-size:13px; color:#0F172A;">{display}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(enriched_df, timeseries_df, pid):
    p = get_patient_row(enriched_df, pid)
    if p is None:
        st.warning("No patient selected.")
        return

    # Computed once here, reused for the "why you're here" callout (if
    # arriving via a Priority Alert jump), the Clinical Summary at the
    # top of the page, and the Medication Alerts section further down.
    alerts = get_medication_alerts(p, aki_risk=p["aki_risk"])
    critical_alerts = [a for a in alerts if a["severity"] in ("CRITICAL", "ACTION NEEDED")]

    # 7-domain scores computed once here too (score, level, reasons per
    # domain) -- feeds the Clinical Summary at the top AND the 7-Domain
    # Risk Panel further down, so it's not calculated twice.
    domain_data = {domain: fn(p) for domain, fn in DOMAIN_FUNCTIONS.items()}

    # Time-series fetched once here, reused by both the Vitals & Labs
    # trend arrows and the Vital Details charts further down.
    ts = get_patient_timeseries(timeseries_df, pid)

    st.markdown('<div class="optoc-section-title">Individual Patient Profile</div>', unsafe_allow_html=True)

    if st.session_state.pop("jump_to_med_alerts", False) and critical_alerts:
        callout_lines = "".join(f"<div style='margin-top:4px;'>&bull; {a['text']}</div>" for a in critical_alerts)
        st.markdown(
            f"""
            <div style="background-color:{RISK_COLORS['HIGH']['bg']}; border-left:4px solid {RISK_COLORS['HIGH']['border']};
                         color:{RISK_COLORS['HIGH']['text']}; border-radius:8px; padding:10px 14px; margin-bottom:12px;">
                <div style="font-weight:800;">OPENED BECAUSE: CRITICAL MEDICATION ALERT</div>
                {callout_lines}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------
    # Rounds Mode: hides secondary sections (7-Domain panel, Comorbidities,
    # Vital Details charts, Print Summary) so everything needed at the
    # bedside -- clinical summary, priority actions, compact outcome
    # tiers, medication alerts, and key lab trends -- fits on one screen
    # without scrolling past deep-dive material.
    # ------------------------------------------------------------
    with st.container(key="rounds_mode_wrap"):
        st.markdown(
            """
            <style>
            div.st-key-rounds_mode_wrap {
                transform: scale(1.3);
                transform-origin: left center;
                margin: 4px 0 14px 0;
            }
            div.st-key-rounds_mode_wrap label p {
                font-weight: 800 !important;
                color: #0F172A !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # key="rounds_mode" directly (not a value= computed from this same
        # session_state key) -- that circular pattern is the exact bug
        # that made the sidebar Patient ID selector need two clicks
        # earlier in this project; same fix applies here.
        rounds_mode = st.toggle(
            "Rounds Mode", key="rounds_mode",
            help="Hides secondary detail (domain panel, comorbidities, vital-sign trend charts, "
                 "print summary) so the essentials fit on one screen.",
        )

    if "archived_patients" not in st.session_state:
        st.session_state["archived_patients"] = set()
    is_archived = pid in st.session_state["archived_patients"]

    header_col, action_col = st.columns([5, 1], vertical_alignment="center")
    with header_col:
        archived_note = " &nbsp;|&nbsp; <b style=\"color:#F59E0B;\">ARCHIVED</b>" if is_archived else ""
        st.markdown(
            f"""
            <div style="background-color:#0F172A; color:white; padding:16px 20px;
                         border-radius:10px; margin-bottom:0; font-size:16px;">
                <span style="font-size:22px; font-weight:800; color:#F59E0B;">Patient ID: {p['patient_id']}</span>
                &nbsp;|&nbsp;
                <b>Age:</b> {p['age']} &nbsp;|&nbsp; <b>Sex:</b> {p['sex']} &nbsp;|&nbsp;
                <b>ICU Unit:</b> {p['icu_unit']} &nbsp;|&nbsp;
                <b>Days in ICU:</b> {p['days_in_icu']} &nbsp;|&nbsp;
                <b>Admitted:</b> {p['admission_date']}{archived_note}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with action_col:
        with st.container(key="discharge_archive_btn_wrap"):
            st.markdown(
                """
                <style>
                div.st-key-discharge_archive_btn_wrap button {
                    background-color: #0F172A !important;
                    color: #FFFFFF !important;
                    border: none !important;
                    border-radius: 10px !important;
                    padding: 14px 8px !important;
                    height: auto !important;
                    font-weight: 700 !important;
                }
                div.st-key-discharge_archive_btn_wrap button:hover {
                    background-color: #1E293B !important;
                    color: #F59E0B !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            if is_archived:
                if st.button("Unarchive", use_container_width=True):
                    st.session_state["archived_patients"].discard(pid)
                    st.rerun()
            else:
                if st.button("Discharge and Archive", use_container_width=True):
                    st.session_state["archived_patients"].add(pid)
                    st.rerun()
    if not is_archived:
        st.caption(
            "Discharging and archiving hides this patient from the dropdown, rankings, and "
            "heatmap by default (check \"Show archived patients\" in the sidebar to bring them "
            "back). No data is deleted, and this resets if the app restarts."
        )

    # ------------------------------------------------------------
    # Clinical Summary & Priority Actions -- the centerpiece, seen
    # immediately on opening the patient. Everything below supports this.
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">Clinical Summary &amp; Priority Actions</div>',
                unsafe_allow_html=True)
    summary_text, interventions = _clinical_summary(p, domain_data, alerts)
    st.markdown(f'<div class="optoc-card" style="font-size:15.5px;">{summary_text}</div>',
                unsafe_allow_html=True)
    if interventions:
        st.markdown("**Pharmacist Proposed Actions**")
        for item in interventions:
            st.markdown(
                f"""
                <div style="background-color:#F8FAFC; border:1px solid #E2E8F0; border-left:4px solid {RISK_COLORS['HIGH']['border']};
                             border-radius:8px; padding:10px 14px; margin-bottom:8px; font-weight:700; color:#0F172A;">
                    {item}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No specific priority actions flagged for this patient at this time.")

    st.markdown("---")

    # ------------------------------------------------------------
    # Compact outcome summary -- tier + % for all 4 outcomes plus
    # composite, replacing 5 full gauge charts. Same information, far
    # less vertical space, leaving room for the clinical content above.
    # ------------------------------------------------------------
    submetric_map = {
        "aki": ("Creatinine", p.get("creatinine_current"), "mg/dL"),
        "sepsis": ("WBC", p.get("wbc_first"), "K/uL"),
        "mortality": ("Lactate Peak", p.get("lactate_peak"), "mmol/L"),
        "readmission": ("Days in ICU", p.get("days_in_icu"), ""),
    }

    outcome_cols = st.columns(5)
    for i, outcome in enumerate(OUTCOME_KEYS):
        risk_pct = p[f"{outcome}_risk"] * 100
        tier = p[f"{outcome}_tier"]
        color = RISK_COLORS[tier]["border"]
        label, val, unit = submetric_map[outcome]
        with outcome_cols[i]:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:4px 0;">
                        <div style="font-size:13px; color:#64748B; font-weight:600;">{OUTCOME_LABELS[outcome]}</div>
                        <div style="font-size:28px; font-weight:800; color:{color}; margin:2px 0;">{risk_pct:.0f}%</div>
                        {risk_badge_html(tier)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                _submetric_badge(label, val, unit)
                if st.button("More Details", key=f"gauge_{outcome}", use_container_width=True):
                    st.session_state["active_tab"] = "Clinical Insights"
                    st.session_state["selected_outcome"] = outcome
                    st.session_state["selected_patient"] = pid
                    st.rerun()

    with outcome_cols[4]:
        with st.container(border=True):
            composite_color = RISK_COLORS[p["composite_tier"]]["border"]
            st.markdown(
                f"""
                <div style="text-align:center; padding:4px 0;">
                    <div style="font-size:13px; color:#64748B; font-weight:600;">Composite Score</div>
                    <div style="font-size:28px; font-weight:800; color:{composite_color}; margin:2px 0;">
                        {p['composite_score']:.0f}%</div>
                    {risk_badge_html(p['composite_tier'])}
                </div>
                """,
                unsafe_allow_html=True,
            )
            _submetric_badge("Nephrotoxic Drugs", p.get("nephrotoxin_count"))
            # A plain button (not st.expander) so this card matches the
            # other four's exact size/style -- an expander renders with
            # its own border/chevron and doesn't line up with them.
            if st.button("More Details", key="composite_more_details", use_container_width=True):
                st.session_state["show_composite_formula"] = not st.session_state.get(
                    "show_composite_formula", False
                )
            if st.session_state.get("show_composite_formula"):
                st.caption("Composite Score = 35% Mortality + 30% Sepsis + 25% AKI + 10% Readmission risk")

    # ------------------------------------------------------------
    # Composite tier badge
    # ------------------------------------------------------------
    st.markdown(
        f"""
        <div style="text-align:center; margin: 10px 0 20px 0;">
            {risk_badge_html(p['composite_tier'], f"{p['composite_tier']} RISK", font_size="22px", padding="8px 22px")}
            <div style="margin-top:10px; font-size:17px;">This patient is classified as
            <b>{p['composite_tier']}</b> risk based on a composite score of
            <b>{p['composite_score']:.0f}%</b>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not rounds_mode:
        st.markdown("---")

        # ------------------------------------------------------------
        # 7-Domain Risk Panel (hidden in Rounds Mode)
        # ------------------------------------------------------------
        st.markdown('<div class="optoc-section-title">7-Domain Risk Panel</div>', unsafe_allow_html=True)
        st.caption(
            "See the \"criteria scorable\" note under each card. A LOW badge on a low-completeness "
            "domain means \"not enough data to flag,\" not \"verified low.\""
        )
        # Cards are informational only (no per-card button) -- a single
        # shared button below covers all 7 domains' details at once,
        # instead of one toggle per card.
        domain_cols = st.columns(7)
        for i, domain in enumerate(DOMAIN_FUNCTIONS.keys()):
            score, level, reasons = domain_data[domain]
            avail, total = DOMAIN_DATA_COMPLETENESS[domain]
            with domain_cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{DOMAIN_LABELS[domain]}**")
                    st.markdown(risk_badge_html(level), unsafe_allow_html=True)
                    st.caption(f"Score: {score} · {avail}/{total} scorable")

        if "show_domain_details" not in st.session_state:
            st.session_state["show_domain_details"] = False

        if st.button("Hide Domain Details" if st.session_state["show_domain_details"] else "Domain Details",
                     use_container_width=True,
                     type="primary" if st.session_state["show_domain_details"] else "secondary"):
            st.session_state["show_domain_details"] = not st.session_state["show_domain_details"]
            st.rerun()

        if st.session_state["show_domain_details"]:
            with st.container(border=True):
                for domain, (score, level, reasons) in domain_data.items():
                    avail, total = DOMAIN_DATA_COMPLETENESS[domain]
                    st.markdown(
                        f"**{DOMAIN_LABELS[domain]}** &mdash; {risk_badge_html(level)} "
                        f"<span style='color:#64748B; font-size:12px;'>Score: {score} · "
                        f"{avail}/{total} scorable</span>",
                        unsafe_allow_html=True,
                    )
                    if reasons:
                        for text, pts in reasons:
                            st.write(f"- {text} (+{pts})")
                    else:
                        st.write("No contributing factors flagged.")
                    if level == "HIGH" and domain in DOMAIN_ACTIONS:
                        st.markdown(f"**Recommended Action:** {DOMAIN_ACTIONS[domain]}")
                    st.markdown("&nbsp;", unsafe_allow_html=True)

                if st.button("View Full Population Domain Heatmap in Clinical Insights →"):
                    st.session_state["domain_card_filter"] = None
                    st.session_state["active_tab"] = "Clinical Insights"
                    st.rerun()

    st.markdown("---")

    # Full SHAP/LIME explainability (population drivers + this patient's
    # factors, for all 4 outcomes) now lives only in the Clinical Insights
    # tab -- reached via the "More Details" button on each outcome above.
    # Keeping a second copy of the same charts here was pure duplication.

    # ------------------------------------------------------------
    # Vitals & labs snapshot
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">Current Vitals & Laboratory Results</div>',
                unsafe_allow_html=True)
    st.caption("Cards outlined in red are outside normal range for that value.")

    def _trend_delta(ts_col):
        """(delta, worse_if_increases) from first->last time-series
        reading, or None if fewer than 2 readings exist for this column."""
        if ts_col not in ts.columns:
            return None
        sub = ts.dropna(subset=[ts_col]).sort_values("hours_since_admission")
        if len(sub) < 2:
            return None
        return float(sub[ts_col].iloc[-1] - sub[ts_col].iloc[0])

    # worse_if_increases: True = an upward trend is clinically bad
    # (Streamlit's delta_color="inverse" flips the usual green-up/red-down
    # so a rise shows red here); MAP is the opposite -- a rise is good.
    TREND_COLS = {"MAP": ("map", False), "Lactate": ("lactate", True), "Creatinine": ("creatinine", True)}

    # Abnormal-range flags for the CURRENT value itself (not just its
    # trend) -- reuses the same cutoffs already used in the 7-domain
    # scoring rules where one exists (MAP<65, SpO2<92, Lactate>2.0,
    # Fibrinogen<150), plus standard normal ranges for HR/RR/WBC/Creatinine.
    ABNORMAL_RULES = {
        "MAP": lambda v: v < 65,
        "HR": lambda v: v < 60 or v > 100,
        "RR": lambda v: v < 12 or v > 20,
        "SpO2": lambda v: v < 92,
        "Lactate": lambda v: v > 2.0,
        "Creatinine": lambda v: v > 1.3,
        "WBC": lambda v: v < 4 or v > 11,
        "Fibrinogen": lambda v: v < 150,
    }

    # MAP is missing (NaN) in the raw data for some patients -- the model
    # and 7-domain scoring both fall back to (SBP + 2*DBP) / 3 in that
    # case (see core/features.py, core/scoring.py), but that fallback was
    # never applied here, so this card was printing the literal string
    # "nan" instead of either a real number or "N/A". Matching the same
    # fallback here, labeled as derived so it's not mistaken for a
    # directly measured reading.
    map_val = p.get("map_min")
    map_label = "MAP"
    if pd.isna(map_val):
        sbp, dbp = p.get("systolic_bp"), p.get("diastolic_bp")
        if pd.notna(sbp) and pd.notna(dbp):
            map_val = (sbp + 2 * dbp) / 3
            map_label = "MAP (derived)"
        else:
            map_val = None

    vitals = [
        (map_label, map_val, "mmHg"), ("HR", p.get("heart_rate"), "bpm"),
        ("RR", p.get("respiratory_rate"), "br/min"), ("SpO2", p.get("spo2"), "%"),
        ("Lactate", p.get("lactate_peak"), "mmol/L"), ("Creatinine", p.get("creatinine_current"), "mg/dL"),
        ("WBC", p.get("wbc_first"), "K/uL"), ("Fibrinogen", p.get("fibrinogen_first"), "mg/dL"),
    ]
    # MAP, HR, RR, SpO2 shown as whole numbers with no unit label (per
    # pharmacist feedback -- these are read at a glance, units add
    # clutter). Lactate/Creatinine/WBC/Fibrinogen keep units + 1 decimal.
    NO_UNIT_ROUND = {"MAP", "HR", "RR", "SpO2"}

    vcols = st.columns(4)
    for i, (label, val, unit) in enumerate(vitals):
        is_missing = val is None or (isinstance(val, float) and pd.isna(val))
        bare_label = label.replace(" (derived)", "")
        if is_missing:
            value_str = "N/A"
        elif bare_label in NO_UNIT_ROUND:
            value_str = f"{val:.0f}"
        else:
            value_str = f"{val:.1f} {unit}"
        rule = ABNORMAL_RULES.get(label.replace(" (derived)", ""))
        is_abnormal = bool(not is_missing and rule and rule(val))
        card_key = f"vital_card_{label.replace(' ', '_').replace('(', '').replace(')', '')}"

        with vcols[i % 4]:
            with st.container(border=True, key=card_key):
                if is_abnormal:
                    st.markdown(
                        f"""
                        <style>
                        div.st-key-{card_key} {{
                            background-color: {RISK_COLORS['HIGH']['bg']} !important;
                            border-color: {RISK_COLORS['HIGH']['border']} !important;
                            border-width: 2px !important;
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                trend_key = label.replace(" (derived)", "")
                if trend_key in TREND_COLS:
                    ts_col, worse_if_increases = TREND_COLS[trend_key]
                    delta = _trend_delta(ts_col)
                    if delta is not None:
                        st.metric(label, value_str, delta=f"{delta:+.2f}",
                                  delta_color="inverse" if worse_if_increases else "normal")
                    else:
                        st.metric(label, value_str)
                else:
                    st.metric(label, value_str)
                if is_abnormal:
                    st.markdown(
                        f'<span style="color:{RISK_COLORS["HIGH"]["text"]}; font-weight:800; '
                        f'font-size:11px;">ABNORMAL</span>',
                        unsafe_allow_html=True,
                    )

    if not rounds_mode:
        st.markdown("---")

        # ------------------------------------------------------------
        # Comorbidities (top 5) -- hidden in Rounds Mode
        # ------------------------------------------------------------
        st.markdown('<div class="optoc-section-title">Comorbidities</div>', unsafe_allow_html=True)
        comorbidity_labels = {
            "comorbidity_MI": "Myocardial Infarction", "comorbidity_CHF": "Congestive Heart Failure",
            "comorbidity_PVD": "Peripheral Vascular Disease", "comorbidity_stroke": "Stroke",
            "comorbidity_dementia": "Dementia", "comorbidity_COPD": "COPD",
            "comorbidity_diabetes_uncomplicated": "Diabetes (Uncomplicated)",
            "comorbidity_diabetes_complicated": "Diabetes (Complicated)",
            "comorbidity_CKD": "Chronic Kidney Disease", "comorbidity_paraplegia": "Paraplegia",
            "comorbidity_cancer": "Cancer", "comorbidity_metastatic_cancer": "Metastatic Cancer",
            "comorbidity_HIV": "HIV", "comorbidity_liver_mild": "Liver Disease (Mild)",
            "comorbidity_liver_severe": "Liver Disease (Severe)",
        }
        active_comorbidities = [label for col, label in comorbidity_labels.items() if p.get(col)]
        if not active_comorbidities:
            st.caption("No comorbidities recorded for this patient.")
        else:
            shown = active_comorbidities[:5]
            st.write(", ".join(shown))
            remaining = len(active_comorbidities) - len(shown)
            if remaining > 0:
                st.caption(f"+ {remaining} more not shown.")

    st.markdown("---")

    # ------------------------------------------------------------
    # Medication profile + alerts (always visible -- core to Rounds Mode)
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

    st.markdown("**Medication Alerts**")
    st.caption(
        "Organized by severity -- Critical (act now) down to Routine (awareness only). Every "
        "alert ends with a recommended pharmacist action, not just a flag."
    )

    # 4-tier severity, mapped from the existing CRITICAL/ACTION NEEDED/
    # WARNING/INFO taxonomy in core/med_alerts.py (SEVERITY_ORDER) --
    # not a new classification, just clearer labels for clinical use.
    critical_items = [a["text"] for a in alerts if a["severity"] == "CRITICAL"]
    high_items = [a["text"] for a in alerts if a["severity"] == "ACTION NEEDED"]
    moderate_items = [a["text"] for a in alerts if a["severity"] == "WARNING"]
    routine_items = [a["text"] for a in alerts if a["severity"] == "INFO"]

    # Same "no real prior-history field" caveat as the Homepage color tags --
    # this uses current AKI/Sepsis risk tier (HIGH) as the closest proxy
    # for a prior episode, since optoc_sample_patients.csv has no actual
    # prior-AKI/prior-sepsis history field.
    prior_items = []
    if p.get("aki_tier") == "HIGH":
        prior_items.append("Patient has Prior AKI during the stay")
    if p.get("sepsis_tier") == "HIGH":
        prior_items.append("Patient has Prior Sepsis during the stay")

    def _alert_box(title, items, bg, border, text_color):
        if not items:
            return
        lines = "".join(f"<div style='margin-top:4px;'>&bull; {t}</div>" for t in items)
        st.markdown(
            f"""
            <div style="background-color:{bg}; border-left:4px solid {border}; color:{text_color};
                         border-radius:8px; padding:10px 14px; margin-bottom:10px;">
                <div style="font-weight:800; letter-spacing:0.03em;">{title}</div>
                {lines}
            </div>
            """,
            unsafe_allow_html=True,
        )

    _alert_box("CRITICAL", critical_items, RISK_COLORS["HIGH"]["bg"], RISK_COLORS["HIGH"]["border"],
                RISK_COLORS["HIGH"]["text"])
    _alert_box("HIGH", high_items, "#FFEDD5", "#EA580C", "#9A3412")
    _alert_box("MODERATE", moderate_items, RISK_COLORS["MEDIUM"]["bg"], RISK_COLORS["MEDIUM"]["border"],
                RISK_COLORS["MEDIUM"]["text"])
    _alert_box("PRIOR AKI/SEPSIS", prior_items, "#F3E9C9", "#D4AF37", "#7A5C00")
    _alert_box("ROUTINE", routine_items, "#E7F1FF", "#2563EB", "#1E3A8A")

    if not (critical_items or high_items or moderate_items or prior_items or routine_items):
        st.success("No medication alerts triggered for this patient's current profile.")

    if not rounds_mode:
        st.markdown("---")

        # ------------------------------------------------------------
        # Vital Details -- hidden in Rounds Mode. Replaced the 6 line
        # charts with a single High/Medium/Low reading-count bar per
        # vital over the last 24h; the line charts weren't actionable
        # for a pharmacist at a glance, this is.
        # ------------------------------------------------------------
        st.markdown('<div class="optoc-section-title">Vital Details</div>', unsafe_allow_html=True)
        st.caption("Count of readings in each risk tier over the last 24 hours since admission.")
        if len(ts) == 0:
            st.info("No time-series data available for this patient.")
        else:
            def _tier_map(v):
                if v < 55:
                    return "HIGH"
                if v < 65:
                    return "MEDIUM"
                return "LOW"

            def _tier_hr(v):
                if v < 50 or v > 120:
                    return "HIGH"
                if v < 60 or v > 100:
                    return "MEDIUM"
                return "LOW"

            def _tier_rr(v):
                if v < 8 or v > 30:
                    return "HIGH"
                if v < 12 or v > 20:
                    return "MEDIUM"
                return "LOW"

            def _tier_spo2(v):
                if v < 88:
                    return "HIGH"
                if v < 92:
                    return "MEDIUM"
                return "LOW"

            def _tier_lactate(v):
                if v >= 4:
                    return "HIGH"
                if v >= 2:
                    return "MEDIUM"
                return "LOW"

            def _tier_creatinine(v):
                if v >= 2.0:
                    return "HIGH"
                if v >= 1.3:
                    return "MEDIUM"
                return "LOW"

            VITAL_TIER_FNS = {
                "map": (_tier_map, "MAP"), "hr": (_tier_hr, "HR"), "rr": (_tier_rr, "RR"),
                "spo2": (_tier_spo2, "SpO2"), "lactate": (_tier_lactate, "Lactate"),
                "creatinine": (_tier_creatinine, "Creatinine"),
            }

            window = ts[ts["hours_since_admission"] <= 24]
            counts = {}
            for col, (fn, label) in VITAL_TIER_FNS.items():
                if col not in window.columns:
                    continue
                readings = window[col].dropna()
                if readings.empty:
                    continue
                tier_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                for v in readings:
                    tier_counts[fn(v)] += 1
                counts[label] = tier_counts

            if not counts:
                st.info("No vital readings within 24 hours of admission on file for this patient.")
            else:
                vital_labels = list(counts.keys())
                # Low -> Medium -> High stacking order means a heavily
                # "High" vital visibly towers in red at the top of its
                # bar -- the thing a pharmacist should spot first.
                fig = go.Figure()
                for tier in ["LOW", "MEDIUM", "HIGH"]:
                    values = [counts[v][tier] for v in vital_labels]
                    fig.add_trace(go.Bar(
                        name=tier.title(),
                        x=vital_labels,
                        y=values,
                        marker_color=RISK_COLORS[tier]["border"],
                        marker_line=dict(color="#FFFFFF", width=1.5),
                        text=[str(v) if v > 0 else "" for v in values],
                        textposition="inside",
                        textfont=dict(color="#FFFFFF", size=13, family="Arial, sans-serif"),
                        hovertemplate="%{x}: %{y} " + tier.title() + " reading(s)<extra></extra>",
                    ))
                fig.update_layout(
                    barmode="stack",
                    height=340,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(family="Arial, sans-serif", size=13, color="#0F172A"),
                    bargap=0.35,
                    xaxis=dict(title=None, tickfont=dict(size=14, color="#0F172A")),
                    yaxis=dict(
                        title="Number of Readings",
                        gridcolor="#EEF1F6",
                        zeroline=False,
                        dtick=1,
                    ),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        font=dict(size=13),
                    ),
                )
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # Pharmacist note -- always visible, it's a rounds tool
    # ------------------------------------------------------------
    note = st.text_area(
        "Pharmacist note for this patient",
        key=f"pharm_note_{pid}",
        help="Persists for this patient for the rest of this browser session (survives switching "
             "patients/tabs) -- not saved to a database, so it's gone if the app restarts or you "
             "close the tab.",
    )

    if rounds_mode:
        return

    st.markdown("---")

    # ------------------------------------------------------------
    # Print / export summary. Hidden in Rounds Mode (an end-of-encounter
    # action, not needed mid-round).
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">Print Summary</div>', unsafe_allow_html=True)
    st.caption("Prints just the box below (composite score, top factors, active alerts) -- "
               "not the whole page.")

    top_factors = sorted(
        (r for score, level, reasons in domain_data.values() for r in reasons),
        key=lambda r: r[1], reverse=True,
    )[:3]
    top_factors_html = "".join(f"<li>{text} (+{pts})</li>" for text, pts in top_factors) or "<li>None flagged.</li>"

    all_active_alerts = critical_alerts + [a for a in alerts if a["severity"] == "WARNING"]
    alerts_html = "".join(f"<li>{a['text']}</li>" for a in all_active_alerts) or "<li>None active.</li>"

    note_html = note.replace("\n", "<br>") if note else "<em>No note entered.</em>"

    # Same font stack Streamlit's own UI uses, so the on-screen card and
    # the printed PDF both match the rest of the dashboard.
    app_font = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Source Sans Pro", sans-serif'

    summary_html = f"""
        <div style="border:1px solid #E2E8F0; border-radius:10px; padding:18px 22px;
                     background-color:#FFFFFF; font-family:{app_font}; color:#0F172A;">
            <h3 style="margin-top:0;">OPTOC Patient Summary &mdash; {p['patient_id']}</h3>
            <p><b>Age:</b> {p['age']} &nbsp;|&nbsp; <b>Sex:</b> {p['sex']} &nbsp;|&nbsp;
               <b>ICU Unit:</b> {p['icu_unit']} &nbsp;|&nbsp; <b>Days in ICU:</b> {p['days_in_icu']}</p>
            <p><b>Composite Score:</b> {p['composite_score']:.0f}% ({p['composite_tier']} risk)</p>
            <p><b>Top Contributing Factors:</b></p>
            <ul>{top_factors_html}</ul>
            <p><b>Active Alerts:</b></p>
            <ul>{alerts_html}</ul>
            <p><b>Pharmacist Note:</b><br>{note_html}</p>
        </div>
    """

    # On-screen preview -- a normal Streamlit-rendered card.
    st.markdown(summary_html, unsafe_allow_html=True)

    # The button lives in its own self-contained HTML document (via
    # components.html, which -- unlike st.markdown -- isn't sanitized,
    # so onclick actually runs). It prints ONLY this iframe's own
    # document (plain window.print(), not window.parent.print()), which
    # contains nothing but this summary -- no sidebar, nav, or the rest
    # of the app. That sidesteps the earlier approach (hiding everything
    # else on the main page via CSS visibility:hidden), which left blank
    # reserved space for every hidden element and produced a mostly-empty
    # multi-page PDF instead of one clean page.
    components.html(
        f"""
        <style>
            body {{ margin: 0; font-family: {app_font}; }}
        </style>
        <div id="print-content" style="display:none;">
            {summary_html}
        </div>
        <button onclick="var c=document.getElementById('print-content'); c.style.display='block'; window.print(); c.style.display='none';"
                style="background-color:#0F172A; color:white; border:none; border-radius:8px;
                       padding:10px 18px; font-weight:700; font-size:14px; cursor:pointer;
                       font-family:{app_font};">
            Print This Summary
        </button>
        """,
        height=50,
    )
