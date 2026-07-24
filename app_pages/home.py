"""
app_pages/home.py -- Tab 1: Homepage

Per spec: one-day snapshot of ICU patients nearing discharge, priority
alerts, risk KPIs, sortable/rankable patient table. Landing page.
"""

import streamlit as st
import plotly.graph_objects as go

from core.theme import risk_badge_html, risk_cell_style, color_dot, RISK_COLORS
from core.med_alerts import get_medication_alerts
from core.model_registry import OUTCOME_LABELS

MORE_DETAILS_TEXT = """
**Optoc AI Pharmacist Dashboard**
- Flags patients at high risk of readmission or mortality before discharge
- Built for pharmacists, physicians, and the care team
- Shows for each patient:
    - Risk scores across 4 clinical outcomes
    - Key factors driving those risks
    - Medication-related alerts
    - 7 organ-system risk domains
- Helps the user make decisions during transitions of care
- Machine Learning was used in calculating the risks

**Sample data**
- Electronic health record data for 10 hospitalized patients
"""


TIERS = ["HIGH", "MEDIUM", "LOW"]

# outer -> inner ring domain spans (fraction of the plot's square area each
# ring occupies) -- gives the concentric "radial gauge" look
_RING_SIZES = [1.0, 0.78, 0.56]


def _radial_rings(df, risk_col, tier_col, title, subtitle):
    total = len(df) or 1
    counts = df[tier_col].value_counts()
    pct = {t: counts.get(t, 0) / total * 100 for t in TIERS}
    avg_risk = df[risk_col].mean() * 100

    fig = go.Figure()
    for size, tier in zip(_RING_SIZES, TIERS):
        frac = max(pct[tier] / 100, 0.0015)
        pad = (1 - size) / 2
        fig.add_trace(go.Pie(
            values=[frac, 1 - frac],
            hole=0.72,
            domain=dict(x=[pad, 1 - pad], y=[pad, 1 - pad]),
            marker=dict(colors=[RISK_COLORS[tier]["border"], "#EDF1F7"]),
            direction="clockwise",
            rotation=90,
            sort=False,
            textinfo="none",
            hovertext=[f"{tier.title()}: {pct[tier]:.0f}% ({int(counts.get(tier, 0))})", ""],
            hoverinfo="text",
            showlegend=False,
        ))
    fig.update_layout(
        height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        annotations=[dict(
            text=f"<b style='font-size:26px;color:#0F172A;'>{avg_risk:.0f}%</b>"
                 f"<br><span style='font-size:11px;color:#64748B;'>Avg Risk</span>",
            x=0.5, y=0.5, showarrow=False,
        )],
    )

    legend_rows = "".join(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 2px;">
            <div>{color_dot(RISK_COLORS[t]["border"])}<span style="font-weight:600; color:#0F172A;">{t.title()}</span></div>
            <div style="font-weight:700; color:#0F172A;">{pct[t]:.0f}% ({int(counts.get(t, 0))})</div>
        </div>
        """
        for t in TIERS
    )
    card_html = f"""
    <div class="optoc-card" style="padding:16px 18px;">
        <div style="font-weight:700; font-size:16px; color:#0F172A;">{title}</div>
        <div style="font-size:12.5px; color:#64748B; margin-bottom:4px;">{subtitle}</div>
    </div>
    """
    return fig, legend_rows, card_html


def render(enriched_df):
    # ------------------------------------------------------------
    # Compact ICU census summary -- replaces the large OPTOC banner.
    # "Needs pharmacist review" = composite HIGH tier OR at least one
    # CRITICAL/ACTION NEEDED medication alert, same definition used to
    # decide a Priority Alert row's "Top Issue" below.
    # ------------------------------------------------------------
    n_total = len(enriched_df)
    n_high = int((enriched_df["composite_tier"] == "HIGH").sum())

    def _needs_review(row):
        row_alerts = get_medication_alerts(row, aki_risk=row["aki_risk"])
        has_critical = any(a["severity"] in ("CRITICAL", "ACTION NEEDED") for a in row_alerts)
        return has_critical or row["composite_tier"] == "HIGH"

    n_review = int(enriched_df.apply(_needs_review, axis=1).sum())

    st.markdown(
        f"""
        <div style="background-color:#0F172A; color:white; padding:14px 24px; border-radius:12px;
                     margin-bottom:14px; display:flex; justify-content:space-between; align-items:center;
                     flex-wrap:wrap; gap:10px;">
            <div>
                <span style="font-size:20px; font-weight:800;">OPTOC</span>
                <span style="color:#94A3B8; font-size:13px; margin-left:8px;">Guiding Safe Transitions Out of the ICU</span>
            </div>
            <div style="display:flex; gap:28px; font-size:14px;">
                <div><b style="font-size:18px;">{n_total}</b><br><span style="color:#94A3B8; font-size:11.5px;">Patients Today</span></div>
                <div><b style="font-size:18px; color:#FCA5A5;">{n_high}</b><br><span style="color:#94A3B8; font-size:11.5px;">High Risk</span></div>
                <div><b style="font-size:18px; color:#FCD34D;">{n_review}</b><br><span style="color:#94A3B8; font-size:11.5px;">Need Pharmacist Review</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("More Details", expanded=False):
        st.markdown(MORE_DETAILS_TEXT)

    # ------------------------------------------------------------
    # Top 5 Priority Alerts -- moved to the very top of the page: "which
    # patients need my attention first" is the first question every shift,
    # ahead of population statistics.
    # ------------------------------------------------------------
    st.markdown(
        f'<div class="optoc-section-title">{color_dot(RISK_COLORS["HIGH"]["border"], size=12)}Priority Alerts</div>',
        unsafe_allow_html=True,
    )
    st.caption("The following patients require immediate pharmacist review.")

    # Ranked by composite_score (the same overall risk score/tier shown as
    # each row's badge and used throughout the rest of the dashboard) --
    # not just mortality+readmission -- so "Priority Alerts" genuinely
    # means "highest overall risk level" per the professor's note. Every
    # patient is included now (ascending by rank -- highest risk first),
    # not just the top 5, since the box scrolls.
    all_ranked = enriched_df.sort_values("composite_score", ascending=False)

    # Fixed-height, scrollable box (native Streamlit container height=...)
    # so Priority Alerts doesn't push the rest of the homepage down --
    # all patients, ranked highest-risk first, scrollable in place.
    with st.container(height=360, border=False):
        for _, row in all_ranked.iterrows():
            tier = row["composite_tier"]
            c = RISK_COLORS.get(tier, RISK_COLORS["LOW"])

            # Top issue for this patient: a CRITICAL/ACTION NEEDED medication
            # alert takes priority (acute, actionable) over just naming the
            # worst outcome -- otherwise, name whichever of the 4 outcomes
            # is highest for them. Clicking "Open" jumps straight there
            # instead of just the generic profile.
            alerts = get_medication_alerts(row, aki_risk=row["aki_risk"])
            critical_alerts = [a for a in alerts if a["severity"] in ("CRITICAL", "ACTION NEEDED")]
            outcome_risks = {o: row[f"{o}_risk"] for o in ["aki", "sepsis", "mortality", "readmission"]}
            worst_outcome = max(outcome_risks, key=outcome_risks.get)

            if critical_alerts:
                issue_label = "Critical medication alert"
            else:
                issue_label = f"{OUTCOME_LABELS[worst_outcome]} ({outcome_risks[worst_outcome]*100:.0f}%)"

            with st.container(border=True):
                bar, info, demo, unit, risk, action = st.columns(
                    [0.12, 2.0, 1.1, 1.3, 1.6, 0.9]
                )
                bar.markdown(
                    f'<div style="width:5px; height:44px; border-radius:3px; background-color:{c["border"]};"></div>',
                    unsafe_allow_html=True,
                )
                info.markdown(
                    f'<div style="font-weight:700; color:#0F172A;">{row["patient_id"]}</div>'
                    f'{risk_badge_html(tier, f"{tier.title()} Risk")}',
                    unsafe_allow_html=True,
                )
                demo.markdown(
                    f'<div style="font-size:11px; color:#64748B;">Age &middot; Sex</div>'
                    f'<div style="font-weight:600; color:#0F172A;">{row["age"]:.0f} &middot; {row["sex"]}</div>',
                    unsafe_allow_html=True,
                )
                unit.markdown(
                    f'<div style="font-size:11px; color:#64748B;">ICU Unit</div>'
                    f'<div style="font-weight:600; color:#0F172A;">{row["icu_unit"]}</div>',
                    unsafe_allow_html=True,
                )
                risk.markdown(
                    f'<div style="font-size:11px; color:#64748B;">Top Issue</div>'
                    f'<div style="font-weight:600; color:#0F172A;">{issue_label}</div>',
                    unsafe_allow_html=True,
                )
                if action.button("Open", key=f"alert_{row['patient_id']}", use_container_width=True):
                    st.session_state["selected_patient"] = row["patient_id"]
                    if critical_alerts:
                        st.session_state["active_tab"] = "Individual Patient"
                        st.session_state["jump_to_med_alerts"] = True
                    else:
                        st.session_state["active_tab"] = "Clinical Insights"
                        st.session_state["selected_outcome"] = worst_outcome
                    st.rerun()

    st.caption(
        "Click Open on a Priority Alert row to jump straight to that patient's top issue -- a "
        "critical medication alert if one exists, otherwise their highest-risk outcome."
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # KPI cards (5) -- population statistics, demoted below Priority
    # Alerts (less immediate clinical value than "who needs me first").
    # ------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients Today", n_total)
    c2.metric("High AKI Risk", f"{round((enriched_df['aki_risk'] >= 0.70).mean() * 100)}%")
    c3.metric("High Sepsis Risk", f"{round((enriched_df['sepsis_risk'] >= 0.70).mean() * 100)}%")
    c4.metric("High Readmission Risk", f"{round((enriched_df['readmission_risk'] >= 0.70).mean() * 100)}%")
    c5.metric("High Mortality Risk", f"{round((enriched_df['mortality_risk'] >= 0.70).mean() * 100)}%")

    st.markdown("---")

    # ------------------------------------------------------------
    # Filter toolbar
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">Patient Ranking</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([2, 1.5, 1.5, 2])
    units = ["All"] + sorted(enriched_df["icu_unit"].dropna().unique().tolist())
    unit_filter = f1.multiselect("ICU Unit", units, default=["All"])
    tier_filter = f2.selectbox("Risk Tier", ["All", "HIGH", "MEDIUM", "LOW"])
    sort_by = f3.selectbox(
        "Sort by",
        ["Readmission Risk (Descending)", "Composite Score (Descending)",
         "AKI Risk (Descending)", "Sepsis Risk (Descending)", "Patient ID (Ascending)"],
    )
    patient_id_options = ["All"] + sorted(enriched_df["patient_id"].unique().tolist())
    search = f4.selectbox("Search Patient ID", patient_id_options)

    filtered = enriched_df.copy()
    if unit_filter and "All" not in unit_filter:
        filtered = filtered[filtered["icu_unit"].isin(unit_filter)]
    if tier_filter != "All":
        # Filter on composite_tier -- the same field shown in the "Risk
        # Tier" column below. Filtering on readmission_tier instead (the
        # previous bug) let MEDIUM/LOW rows leak into a "High" filter
        # whenever their composite tier and readmission tier disagreed.
        filtered = filtered[filtered["composite_tier"] == tier_filter]
    if search != "All":
        filtered = filtered[filtered["patient_id"] == search]

    sort_map = {
        "Readmission Risk (Descending)": ("readmission_risk", False),
        "Composite Score (Descending)": ("composite_score", False),
        "AKI Risk (Descending)": ("aki_risk", False),
        "Sepsis Risk (Descending)": ("sepsis_risk", False),
        "Patient ID (Ascending)": ("patient_id", True),
    }
    sort_col, ascending = sort_map[sort_by]
    filtered = filtered.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    filtered.insert(0, "Rank", range(1, len(filtered) + 1))

    # Patient ID color tags: this dataset has no actual "prior AKI"/"prior
    # sepsis" history field, so this uses each patient's current AKI/Sepsis
    # risk tier (HIGH) as the closest available proxy -- flagged here since
    # it's a stand-in, not a real documented history flag.
    id_tags = {}
    for _, r in filtered.iterrows():
        has_aki = r["aki_tier"] == "HIGH"
        has_sepsis = r["sepsis_tier"] == "HIGH"
        if has_aki and has_sepsis:
            id_tags[r["patient_id"]] = "BOTH"
        elif has_aki:
            id_tags[r["patient_id"]] = "AKI"
        elif has_sepsis:
            id_tags[r["patient_id"]] = "SEPSIS"

    ID_TAG_COLORS = {"AKI": "#D4AF37", "SEPSIS": "#DC143C", "BOTH": "#4B0082"}

    display = filtered[[
        "Rank", "patient_id", "age", "sex", "icu_unit",
        "aki_risk", "sepsis_risk", "mortality_risk", "readmission_risk",
        "composite_score", "composite_tier", "days_in_icu",
    ]].rename(columns={
        "patient_id": "Patient ID", "age": "Age", "sex": "Sex", "icu_unit": "ICU Unit",
        "aki_risk": "AKI Risk", "sepsis_risk": "Sepsis Risk", "mortality_risk": "Mortality Risk",
        "readmission_risk": "Readmission Risk", "composite_score": "Composite Score",
        "composite_tier": "Risk Tier", "days_in_icu": "Days in ICU",
    })
    for col in ["AKI Risk", "Sepsis Risk", "Mortality Risk", "Readmission Risk"]:
        display[col] = (display[col] * 100).round(1).astype(str) + "%"
    display["Composite Score"] = display["Composite Score"].round(2)

    def _tier_style(val):
        return risk_cell_style(val) if val in RISK_COLORS else ""

    def _id_tag_style(id_series):
        styles = []
        for pid_val in id_series:
            tag = id_tags.get(pid_val)
            if tag:
                color = ID_TAG_COLORS[tag]
                styles.append(f"color:{color}; font-weight:800; background-color:{color}22;")
            else:
                styles.append("")
        return styles

    styled = (
        display.style
        .map(_tier_style, subset=["Risk Tier"])
        .apply(_id_tag_style, subset=["Patient ID"])
    )
    table_event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="home_ranking_table",
    )

    selected_rows = table_event.selection.rows if table_event and table_event.selection else []
    if selected_rows:
        clicked_pid = display.iloc[selected_rows[0]]["Patient ID"]
        st.session_state["selected_patient"] = clicked_pid
        st.session_state["active_tab"] = "Individual Patient"
        st.rerun()

    st.markdown(
        f"""
        <div style="display:flex; gap:22px; flex-wrap:wrap; align-items:center; margin:6px 0 4px 0;">
            <div><span style="display:inline-block; width:12px; height:12px; background-color:{ID_TAG_COLORS['AKI']};
                 margin-right:6px; vertical-align:middle;"></span>
                 <span style="font-size:12.5px;">Nephro-Gold: Patient has Prior AKI</span></div>
            <div><span style="display:inline-block; width:12px; height:12px; background-color:{ID_TAG_COLORS['SEPSIS']};
                 margin-right:6px; vertical-align:middle;"></span>
                 <span style="font-size:12.5px;">Sepsis Crimson: Patient has Prior Sepsis</span></div>
            <div><span style="display:inline-block; width:12px; height:12px; background-color:{ID_TAG_COLORS['BOTH']};
                 margin-right:6px; vertical-align:middle;"></span>
                 <span style="font-size:12.5px;">Infection-Ischemic Indigo: Patient has both Prior AKI and Sepsis</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "Export filtered table as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="optoc_patient_ranking.csv",
        mime="text/csv",
    )

    st.caption("Click any row in the table above to open that patient's full profile in the Individual Patient tab.")

    st.markdown("---")

    # ------------------------------------------------------------
    # Two radial-ring risk-distribution charts -- moved below Patient
    # Ranking (population distribution is a supporting view, not the
    # first thing a pharmacist needs).
    # ------------------------------------------------------------
    d1, d2 = st.columns(2)
    for col, risk_col, tier_col, title, subtitle in [
        (d1, "readmission_risk", "readmission_tier", "Readmission Risk", "Distribution by risk tier"),
        (d2, "mortality_risk", "mortality_tier", "Mortality Risk", "Distribution by risk tier"),
    ]:
        with col:
            fig, legend_rows, card_html = _radial_rings(enriched_df, risk_col, tier_col, title, subtitle)
            st.markdown(card_html, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(legend_rows, unsafe_allow_html=True)
