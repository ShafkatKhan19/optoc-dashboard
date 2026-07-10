"""
app_pages/home.py -- Tab 1: Homepage

Per spec: one-day snapshot of ICU patients nearing discharge, priority
alerts, risk KPIs, sortable/rankable patient table. Landing page.
"""

import streamlit as st
import plotly.graph_objects as go

from core.theme import page_header, risk_badge_html, risk_cell_style, color_dot, RISK_COLORS

INTRO_TEXT = (
    "Optoc AI Pharmacist Dashboard flags ICU patients at elevated risk of "
    "readmission and mortality as they approach discharge. It is designed "
    "for pharmacists, physicians, and other members of the care team "
    "involved in transitions of care. For each patient, it shows risk "
    "scores across five clinical outcomes, the clinical factors driving "
    "those scores, medication-related alerts, and 7 organ-system risk "
    "domains, helping the care team decide who needs review before discharge."
)


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
            hoverinfo="skip",
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
            <div style="font-weight:700; color:#0F172A;">{pct[t]:.0f}%</div>
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
    page_header(
        "OPTOC",
        "Guiding Safe Transitions Out of the ICU",
    )

    # ------------------------------------------------------------
    # KPI cards (5)
    # ------------------------------------------------------------
    n = len(enriched_df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients Today", n)
    c2.metric("HIGH AKI Risk", f"{round((enriched_df['aki_risk'] >= 0.70).mean() * 100)}%")
    c3.metric("HIGH Sepsis Risk", f"{round((enriched_df['sepsis_risk'] >= 0.70).mean() * 100)}%")
    c4.metric("HIGH Readmission Risk", f"{round((enriched_df['readmission_risk'] >= 0.70).mean() * 100)}%")
    c5.metric("HIGH Mortality Risk", f"{round((enriched_df['mortality_risk'] >= 0.70).mean() * 100)}%")

    st.markdown("---")

    # ------------------------------------------------------------
    # Two radial-ring risk-distribution charts
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

    st.markdown("---")

    # ------------------------------------------------------------
    # Top 5 Priority Alerts (combined Mortality + Readmission)
    # ------------------------------------------------------------
    st.markdown(
        f'<div class="optoc-section-title">{color_dot(RISK_COLORS["HIGH"]["border"], size=12)}Priority Alerts</div>',
        unsafe_allow_html=True,
    )
    st.caption("The following patients require immediate pharmacist review.")

    ranked = enriched_df.copy()
    ranked["combined_risk"] = ranked["mortality_risk"] + ranked["readmission_risk"]
    top5 = ranked.sort_values("combined_risk", ascending=False).head(5)

    for _, row in top5.iterrows():
        tier = row["composite_tier"]
        c = RISK_COLORS.get(tier, RISK_COLORS["LOW"])
        avatar_text = row["patient_id"].split("-")[-1][-2:]

        with st.container(border=True):
            bar, avatar, info, demo, unit, risk, action = st.columns(
                [0.12, 0.5, 2.0, 1.1, 1.3, 1.6, 0.9]
            )
            bar.markdown(
                f'<div style="width:5px; height:44px; border-radius:3px; background-color:{c["border"]};"></div>',
                unsafe_allow_html=True,
            )
            avatar.markdown(
                f'<div style="width:36px; height:36px; border-radius:50%; background-color:{c["bg"]}; '
                f'color:{c["text"]}; display:flex; align-items:center; justify-content:center; '
                f'font-weight:700; font-size:13px;">{avatar_text}</div>',
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
                f'<div style="font-size:11px; color:#64748B;">Mortality &middot; Readmission</div>'
                f'<div style="font-weight:600; color:#0F172A;">'
                f'{row["mortality_risk"]*100:.0f}% &middot; {row["readmission_risk"]*100:.0f}%</div>',
                unsafe_allow_html=True,
            )
            if action.button("Open", key=f"alert_{row['patient_id']}", use_container_width=True):
                st.session_state["selected_patient"] = row["patient_id"]
                st.session_state["active_tab"] = "Individual Patient"
                st.rerun()

    st.caption("Click Open on a Priority Alert row to open that patient's full profile in the Individual Patient tab.")

    st.markdown("---")

    # ------------------------------------------------------------
    # Intro paragraph
    # ------------------------------------------------------------
    st.markdown(f'<div class="optoc-card">{INTRO_TEXT}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------
    # Filter toolbar
    # ------------------------------------------------------------
    st.markdown('<div class="optoc-section-title">Patient Ranking (by Readmission Risk)</div>',
                unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([2, 1.5, 1.5, 2])
    units = ["All"] + sorted(enriched_df["icu_unit"].dropna().unique().tolist())
    unit_filter = f1.multiselect("ICU Unit", units, default=["All"])
    tier_filter = f2.selectbox("Risk Tier", ["All", "HIGH", "MEDIUM", "LOW"])
    sort_by = f3.selectbox(
        "Sort by",
        ["Rank - Readmission Risk desc", "Composite Score desc", "AKI risk desc",
         "Sepsis risk desc", "Patient ID asc"],
    )
    search = f4.text_input("Search Patient ID")

    filtered = enriched_df.copy()
    if unit_filter and "All" not in unit_filter:
        filtered = filtered[filtered["icu_unit"].isin(unit_filter)]
    if tier_filter != "All":
        filtered = filtered[filtered["readmission_tier"] == tier_filter]
    if search:
        filtered = filtered[filtered["patient_id"].str.contains(search, case=False, na=False)]

    sort_map = {
        "Rank - Readmission Risk desc": ("readmission_risk", False),
        "Composite Score desc": ("composite_score", False),
        "AKI risk desc": ("aki_risk", False),
        "Sepsis risk desc": ("sepsis_risk", False),
        "Patient ID asc": ("patient_id", True),
    }
    sort_col, ascending = sort_map[sort_by]
    filtered = filtered.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    filtered.insert(0, "Rank", range(1, len(filtered) + 1))

    display = filtered[[
        "Rank", "patient_id", "age", "sex", "icu_unit",
        "aki_risk", "sepsis_risk", "mortality_risk", "readmission_risk",
        "composite_score", "composite_tier", "days_in_icu", "discharge_due_date",
    ]].rename(columns={
        "patient_id": "Patient ID", "age": "Age", "sex": "Sex", "icu_unit": "ICU Unit",
        "aki_risk": "AKI Risk", "sepsis_risk": "Sepsis Risk", "mortality_risk": "Mortality Risk",
        "readmission_risk": "Readmission Risk", "composite_score": "Composite Score",
        "composite_tier": "Risk Tier", "days_in_icu": "Days in ICU", "discharge_due_date": "Discharge Due",
    })
    for col in ["AKI Risk", "Sepsis Risk", "Mortality Risk", "Readmission Risk"]:
        display[col] = (display[col] * 100).round(1).astype(str) + "%"

    def _tier_style(val):
        return risk_cell_style(val) if val in RISK_COLORS else ""

    styled = display.style.map(_tier_style, subset=["Risk Tier"])
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

    st.download_button(
        "Export filtered table as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="optoc_patient_ranking.csv",
        mime="text/csv",
    )

    st.caption("Click any row in the table above to open that patient's full profile in the Individual Patient tab.")
