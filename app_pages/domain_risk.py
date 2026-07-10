"""
app_pages/domain_risk.py -- Tab 4: Domain Specific Risk

Per spec: population-level heatmap of the 7 pharmacist risk domains,
domain summary cards (clickable to filter), unit filter + sort-by-domain.
"""

import streamlit as st

from core.theme import page_header, risk_cell_style, RISK_COLORS
from core.scoring import DOMAIN_FUNCTIONS, DOMAIN_LABELS


def render(enriched_df, pid):
    page_header("Domain-Specific Risk", "Population-level view of the 7 organ-system risk domains")

    domain_names = list(DOMAIN_FUNCTIONS.keys())

    # ------------------------------------------------------------
    # Domain summary cards (top) — count of HIGH patients per domain
    # ------------------------------------------------------------
    if "domain_card_filter" not in st.session_state:
        st.session_state["domain_card_filter"] = None

    card_cols = st.columns(7)
    for i, domain in enumerate(domain_names):
        level_col = f"domain_{domain}_level"
        high_count = (enriched_df[level_col] == "HIGH").sum()
        with card_cols[i]:
            st.markdown(f"**{DOMAIN_LABELS[domain]}**")
            st.metric("HIGH", high_count, label_visibility="collapsed")
            if st.button("Filter", key=f"domain_card_{domain}"):
                st.session_state["domain_card_filter"] = domain
                st.rerun()

    if st.session_state["domain_card_filter"]:
        st.info(
            f"Showing only patients at HIGH risk in "
            f"{DOMAIN_LABELS[st.session_state['domain_card_filter']]}. "
            f"[Clear filter]"
        )
        if st.button("Clear domain filter"):
            st.session_state["domain_card_filter"] = None
            st.rerun()

    st.markdown("---")

    # ------------------------------------------------------------
    # Filter / sort controls
    # ------------------------------------------------------------
    f1, f2 = st.columns(2)
    units = ["All"] + sorted(enriched_df["icu_unit"].dropna().unique().tolist())
    unit_filter = f1.selectbox("ICU Unit", units)
    sort_domain = f2.selectbox("Sort by domain", domain_names, format_func=lambda d: DOMAIN_LABELS[d])

    filtered = enriched_df.copy()
    if unit_filter != "All":
        filtered = filtered[filtered["icu_unit"] == unit_filter]
    if st.session_state["domain_card_filter"]:
        d = st.session_state["domain_card_filter"]
        filtered = filtered[filtered[f"domain_{d}_level"] == "HIGH"]

    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    filtered = filtered.sort_values(
        by=f"domain_{sort_domain}_level", key=lambda s: s.map(order)
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------
    st.markdown("### Domain Heatmap")

    display_cols = ["patient_id"] + [f"domain_{d}_level" for d in domain_names]
    display = filtered[display_cols].rename(
        columns={f"domain_{d}_level": DOMAIN_LABELS[d] for d in domain_names}
    ).rename(columns={"patient_id": "Patient ID"})

    def _style(val):
        return risk_cell_style(val) if val in RISK_COLORS else ""

    styled = display.style.map(_style, subset=[DOMAIN_LABELS[d] for d in domain_names])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.caption(
        f"Highlighted patient: {pid}. Click a Patient ID via the sidebar selector to jump to "
        "their Individual Patient profile."
    )
