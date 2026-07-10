"""
app.py -- Optoc AI Pharmacist Dashboard, main entry point.

Uses a session-state-driven top nav (buttons styled to look like tabs)
instead of st.tabs(), because st.tabs() cannot be switched
programmatically in Streamlit -- and the spec explicitly requires
click-through navigation (patient IDs -> Tab 2, gauges -> Tab 3, domain
cards -> Tab 4), which native st.tabs can't support.

Run with:
    streamlit run app.py
"""

import streamlit as st

from core.theme import inject_global_css
from core.data_io import build_enriched_cohort
from app_pages import home, individual_patient, risk_profile, domain_risk, try_your_data

st.set_page_config(
    page_title="OPTOC",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()

TABS = ["Homepage", "Individual Patient", "Risk Profile", "Domain Specific Risk", "Try Your Data!"]

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Homepage"

# ------------------------------------------------------------
# Load + enrich data (cached)
# ------------------------------------------------------------
with st.spinner("Loading patient data and running risk models..."):
    enriched_df, feature_df, timeseries_df = build_enriched_cohort()

if "selected_patient" not in st.session_state:
    st.session_state["selected_patient"] = enriched_df["patient_id"].iloc[0]

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("OPTOC")

st.sidebar.markdown("### Filters")
unit_options = ["All"] + sorted(enriched_df["icu_unit"].dropna().unique().tolist())
sidebar_unit = st.sidebar.selectbox("ICU Unit", unit_options)

sidebar_date = st.sidebar.date_input("Census Date")

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Patient Selector")
patient_ids = enriched_df["patient_id"].tolist()
idx = patient_ids.index(st.session_state["selected_patient"]) if st.session_state["selected_patient"] in patient_ids else 0
picked = st.sidebar.selectbox("Patient ID", patient_ids, index=idx)
st.session_state["selected_patient"] = picked

short_id = picked.split("-")[-1] if "-" in picked else picked
st.sidebar.markdown(
    f"""
    <div style="text-align:center; background-color:#1E293B; border:1px solid #334155;
                border-radius:10px; padding:12px 0; margin-top:10px;">
        <div style="font-size:11px; letter-spacing:0.05em; color:#94A3B8; text-transform:uppercase;">
            Selected Patient
        </div>
        <div style="font-size:20px; font-weight:700; color:#F8FAFC; margin-top:2px;">
            Patient ID: {short_id}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Apply sidebar unit filter to the cohort used everywhere
display_df = enriched_df.copy()
if sidebar_unit != "All":
    display_df = display_df[display_df["icu_unit"] == sidebar_unit]
    if len(display_df) and st.session_state["selected_patient"] not in display_df["patient_id"].values:
        st.session_state["selected_patient"] = display_df["patient_id"].iloc[0]

# ------------------------------------------------------------
# Top nav (session-state driven, styled like tabs)
# ------------------------------------------------------------
nav_cols = st.columns(len(TABS))
for i, tab_name in enumerate(TABS):
    is_active = st.session_state["active_tab"] == tab_name
    clicked = nav_cols[i].button(
        tab_name,
        key=f"nav_{tab_name}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    )
    if clicked:
        st.session_state["active_tab"] = tab_name
        st.rerun()

st.markdown("---")

# ------------------------------------------------------------
# Render active tab
# ------------------------------------------------------------
active = st.session_state["active_tab"]
pid = st.session_state["selected_patient"]

if active == "Homepage":
    home.render(display_df)
elif active == "Individual Patient":
    individual_patient.render(enriched_df, feature_df, timeseries_df, pid)
elif active == "Risk Profile":
    risk_profile.render(enriched_df, feature_df, pid)
elif active == "Domain Specific Risk":
    domain_risk.render(display_df, pid)
elif active == "Try Your Data!":
    try_your_data.render()
