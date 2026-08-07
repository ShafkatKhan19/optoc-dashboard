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
from app_pages import home, individual_patient, risk_profile, try_your_data

st.set_page_config(
    page_title="OPTOC",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()

TABS = ["Homepage", "Individual Patient", "Clinical Insights", "Try Your Data!"]

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Homepage"

# ------------------------------------------------------------
# Load + enrich data (cached)
# ------------------------------------------------------------
with st.spinner("Loading patient data and running risk models..."):
    enriched_df, feature_df, timeseries_df = build_enriched_cohort()

if "selected_patient" not in st.session_state:
    st.session_state["selected_patient"] = enriched_df["patient_id"].iloc[0]

# Manually archived patients (e.g. discharged) -- session-only, since this
# demo has no backing database. Archiving hides a patient from the
# dropdown/rankings/heatmap by default without deleting any data; nothing
# is ever removed from the underlying CSV. This resets if the app process
# restarts -- flagged here rather than silently pretending it's permanent.
if "archived_patients" not in st.session_state:
    st.session_state["archived_patients"] = set()

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

n_archived = len(st.session_state["archived_patients"])
show_archived = st.sidebar.checkbox(f"Show archived patients ({n_archived})", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### Patient Selector")
all_patient_ids = enriched_df["patient_id"].tolist()
patient_ids = all_patient_ids if show_archived else [
    p for p in all_patient_ids if p not in st.session_state["archived_patients"]
]
if not patient_ids:
    # don't let an "archive everyone" state break patient selection entirely
    patient_ids = all_patient_ids

# One-way sync, external -> sidebar widget, using a sentinel to tell "you
# just clicked this widget" apart from "selected_patient changed elsewhere
# (e.g. a Priority Alert click on another tab)". Without the sentinel,
# both cases look identical (sidebar_patient_select != selected_patient)
# at this point in the script, so a naive sync stomps your own click back
# to the old value on every interaction -- that was the bug.
if "sidebar_patient_select" not in st.session_state or st.session_state["sidebar_patient_select"] not in patient_ids:
    default_patient = (
        st.session_state["selected_patient"]
        if st.session_state["selected_patient"] in patient_ids
        else patient_ids[0]
    )
    st.session_state["sidebar_patient_select"] = default_patient
    st.session_state["_last_synced_patient"] = default_patient
elif st.session_state["selected_patient"] != st.session_state.get("_last_synced_patient"):
    # selected_patient moved without this widget's involvement -> catch up.
    st.session_state["sidebar_patient_select"] = st.session_state["selected_patient"]
    st.session_state["_last_synced_patient"] = st.session_state["selected_patient"]

picked = st.sidebar.selectbox("Patient ID", patient_ids, key="sidebar_patient_select")
st.session_state["selected_patient"] = picked
st.session_state["_last_synced_patient"] = picked

# Manual entry, for typing a Patient ID directly instead of scrolling the
# dropdown. Matches on the patient's actual NUMBER, not a text fragment --
# "0107", "P-0107", and "107" (no leading zero) all correctly find
# "P-0107" because 107 == 107 as numbers, but "7" does NOT match "P-0107"
# just because "0107" happens to end in "7". Comparing as integers (not
# substring/suffix text matching) is what keeps that distinction correct.
manual_id = st.sidebar.text_input("Or type Patient ID directly", placeholder="e.g. P-0107 or just 107")
if manual_id.strip():
    typed_digits = "".join(ch for ch in manual_id if ch.isdigit())
    typed_alnum = "".join(ch for ch in manual_id if ch.isalnum()).upper()

    exact = [p for p in patient_ids if "".join(ch for ch in p if ch.isalnum()).upper() == typed_alnum]

    numeric_matches = []
    if not exact and typed_digits:
        typed_num = int(typed_digits)
        for p in patient_ids:
            pid_digits = "".join(ch for ch in p if ch.isdigit())
            if pid_digits and int(pid_digits) == typed_num:
                numeric_matches.append(p)

    if len(exact) == 1:
        if exact[0] != st.session_state["selected_patient"]:
            st.session_state["selected_patient"] = exact[0]
            st.rerun()
    elif len(numeric_matches) == 1:
        if numeric_matches[0] != st.session_state["selected_patient"]:
            st.session_state["selected_patient"] = numeric_matches[0]
            st.rerun()
    elif numeric_matches:
        st.sidebar.caption("Multiple matches -- did you mean:")
        for candidate in numeric_matches[:6]:
            if st.sidebar.button(candidate, key=f"manual_match_{candidate}", use_container_width=True):
                st.session_state["selected_patient"] = candidate
                st.rerun()
    else:
        st.sidebar.caption(f'No patient found matching "{manual_id}".')

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

# Apply sidebar unit + archive filters to the cohort used everywhere
display_df = enriched_df.copy()
if not show_archived:
    display_df = display_df[~display_df["patient_id"].isin(st.session_state["archived_patients"])]
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
    individual_patient.render(enriched_df, timeseries_df, pid)
elif active == "Clinical Insights":
    risk_profile.render(enriched_df, feature_df, display_df, pid)
elif active == "Try Your Data!":
    try_your_data.render()
