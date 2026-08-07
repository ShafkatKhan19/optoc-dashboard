"""
core/theme.py

Color standards are copied EXACTLY from the TOC v4 spec's "COLOR
STANDARDS & RISK DISPLAY RULES" section ("Apply these consistently
across ALL tabs. Do not deviate.") -- do not change these hex values
without checking with the professor.

The overall visual language (card layout, left-accent-border cards,
pill badges, grouped/clean spacing, dark-blue header) is inspired by
the Ember dashboard template you shared, adapted with Streamlit-native
CSS rather than copied Ember code -- Ember is a paid commercial
template (dashboardpack.com), so this recreates the *feel* (clean
cards, colored accents, generous whitespace, muted palette) without
reusing their actual markup/CSS, which would need a license.
"""

import streamlit as st

# ---------------------------------------------------------------------
# EXACT color standards from the spec -- do not deviate
# ---------------------------------------------------------------------
RISK_COLORS = {
    "HIGH": {
        "bg": "#FFE6E6",
        "text": "#C00000",
        "border": "#C00000",
    },
    "MEDIUM": {
        "bg": "#FFF2CC",
        "text": "#7B5E00",
        "border": "#7B5E00",
    },
    "LOW": {
        "bg": "#E2EFDA",
        "text": "#375623",
        "border": "#375623",
    },
}

# Ember-inspired accent palette (not from the spec, chosen to match the
# Ember screenshot's dark-blue header + muted slate body)
EMBER_HEADER_BG = "#0F172A"
EMBER_ACCENT = "#2563EB"
EMBER_BG = "#E7EBF2"
EMBER_CARD_BG = "#FFFFFF"
EMBER_BORDER = "#CBD5E1"
EMBER_TEXT_MUTED = "#64748B"

# Warm accent used to mark the active top-nav tab (amber) -- deliberately
# distinct from the cool blue accent and from the HIGH/MEDIUM/LOW risk
# colors above so it never reads as a clinical signal.
ACTIVE_TAB_BG = "#F59E0B"
ACTIVE_TAB_TEXT = "#1C1300"


def risk_tier(prob, threshold):
    """Exact tiering rule from spec Appendix A4."""
    if prob >= threshold:
        return "HIGH"
    elif prob >= threshold * 0.6:
        return "MEDIUM"
    return "LOW"


def risk_badge_html(tier, label=None, font_size="12px", padding="3px 10px"):
    c = RISK_COLORS.get(tier, RISK_COLORS["LOW"])
    label = label or tier
    return (
        f'<span style="background-color:{c["bg"]}; color:{c["text"]}; '
        f'padding:{padding}; border-radius:999px; font-weight:700; '
        f'font-size:{font_size}; letter-spacing:0.02em;">{label}</span>'
    )


def risk_cell_style(tier):
    c = RISK_COLORS.get(tier, RISK_COLORS["LOW"])
    return f"background-color:{c['bg']}; color:{c['text']}; font-weight:600;"


def color_dot(hex_color, size=10):
    """Small solid circle used in place of emoji markers/icons."""
    return (
        f'<span style="display:inline-block; width:{size}px; height:{size}px; '
        f'border-radius:50%; background-color:{hex_color}; margin-right:8px; '
        f'vertical-align:middle;"></span>'
    )


def inject_global_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {EMBER_BG};
        }}
        div[data-testid="stMetric"] {{
            background-color: {EMBER_CARD_BG};
            border: 1px solid {EMBER_BORDER};
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 2px 5px rgba(15, 23, 42, 0.12);
        }}
        /* Let metric labels/values wrap and shrink instead of being
           clipped with an ellipsis -- affects every st.metric card in
           the app (Homepage KPIs, Vitals & Labs, Domain summary cards)
           since they all use the same Streamlit component. */
        div[data-testid="stMetricLabel"] {{
            white-space: normal !important;
            overflow: visible !important;
            font-size: 13px !important;
            line-height: 1.25 !important;
        }}
        div[data-testid="stMetricValue"] {{
            white-space: normal !important;
            overflow: visible !important;
            font-size: 22px !important;
            line-height: 1.25 !important;
        }}
        div[data-testid="stElementContainer"] div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            box-shadow: 0 2px 5px rgba(15, 23, 42, 0.10);
        }}
        .optoc-card, .js-plotly-plot {{
            box-shadow: 0 2px 5px rgba(15, 23, 42, 0.10);
        }}
        section[data-testid="stSidebar"] {{
            background-color: {EMBER_HEADER_BG};
        }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span:not([style]) {{
            color: #E2E8F0 !important;
        }}
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input {{
            background-color: #1E293B !important;
            color: #F1F5F9 !important;
        }}
        /* Sidebar buttons (e.g. Refresh Data) sit on the dark navy panel --
           give them their own dark-on-dark styling instead of inheriting
           the default light Streamlit button surface, which made the
           label unreadable. */
        section[data-testid="stSidebar"] button[kind="secondary"] {{
            background-color: #1E293B !important;
            border: 1px solid #475569 !important;
        }}
        section[data-testid="stSidebar"] button[kind="secondary"] p {{
            color: #F1F5F9 !important;
        }}
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {{
            background-color: #334155 !important;
            border-color: {ACTIVE_TAB_BG} !important;
        }}
        section[data-testid="stSidebar"] button[kind="secondary"]:hover p {{
            color: {ACTIVE_TAB_BG} !important;
        }}
        .optoc-header {{
            background-color: {EMBER_HEADER_BG};
            color: white;
            padding: 20px 28px;
            border-radius: 14px;
            margin-bottom: 18px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.15);
        }}
        /* Active top-nav tab: amber fill instead of a bullet character */
        button[kind="primary"] {{
            background-color: {ACTIVE_TAB_BG} !important;
            border-color: {ACTIVE_TAB_BG} !important;
            color: {ACTIVE_TAB_TEXT} !important;
            font-weight: 700 !important;
        }}
        button[kind="primary"]:hover {{
            background-color: #D97706 !important;
            border-color: #D97706 !important;
        }}
        .optoc-header h1 {{
            margin: 0;
            font-size: 26px;
            font-weight: 700;
        }}
        .optoc-header p {{
            margin: 4px 0 0 0;
            color: #CBD5E1;
            font-size: 14px;
        }}
        .optoc-card {{
            background-color: {EMBER_CARD_BG};
            border: 1px solid {EMBER_BORDER};
            border-left: 4px solid {EMBER_ACCENT};
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }}
        .optoc-alert-critical {{
            background-color: #FFE6E6;
            border-left: 4px solid #C00000;
            color: #C00000;
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
        }}
        .optoc-alert-warning {{
            background-color: #FFF2CC;
            border-left: 4px solid #7B5E00;
            color: #7B5E00;
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
        }}
        .optoc-alert-info {{
            background-color: #E7F1FF;
            border-left: 4px solid #2563EB;
            color: #1E3A8A;
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
        }}
        .optoc-section-title {{
            font-weight: 700;
            font-size: 18px;
            color: #0F172A;
            margin-top: 6px;
            margin-bottom: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title, subtitle=""):
    st.markdown(
        f"""
        <div class="optoc-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
