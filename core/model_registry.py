"""
core/model_registry.py

Loads all pre-trained model artifacts once at startup (cached via
st.cache_resource per spec Appendix A3).

KNOWN ISSUE, flagged rather than silently worked around:
models/model_metadata.csv lists "Best_Model" / "PKL_Filename" for
Sepsis as "Random Forest" / "sepsis_rf_model.pkl", and for Mortality as
"Random Forest" / "mortality_rf_model.pkl" -- but models.zip does NOT
contain either of those files. Only sepsis_xgb_model.pkl and
mortality_xgb_model.pkl exist, and inspecting them directly confirms
both are XGBClassifier pipelines, not Random Forests. This registry
uses the actual *_xgb_model.pkl files that were supplied for all 4
outcomes (matching what Appendix A3's own load code does), and keeps
the Youden thresholds from model_metadata.csv (those are keyed by
outcome name, independent of the filename mismatch). Flag this to the
professor/Buchi -- either the metadata CSV is stale, or two files are
missing from the handoff.

Also: the spec's prose says "38 model-input columns"; the actual
required_feature_columns.txt (the technical ground truth) has 35. This
code uses the file, not the prose count.
"""

import os
import warnings
import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

OUTCOMES = ["aki", "sepsis", "mortality", "readmission"]

OUTCOME_LABELS = {
    "aki": "AKI Risk",
    "sepsis": "Sepsis Risk",
    "mortality": "Mortality Risk",
    "readmission": "Readmission Risk",
}

OUTCOME_PHRASES = {
    "aki": "develop acute kidney injury",
    "sepsis": "develop sepsis",
    "mortality": "die during this hospitalization",
    "readmission": "be readmitted within 30 days",
}

# Maps our short outcome keys to model_metadata.csv's "Outcome" values
METADATA_OUTCOME_NAMES = {
    "aki": "AKI",
    "sepsis": "Sepsis",
    "mortality": "In-Hospital Mortality",
    "readmission": "30-Day Readmission",
}

with open(os.path.join(MODELS_DIR, "required_feature_columns.txt")) as _f:
    REQUIRED_FEATURE_COLUMNS = [line.strip() for line in _f if line.strip()]


@st.cache_resource
def load_registry():
    warnings.filterwarnings("ignore")  # suppress sklearn cross-version pickle warnings

    registry = {}
    meta = pd.read_csv(os.path.join(MODELS_DIR, "model_metadata.csv")).set_index("Outcome")

    for outcome in OUTCOMES:
        registry[outcome] = {
            "model": joblib.load(os.path.join(MODELS_DIR, f"{outcome}_xgb_model.pkl")),
            "shap_explainer": joblib.load(os.path.join(MODELS_DIR, f"{outcome}_xgb_shap_explainer.pkl")),
            "feature_names": joblib.load(os.path.join(MODELS_DIR, f"{outcome}_xgb_feature_names.pkl")),
            "lime_background": joblib.load(os.path.join(MODELS_DIR, f"{outcome}_xgb_lime_background.pkl")),
            "threshold": float(meta.loc[METADATA_OUTCOME_NAMES[outcome], "Youden_Threshold"]),
        }

    return registry


def predict_all(feature_df):
    """feature_df: DataFrame with REQUIRED_FEATURE_COLUMNS, one row per patient.
    Returns dict {outcome: np.array of probabilities}."""
    registry = load_registry()
    return {
        outcome: registry[outcome]["model"].predict_proba(feature_df)[:, 1]
        for outcome in OUTCOMES
    }
