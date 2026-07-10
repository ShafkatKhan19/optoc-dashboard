"""
core/data_io.py

Central place every tab pulls data from: loads the sample census,
computes model features, runs all 4 outcome models, computes the 7
domain scores, and assembles one enriched DataFrame the whole app
shares -- so every tab's numbers are guaranteed consistent.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from core.features import build_features_df
from core.model_registry import predict_all, load_registry
from core.theme import risk_tier
from core.scoring import score_all_domains

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@st.cache_data
def load_raw_data():
    patients = pd.read_csv(os.path.join(DATA_DIR, "optoc_sample_patients.csv"))
    timeseries = pd.read_csv(os.path.join(DATA_DIR, "optoc_timeseries_vitals.csv"))
    return patients, timeseries


@st.cache_data
def build_enriched_cohort():
    """Returns (patients_df, feature_df, timeseries_df) where patients_df
    has the 4 outcome risk probabilities + tiers + composite score + 7
    domain scores all attached, ready for every tab to use directly."""
    patients, timeseries = load_raw_data()
    feature_df, patient_ids = build_features_df(patients, timeseries)

    registry = load_registry()
    preds = predict_all(feature_df)

    enriched = patients.copy()
    for outcome, probs in preds.items():
        enriched[f"{outcome}_risk"] = probs
        threshold = registry[outcome]["threshold"]
        enriched[f"{outcome}_tier"] = [risk_tier(p, threshold) for p in probs]

    # Composite score: mean of the 4 outcome probabilities (0-100 scale)
    enriched["composite_score"] = (
        enriched[["aki_risk", "sepsis_risk", "mortality_risk", "readmission_risk"]].mean(axis=1) * 100
    ).round(1)
    enriched["composite_tier"] = enriched["composite_score"].apply(
        lambda s: "HIGH" if s >= 70 else ("MEDIUM" if s >= 42 else "LOW")
    )

    # 7-domain scores
    domain_rows = []
    for _, row in enriched.iterrows():
        domains = score_all_domains(row.to_dict())
        domain_rows.append({name: (score, level) for name, (score, level, _) in domains.items()})
    for domain_name in domain_rows[0].keys():
        enriched[f"domain_{domain_name}_score"] = [d[domain_name][0] for d in domain_rows]
        enriched[f"domain_{domain_name}_level"] = [d[domain_name][1] for d in domain_rows]

    # Days in ICU / Discharge Due, formatted for display
    enriched["days_in_icu"] = enriched["icu_los_days"]

    return enriched, feature_df, timeseries


def get_patient_row(enriched_df, pid):
    match = enriched_df[enriched_df["patient_id"] == pid]
    if len(match) == 0:
        return None
    return match.iloc[0].to_dict()


def get_patient_feature_row(feature_df, enriched_df, pid):
    """Returns the single-row feature DataFrame (for SHAP/LIME) for one patient."""
    idx = enriched_df.index[enriched_df["patient_id"] == pid]
    if len(idx) == 0:
        return None
    return feature_df.loc[[idx[0]]]


def get_patient_timeseries(timeseries_df, pid):
    return timeseries_df[timeseries_df["patient_id"] == pid].sort_values("hours_since_admission")
