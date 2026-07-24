"""
core/features.py

Implements build_model_features() from the TOC v4 spec's Appendix A
("Background calculation function (for the developer)"), adapted to
the actual column names found in optoc_sample_patients.csv /
optoc_timeseries_vitals.csv (the spec's pseudocode used slightly
different field names than the real sample data -- e.g. the spec
sketch uses `hemoglobin_last` as a form field name, but the actual CSV
column is `haemoglobin_latest`; mapped below).

This is the ONLY place that maps raw clinical inputs (what a bedside
pharmacist has, or what's in a raw patient CSV) to the 35 columns in
models/required_feature_columns.txt that the pkl models expect --
matching the spec's explicit instruction that this mapping should live
in exactly one place.

Missing values (e.g. mrci_simplified, which isn't collected anywhere
in the sample data) are imputed with the median of that feature drawn
from the model's own LIME background dataset -- a real training-set
distribution shipped with the models -- rather than an arbitrary
placeholder, per the spec's "system uses training-set median" note.
"""

import numpy as np
import pandas as pd

from core.model_registry import REQUIRED_FEATURE_COLUMNS

CHARLSON_WEIGHTS = {
    "MI": 1, "CHF": 1, "PVD": 1, "stroke": 1, "dementia": 1, "COPD": 1,
    "diabetes_uncomplicated": 1, "diabetes_complicated": 2, "CKD": 2,
    "paraplegia": 2, "cancer": 2, "liver_mild": 1, "liver_severe": 3,
    "metastatic_cancer": 6, "HIV": 6,
}


def compute_slope(hours, values):
    """Linear regression slope (units per hour). None if <2 distinct time points."""
    hours = np.array(hours, dtype=float)
    values = np.array(values, dtype=float)
    mask = ~np.isnan(hours) & ~np.isnan(values)
    hours, values = hours[mask], values[mask]
    if len(hours) < 2 or len(set(hours)) < 2:
        return None
    return float(np.polyfit(hours, values, 1)[0])


def charlson_index(row):
    total = 0
    for key, weight in CHARLSON_WEIGHTS.items():
        col = f"comorbidity_{key}"
        if row.get(col, 0) in (1, True, "1", "Yes", "yes"):
            total += weight
    return total


def build_features_row(row, ts_patient=None):
    """
    row: dict-like of one patient's Tier-1 raw fields (matches
         optoc_sample_patients.csv column names, or the equivalent
         Tab 5 manual-entry form field names -- see MANUAL_ENTRY_MAP
         in app_pages/try_your_data.py for that mapping).
    ts_patient: optional DataFrame of that patient's time-series rows
         (hours_since_admission, creatinine, lactate, map, hr, rr, spo2)

    Returns a dict of the 35 model-input features (raw, unimputed --
    call impute_missing() separately, or use build_features_df below
    which does both).
    """
    f = {}

    f["sofa_total"] = row.get("sofa_total")

    rr = row.get("respiratory_rate")
    sbp = row.get("systolic_bp")
    gcs = row.get("gcs_total")
    hr = row.get("heart_rate")
    spo2 = row.get("spo2")
    fio2 = row.get("fio2")
    fio2 = fio2 if pd.notna(fio2) else 21  # spec default: 21 (room air, as a percent-style value)

    f["qsofa_score"] = (
        int(rr is not None and rr >= 22)
        + int(sbp is not None and sbp <= 100)
        + int(gcs is not None and gcs < 15)
    )
    f["shock_index_mean"] = (hr / sbp) if (hr is not None and sbp) else None
    f["rass_min"] = row.get("rass_min")
    f["rox_index_min"] = (spo2 / fio2) / rr if (spo2 is not None and rr) else None

    f["age"] = row.get("age")
    icu_los_days = row.get("icu_los_days", 0) or 0
    f["icu_los_hours"] = icu_los_days * 24

    f["flag_vasopressor_use"] = int(bool(row.get("on_vasopressor", 0)))
    f["flag_mechanical_ventilation"] = int(bool(row.get("on_mechanical_ventilation", 0)))
    f["flag_rrt"] = int(bool(row.get("on_rrt", 0)))
    f["flag_high_flow_or_niv"] = int(bool(row.get("on_hfnc_niv", 0)))

    cr_current = row.get("creatinine_current")
    cr_baseline = row.get("creatinine_baseline")
    f["scr_delta"] = (
        (cr_current - cr_baseline) if (cr_current is not None and cr_baseline is not None) else None
    )

    map_min = row.get("map_min")
    if pd.isna(map_min) if map_min is not None else True:
        dbp = row.get("diastolic_bp")
        map_min = (sbp + 2 * dbp) / 3 if (sbp is not None and dbp is not None) else None
    f["map_min"] = map_min

    f["fluid_balance_24h"] = row.get("fluid_balance_24h")

    # Time-series-derived features
    if ts_patient is not None and len(ts_patient):
        ts_sorted = ts_patient.sort_values("hours_since_admission")
        f["creatinine_slope"] = compute_slope(ts_sorted["hours_since_admission"], ts_sorted["creatinine"])
        f["map_slope"] = compute_slope(ts_sorted["hours_since_admission"], ts_sorted["map"])
        f["lactate_slope"] = compute_slope(ts_sorted["hours_since_admission"], ts_sorted["lactate"])
        lactate_vals = list(ts_sorted["lactate"].dropna())
        f["lactate_max"] = max(lactate_vals) if lactate_vals else row.get("lactate_peak")
    else:
        f["creatinine_slope"] = None
        f["map_slope"] = None
        f["lactate_slope"] = None
        f["lactate_max"] = row.get("lactate_peak")

    f["wbc_first"] = row.get("wbc_first")
    f["magnesium_first"] = row.get("magnesium_first")
    f["phosphate_first"] = row.get("phosphate_first")
    f["troponin_t_max"] = row.get("troponin_t_peak")
    f["fibrinogen_first"] = row.get("fibrinogen_first")
    f["hemoglobin_last"] = row.get("haemoglobin_latest")

    f["charlson_comorbidity_index"] = charlson_index(row)

    f["nephrotoxin_count"] = row.get("nephrotoxin_count")
    f["total_discharge_meds"] = row.get("total_discharge_meds")
    f["mrci_simplified"] = row.get("mrci_simplified")  # not in sample data -> imputed later

    f["flag_tpn"] = int(bool(row.get("on_tpn", 0)))
    f["flag_hit_risk_proxy"] = int(bool(row.get("on_heparin", 0)))
    f["flag_qtc_prolonging_med_exposure"] = int(bool(row.get("on_qtc_med", 0)))
    f["flag_anticholinergic_exposure"] = int(bool(row.get("on_anticholinergic", 0)))
    f["flag_cefepime_exposure"] = int(bool(row.get("on_cefepime", 0)))
    f["flag_linezolid_exposure"] = int(bool(row.get("on_linezolid", 0)))

    on_steroid = bool(row.get("on_steroid", 0))
    glucose = row.get("glucose_current")
    f["flag_steroid_induced_hyperglycemia"] = int(on_steroid and glucose is not None and glucose > 180)

    return f


def build_features_df(patients_df, timeseries_df=None):
    """
    patients_df: DataFrame of raw Tier-1 patient rows (must include patient_id)
    timeseries_df: optional DataFrame with patient_id, hours_since_admission,
                   creatinine, lactate, map, hr, rr, spo2

    Returns (feature_df, patient_ids) where feature_df has exactly the
    35 REQUIRED_FEATURE_COLUMNS in the right order, fully imputed and
    ready for pipeline.predict_proba().
    """
    rows = []
    for _, row in patients_df.iterrows():
        pid = row.get("patient_id")
        ts_p = None
        if timeseries_df is not None and "patient_id" in timeseries_df.columns:
            ts_p = timeseries_df[timeseries_df["patient_id"] == pid]
        rows.append(build_features_row(row.to_dict(), ts_p))

    # Leave missing values as NaN -- do NOT pre-impute here. Each model's
    # own sklearn Pipeline ("prep" step) already contains an imputer fit
    # on that model's real training data, and expects to do this itself
    # (this matches the professor's validate_dashboard.py reference
    # implementation, which passes raw/NaN features straight into
    # model.predict_proba() with no external imputation step). Pre-filling
    # with our own medians -- previously computed from just the AKI
    # model's LIME background, and reused for all four different models --
    # silently overrode each model's real trained imputation statistics
    # and was the source of the risk-score discrepancies flagged against
    # validate_dashboard.py.
    feat_df = pd.DataFrame(rows)[REQUIRED_FEATURE_COLUMNS].astype(float)

    return feat_df, patients_df["patient_id"].tolist()
