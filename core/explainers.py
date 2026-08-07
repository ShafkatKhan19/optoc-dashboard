"""
core/explainers.py

SHAP (population-level, Tab 3 "Key Risk Factors — All Patients") and
LIME (patient-level, Tab 2 Personalized Risk Summary + Tab 3
"Contributing Factors — This Patient") wrappers, following Appendix A5
/ A6 of the spec exactly.

Feature labels are converted from raw column names (e.g.
"num__scr_delta > 0.24") to plain clinical language per the spec's
explicit instruction that factors must be "labeled in plain clinical
language... not variable/code names."
"""

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from core.model_registry import load_registry, OUTCOME_LABELS

# Plain-language labels for every raw feature name
FEATURE_LABELS = {
    "sofa_total": "SOFA score",
    "qsofa_score": "qSOFA score",
    "shock_index_mean": "Shock index (HR/SBP)",
    "rass_min": "RASS sedation score",
    "rox_index_min": "ROX index (oxygenation)",
    "age": "Age",
    "icu_los_hours": "Time in ICU",
    "flag_vasopressor_use": "Vasopressor use",
    "flag_mechanical_ventilation": "Mechanical ventilation",
    "flag_rrt": "Renal replacement therapy",
    "flag_high_flow_or_niv": "High-flow O2 / NIV",
    "scr_delta": "Rising creatinine (\u0394 from baseline)",
    "creatinine_slope": "Creatinine trend",
    "map_min": "MAP (blood pressure)",
    "map_slope": "MAP trend",
    "fluid_balance_24h": "24h fluid balance",
    "lactate_max": "Peak lactate",
    "lactate_slope": "Lactate trend",
    "wbc_first": "White blood cell count",
    "magnesium_first": "Magnesium level",
    "phosphate_first": "Phosphate level",
    "troponin_t_max": "Peak troponin",
    "fibrinogen_first": "Fibrinogen level",
    "hemoglobin_last": "Hemoglobin",
    "charlson_comorbidity_index": "Comorbidity burden (Charlson index)",
    "nephrotoxin_count": "Number of nephrotoxic drugs",
    "total_discharge_meds": "Total discharge medications",
    "mrci_simplified": "Medication regimen complexity",
    "flag_tpn": "TPN (parenteral nutrition)",
    "flag_hit_risk_proxy": "Heparin exposure",
    "flag_qtc_prolonging_med_exposure": "QTc-prolonging medication",
    "flag_anticholinergic_exposure": "Anticholinergic medication",
    "flag_cefepime_exposure": "Cefepime exposure",
    "flag_linezolid_exposure": "Linezolid exposure",
    "flag_steroid_induced_hyperglycemia": "Steroid-associated hyperglycemia",
}


def _clean_feature_name(raw_name):
    """Strip the ColumnTransformer's 'num__' prefix and translate to a
    plain clinical label."""
    key = raw_name.replace("num__", "")
    return FEATURE_LABELS.get(key, key.replace("_", " "))


def shap_population_drivers(outcome, cohort_feature_df, top_n=10):
    """Tab 3 'Key Risk Factors (All Patients)' -- per spec Appendix A5."""
    registry = load_registry()
    entry = registry[outcome]
    model = entry["model"]
    explainer = entry["shap_explainer"]
    feature_names = entry["feature_names"]

    preprocessor = model.named_steps["prep"]
    X_transformed = preprocessor.transform(cohort_feature_df)

    shap_values = explainer.shap_values(X_transformed)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]

    return pd.DataFrame({
        "feature": [_clean_feature_name(feature_names[i]) for i in top_idx],
        "raw": [feature_names[i].replace("num__", "").replace("cat__", "") for i in top_idx],
        "impact": mean_abs[top_idx],
    })


def shap_patient_factors(outcome, patient_feature_row, cohort_feature_df=None, top_n=10):
    """
    Tab 2 'Contributing Factors' -- the first of the two independent
    explanation methods (LIME is the second, in lime_patient_factors
    below, matching the spec's "generated with a second, independent
    explanation method as a cross-check on the first"). Uses the same
    pre-fitted SHAP TreeExplainer as the population chart, but reads a
    single patient's row instead of averaging across the cohort, so
    it's a real per-patient local explanation, not an aggregate.
    """
    registry = load_registry()
    entry = registry[outcome]
    model = entry["model"]
    explainer = entry["shap_explainer"]
    feature_names = entry["feature_names"]

    preprocessor = model.named_steps["prep"]
    X_transformed = preprocessor.transform(patient_feature_row)

    shap_values = explainer.shap_values(X_transformed)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    row_values = shap_values[0]
    top_idx = np.argsort(np.abs(row_values))[::-1][:top_n]

    return pd.DataFrame({
        "feature": [_clean_feature_name(feature_names[i]) for i in top_idx],
        "impact": row_values[top_idx],
        "direction": ["Increases risk" if v > 0 else "Decreases risk" for v in row_values[top_idx]],
    })


def lime_patient_factors(outcome, patient_feature_row, top_n=10):
    """Tab 2 Personalized Risk Summary / Tab 3 'Contributing Factors
    (This Patient)' -- per spec Appendix A6. patient_feature_row: a
    single-row DataFrame (REQUIRED_FEATURE_COLUMNS)."""
    registry = load_registry()
    entry = registry[outcome]
    model = entry["model"]
    lime_bg = entry["lime_background"]

    preprocessor = model.named_steps["prep"]
    classifier = model.named_steps["clf"]

    explainer = LimeTabularExplainer(
        lime_bg["background"],
        feature_names=lime_bg["feature_names"],
        class_names=[f"No {OUTCOME_LABELS[outcome]}", OUTCOME_LABELS[outcome]],
        mode="classification",
        random_state=42,
    )

    row_transformed = preprocessor.transform(patient_feature_row)[0]
    explanation = explainer.explain_instance(
        row_transformed, classifier.predict_proba, num_features=top_n
    )

    factors = explanation.as_list()
    df = pd.DataFrame(factors, columns=["condition", "weight"])
    df["feature"] = df["condition"].apply(_clean_condition_label)
    df["raw"] = df["condition"].apply(_raw_key_from_condition)
    df["direction"] = np.where(df["weight"] > 0, "Increases risk", "Decreases risk")
    return df[["feature", "raw", "weight", "direction"]]


def _raw_key_from_condition(condition_str):
    """Which raw feature column a LIME condition string (e.g.
    'num__scr_delta > 0.24') refers to, or None if it doesn't match a
    known column -- used to check actionability, separate from the
    plain-language label."""
    for raw in FEATURE_LABELS:
        if raw in condition_str:
            return raw
    return None


def _clean_condition_label(condition_str):
    """LIME returns strings like 'num__scr_delta > 0.24' -- translate
    the variable part to plain language, keep the threshold."""
    for raw, label in FEATURE_LABELS.items():
        if raw in condition_str:
            return condition_str.replace(f"num__{raw}", label).replace(raw, label)
    return condition_str.replace("num__", "")
