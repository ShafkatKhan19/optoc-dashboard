"""
core/scoring.py

7-domain rule-based risk scoring engine. Point values transcribed
EXACTLY from the TOC v4 spec's "7-Domain Risk Scoring Guide."

IMPORTANT CHANGE FROM AN EARLIER PROTOTYPE: a previous iteration of
this dashboard normalized Low/Moderate/High thresholds as a % of each
domain's max possible score, reasoning that a domain with a low
ceiling (e.g. Infectious Disease, max 8) shouldn't hit "High" as
easily as one with a high ceiling (e.g. Cardio, max 18). The TOC v4
spec explicitly overrides that: "The score-to-label conversion is the
SAME for all 7 domains" -- 0-3 Low, 4-6 Moderate, 7+ High, no
normalization. This file follows the spec, not the earlier prototype.

KNOWN DATA GAP: optoc_sample_patients.csv does not include several
fields these rules reference (platelet count/trend, INR, fibrinogen
"current", measured QTc in ms, sodium, glucose baseline, bilirubin,
primary diagnosis / sepsis flag, prior-AKI history, OSA history,
CAD/AFib flags, benzodiazepine-for-sedation flag, SBT readiness, and
per-drug vancomycin/piperacillin-tazobactam "active" flags). Every
function below uses .get(key, default) so a missing field just means
that specific condition can't fire (0 points), rather than crashing --
but that also means some domains will under-score on THIS sample data
specifically. See README "Data Coverage" section for the full list.
Once the census data includes these fields, the same functions will
pick them up automatically -- no code changes needed.
"""

DOMAIN_LABELS = {
    "Neuro": "Neurological",
    "Pulmonary": "Pulmonary",
    "Cardio": "Cardiovascular",
    "Renal": "Renal",
    "GI": "GI / Metabolic",
    "Heme": "Hematology",
    "ID": "Infectious Disease",
}

DOMAIN_COLORS = {
    "Neuro": "#9b59b6", "Pulmonary": "#3498db", "Cardio": "#e74c3c",
    "Renal": "#16a085", "GI": "#f39c12", "Heme": "#c0392b", "ID": "#27ae60",
}


def risk_level(score):
    """Uniform Low/Moderate/High conversion -- SAME for every domain, per spec."""
    if score <= 3:
        return "LOW"
    elif score <= 6:
        return "MODERATE"
    return "HIGH"


# ---------------------------------------------------------------------
# 1. NEURO
# ---------------------------------------------------------------------
def score_neuro(p):
    score, reasons = 0, []
    rass = p.get("rass_min")
    if rass is not None:
        if rass < -3:
            score += 3; reasons.append(("RASS < -3 (deep sedation)", 3))
        elif rass <= -2:
            score += 1; reasons.append(("RASS -3 to -2 (moderate sedation)", 1))

    if p.get("on_benzodiazepine_sedation", 0):
        score += 3; reasons.append(("Benzodiazepine sedation", 3))

    if p.get("on_anticholinergic", 0):
        score += 2; reasons.append(("Anticholinergic medication present", 2))

    mg = p.get("magnesium_first") if p.get("magnesium_current") is None else p.get("magnesium_current")
    if mg is not None and mg < 1.7:
        score += 1; reasons.append(("Mg < 1.7 mg/dL", 1))

    na = p.get("sodium_current")
    if na is not None and na < 130:
        score += 1; reasons.append(("Hyponatremia (Na < 130)", 1))

    if p.get("comorbidity_dementia", 0):
        score += 1; reasons.append(("Dementia", 1))

    return score, risk_level(score), reasons


# ---------------------------------------------------------------------
# 2. PULMONARY
# ---------------------------------------------------------------------
def score_pulmonary(p):
    score, reasons = 0, []
    rr = p.get("respiratory_rate")
    if rr is not None:
        if rr < 10:
            score += 3; reasons.append(("RR < 10 (respiratory depression)", 3))
        elif rr <= 12:
            score += 2; reasons.append(("RR 10-12", 2))

    spo2 = p.get("spo2")
    if spo2 is not None:
        if spo2 < 88:
            score += 3; reasons.append(("SpO2 < 88%", 3))
        elif spo2 < 92:
            score += 1; reasons.append(("SpO2 88-92%", 1))

    if p.get("comorbidity_COPD", 0):
        score += 2; reasons.append(("COPD", 2))

    if p.get("osa_flag", 0):
        score += 1; reasons.append(("OSA", 1))

    if p.get("on_benzodiazepine_sedation", 0):
        score += 2; reasons.append(("Benzodiazepines increase respiratory depression risk", 2))

    on_vent = bool(p.get("on_mechanical_ventilation", 0)) or bool(p.get("on_hfnc_niv", 0))
    if on_vent:
        score += 2; reasons.append(("On ventilator / NIV support", 2))

    if p.get("on_mechanical_ventilation", 0) and p.get("sbt_ready", 1) == 0:
        score += 2; reasons.append(("Invasive vent, not SBT-ready", 2))

    return score, risk_level(score), reasons


# ---------------------------------------------------------------------
# 3. CARDIO
# ---------------------------------------------------------------------
def score_cardio(p):
    score, reasons = 0, []
    map_min = p.get("map_min")
    if map_min is not None and map_min < 65:
        score += 3; reasons.append(("MAP < 65 mmHg", 3))

    hr = p.get("heart_rate")
    if hr is not None and map_min:
        si = hr / map_min
        if si > 1.0:
            score += 2; reasons.append(("Shock Index > 1.0", 2))

    lact = p.get("lactate_peak") if p.get("lactate_current") is None else p.get("lactate_current")
    if lact is not None:
        if lact >= 4:
            score += 3; reasons.append(("Lactate >= 4.0", 3))
        elif lact >= 2:
            score += 2; reasons.append(("Lactate 2.0-3.9", 2))

    # Sample data only has a boolean on_vasopressor (no dose tier) --
    # approximated as "medium" dose when true. Flagged as an approximation.
    if p.get("pressor_dose_level") == "high":
        score += 3; reasons.append(("High-dose pressors", 3))
    elif p.get("pressor_dose_level") == "medium":
        score += 2; reasons.append(("Medium-dose pressors", 2))
    elif p.get("on_vasopressor", 0):
        score += 2; reasons.append(("On vasopressor (dose tier unavailable, treated as medium)", 2))

    trop = p.get("troponin_t_peak") if p.get("troponin_t_current") is None else p.get("troponin_t_current")
    if trop is not None and trop > 0.04:
        score += 2; reasons.append(("Troponin elevation", 2))

    qtc = p.get("qtc_current")
    if qtc is not None:
        if qtc > 500:
            score += 2; reasons.append(("QTc > 500 ms", 2))
        elif qtc > 470:
            score += 1; reasons.append(("QTc borderline elevated", 1))

    if p.get("comorbidity_CHF", 0):
        score += 1; reasons.append(("Heart Failure", 1))
    if p.get("cad_flag", 0):
        score += 1; reasons.append(("CAD", 1))
    if p.get("afib_flag", 0):
        score += 1; reasons.append(("AFib", 1))

    return score, risk_level(score), reasons


# ---------------------------------------------------------------------
# 4. RENAL
# ---------------------------------------------------------------------
def score_renal(p):
    score, reasons = 0, []
    cr_cur, cr_base = p.get("creatinine_current"), p.get("creatinine_baseline")
    if cr_cur is not None and cr_base is not None and (cr_cur - cr_base) >= 0.3:
        score += 3; reasons.append(("\u0394Cr >= 0.3 mg/dL", 3))

    if p.get("comorbidity_CKD", 0):
        score += 2; reasons.append(("Baseline CKD", 2))

    if p.get("aki_history_flag", 0):
        score += 1; reasons.append(("Prior AKI episode", 1))

    if p.get("on_pip_tazo", 0):
        score += 2; reasons.append(("Piperacillin/Tazobactam active", 2))
    if p.get("on_vancomycin", 0):
        score += 2; reasons.append(("Vancomycin active", 2))

    map_min = p.get("map_min")
    if map_min is not None and map_min < 65:
        score += 2; reasons.append(("MAP < 65", 2))

    fb = p.get("fluid_balance_24h")
    if fb is not None and fb > 2000:
        score += 2; reasons.append(("Positive fluid balance > 2000 mL", 2))

    uop_24h = p.get("urine_output_24h")
    uop_hr = (uop_24h / 24) if uop_24h is not None else p.get("uop_ml_hr")
    if uop_hr is not None and uop_hr < 30:
        score += 2; reasons.append(("Urine output < 30 mL/hr", 2))

    return score, risk_level(score), reasons


# ---------------------------------------------------------------------
# 5. GI / METABOLIC
# ---------------------------------------------------------------------
def score_gi(p):
    score, reasons = 0, []
    glu = p.get("glucose_current")
    if glu is not None and (glu < 70 or glu > 180):
        score += 2; reasons.append(("Glucose < 70 or > 180 mg/dL", 2))

    base = p.get("glucose_baseline")
    if glu is not None and base is not None and abs(glu - base) > 50:
        score += 1; reasons.append(("Glucose variability > 50 mg/dL", 1))

    lact = p.get("lactate_peak") if p.get("lactate_current") is None else p.get("lactate_current")
    if lact is not None and lact > 2:
        score += 1; reasons.append(("Lactate > 2.0", 1))

    phos = p.get("phosphate_first") if p.get("phosphate_current") is None else p.get("phosphate_current")
    if phos is not None and phos < 2.5:
        score += 2; reasons.append(("Phosphate < 2.5 mg/dL", 2))

    mg = p.get("magnesium_first") if p.get("magnesium_current") is None else p.get("magnesium_current")
    if mg is not None and mg < 1.8:
        score += 1; reasons.append(("Magnesium < 1.8 mg/dL", 1))

    bili = p.get("bilirubin_current")
    if bili is not None and bili > 2.0:
        score += 2; reasons.append(("Bilirubin > 2.0 mg/dL", 2))

    if p.get("on_steroid", 0) and glu is not None and glu > 180:
        score += 1; reasons.append(("Steroid with hyperglycemia risk", 1))

    return score, risk_level(score), reasons


# ---------------------------------------------------------------------
# 6. HEMATOLOGY
# ---------------------------------------------------------------------
def score_heme(p):
    score, reasons = 0, []
    plt_cur = p.get("platelet_current")
    if plt_cur is not None:
        if plt_cur < 50000:
            score += 3; reasons.append(("Platelets < 50,000", 3))
        elif plt_cur < 100000:
            score += 2; reasons.append(("Platelets 50,000-99,999", 2))

    plt_prev = p.get("platelet_48h_ago")
    if plt_cur is not None and plt_prev is not None and (plt_prev - plt_cur) > 50000:
        score += 2; reasons.append(("Platelet drop > 50,000 in 48h", 2))

    if p.get("on_heparin", 0):
        score += 2; reasons.append(("Heparin exposure (HIT risk)", 2))

    inr = p.get("inr_current")
    if inr is not None and inr > 1.5:
        score += 2; reasons.append(("INR > 1.5", 2))

    fib = p.get("fibrinogen_first") if p.get("fibrinogen_current") is None else p.get("fibrinogen_current")
    if fib is not None and fib < 150:
        score += 2; reasons.append(("Fibrinogen < 150 mg/dL", 2))

    if p.get("sepsis_flag", 0):
        score += 1; reasons.append(("Sepsis present (coagulopathy risk)", 1))

    return score, risk_level(score), reasons


# ---------------------------------------------------------------------
# 7. INFECTIOUS DISEASE
# ---------------------------------------------------------------------
def score_id(p):
    score, reasons = 0, []
    if p.get("sepsis_flag", 0):
        score += 2; reasons.append(("Sepsis is primary diagnosis", 2))

    lact = p.get("lactate_peak") if p.get("lactate_current") is None else p.get("lactate_current")
    if lact is not None and lact > 2:
        score += 1; reasons.append(("Lactate > 2.0", 1))

    if p.get("on_cefepime", 0):
        score += 1; reasons.append(("Cefepime active", 1))
    if p.get("on_pip_tazo", 0):
        score += 2; reasons.append(("Piperacillin/Tazobactam active", 2))
    if p.get("on_vancomycin", 0):
        score += 1; reasons.append(("Vancomycin active", 1))

    if p.get("on_tpn", 0):
        score += 1; reasons.append(("TPN (raises C. diff risk)", 1))

    return score, risk_level(score), reasons


DOMAIN_FUNCTIONS = {
    "Neuro": score_neuro,
    "Pulmonary": score_pulmonary,
    "Cardio": score_cardio,
    "Renal": score_renal,
    "GI": score_gi,
    "Heme": score_heme,
    "ID": score_id,
}


def score_all_domains(p):
    return {name: fn(p) for name, fn in DOMAIN_FUNCTIONS.items()}
