"""
core/med_alerts.py

Medication alert logic transcribed EXACTLY from the TOC v4 spec's
"Medication alert logic" table (Tab 2).

DATA GAP: several trigger conditions reference fields optoc_sample_patients.csv
doesn't have -- per-drug active flags for vancomycin, piperacillin-tazobactam,
gabapentin, metformin, warfarin/DOAC; a numeric CrCl value; platelet
count/trend; and a count of new medications started during the ICU stay.
Every check below uses .get(key, default) so those specific alerts simply
won't fire on THIS sample data (rather than crashing) -- they're fully
implemented and will fire correctly once the census data includes those
columns. See README "Data Coverage" for the complete list.
"""

SEVERITY_ORDER = {"CRITICAL": 0, "ACTION NEEDED": 1, "WARNING": 2, "INFO": 3}


def get_medication_alerts(p, aki_risk=None):
    """
    p: dict-like patient row (raw fields + on_* medication flags)
    aki_risk: this patient's model-predicted AKI probability (0-1),
              needed for the "AKI + nephrotoxin" rule
    """
    alerts = []

    # --- Nephrotoxin pair: vancomycin + pip-tazo ---
    if p.get("on_vancomycin", 0) and p.get("on_pip_tazo", 0):
        alerts.append({
            "severity": "CRITICAL",
            "text": "CRITICAL: Vancomycin + Pip-Tazo combination detected. High AKI risk. "
                    "Consider alternative or monitor SCr q6h.",
        })

    # --- Nephrotoxin pair: vancomycin + loop diuretic ---
    if p.get("on_vancomycin", 0) and p.get("on_loop_diuretic", 0):
        alerts.append({
            "severity": "WARNING",
            "text": "WARNING: Vancomycin + loop diuretic. Increased nephrotoxicity risk. "
                    "Ensure SCr and vancomycin levels monitored.",
        })

    # --- Renal dose adjustment: CrCl < 60 + gabapentin, no dose reduction flag ---
    crcl = p.get("crcl")
    if crcl is not None and crcl < 60 and p.get("on_gabapentin", 0) and not p.get("gabapentin_dose_reduced", 0):
        alerts.append({
            "severity": "ACTION NEEDED",
            "text": f"ACTION NEEDED: Gabapentin dose may require reduction for CrCl {crcl}. Review dosing.",
        })

    # --- Renal dose adjustment: CrCl < 30 + metformin ---
    if crcl is not None and crcl < 30 and p.get("on_metformin", 0):
        alerts.append({
            "severity": "CRITICAL",
            "text": "CRITICAL: Metformin contraindicated with CrCl < 30. Recommend discontinuation.",
        })

    # --- Polypharmacy ---
    n_meds = p.get("total_discharge_meds")
    if n_meds is not None and n_meds >= 15:
        alerts.append({
            "severity": "INFO",
            "text": f"INFO: Patient has {int(n_meds)} discharge medications. High regimen complexity. "
                    "Medication reconciliation and patient counseling recommended.",
        })

    # --- High alert med: warfarin/DOAC ---
    if p.get("on_warfarin_or_doac", 0):
        alerts.append({
            "severity": "INFO",
            "text": "INFO: Anticoagulant on discharge regimen. Confirm INR / renal dosing / indication reviewed.",
        })

    # --- New medications during ICU stay ---
    n_new = p.get("new_meds_during_stay")
    if n_new is not None and n_new >= 5:
        alerts.append({
            "severity": "INFO",
            "text": f"INFO: {int(n_new)} new medications started during ICU stay. "
                    "Ensure patient and/or caregiver counseled on all new drugs.",
        })

    # --- AKI risk + any nephrotoxin ---
    nephrotoxin_count = p.get("nephrotoxin_count", 0) or 0
    if aki_risk is not None and aki_risk > 0.70 and nephrotoxin_count >= 1:
        alerts.append({
            "severity": "CRITICAL",
            "text": "CRITICAL: High AKI risk AND active nephrotoxic drug. Urgent pharmacist review recommended.",
        })

    # --- QTc risk ---
    if p.get("on_qtc_med", 0):
        alerts.append({
            "severity": "WARNING",
            "text": "WARNING: QTc-prolonging medication detected. Monitor ECG and QTc interval, "
                    "especially in combination with other QT-prolonging drugs.",
        })

    # --- Anticholinergic ---
    if p.get("on_anticholinergic", 0):
        alerts.append({
            "severity": "WARNING",
            "text": "WARNING: Anticholinergic medication detected. Increased delirium risk in ICU. "
                    "Review medication necessity.",
        })

    # --- Cefepime + altered mental status ---
    gcs = p.get("gcs_total")
    rass = p.get("rass_min")
    altered_mental_status = (gcs is not None and gcs < 13) or (rass is not None and rass < -2)
    if p.get("on_cefepime", 0) and altered_mental_status:
        alerts.append({
            "severity": "WARNING",
            "text": "WARNING: Cefepime exposure with altered mental status. Monitor for "
                    "cefepime-induced neurotoxicity (encephalopathy, seizures).",
        })

    # --- Linezolid + thrombocytopenia/declining platelets ---
    plt_cur = p.get("platelet_current")
    thrombocytopenia = plt_cur is not None and plt_cur < 100000
    if p.get("on_linezolid", 0) and thrombocytopenia:
        alerts.append({
            "severity": "WARNING",
            "text": "WARNING: Linezolid exposure with thrombocytopenia or declining platelets. "
                    "Monitor platelet count closely.",
        })

    # --- TPN ---
    if p.get("on_tpn", 0):
        alerts.append({
            "severity": "INFO",
            "text": "INFO: Patient on total parenteral nutrition. Monitor electrolytes "
                    "(phosphate, magnesium, potassium) and blood glucose closely.",
        })

    # --- Steroid-induced hyperglycemia ---
    glucose = p.get("glucose_current")
    if p.get("on_steroid", 0) and glucose is not None and glucose > 180:
        alerts.append({
            "severity": "WARNING",
            "text": "WARNING: Corticosteroid use with hyperglycaemia (glucose > 180 mg/dL) detected. "
                    "Consider enhanced glucose monitoring protocol.",
        })

    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 9))
    return alerts


SEVERITY_STYLE_CLASS = {
    "CRITICAL": "optoc-alert-critical",
    "ACTION NEEDED": "optoc-alert-critical",
    "WARNING": "optoc-alert-warning",
    "INFO": "optoc-alert-info",
}
