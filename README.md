# Optoc AI Pharmacist Dashboard

Built from `TOC_Specifications_v4.docx`, the 18 pre-trained model files in
`models.zip`, and the sample data your professor provided
(`optoc_sample_patients.csv`, `optoc_timeseries_vitals.csv`).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**Read the scikit-learn version note in `requirements.txt` before you `pip install` anything else in this environment** — the supplied `.pkl` files were pickled with scikit-learn 1.5.1, and a newer scikit-learn breaks unpickling (`AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`). This was caught by actually running the pipeline against a newer sklearn during testing, not assumed.

## What's real vs. simplified in this build

Every render function, the full feature-engineering pipeline, and all 4 XGBoost models were actually executed end-to-end against the sample data before this was packaged (not just written and assumed to work). Two things are simplified due to scope:

- **PDF "Print Summary" button** (Tab 2) is a placeholder — no PDF-rendering library was wired in.
- **Tab 5 manual entry form** covers the core Tier-1 fields but not the full 28-checkbox comorbidity list or the Tier-2 time-series add-row tables. CSV upload supports the full schema already, so it's the more complete path today.

## Discrepancies found between the spec, the models, and the sample data

These are worth a quick conversation with the professor/Buchi — none of them blocked the build (the app runs and produces real predictions), but they affect what the numbers mean.

1. **`model_metadata.csv` names two files that don't exist.** It lists Sepsis's best model as `sepsis_rf_model.pkl` (Random Forest) and Mortality's as `mortality_rf_model.pkl` — neither file is in `models.zip`. Only `sepsis_xgb_model.pkl` and `mortality_xgb_model.pkl` exist, and loading them confirms both are actually `XGBClassifier` pipelines. This app uses the XGBoost files that were actually supplied for all 4 outcomes (matching Appendix A3's own example code) and keeps the Youden thresholds from the metadata CSV, since those are keyed by outcome name independent of the filename issue. Either the metadata is stale, or two files are missing from the handoff.

2. **Feature count: spec prose says 38, the actual file has 35.** `required_feature_columns.txt` — the technical ground truth — lists 35 columns. This app uses the file, not the prose number.

3. **Domain-scoring threshold logic (flagging my own earlier work):** an earlier prototype I built normalized each domain's Low/Moderate/High cutoff as a % of that domain's max possible score. This spec explicitly says the opposite — "*The score-to-label conversion is the same for all 7 domains*," using the same absolute 0–3/4–6/7+ cutoffs everywhere. This build follows the spec, not the earlier prototype. Said plainly: I changed my mind based on your professor's authoritative document, not because the earlier reasoning was wrong in general — just overridden here.

4. **Two independent explanation methods, resolved.** The spec's Tab 2 asks for "Contributing Factors" and a "Personalized Risk Summary... generated with a second, independent explanation method as a cross-check on the first (Powered by LIME)," but only names LIME explicitly, leaving the first method unspecified. Since a per-instance SHAP value (not just the population-mean SHAP importance Tab 3 uses) is available from the same pre-fitted explainer, this build uses **SHAP** for "Contributing Factors" and **LIME** for "Personalized Risk Summary" — two genuinely different, real methods, matching the same SHAP/LIME split the spec already uses on Tab 3.

## Data Coverage — fields the sample CSV doesn't have

`optoc_sample_patients.csv` doesn't include every field the 7-domain rules and medication alerts reference. Every scoring/alert function uses `.get(field, default)`, so missing fields just mean that specific condition can't fire (0 points / no alert) rather than crashing — but it also means some domains and alerts will under-report on **this sample data specifically**, not because patients are actually low-risk there. Once the real census data includes these columns, the same code picks them up automatically — no changes needed.

Missing from the sample data: platelet count (current + 48h-ago), INR, measured QTc (ms — only a medication-exposure flag exists), sodium, glucose baseline, bilirubin, primary diagnosis / sepsis flag, prior-AKI history, OSA history, CAD/AFib history, benzodiazepine-for-sedation flag, SBT readiness, per-drug vancomycin/piperacillin-tazobactam/gabapentin/metformin/warfarin/DOAC "active" flags, a numeric CrCl value, and a count of new medications started during the stay.

Because of this, on the current sample data: the Hematology domain will read entirely LOW (no platelet/INR data at all), several Cardio and Renal rules are inactive, and most of the medication-alert table's specific drug-combination rules (vancomycin+pip-tazo, vancomycin+loop diuretic, CrCl+gabapentin, CrCl+metformin) won't fire. The rules everyone actually asked about — cefepime+altered mental status, QTc-prolonging meds, anticholinergic burden, TPN, steroid hyperglycemia, polypharmacy, AKI+nephrotoxin — all work correctly against this sample data.

## One data observation, not a bug

AKI risk comes back very high (0.87–0.997) for essentially every sample patient. That might just mean this 10-patient sample is a deliberately high-acuity test batch — but it's worth a sanity check with whoever generated the sample data before trusting the AKI numbers at face value.

## Ember UI note

The visual style (dark header bar, card layout with colored left accents, pill-shaped risk badges) is inspired by the Ember dashboard template you shared — recreated with Streamlit-native CSS, not copied Ember markup/CSS, since Ember is a paid commercial template (dashboardpack.com) and its actual code would need a license to reuse directly.

## Project layout

```
optoc_dashboard/
├── app.py                        # Entry point — session-state-driven top nav
├── app_pages/
│   ├── home.py                   # Tab 1: Homepage
│   ├── individual_patient.py     # Tab 2: Individual Patient Profile
│   ├── risk_profile.py           # Tab 3: Risk Profile (4-outcome selector)
│   ├── domain_risk.py            # Tab 4: Domain Specific Risk (heatmap)
│   └── try_your_data.py          # Tab 5: Try Your Data! (CSV upload / manual entry)
├── core/
│   ├── features.py               # build_model_features() — raw fields -> 35 model columns
│   ├── model_registry.py         # Loads all 18 pkl files, cached at startup
│   ├── explainers.py             # SHAP (population + per-patient) and LIME wrappers
│   ├── scoring.py                # 7-domain rule engine, exact spec point values
│   ├── med_alerts.py             # Exact medication alert rule table from the spec
│   ├── data_io.py                # Central data + prediction pipeline every tab shares
│   └── theme.py                  # Exact color standards from spec + Ember-inspired CSS
├── data/                         # optoc_sample_patients.csv, optoc_timeseries_vitals.csv
├── models/                       # All 18 supplied .pkl/.csv/.txt files
└── requirements.txt
```
