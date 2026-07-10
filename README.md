# Healthcare-Patient-Risk-and-Readmission-Analytics

An end-to-end data analyst portfolio project: SQL → Python (EDA + ML) → Tableau dashboard.
Built to demonstrate the full analyst workflow a fresher is expected to show: data cleaning,
querying, statistical/ML analysis, and business-facing visualization.

## Problem Statement
A hospital network wants to reduce 30-day readmissions. This project identifies which patient
segments (age, comorbidities, department, insurance type) carry the highest readmission risk,
builds a predictive risk score, and presents it as an interactive dashboard for the ops team.

## Dataset
`healthcare_encounters.csv` — 5,000 synthetic-but-realistic patient encounters (2024-2025) across
3 hospital branches and 6 departments, with demographics, vitals, billing, and readmission outcome.
Realistic nulls and correlations are built in on purpose (see `generate_data.py`).

## Project Structure
```
healthcare_project/
├── generate_data.py          # synthetic dataset generator
├── healthcare_encounters.csv # raw dataset
├── sql_analysis.sql          # schema, cleaning, 6 business-question queries (MySQL)
├── analysis_and_model.py     # pandas EDA + Random Forest readmission risk model
├── tableau_ready.csv         # scored dataset (output of the model) — connect Tableau here
├── feature_importance.csv    # top risk drivers from the model
└── README.md
```

## How to Run
1. `pip install pandas numpy scikit-learn`
2. `python generate_data.py` → creates `healthcare_encounters.csv`
3. Load `sql_analysis.sql` into MySQL Workbench (or any MySQL instance) and run through it —
   this is your SQL portfolio piece (window functions, LOD-style aggregations, NULL handling,
   CHECK constraints).
4. `python analysis_and_model.py` → runs EDA, trains the model, outputs `tableau_ready.csv`.
5. Open Tableau → follow `TABLEAU_GUIDE.md` → publish to Tableau Public.

## Key Findings (from this run)
- Overall 30-day readmission rate: **31.9%**
- Oncology and Cardiology have the highest readmission rates (~36-40%)
- Comorbidity count, age, and length of stay are the strongest predictors (correlation ~0.28-0.30)
- Random Forest risk model achieves **ROC-AUC 0.74**
- ~14% of patients fall into the "High Risk" tier and are flagged for proactive follow-up

## Tech Stack
SQL (MySQL) · Python (pandas, scikit-learn) · Tableau · Git/GitHub

## Resume Bullets (copy-paste and tweak)
- Built an end-to-end healthcare analytics pipeline (SQL → Python → Tableau) analyzing 5,000
  patient encounters to identify 30-day readmission drivers, achieving 0.74 ROC-AUC with a
  Random Forest model.
- Wrote SQL queries using window functions (RANK, ROW_NUMBER, rolling averages) and LOD-style
  aggregations to surface department- and segment-level readmission patterns.
- Designed an interactive Tableau dashboard with LOD expressions, parameter-driven metrics, and
  dashboard actions to help clinical ops teams prioritize high-risk patients for follow-up.

## Next Steps (say these in interviews to show initiative)
- Add a cost-savings estimate: $ saved per prevented readmission × high-risk patients flagged.
- A/B test framing: simulate what readmission rate would look like if top-risk patients got
  a follow-up call (needs an assumed intervention effect size).
- Swap Random Forest for XGBoost + SHAP values for more interpretable per-patient explanations.


## 📊 Live Dashboard

View the interactive Tableau dashboard here:

**🔗 [Hospital Readmission Risk & Cost Analytics Dashboard](https://public.tableau.com/views/HospitalReadmissionRiskCostAnalyticsDashboard/Dashboard1)**
