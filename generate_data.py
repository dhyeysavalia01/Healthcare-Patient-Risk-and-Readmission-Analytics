"""
Generates a realistic synthetic Healthcare Patient Risk & Readmission dataset.
5000 patient encounters across 6 departments, 3 hospital branches, 2 years.
Correlations are built in on purpose (age/comorbidity -> risk, LOS -> readmission)
so that the SQL / EDA / ML / Tableau work on top of it produces genuine insights
instead of pure noise.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

departments = ["Cardiology", "Orthopedics", "General Medicine", "Oncology", "Neurology", "Pediatrics"]
dept_weights = [0.22, 0.18, 0.25, 0.10, 0.15, 0.10]

branches = ["City Central", "Riverside", "North Campus"]
branch_weights = [0.5, 0.3, 0.2]

insurance_types = ["Private", "Government Scheme", "Self-Pay", "Corporate"]
insurance_weights = [0.35, 0.30, 0.15, 0.20]

genders = ["Male", "Female"]

admission_types = ["Emergency", "Elective", "Referral"]

# date range: 2 years
start = pd.Timestamp("2024-01-01")
end = pd.Timestamp("2025-12-31")
date_range_days = (end - start).days

rows = []
for i in range(N):
    patient_id = f"P{100000+i}"
    age = int(np.clip(np.random.normal(52, 18), 1, 95))
    gender = np.random.choice(genders, p=[0.52, 0.48])
    dept = np.random.choice(departments, p=dept_weights)
    branch = np.random.choice(branches, p=branch_weights)
    insurance = np.random.choice(insurance_types, p=insurance_weights)
    admission_type = np.random.choice(admission_types, p=[0.45, 0.40, 0.15])

    admit_offset = np.random.randint(0, date_range_days)
    admit_date = start + pd.Timedelta(days=admit_offset)

    # comorbidity count correlates with age
    base_comorbid = 0.02 * age + np.random.normal(0, 1.2)
    comorbidities = int(np.clip(round(base_comorbid), 0, 6))

    # length of stay correlates with age, comorbidities, dept
    dept_los_bump = {"Oncology": 3, "Cardiology": 2, "Neurology": 2,
                      "General Medicine": 1, "Orthopedics": 1.5, "Pediatrics": 0.5}
    los_base = 2 + 0.05 * age + 0.8 * comorbidities + dept_los_bump[dept]
    length_of_stay = int(np.clip(round(np.random.normal(los_base, 2)), 1, 45))

    discharge_date = admit_date + pd.Timedelta(days=length_of_stay)

    # billing correlates with LOS, dept, branch
    dept_cost_mult = {"Oncology": 2.4, "Cardiology": 2.0, "Neurology": 1.8,
                       "General Medicine": 1.0, "Orthopedics": 1.6, "Pediatrics": 0.9}
    base_cost_per_day = 4200 * dept_cost_mult[dept]
    billing_amount = round(max(3000, np.random.normal(base_cost_per_day * length_of_stay, base_cost_per_day * 1.5)), 2)

    # vitals
    systolic_bp = int(np.clip(np.random.normal(122 + 0.3 * age * (dept == "Cardiology"), 15), 85, 210))
    bmi = round(float(np.clip(np.random.normal(26 + 0.02*age, 5), 14, 48)), 1)
    glucose = int(np.clip(np.random.normal(105 + 1.5*comorbidities, 25), 60, 350))

    # readmission risk: logistic function of age, comorbidities, LOS, admission_type, insurance
    risk_score = (
        -4.2
        + 0.028 * age
        + 0.55 * comorbidities
        + 0.10 * length_of_stay
        + (0.6 if admission_type == "Emergency" else 0.0)
        + (0.4 if insurance == "Self-Pay" else 0.0)
        + (0.3 if dept in ["Oncology", "Cardiology"] else 0.0)
        + np.random.normal(0, 0.6)
    )
    prob_readmit = 1 / (1 + np.exp(-risk_score))
    readmitted_30d = np.random.binomial(1, np.clip(prob_readmit, 0.02, 0.95))

    satisfaction_score = int(np.clip(round(np.random.normal(
        8.2 - 0.3*comorbidities - 0.05*length_of_stay + (0.4 if admission_type == "Elective" else 0), 1.3)), 1, 10))

    rows.append([
        patient_id, age, gender, dept, branch, insurance, admission_type,
        admit_date.date().isoformat(), discharge_date.date().isoformat(), length_of_stay,
        comorbidities, systolic_bp, bmi, glucose, billing_amount,
        satisfaction_score, readmitted_30d
    ])

cols = [
    "patient_id", "age", "gender", "department", "branch", "insurance_type", "admission_type",
    "admit_date", "discharge_date", "length_of_stay_days",
    "comorbidity_count", "systolic_bp", "bmi", "glucose_level", "billing_amount",
    "satisfaction_score", "readmitted_30d"
]

df = pd.DataFrame(rows, columns=cols)

# introduce a little realistic messiness (nulls) that you clean in SQL/Python — good talking point in interviews
null_idx = np.random.choice(df.index, size=int(0.02 * N), replace=False)
df.loc[null_idx, "satisfaction_score"] = np.nan
null_idx2 = np.random.choice(df.index, size=int(0.015 * N), replace=False)
df.loc[null_idx2, "bmi"] = np.nan

df.to_csv("/home/claude/healthcare_project/healthcare_encounters.csv", index=False)
print(df.shape)
print(df.head())
print(df["readmitted_30d"].mean())
