"""
Healthcare Patient Risk & Readmission Analytics
Python: cleaning, EDA, and a readmission-risk ML model.
Outputs `tableau_ready.csv` — the file you connect Tableau to.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

pd.set_option("display.max_columns", None)

# ---------- 1. LOAD & CLEAN ----------
df = pd.read_csv("healthcare_encounters.csv", parse_dates=["admit_date", "discharge_date"])

print("Shape:", df.shape)
print("\nNulls:\n", df.isnull().sum())

# Impute missing bmi / satisfaction_score with department-level median
df["bmi"] = df.groupby("department")["bmi"].transform(lambda x: x.fillna(x.median()))
df["satisfaction_score"] = df.groupby("department")["satisfaction_score"].transform(lambda x: x.fillna(x.median()))

# Feature engineering
df["age_bucket"] = pd.cut(df["age"], bins=[0, 30, 50, 70, 120],
                           labels=["<30", "30-50", "51-70", "70+"])
df["admit_month"] = df["admit_date"].dt.to_period("M").astype(str)
df["is_high_cost"] = (df["billing_amount"] > df["billing_amount"].quantile(0.75)).astype(int)
df["high_comorbidity"] = (df["comorbidity_count"] >= 3).astype(int)

# ---------- 2. EDA SUMMARY (talking points for your resume/interview) ----------
print("\nOverall readmission rate: {:.1f}%".format(100 * df["readmitted_30d"].mean()))
print("\nReadmission rate by department:\n",
      df.groupby("department")["readmitted_30d"].mean().sort_values(ascending=False) * 100)
print("\nCorrelation of numeric features with readmission:\n",
      df[["age", "comorbidity_count", "length_of_stay_days", "billing_amount",
          "readmitted_30d"]].corr()["readmitted_30d"].sort_values(ascending=False))

# ---------- 3. RISK PREDICTION MODEL ----------
features_num = ["age", "comorbidity_count", "length_of_stay_days", "systolic_bp",
                 "bmi", "glucose_level", "billing_amount"]
features_cat = ["gender", "department", "branch", "insurance_type", "admission_type"]
target = "readmitted_30d"

X = df[features_num + features_cat]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat)
], remainder="passthrough")

model = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42))
])

model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_test)[:, 1]
pred = model.predict(X_test)

print("\nROC-AUC:", round(roc_auc_score(y_test, pred_proba), 3))
print("\nClassification report:\n", classification_report(y_test, pred))

# Feature importance
ohe_cols = model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(features_cat)
all_cols = list(ohe_cols) + features_num
importances = pd.Series(model.named_steps["clf"].feature_importances_, index=all_cols).sort_values(ascending=False)
print("\nTop 10 features driving readmission risk:\n", importances.head(10))

# ---------- 4. SCORE THE FULL DATASET → EXPORT FOR TABLEAU ----------
df["predicted_readmit_risk"] = model.predict_proba(X)[:, 1]
df["risk_tier"] = pd.cut(df["predicted_readmit_risk"], bins=[0, 0.33, 0.66, 1.0],
                          labels=["Low", "Medium", "High"])

df.to_csv("tableau_ready.csv", index=False)
importances.head(15).reset_index().rename(columns={"index": "feature", 0: "importance"}) \
    .to_csv("feature_importance.csv", index=False)

print("\nSaved tableau_ready.csv and feature_importance.csv")
print(df["risk_tier"].value_counts())
