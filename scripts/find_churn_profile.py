"""Find a payload that triggers churn prediction."""
import warnings
warnings.filterwarnings("ignore")
import joblib
import pandas as pd
import numpy as np

model = joblib.load("models/best_random_forest.joblib")

features = list(model.named_steps["selector"].feature_names_in_)
mask = model.named_steps["selector"].get_support()
selected = [f for f, m in zip(features, mask) if m]
rf = model.named_steps["model"]

imp = pd.Series(rf.feature_importances_, index=selected)
print("Top 10 features by importance:")
for name, val in imp.sort_values(ascending=False).head(10).items():
    print(f"  {val:.4f}  {name}")

print()

# Profile: high churn risk
row = {f: 0 for f in features}
row["Churn Score"] = 95
row["CLTV"] = 2000
row["Tenure Months"] = 1
row["Monthly Charges"] = 105.0
row["Total Charges"] = 105.0
row["Churn Label_Yes"] = 1
row["Internet Service_Fiber optic"] = 1
row["Internet Service_No"] = 0
row["Online Security_No internet service"] = 0
row["Online Security_Yes"] = 0
row["Online Backup_No internet service"] = 0
row["Online Backup_Yes"] = 0
row["Device Protection_No internet service"] = 0
row["Device Protection_Yes"] = 0
row["Tech Support_No internet service"] = 0
row["Tech Support_Yes"] = 0
row["Streaming TV_No internet service"] = 0
row["Streaming TV_Yes"] = 0
row["Streaming Movies_No internet service"] = 0
row["Streaming Movies_Yes"] = 0
row["Contract_One year"] = 0
row["Contract_Two year"] = 0
row["Paperless Billing_Yes"] = 1
row["Payment Method_Electronic check"] = 1
row["Payment Method_Credit card (automatic)"] = 0
row["Engineered Monthly Charges"] = 105.0 ** 2
row["charge_rel"] = 105.0 / (105.0 + 1) - 1

df = pd.DataFrame([row])[features]
proba = model.predict_proba(df)[0, 1]
print(f"Churn probability: {proba:.4f}")
label = "CHURN" if proba >= 0.5 else "NO CHURN"
print(f"Prediction: {label}")

if proba < 0.5:
    print("\nTrying with Churn Reason features...")
    row["Churn Reason_Network reliability"] = 1
    row["Churn Reason_Product dissatisfaction"] = 1
    df2 = pd.DataFrame([row])[features]
    proba2 = model.predict_proba(df2)[0, 1]
    label2 = "CHURN" if proba2 >= 0.5 else "NO CHURN"
    print(f"Churn probability: {proba2:.4f}")
    print(f"Prediction: {label2}")
