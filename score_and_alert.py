import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Define the features (must match the features used for training!)
feature_cols = ['Current_Utilization_Rate', 'Minimum_Payment_Flag', 'Days_Since_Last_Payment_Proxy']
target_col = 'Target_Delinquent_Next_Month'

# 1. Load Data and Model
df = pd.read_csv('training_data.csv')
model = joblib.load('delinquency_model.joblib')

# 2. Separate the Test Data (Simulating new, unseen accounts)
# We recreate the exact 30% split from the training script for consistency
X = df[feature_cols]
y = df[target_col]
_, X_current, _, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
_, df_current = train_test_split(df[['ID'] + feature_cols], test_size=0.3, random_state=42, stratify=y)

print(f"Loaded {len(df_current)} accounts for scoring simulation.")

# 3. Generate Risk Scores
# Predict the probability of delinquency (class 1)
risk_scores = model.predict_proba(X_current)[:, 1]

df_current = df_current.copy()
df_current['Risk_Score'] = risk_scores

# 4. Generate Alert List (Top 10% highest risk scores)
# The 0.90 quantile finds the score cutoff for the riskiest 10% of customers
alert_threshold = df_current['Risk_Score'].quantile(0.90)
high_risk_accounts = df_current[df_current['Risk_Score'] >= alert_threshold]

# Select and sort the final list for the operations team
final_alert_list = high_risk_accounts[['ID', 'Risk_Score']].sort_values(by='Risk_Score', ascending=False)

# 5. Save Alert List (The final project deliverable)
df_current.to_csv('full_scored_data.csv', index=False)

print("\n--- Early Warning Alert List Generated ---")
print(f"Number of accounts identified as high risk: {len(final_alert_list)}")
print("Alert List saved to: early_warning_list.csv")
print("\nTop 5 High-Risk Accounts:")
print(final_alert_list[['ID', 'Risk_Score']].head().to_string(index=False))