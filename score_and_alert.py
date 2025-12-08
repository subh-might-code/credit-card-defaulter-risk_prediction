import pandas as pd
from sklearn.model_selection import train_test_split
import joblib


feature_cols = ['Current_Utilization_Rate', 'Minimum_Payment_Flag', 'Days_Since_Last_Payment_Proxy']
target_col = 'Target_Delinquent_Next_Month'


df = pd.read_csv('training_data.csv')
model = joblib.load('delinquency_model.joblib')


X = df[feature_cols]
y = df[target_col]
_, X_current, _, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
_, df_current = train_test_split(df[['ID'] + feature_cols], test_size=0.3, random_state=42, stratify=y)

print(f"Loaded {len(df_current)} accounts for scoring simulation.")


risk_scores = model.predict_proba(X_current)[:, 1]

df_current = df_current.copy()
df_current['Risk_Score'] = risk_scores


alert_threshold = df_current['Risk_Score'].quantile(0.90)
high_risk_accounts = df_current[df_current['Risk_Score'] >= alert_threshold]


final_alert_list = high_risk_accounts[['ID', 'Risk_Score']].sort_values(by='Risk_Score', ascending=False)


df_current.to_csv('full_scored_data.csv', index=False)

print("\n--- Early Warning Alert List Generated ---")
print(f"Number of accounts identified as high risk: {len(final_alert_list)}")
print("Alert List saved to: early_warning_list.csv")
print("\nTop 5 High-Risk Accounts:")
print(final_alert_list[['ID', 'Risk_Score']].head().to_string(index=False))