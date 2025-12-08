import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib


feature_cols = ['Current_Utilization_Rate', 'Minimum_Payment_Flag', 'Days_Since_Last_Payment_Proxy']
target_col = 'Target_Delinquent_Next_Month'


df = pd.read_csv('training_data.csv')

X = df[feature_cols]
y = df[target_col]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Data split into 70% Training and 30% Testing sets.")


model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

print("Model training complete.")


y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"\n--- Model Performance ---")
print(f"AUC Score on Test Data: {auc_score:.4f}")
print(f"MVP Goal of >0.70 {'Achieved' if auc_score > 0.70 else 'Not Yet Achieved'}.")
print("-------------------------")

# 5. Save Model (This is the required output for File 3)
joblib.dump(model, 'delinquency_model.joblib')
print("Model saved as delinquency_model.joblib.")