import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np  # Added for robust type handling

# --- 1. Load Model and Initialize API ---
# Load the model once when the application starts
MODEL_PATH = 'delinquency_model.joblib'
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}. Ensure 'delinquency_model.joblib' is in this directory.")
    model = None

# Initialize FastAPI application. The object name MUST be 'app'.
app = FastAPI(
    title="Delinquency Risk Prediction API",
    version="1.0",
    description="Backend API for real-time credit card delinquency risk prediction."
)


# --- 2. Define Input Data Schema ---
# This Pydantic class ensures the data sent to the API is structured correctly.
class PredictionRequest(BaseModel):
    Current_Utilization_Rate: float = 0.50  # Default for example
    Minimum_Payment_Flag: int = 1  # Default for example (1=paid minimum or late)
    Days_Since_Last_Payment_Proxy: int = 0  # Default for example (0=used revolving credit)


# --- 3. Define the Prediction Endpoint ---
@app.post("/predict")
def predict_risk(data: PredictionRequest):
    if model is None:
        return {"error": "Model failed to load. Check server console for details."}

    # Prepare input data in the exact format the model expects
    input_data = pd.DataFrame({
        'Current_Utilization_Rate': [data.Current_Utilization_Rate],
        'Minimum_Payment_Flag': [data.Minimum_Payment_Flag],
        'Days_Since_Last_Payment_Proxy': [data.Days_Since_Last_Payment_Proxy]
    })

    # Run the prediction
    # Get the probability for the positive class (class 1: delinquent)
    prediction_proba = model.predict_proba(input_data)[:, 1][0]

    # Define a simple risk tier based on the prediction
    risk_tier = "Extreme Risk" if prediction_proba >= 0.80 else \
        "High Risk" if prediction_proba >= 0.60 else \
            "Monitor"

    # Format the response
    return {
        "prediction_probability": round(float(prediction_proba), 4),
        "risk_tier": risk_tier,
        "features_used": data.dict()
    }


# --- 4. Define Health Check Endpoint (Optional but Recommended) ---
@app.get("/")
def read_root():
    return {"status": "API is running", "model_loaded": model is not None}