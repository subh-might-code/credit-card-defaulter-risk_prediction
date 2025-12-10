import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load Model and Initialize App ---
MODEL_PATH = 'delinquency_model.joblib'
try:
    # Load the model directly into the Streamlit app
    model = joblib.load(MODEL_PATH)


    # Cache the model loading for performance
    @st.cache_resource
    def get_model():
        return joblib.load(MODEL_PATH)


    model = get_model()

except Exception as e:
    st.error(f"Error loading model: {e}. Ensure 'delinquency_model.joblib' is pushed to the repository.")
    model = None

st.set_page_config(page_title="Credit Risk Predictor (Cloud)", layout="centered")

# --- 2. Application Layout ---
st.title("💳 Real-Time Credit Delinquency Risk Predictor")
#st.markdown("This application provides real-time predictions directly from the hosted model.")

# --- 3. User Input Form ---
st.header("Input Customer Features")

with st.form("risk_prediction_form"):
    # F1: Current Utilization Rate
    utilization = st.slider(
        "Current Utilization Rate (Balance / Limit)",
        min_value=0.0, max_value=1.5, value=0.5, step=0.01
    )

    # F2: Minimum Payment Flag
    min_payment = st.radio(
        "Did the customer use minimum payment or pay late recently?",
        options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
    )

    # F3: Days Since Last Payment Proxy
    days_since_payment = st.number_input(
        "Repayment Status (input e.g., 0=Revolving, 2=2 months late)",
        min_value=-2, max_value=8, value=0, step=1, help="Input the numeric code for repayment status."
    )

    submitted = st.form_submit_button("Predict Risk Score")

# --- 4. Prediction Logic (Runs model locally) ---
if submitted and model is not None:
    # Prepare input data in the exact format the model expects
    input_data = pd.DataFrame({
        'Current_Utilization_Rate': [utilization],
        'Minimum_Payment_Flag': [min_payment],
        'Days_Since_Last_Payment_Proxy': [days_since_payment]
    })

    # Run the prediction directly using the loaded model
    prediction_proba = model.predict_proba(input_data)[:, 1][0]

    # Define a simple risk tier based on the prediction
    risk_tier = "Extreme Risk" if prediction_proba >= 0.80 else \
        "High Risk" if prediction_proba >= 0.60 else \
            "Monitor"

    score = round(float(prediction_proba), 4)

    # Display results
    if "Extreme" in risk_tier:
        st.error(f"⚠️ **{risk_tier}**: **{score}**")
    elif "High" in risk_tier:
        st.warning(f"🔔 **{risk_tier}**: **{score}**")
    else:
        st.success(f"✅ **{risk_tier}**: **{score}**")

    st.markdown(f"**Intervention Suggested:** The account falls into the **{risk_tier}** category.")