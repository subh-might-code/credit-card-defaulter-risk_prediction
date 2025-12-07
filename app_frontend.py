import streamlit as st
import requests
import json

# The URL of your running FastAPI service endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

# --- 1. Application Layout ---
st.title("💳 Real-Time Credit Delinquency Risk Predictor")
st.markdown("Use this tool to get an immediate risk score from the deployed machine learning model.")

# --- 2. User Input Form ---
st.header("Input Customer Features")

# Create three input fields that mirror the features the model was trained on.
with st.form("risk_prediction_form"):
    # F1: Current Utilization Rate (Ratio: Balance / Limit)
    utilization = st.slider(
        "Current Utilization Rate (Balance / Limit)",
        min_value=0.0, max_value=1.5, value=0.5, step=0.01
    )

    # F2: Minimum Payment Flag (Binary: 1=Minimum/Late, 0=Paid on time/in full)
    min_payment = st.radio(
        "Did the customer use minimum payment or pay late recently?",
        options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
    )

    # F3: Days Since Last Payment Proxy (Repayment Status)
    # Using the same scale as the PAY_0 column (-2 to 8)
    days_since_payment = st.number_input(
        "Repayment Status (PAY_0 code: e.g., 0=Revolving, 2=2 months late)",
        min_value=-2, max_value=8, value=0, step=1, help="Input the numeric code for repayment status."
    )

    # Submit button
    submitted = st.form_submit_button("Predict Risk Score")

# --- 3. Prediction Logic (Calls the API) ---
if submitted:
    # 3a. Prepare data payload to match the FastAPI Pydantic schema
    payload = {
        "Current_Utilization_Rate": utilization,
        "Minimum_Payment_Flag": min_payment,
        "Days_Since_Last_Payment_Proxy": days_since_payment
    }

    # 3b. Call the FastAPI endpoint
    try:
        response = requests.post(API_URL, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raise exception for bad status codes (4xx or 5xx)

        # 3c. Process and display results
        result = response.json()

        score = result["prediction_probability"]
        tier = result["risk_tier"]

        # Color coding the output based on risk tier
        if "Extreme" in tier:
            st.error(f"⚠️ **{tier}**: **{score:.4f}**")
        elif "High" in tier:
            st.warning(f"🔔 **{tier}**: **{score:.4f}**")
        else:
            st.success(f"✅ **{tier}**: **{score:.4f}**")

        st.markdown(f"**Intervention Suggested:** The account falls into the **{tier}** category.")

    except requests.exceptions.ConnectionError:
        st.error("❌ **Connection Error:** Could not connect to the Prediction API.")
        st.info("Please ensure your FastAPI server is running at http://127.0.0.1:8000 in a separate terminal.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")