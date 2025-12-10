# Credit Card Delinquency Early Warning System

## Project Overview

**Live Demo:** https://credit-card-defaulter-riskprediction-gsudhvadgmcyyle3inbswp.streamlit.app/

This project implements a **Real-Time Credit Delinquency Risk Predictor** as a Minimum Viable Product (MVP). The primary goal was to deploy a trained Logistic Regression model into a single, publicly accessible web application.

The solution demonstrates the end-to-end process from model training to direct deployment on the Streamlit Community Cloud.

---

## Architecture

The application uses a **monolithic architecture** for simplicity and ease of deployment on the Streamlit Cloud platform.

* The Streamlit app (`app_cloud.py`) directly loads the saved model (`delinquency_model.joblib`) into memory upon startup.
* All data processing and prediction logic are contained within this single file.



---

## How to Deploy and Run

This application is designed for one-click deployment via the Streamlit Community Cloud.

### Prerequisites (For Local Testing)

To run or test locally, you need Python and all required dependencies:

```bash

pip install -r requirements.txt
```
# Local Execution (For Testing)

1) Ensure the file delinquency_model.joblib is present.

2) Run the application in your terminal:

```Bash

streamlit run app_cloud.py
```
# Cloud Deployment Steps (Streamlit Community Cloud)

1) Commit Files:Ensure the following files are pushed to this public GitHub repository:

* app_cloud.py

* delinquency_model.joblib

* requirements.txt

2) Deploy: Go to the Streamlit Community Cloud, click "New App," and point to this repository with the main file set to app_cloud.py.

## Model Details and Lineage
1) Model Type	- Logistic Regression
2) Problem Type	- Binary Classification (Delinquent / Not Delinquent)
3) Objective	-Maximize the Area Under the ROC Curve (AUC)
4) Training Data -	Cleaned and feature-engineered version of the UCI default of credit card clients dataset.
5) Key Parameters -	Default scikit-learn parameters (e.g., L2 regularization).
6) Deployment File  -	delinquency_model.joblib

## Core Features and Performance
### Features Used
The MVP model relies on three highly predictive features:
* Current_Utilization_Rate (Most Recent Bill Amount / Credit Limit)

* Minimum_Payment_Flag (Proxy for financial stress)

* Days_Since_Last_Payment_Proxy (The most recent numeric repayment status code)

### Model Performance
* Evaluation Metric: Area Under the ROC Curve (AUC)
* Result: AUC approx **~0.7097** on the test set.

## Potential Enhancements
### To upgrade this MVP:
1) Model Upgrade: Transition to a more powerful algorithm like XGBoost to boost the AUC score.

2) Advanced Features: Implement velocity features (tracking rate of change in balances/payments).

3) Scalability: Revert to the decoupled FastAPI/Streamlit architecture for production environments.