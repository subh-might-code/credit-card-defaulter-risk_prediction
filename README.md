# Credit Card Delinquency Early Warning System (Full-Stack MVP)

## Problem statement:
Early risk signals: credit card deliquency watch

## Project Overview
This project implements a **real-time machine learning service** that predicts the probability of **credit card delinquency**.  
It is built as a **Full-Stack Minimum Viable Product (MVP)** with a clean, decoupled architecture.

The solution demonstrates how to take a **Logistic Regression model** and deploy it as a robust API service with an **interactive web frontend**.

---

## System Architecture

The system follows a **three-tier architecture**:

### **1. Backend API (FastAPI)**
- Hosts the trained Logistic Regression model.
- Exposes a `/predict` HTTP endpoint.
- Loads model from `.joblib` storage.

### **2. Frontend App (Streamlit)**
- Interactive UI for entering user inputs.
- Sends requests to FastAPI server.
- Displays final delinquency risk score.

### **3. Data Persistence**
- `joblib` is used for model serialization.
- Dependencies managed via `requirements.txt`.

---

## Local Deployment Guide

This application requires **two terminals** running simultaneously.

### **Prerequisites**
- Python installed
- Docker Desktop (optional for containerized execution)

Install required dependencies:
```bash
pip install -r requirements.txt
```

### Step 1: Start the Backend API (Terminal 1)

The FastAPI server must be running to serve predictions.

Run:
```bash
uvicorn api_service:app --reload
```

Expected:
Server starts at → http://127.0.0.1:8000

### Step 2: Start the Frontend App (Terminal 2)

Run the Streamlit app:

```bash
streamlit run app_frontend.py
```

Expected:
Browser opens Streamlit UI → http://localhost:8501

## Project Structure
api_service.py - FastAPI backend to load model, expose /predict

app_frontend.py -	Streamlit frontend for web interaction

requirements.txt -	Python dependencies

delinquency_model.joblib -	Serialized Logistic Regression model


## Core ML Features & Modeling
Features Used

1) Current_Utilization_Rate
(Most Recent Bill Amount / Credit Limit)

2) Minimum_Payment_Flag
(Indicator of financial stress)

3) Days_Since_Last_Payment_Proxy
(Latest numeric repayment status code)

## Model Performance

Model: Logistic Regression

Metric: AUC (Area Under ROC Curve)

Score: ≈ 0.7097 on test data

## Future Enhancements

To make this system enterprise-ready:

1) Model Improvements

Switch to XGBoost for higher accuracy.

2) Feature Engineering

Add velocity features (rate-of-change patterns).

3) Security

Add API Key authentication to FastAPI routes.

## Summary

This MVP provides a clean, scalable, and production-style design for real-time credit risk prediction - combining FastAPI, Streamlit, and a machine learning model into a cohesive full-stack workflow.
