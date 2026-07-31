import streamlit as st

st.title("ℹ️ About Sentinel")

st.markdown("""
## Overview

Sentinel is an end-to-end machine learning system for real-time financial fraud detection using the PaySim dataset.

The project demonstrates the complete ML lifecycle, from data preprocessing and feature engineering to model deployment with a production-ready REST API and interactive dashboard.

---

## Tech Stack

- Python
- XGBoost
- SHAP
- FastAPI
- Streamlit
- Docker
- GitHub Actions
- Pandas
- Scikit-learn
- Plotly

---

## Machine Learning Pipeline

1. Data preprocessing
2. Feature engineering
3. Train / Validation split
4. XGBoost training
5. Threshold optimization
6. Model evaluation
7. SHAP explainability
8. FastAPI deployment
9. Streamlit dashboard

---

## Feature Engineering

The model uses engineered features to improve fraud detection, including:

- Amount-to-balance ratio
- Origin account emptied
- Destination balance growth ratio
- Hour of day
- Night transaction flag

These features provide behavioral signals beyond the raw transaction data.

---

## Why XGBoost?

XGBoost was selected because it performs well on structured tabular data, handles nonlinear relationships, and provides feature importance for model explainability.

---

## Deployment

The project is fully containerized using Docker.

Services:

- FastAPI inference API
- Streamlit dashboard

Continuous Integration is handled with GitHub Actions.

---

## Dataset

PaySim

A synthetic mobile money transaction dataset designed for fraud detection research.

The dataset is downloaded automatically during the first run and is therefore not stored inside the repository.

---

## Repository

GitHub: https://github.com/MiladGolchinpour/sentinel-fraud-platform

Author: Milad Golchinpour
""")