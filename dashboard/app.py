import streamlit as st
import requests
import os
API_URL = os.getenv("API_URL","http://localhost:8000/predict")

st.set_page_config(
    page_title="Sentinel Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Sentinel Fraud Detection")
st.write("Real-time fraud detection system powered by XGBoost")
st.sidebar.header("Transaction")

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    [
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "DEBIT"
    ]
)

amount = st.sidebar.number_input("Amount", min_value=0.0, value=1000.0)
old_balance = st.sidebar.number_input("Old Balance Origin", min_value=0.0, value=5000.0)
new_balance = st.sidebar.number_input("New Balance Origin", min_value=0.0, value=4000.0)
old_dest = st.sidebar.number_input("Old Balance Destination", min_value=0.0, value=0.0)
new_dest = st.sidebar.number_input("New Balance Destination", min_value=0.0, value=1000.0)
step = st.sidebar.number_input("Step", min_value=0, value=100)

if st.sidebar.button("Predict"):
    payload = {
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": new_balance,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "step": step
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        probability = result["fraud_probability"]
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediction", result["prediction"])
        with col2:
            st.metric("Fraud Probability", f"{probability:.2%}")
        with col3:
            st.metric("Risk Level", result["risk_level"])

        st.subheader("Why?")

        if result["top_reasons"]:
            for reason in result["top_reasons"]:
                st.write(f"⚠️ {reason}")
        else:
            st.write("No strong fraud indicators detected.")
    else:
        st.error("API Error")