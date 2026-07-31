import streamlit as st
import os, requests, json, random

API_URL = os.getenv("API_URL","http://localhost:8000/predict")
HEALTH_URL = os.getenv("HEALTH_URL","http://localhost:8000/status")

with open("models/demo_samples.json") as f:
    SAMPLES = json.load(f)

if "initialized" not in st.session_state:
    default = random.choice(list(SAMPLES.values()))
    for key, value in default.items():
        st.session_state[key] = value
    st.session_state.initialized = True

st.set_page_config(
    page_title="Sentinel Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

try:
    status_code = requests.get(HEALTH_URL, timeout=2).status_code
except:
    status_code = 500
status_icon = ("🟢" if status_code == 200 else "🔴")
status_explain = ("Connected" if status_code == 200 else "Disconnected")

st.title(f"🛡️ Sentinel Fraud Detection - {status_icon} API {status_explain}")
st.caption("Real-time fraud detection system powered by XGBoost")

st.subheader("Try Sample Transactions")

if st.button("🎲 Random Sample", width="stretch"):
    st.session_state.pending_sample = random.choice(list(SAMPLES.values()))
    st.session_state.auto_predict = True
    st.rerun()

cols = st.columns(5)

for col, (name, sample) in zip(cols, SAMPLES.items()):
    if col.button(name, width="stretch"):
        st.session_state.pending_sample = sample
        st.session_state.auto_predict = True
        st.rerun()

st.sidebar.header("Transaction")

if "pending_sample" in st.session_state:
    for key, value in st.session_state.pending_sample.items():
        st.session_state[key] = value

    del st.session_state.pending_sample

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"],
    key="type",
)
amount = st.sidebar.number_input("Amount", min_value=0.0, key="amount")
old_balance = st.sidebar.number_input("Old Balance Origin", min_value=0.0, key="oldbalanceOrg")
new_balance = st.sidebar.number_input("New Balance Origin", min_value=0.0,key="newbalanceOrig")
old_dest = st.sidebar.number_input("Old Balance Destination", min_value=0.0, key="oldbalanceDest")
new_dest = st.sidebar.number_input("New Balance Destination", min_value=0.0, key="newbalanceDest")
step = st.sidebar.number_input("Step", min_value=0, key="step")

def predict_transaction(data=None):
    if data is None:
        data = {
            "type": st.session_state["type"],
            "amount": st.session_state["amount"],
            "oldbalanceOrg": st.session_state["oldbalanceOrg"],
            "newbalanceOrig": st.session_state["newbalanceOrig"],
            "oldbalanceDest": st.session_state["oldbalanceDest"],
            "newbalanceDest": st.session_state["newbalanceDest"],
            "step": st.session_state["step"]
        }

    response = requests.post(API_URL, json=data)

    if response.status_code == 200:
        result = response.json()
        probability = result["fraud_probability"]
        col1, col2, col3 = st.columns(3)

        st.progress(min(probability, 1.0))

        with col1:
            st.metric("Prediction", result["prediction"].title())
        with col2:
            st.metric("Fraud Probability", f"{probability:.2%}")
        with col3:
            st.metric("Risk Level", result["risk_level"].title())

        if result["prediction"] == "fraud":
            st.error("🚨 Fraud detected")
            st.subheader("Why?")
            if result["top_reasons"]:
                for reason in result["top_reasons"]:
                    st.write(f"⚠️ {reason}")
            else:
                st.write("No strong fraud indicators detected.")
        else:
            st.success("✅ Legitimate transaction")
    else:
        st.error("API Error")


if st.session_state.get("auto_predict"):
    predict_transaction()
    st.session_state.auto_predict = False

if st.sidebar.button("Predict", width="stretch"):
    predict_transaction()