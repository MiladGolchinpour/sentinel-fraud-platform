import streamlit as st
import json

st.title("📊 Model Performance")
st.caption("Evaluation metrics computed on the held-out test set using the final XGBoost model.")

with open("models/metrics.json") as f:
    metrics = json.load(f)

st.divider()

st.subheader("Metric Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Precision", f"{metrics['precision']*100:.2f}%")
col2.metric("Recall", f"{metrics['recall']*100:.2f}%")
col3.metric("F1 Score", f"{metrics['f1']*100:.2f}%")

col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC", f"{metrics['roc_auc']*100:.2f}%")
col2.metric("PR-AUC", f"{metrics['pr_auc']*100:.2f}%")
col3.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")

st.divider()

with open("models/config.json") as f:
    config = json.load(f)

col1, col2 = st.columns(2)

with col1:
    st.metric("Model", config["model"].upper())
    st.metric("Features", config["features"])
with col2:
    st.metric("Threshold", config["threshold"])
    st.metric("Version", config["version"])

