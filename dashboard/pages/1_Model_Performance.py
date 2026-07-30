import streamlit as st
import json

st.title("📊 Model Performance")
with open("models/metrics.json") as f:
    metrics = json.load(f)

col1, col2, col3 = st.columns(3)
col1.metric("Precision", f"{metrics['precision']:.4f}")
col2.metric("Recall", f"{metrics['recall']:.4f}")
col3.metric("F1 Score", f"{metrics['f1']:.4f}")

col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
col2.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
col3.metric("Accuracy", f"{metrics['accuracy']:.4f}")