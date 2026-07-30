import streamlit as st
import pandas as pd

st.title("🔎 Transaction Explorer")
df = pd.read_parquet("data/processed/paysim_features.parquet")
total = len(df)
fraud = df["isFraud"].sum()
rate = fraud / total
col1, col2, col3 = st.columns(3)

col1.metric("Transactions", f"{total:,}")
col2.metric("Fraud Cases", f"{fraud:,}")
col3.metric("Fraud Rate", f"{rate:.3%}")

st.subheader("Transaction Types")
st.bar_chart(df["type"].value_counts())
st.subheader("Amount Distribution")
st.line_chart(df["amount"].describe())