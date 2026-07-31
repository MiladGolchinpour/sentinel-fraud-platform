import streamlit as st
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/paysim_features.parquet")

st.title("Transaction Explorer")

if DATA_PATH.exists():
    df = pd.read_parquet(DATA_PATH)

    st.dataframe(df.head())

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

else:
    st.warning(
        "Dataset is not included in deployment. "
        "Data exploration is available only in the local environment."
    )