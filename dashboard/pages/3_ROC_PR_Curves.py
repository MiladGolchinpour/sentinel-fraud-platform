import json
import plotly.graph_objects as go
import streamlit as st

st.title("📈 ROC & Precision-Recall Curves")

with open("models/curves.json") as f:
    curves = json.load(f)

# ---------------- ROC ----------------

st.subheader("ROC Curve")

roc_fig = go.Figure()

roc_fig.add_trace(
    go.Scatter(
        x=curves["roc"]["fpr"],
        y=curves["roc"]["tpr"],
        mode="lines",
        name="Model",
    )
)

roc_fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(dash="dash"),
    )
)

roc_fig.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    template="plotly_white",
)

st.plotly_chart(roc_fig, width="stretch")

# ---------------- PR ----------------

st.subheader("Precision-Recall Curve")

pr_fig = go.Figure()

pr_fig.add_trace(
    go.Scatter(
        x=curves["pr"]["recall"],
        y=curves["pr"]["precision"],
        mode="lines",
        name="Model",
    )
)

pr_fig.update_layout(
    xaxis_title="Recall",
    yaxis_title="Precision",
    template="plotly_white",
)

st.plotly_chart(pr_fig, width="stretch")