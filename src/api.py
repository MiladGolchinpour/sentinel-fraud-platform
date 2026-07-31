import joblib, json
import pandas as pd
from fastapi import FastAPI
from src.features import build_features
from src.schemas import PredictionResponse, Transaction

with open("models/config.json") as f:
    config = json.load(f)

THRESHOLD = config["threshold"]

app = FastAPI(
    title="Sentinel Fraud Detection API",
    version="0.1.0",
)

# load model
pipeline = joblib.load("models/xgb.pkl")
explainer = joblib.load("models/explainer.pkl")

@app.get("/status")
def status():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    # convert request to dataframe
    df = pd.DataFrame([transaction.model_dump()])

    # feature engineering
    df = build_features(df)

    # predict
    probability = float(pipeline.predict_proba(df)[0, 1])

    # shap
    X_transformed = pipeline.named_steps["preprocess"].transform(df)
    shap_values = explainer.shap_values(X_transformed)

    FRIENDLY_NAMES = {
        # transaction type
        "type_CASH_IN": "Cash deposit transaction",
        "type_CASH_OUT": "Cash withdrawal transaction",
        "type_DEBIT": "Debit transaction",
        "type_PAYMENT": "Payment transaction",
        "type_TRANSFER": "Transfer transaction",

        # original features
        "step": "Transaction timing",
        "amount": "High transaction amount",
        "oldbalanceOrg": "High origin account balance",
        "newbalanceOrig": "Low remaining origin balance",
        "oldbalanceDest": "Destination account balance",
        "newbalanceDest": "High destination account balance",

        # other features
        "amount_to_balance_ratio": "Large transaction relative to account balance",
        "origin_account_emptied": "Origin account emptied",
        "destination_balance_growth_ratio": "Large destination balance increase",
        "hour_of_day": "Unusual transaction hour",
        "is_night_transaction": "Night-time transaction",
    }

    # classification
    prediction = (
        "fraud"
        if probability >= THRESHOLD
        else "legitimate"
    )

    # risk level
    if probability < 0.30:
        risk = "low"
    elif probability < 0.70:
        risk = "medium"
    else:
        risk = "high"

    # shap
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame({
        "feature": feature_names,
        "impact": shap_values[0],
    })
    
    # only keep features that increase fraud risk
    importance = importance[importance["impact"] > 0]
    importance = importance.sort_values("impact", ascending=False)
    reasons = []
    for _, row in importance.head(5).iterrows():
        name = FRIENDLY_NAMES.get(row["feature"], row["feature"])
    def clean_feature_name(name):
        return name.replace("remainder__", "").replace("categorical__", "")
    feature = clean_feature_name(row["feature"])
    reasons.append(FRIENDLY_NAMES.get(feature, feature))

    return PredictionResponse(
        prediction=prediction,
        fraud_probability=probability,
        risk_level=risk,
        top_reasons=reasons,
    )
