import joblib
import pandas as pd
from src.features import build_features

def test_model_prediction():
    model = joblib.load("models/xgb.pkl")

    sample = pd.DataFrame({
        "type": ["TRANSFER"],
        "amount": [500000],
        "oldbalanceOrg": [500000],
        "newbalanceOrig": [0],
        "oldbalanceDest": [0],
        "newbalanceDest": [500000],
        "step": [100],
    })

    # same preprocessing as production
    sample = build_features(sample)
    probability = model.predict_proba(sample)[0][1]
    assert 0 <= probability <= 1