import pandas as pd
from src.features import build_features

def test_build_features():
    data = pd.DataFrame({
        "amount": [1000],
        "oldbalanceOrg": [5000],
        "newbalanceOrig": [4000],
        "oldbalanceDest": [0],
        "newbalanceDest": [1000],
        "step": [50],
        "type": ["TRANSFER"]
    })

    result = build_features(data)

    assert "amount_to_balance_ratio" in result.columns
    assert "origin_account_emptied" in result.columns
    assert "destination_balance_growth_ratio" in result.columns
    assert "hour_of_day" in result.columns
    assert "is_night_transaction" in result.columns

def test_hour_of_day():
    data = pd.DataFrame({
        "amount": [1000],
        "oldbalanceOrg": [5000],
        "newbalanceOrig": [4000],
        "oldbalanceDest": [0],
        "newbalanceDest": [1000],
        "step": [50],
        "type": ["TRANSFER"],
    })

    result = build_features(data)

    assert result["hour_of_day"].iloc[0] == 2