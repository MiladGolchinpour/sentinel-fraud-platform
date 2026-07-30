import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features used during training and inference."""

    df = df.copy()

    # amount_to_balance_ratio -> amount / oldbalanceOrg
    # why? large transactions relative to balance may indicate fraud
    # leakage risk: no
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # origin_account_emptied -> newbalanceOrig == 0
    # why? fraudsters often empty accounts
    # leakage risk = low
    df["origin_account_emptied"] = (df["newbalanceOrig"] == 0).astype("int8")

    # destination_balance_growth_ratio -> (newbalanceDest - oldbalanceDest) / (oldbalanceDest + 1)
    # why? detects unusually large increases in the destination account balance after a transaction
    # leakage: low
    df["destination_balance_growth_ratio"] = ((df["newbalanceDest"] - df["oldbalanceDest"]) / (df["oldbalanceDest"] + 1))

    # hour_of_day -> step % 24
    df["hour_of_day"] = (df["step"] % 24).astype("int8")

    # is_night_transaction -> (hour_of_day >= 22) | (hour_of_day <= 6)
    df["is_night_transaction"] = (((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6))).astype("int8")

    return df
