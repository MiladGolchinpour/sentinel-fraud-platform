from typing import Literal
from pydantic import BaseModel, Field

class Transaction(BaseModel):
    type: Literal[
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT",
    ]

    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    step: int = Field(ge=1)


class PredictionResponse(BaseModel):
    prediction: Literal["fraud", "legitimate"]
    fraud_probability: float = Field(ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    top_reasons: list[str]