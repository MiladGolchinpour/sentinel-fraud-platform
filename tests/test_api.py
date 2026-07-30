from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={
            "type": "TRANSFER",
            "amount": 500000,
            "oldbalanceOrg": 500000,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 500000,
            "step": 100
        }
    )

    assert response.status_code == 200
    data = response.json()
    
    assert "prediction" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "top_reasons" in data