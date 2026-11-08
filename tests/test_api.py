from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_normal():
    response = client.post(
        "/predict",
        json={
            "step": 1,
            "amount": 20000,
            "oldbalanceOrg": 100000,
            "newbalanceOrig": 80000,
            "oldbalanceDest": 0,
            "newbalanceDest": 20000,
        },
    )
    assert response.status_code == 200
    assert response.json()["is_fraud"] is False
    assert response.json()["fraud_probability"] < 0.5