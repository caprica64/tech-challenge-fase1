from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health_check():
    """GET /health retorna status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload():
    """POST /predict com payload válido retorna 200."""
    payload = {
        "Tenure Months": 24,
        "Monthly Charges": 80.0,
        "Total Charges": 1920.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in (0, 1, 0.0, 1.0)


def test_predict_missing_field_returns_422():
    """POST /predict sem campo obrigatório retorna 422."""
    payload = {
        "Monthly Charges": 80.0,
        # Missing: Tenure Months, Total Charges
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_value_returns_422():
    """POST /predict com valor fora do range retorna 422."""
    payload = {
        "Tenure Months": -5,  # ge=0 violado
        "Monthly Charges": 80.0,
        "Total Charges": 1920.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    """POST /predict com tipo errado retorna 422."""
    payload = {
        "Tenure Months": "abc",
        "Monthly Charges": 80.0,
        "Total Charges": 1920.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
