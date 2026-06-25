from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

# Payload válido com todas as features que o modelo espera
VALID_PAYLOAD = {
    "tenure_months": 24,
    "monthly_charges": 80.0,
    "total_charges": 1920.0,
    "senior_citizen": 0,
    "partner": 1,
    "dependents": 0,
    "phone_services": 1,
    "multiples_lines": 0,
    "internet_dsl": 0,
    "internet_fiber": 1,
    "online_security": 0,
    "online_backup": 1,
    "device_protection": 0,
    "tech_support": 0,
    "streaming_tv": 1,
    "streaming_movies": 1,
    "contract_month_to_month": 1,
    "contract_one_year": 0,
    "contract_two_year": 0,
    "paperless_billing": 1,
    "payment_method_mailed_check": 0,
    "payment_method_electronic_check": 1,
    "payment_method_bank_transfer_automatic": 0,
    "payment_method_credit_card_automatic": 0,
}


def test_health_check():
    """GET /health retorna status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_valid_payload():
    """POST /predict com payload válido retorna 200
    e faz inferência real."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in (0.0, 1.0)


def test_predict_missing_field_returns_422():
    """POST /predict sem campo obrigatório retorna 422."""
    partial = {"monthly_charges": 80.0}
    response = client.post("/predict", json=partial)
    assert response.status_code == 422


def test_predict_invalid_value_returns_422():
    """POST /predict com valor fora do range retorna 422."""
    bad = {**VALID_PAYLOAD, "tenure_months": -5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    """POST /predict com tipo errado retorna 422."""
    bad = {**VALID_PAYLOAD, "tenure_months": "abc"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
