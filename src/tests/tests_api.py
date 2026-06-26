"""Testes automatizados da API de inferência de churn.

Cobre os 3 tipos exigidos:
- Smoke test: /health responde corretamente
- Schema: Pydantic rejeita payloads inválidos (422)
- API: /predict retorna inferência correta com payload válido
"""

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

# Payload válido usando os validation_alias do schema atual
VALID_PAYLOAD = {
    "Zip Code": 90003,
    "Latitude": 33.964,
    "Longitude": -118.272,
    "Tenure Months": 24,
    "Monthly Charges": 80.0,
    "Total Charges": 1920.0,
    "Churn Score": 65,
    "CLTV": 4500,
    "Gender_Male": 1,
    "Senior Citizen_Yes": 0,
    "Partner_Yes": 1,
    "Dependents_Yes": 0,
    "Phone Service_Yes": 1,
    "Multiple Lines_No phone service": 0,
    "Multiple Lines_Yes": 0,
    "Internet Service_Fiber optic": 1,
    "Internet Service_No": 0,
    "Online Security_No internet service": 0,
    "Online Security_Yes": 0,
    "Online Backup_No internet service": 0,
    "Online Backup_Yes": 1,
    "Device Protection_No internet service": 0,
    "Device Protection_Yes": 0,
    "Tech Support_No internet service": 0,
    "Tech Support_Yes": 0,
    "Streaming TV_No internet service": 0,
    "Streaming TV_Yes": 1,
    "Streaming Movies_No internet service": 0,
    "Streaming Movies_Yes": 1,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "Paperless Billing_Yes": 1,
    "Payment Method_Credit card (automatic)": 0,
    "Payment Method_Electronic check": 1,
    "Churn Label_Yes": 0,
    "Churn Reason_Extra data charges": 0,
    "Churn Reason_Lack of affordable download/upload speed": 0,
    "Churn Reason_Lack of self-service on Website": 0,
    "Churn Reason_Limited range of services": 0,
    "Churn Reason_Long distance charges": 0,
    "Churn Reason_Network reliability": 0,
    "Churn Reason_Poor expertise of online support": 0,
    "Churn Reason_Product dissatisfaction": 0,
}


# ── Smoke test ─────────────────────────────────────────────────────
def test_health_check():
    """GET /health retorna status ok e indica se modelo carregou."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


# ── API test (inferência end-to-end) ───────────────────────────────
def test_predict_valid_payload():
    """POST /predict com payload válido retorna 200 com
    probabilidade e predição dentro dos ranges esperados."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in (0.0, 1.0)


def test_predict_response_format():
    """POST /predict retorna JSON com exatamente 2 campos."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "churn_probability",
        "churn_prediction",
    }


# ── Schema tests (validação Pydantic) ─────────────────────────────
def test_schema_missing_field_returns_422():
    """POST /predict sem campo obrigatório retorna 422."""
    partial = {"Monthly Charges": 80.0, "CLTV": 4500}
    response = client.post("/predict", json=partial)
    assert response.status_code == 422


def test_schema_invalid_range_returns_422():
    """POST /predict com Tenure Months negativo retorna 422."""
    bad = {**VALID_PAYLOAD, "Tenure Months": -5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_schema_invalid_type_returns_422():
    """POST /predict com tipo string onde espera float retorna 422."""
    bad = {**VALID_PAYLOAD, "Monthly Charges": "abc"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_schema_binary_field_out_of_range_returns_422():
    """POST /predict com flag binária > 1 retorna 422."""
    bad = {**VALID_PAYLOAD, "Gender_Male": 5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_schema_empty_body_returns_422():
    """POST /predict com body vazio retorna 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422
