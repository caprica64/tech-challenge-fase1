import json
import logging
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Logging estruturado ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("churn_api")


def log_json(**kwargs) -> None:
    """Emite log line como JSON estruturado."""
    logger.info(json.dumps(kwargs, default=str))


# ── Features esperadas pelo modelo treinado ────────────────────────
MODEL_FEATURES = [
    "Zip Code",
    "Latitude",
    "Longitude",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Churn Score",
    "CLTV",
    "Gender_Male",
    "Senior Citizen_Yes",
    "Partner_Yes",
    "Dependents_Yes",
    "Phone Service_Yes",
    "Multiple Lines_No phone service",
    "Multiple Lines_Yes",
    "Internet Service_Fiber optic",
    "Internet Service_No",
    "Online Security_No internet service",
    "Online Security_Yes",
    "Online Backup_No internet service",
    "Online Backup_Yes",
    "Device Protection_No internet service",
    "Device Protection_Yes",
    "Tech Support_No internet service",
    "Tech Support_Yes",
    "Streaming TV_No internet service",
    "Streaming TV_Yes",
    "Streaming Movies_No internet service",
    "Streaming Movies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "Paperless Billing_Yes",
    "Payment Method_Credit card (automatic)",
    "Payment Method_Electronic check",
    "Churn Label_Yes",
    "Churn Reason_Extra data charges",
    "Churn Reason_Lack of affordable download/upload speed",
    "Churn Reason_Lack of self-service on Website",
    "Churn Reason_Limited range of services",
    "Churn Reason_Long distance charges",
    "Churn Reason_Network reliability",
    "Churn Reason_Poor expertise of online support",
    "Churn Reason_Product dissatisfaction",
    "Engineered Monthly Charges",
    "charge_rel",
]


# ── Carregar modelo na inicialização ───────────────────────────────
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "best_random_forest.joblib"
)

model_pipeline = None

try:
    model_pipeline = joblib.load(MODEL_PATH)
    log_json(event="model_loaded", path=str(MODEL_PATH))
except FileNotFoundError:
    log_json(
        event="model_not_found",
        path=str(MODEL_PATH),
        warning="API iniciará em modo mock",
    )


# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="Churn Prediction API")


# ── Middleware de latência ─────────────────────────────────────────
@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    """Mede latência e loga em JSON estruturado."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    log_json(
        event="http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=round(elapsed_ms, 2),
    )
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ── Schemas ────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    """Input com as features que o modelo espera.

    Campos numéricos brutos + flags binárias one-hot encoded.
    Usa validation_alias para aceitar os nomes originais das
    features (com espaços e caracteres especiais) no JSON.
    """

    # Numéricas contínuas
    zip_code: float = Field(
        ..., validation_alias="Zip Code"
    )
    latitude: float = Field(
        ..., validation_alias="Latitude"
    )
    longitude: float = Field(
        ..., validation_alias="Longitude"
    )
    tenure_months: float = Field(
        ..., validation_alias="Tenure Months", ge=0, le=100
    )
    monthly_charges: float = Field(
        ..., validation_alias="Monthly Charges", ge=0
    )
    total_charges: float = Field(
        ..., validation_alias="Total Charges", ge=0
    )
    churn_score: float = Field(
        ..., validation_alias="Churn Score", ge=0, le=100
    )
    cltv: float = Field(
        ..., validation_alias="CLTV", ge=0
    )

    # One-hot encoded flags (0 ou 1)
    gender_male: int = Field(
        ..., validation_alias="Gender_Male", ge=0, le=1
    )
    senior_citizen_yes: int = Field(
        ..., validation_alias="Senior Citizen_Yes",
        ge=0, le=1,
    )
    partner_yes: int = Field(
        ..., validation_alias="Partner_Yes", ge=0, le=1
    )
    dependents_yes: int = Field(
        ..., validation_alias="Dependents_Yes", ge=0, le=1
    )
    phone_service_yes: int = Field(
        ..., validation_alias="Phone Service_Yes",
        ge=0, le=1,
    )
    multiple_lines_no_phone: int = Field(
        ...,
        validation_alias="Multiple Lines_No phone service",
        ge=0, le=1,
    )
    multiple_lines_yes: int = Field(
        ..., validation_alias="Multiple Lines_Yes",
        ge=0, le=1,
    )
    internet_fiber: int = Field(
        ...,
        validation_alias="Internet Service_Fiber optic",
        ge=0, le=1,
    )
    internet_no: int = Field(
        ..., validation_alias="Internet Service_No",
        ge=0, le=1,
    )
    online_security_no_internet: int = Field(
        ...,
        validation_alias=(
            "Online Security_No internet service"
        ),
        ge=0, le=1,
    )
    online_security_yes: int = Field(
        ..., validation_alias="Online Security_Yes",
        ge=0, le=1,
    )
    online_backup_no_internet: int = Field(
        ...,
        validation_alias=(
            "Online Backup_No internet service"
        ),
        ge=0, le=1,
    )
    online_backup_yes: int = Field(
        ..., validation_alias="Online Backup_Yes",
        ge=0, le=1,
    )
    device_protection_no_internet: int = Field(
        ...,
        validation_alias=(
            "Device Protection_No internet service"
        ),
        ge=0, le=1,
    )
    device_protection_yes: int = Field(
        ..., validation_alias="Device Protection_Yes",
        ge=0, le=1,
    )
    tech_support_no_internet: int = Field(
        ...,
        validation_alias=(
            "Tech Support_No internet service"
        ),
        ge=0, le=1,
    )
    tech_support_yes: int = Field(
        ..., validation_alias="Tech Support_Yes",
        ge=0, le=1,
    )
    streaming_tv_no_internet: int = Field(
        ...,
        validation_alias=(
            "Streaming TV_No internet service"
        ),
        ge=0, le=1,
    )
    streaming_tv_yes: int = Field(
        ..., validation_alias="Streaming TV_Yes",
        ge=0, le=1,
    )
    streaming_movies_no_internet: int = Field(
        ...,
        validation_alias=(
            "Streaming Movies_No internet service"
        ),
        ge=0, le=1,
    )
    streaming_movies_yes: int = Field(
        ..., validation_alias="Streaming Movies_Yes",
        ge=0, le=1,
    )
    contract_one_year: int = Field(
        ..., validation_alias="Contract_One year",
        ge=0, le=1,
    )
    contract_two_year: int = Field(
        ..., validation_alias="Contract_Two year",
        ge=0, le=1,
    )
    paperless_billing_yes: int = Field(
        ..., validation_alias="Paperless Billing_Yes",
        ge=0, le=1,
    )
    payment_credit_card: int = Field(
        ...,
        validation_alias=(
            "Payment Method_Credit card (automatic)"
        ),
        ge=0, le=1,
    )
    payment_electronic_check: int = Field(
        ...,
        validation_alias="Payment Method_Electronic check",
        ge=0, le=1,
    )
    churn_label_yes: int = Field(
        ..., validation_alias="Churn Label_Yes",
        ge=0, le=1,
    )
    churn_reason_extra_data: int = Field(
        ...,
        validation_alias=(
            "Churn Reason_Extra data charges"
        ),
        ge=0, le=1,
    )
    churn_reason_speed: int = Field(
        ...,
        validation_alias=(
            "Churn Reason_Lack of affordable"
            " download/upload speed"
        ),
        ge=0, le=1,
    )
    churn_reason_self_service: int = Field(
        ...,
        validation_alias=(
            "Churn Reason_Lack of self-service"
            " on Website"
        ),
        ge=0, le=1,
    )
    churn_reason_limited_services: int = Field(
        ...,
        validation_alias=(
            "Churn Reason_Limited range of services"
        ),
        ge=0, le=1,
    )
    churn_reason_long_distance: int = Field(
        ...,
        validation_alias=(
            "Churn Reason_Long distance charges"
        ),
        ge=0, le=1,
    )
    churn_reason_network: int = Field(
        ...,
        validation_alias=(
            "Churn Reason_Network reliability"
        ),
        ge=0, le=1,
    )
    churn_reason_online_support: int = Field(
        ...,
        validation_alias=(
            "Churn Reason_Poor expertise"
            " of online support"
        ),
        ge=0, le=1,
    )
    churn_reason_product: int = Field(
        ...,
        validation_alias=(
            "Churn Reason_Product dissatisfaction"
        ),
        ge=0, le=1,
    )


class PredictionResponse(BaseModel):
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0
    )
    churn_prediction: float = Field(..., ge=0, le=1)


# ── Build feature vector ───────────────────────────────────────────
# Map Python field names → model feature names
_FIELD_TO_FEATURE = {
    "zip_code": "Zip Code",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "tenure_months": "Tenure Months",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
    "churn_score": "Churn Score",
    "cltv": "CLTV",
    "gender_male": "Gender_Male",
    "senior_citizen_yes": "Senior Citizen_Yes",
    "partner_yes": "Partner_Yes",
    "dependents_yes": "Dependents_Yes",
    "phone_service_yes": "Phone Service_Yes",
    "multiple_lines_no_phone": "Multiple Lines_No phone service",
    "multiple_lines_yes": "Multiple Lines_Yes",
    "internet_fiber": "Internet Service_Fiber optic",
    "internet_no": "Internet Service_No",
    "online_security_no_internet": (
        "Online Security_No internet service"
    ),
    "online_security_yes": "Online Security_Yes",
    "online_backup_no_internet": (
        "Online Backup_No internet service"
    ),
    "online_backup_yes": "Online Backup_Yes",
    "device_protection_no_internet": (
        "Device Protection_No internet service"
    ),
    "device_protection_yes": "Device Protection_Yes",
    "tech_support_no_internet": (
        "Tech Support_No internet service"
    ),
    "tech_support_yes": "Tech Support_Yes",
    "streaming_tv_no_internet": (
        "Streaming TV_No internet service"
    ),
    "streaming_tv_yes": "Streaming TV_Yes",
    "streaming_movies_no_internet": (
        "Streaming Movies_No internet service"
    ),
    "streaming_movies_yes": "Streaming Movies_Yes",
    "contract_one_year": "Contract_One year",
    "contract_two_year": "Contract_Two year",
    "paperless_billing_yes": "Paperless Billing_Yes",
    "payment_credit_card": (
        "Payment Method_Credit card (automatic)"
    ),
    "payment_electronic_check": (
        "Payment Method_Electronic check"
    ),
    "churn_label_yes": "Churn Label_Yes",
    "churn_reason_extra_data": (
        "Churn Reason_Extra data charges"
    ),
    "churn_reason_speed": (
        "Churn Reason_Lack of affordable"
        " download/upload speed"
    ),
    "churn_reason_self_service": (
        "Churn Reason_Lack of self-service on Website"
    ),
    "churn_reason_limited_services": (
        "Churn Reason_Limited range of services"
    ),
    "churn_reason_long_distance": (
        "Churn Reason_Long distance charges"
    ),
    "churn_reason_network": (
        "Churn Reason_Network reliability"
    ),
    "churn_reason_online_support": (
        "Churn Reason_Poor expertise of online support"
    ),
    "churn_reason_product": (
        "Churn Reason_Product dissatisfaction"
    ),
}


def build_feature_vector(data: CustomerData) -> pd.DataFrame:
    """Converte payload em DataFrame com as features
    na ordem exata que o modelo espera, incluindo
    features derivadas."""
    raw = data.model_dump()

    # Renomeia Python field names → model feature names
    row = {
        _FIELD_TO_FEATURE[k]: v
        for k, v in raw.items()
    }

    # Features engenheiradas
    row["Engineered Monthly Charges"] = (
        row["Monthly Charges"] ** 2
    )
    eps = 1
    row["charge_rel"] = (
        row["Total Charges"]
        / (row["Monthly Charges"] + eps)
        - row["Tenure Months"]
    )

    df = pd.DataFrame([row])
    return df[MODEL_FEATURES]


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Retorna 204 No Content para requisições de favicon."""
    return JSONResponse(content=None, status_code=204)


@app.get("/health")
def health_check() -> JSONResponse:
    return JSONResponse(
        content={
            "status": "ok",
            "model_loaded": model_pipeline is not None,
        },
        status_code=200,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerData) -> JSONResponse:
    try:
        X = build_feature_vector(data)

        if model_pipeline is not None:
            proba = model_pipeline.predict_proba(X)[0, 1]
        else:
            proba = 0.5

        prob = float(np.clip(proba, 0.0, 1.0))
        prediction = 1.0 if prob >= 0.5 else 0.0

        log_json(
            event="prediction",
            churn_probability=prob,
            churn_prediction=prediction,
            tenure_months=data.tenure_months,
            monthly_charges=data.monthly_charges,
            model_loaded=model_pipeline is not None,
        )

        return JSONResponse(
            content={
                "churn_probability": round(prob, 4),
                "churn_prediction": prediction,
            },
            status_code=200,
        )
    except Exception as e:
        log_json(event="prediction_error", error=str(e))
        raise HTTPException(
            status_code=500, detail=str(e)
        )
