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


# ── Carregar modelo na inicialização ───────────────────────────────
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "best_random_forest.joblib"
)

model_pipeline = None

try:
    model_pipeline = joblib.load(MODEL_PATH)
    log_json(
        event="model_loaded",
        path=str(MODEL_PATH),
    )
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

    Pydantic valida tipos e ranges. FastAPI retorna 422
    com detalhes se alguma constraint for violada.
    """

    tenure_months: float = Field(
        ..., ge=0, le=100,
        description="Meses como cliente",
    )
    monthly_charges: float = Field(
        ..., ge=0,
        description="Cobrança mensal em USD",
    )
    total_charges: float = Field(
        ..., ge=0,
        description="Total cobrado acumulado em USD",
    )
    senior_citizen: int = Field(
        ..., ge=0, le=1,
        description="Idoso (1=sim, 0=não)",
    )
    partner: int = Field(
        ..., ge=0, le=1,
        description="Possui parceiro (1=sim)",
    )
    dependents: int = Field(
        ..., ge=0, le=1,
        description="Possui dependentes (1=sim)",
    )
    phone_services: int = Field(
        ..., ge=0, le=1,
        description="Serviço de telefone (1=sim)",
    )
    multiples_lines: int = Field(
        ..., ge=0, le=1,
        description="Múltiplas linhas (1=sim)",
    )
    internet_dsl: int = Field(
        ..., ge=0, le=1,
        description="Internet DSL (1=sim)",
    )
    internet_fiber: int = Field(
        ..., ge=0, le=1,
        description="Internet Fibra (1=sim)",
    )
    online_security: int = Field(
        ..., ge=0, le=1,
        description="Segurança online (1=sim)",
    )
    online_backup: int = Field(
        ..., ge=0, le=1,
        description="Backup online (1=sim)",
    )
    device_protection: int = Field(
        ..., ge=0, le=1,
        description="Proteção de dispositivo (1=sim)",
    )
    tech_support: int = Field(
        ..., ge=0, le=1,
        description="Suporte técnico (1=sim)",
    )
    streaming_tv: int = Field(
        ..., ge=0, le=1,
        description="Streaming TV (1=sim)",
    )
    streaming_movies: int = Field(
        ..., ge=0, le=1,
        description="Streaming filmes (1=sim)",
    )
    contract_month_to_month: int = Field(
        ..., ge=0, le=1,
        description="Contrato mensal (1=sim)",
    )
    contract_one_year: int = Field(
        ..., ge=0, le=1,
        description="Contrato anual (1=sim)",
    )
    contract_two_year: int = Field(
        ..., ge=0, le=1,
        description="Contrato bienal (1=sim)",
    )
    paperless_billing: int = Field(
        ..., ge=0, le=1,
        description="Fatura digital (1=sim)",
    )
    payment_method_mailed_check: int = Field(
        ..., ge=0, le=1,
    )
    payment_method_electronic_check: int = Field(
        ..., ge=0, le=1,
    )
    payment_method_bank_transfer_automatic: int = Field(
        ..., ge=0, le=1,
    )
    payment_method_credit_card_automatic: int = Field(
        ..., ge=0, le=1,
    )


class PredictionResponse(BaseModel):
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0
    )
    churn_prediction: float = Field(..., ge=0, le=1)


# ── Feature engineering (replica data_ingestion) ───────────────────
MODEL_FEATURES = [
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "is_new_customer",
    "phone_services",
    "multiples_lines",
    "internet_dsl",
    "internet_fiber",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract_month_to_month",
    "contract_one_year",
    "contract_two_year",
    "paperless_billing",
    "payment_method_mailed_check",
    "payment_method_electronic_check",
    "payment_method_bank_transfer_automatic",
    "payment_method_credit_card_automatic",
    "monthly_charges",
    "total_charges",
    "avg_ticket",
    "high_risk_profile",
]


def build_feature_vector(data: CustomerData) -> pd.DataFrame:
    """Converte o payload em um DataFrame com as features
    que o modelo espera, incluindo features derivadas."""
    row = data.model_dump()

    # Features derivadas (mesma lógica de data_ingestion.py)
    row["is_new_customer"] = int(row["tenure_months"] in (1, 2))
    row["high_risk_profile"] = int(
        row["is_new_customer"] == 1
        and row["contract_month_to_month"] == 1
        and row["internet_fiber"] == 1
    )
    row["avg_ticket"] = (
        row["total_charges"] / row["tenure_months"]
        if row["tenure_months"] > 0
        else 0.0
    )

    df = pd.DataFrame([row])
    return df[MODEL_FEATURES]


# ── Endpoints ──────────────────────────────────────────────────────
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
            # Fallback mock se modelo não foi carregado
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
