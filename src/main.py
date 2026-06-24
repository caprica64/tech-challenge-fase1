import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Logging estruturado ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # JSON puro — CloudWatch parseia direto
)
logger = logging.getLogger("churn_api")


def log_json(**kwargs) -> None:
    """Emite um log line como JSON estruturado."""
    logger.info(json.dumps(kwargs, default=str))


# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Churn Prediction API")


# ── Middleware de latência ─────────────────────────────────────────────
@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    """Mede latência de cada request e loga em formato JSON."""
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

    # Injeta latência no header para observabilidade
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"

    return response


# ── Schemas ────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    """Input payload for churn prediction.

    Pydantic validates types and value ranges. FastAPI returns
    a 422 with detailed errors if any constraint is violated.
    """

    tenure_months: float = Field(
        ...,
        alias="Tenure Months",
        ge=0,
        le=100,
        description="Meses como cliente (0-100)",
    )
    monthly_charges: float = Field(
        ...,
        alias="Monthly Charges",
        ge=0,
        description="Cobrança mensal em USD",
    )
    total_charges: float = Field(
        ...,
        alias="Total Charges",
        ge=0,
        description="Total cobrado acumulado em USD",
    )

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0
    )
    churn_prediction: float = Field(..., ge=0, le=1)


# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
def health_check() -> JSONResponse:
    return JSONResponse(
        content={"status": "ok"}, status_code=200
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerData) -> JSONResponse:
    try:
        # TODO: replace mock with real model inference
        prob = 0.85

        log_json(
            event="prediction",
            tenure_months=data.tenure_months,
            monthly_charges=data.monthly_charges,
            total_charges=data.total_charges,
            churn_probability=prob,
            churn_prediction=1 if prob > 0.5 else 0,
        )

        return JSONResponse(
            content={
                "churn_probability": prob,
                "churn_prediction": 1 if prob > 0.5 else 0,
            },
            status_code=200,
        )
    except Exception as e:
        log_json(
            event="prediction_error",
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail=str(e)
        )
