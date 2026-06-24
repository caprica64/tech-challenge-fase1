from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Churn Prediction API")


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

        return JSONResponse(
            content={
                "churn_probability": prob,
                "churn_prediction": 1 if prob > 0.5 else 0,
            },
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e)
        )
