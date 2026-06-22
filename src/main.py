from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(title="Churn Prediction API")


class CustomerData(BaseModel):
    """Input payload for churn prediction.

    All fields are optional during development. Replace with
    required fields matching the trained model's features before
    production.
    """

    tenure_months: Optional[float] = Field(None, alias="Tenure Months")
    monthly_charges: Optional[float] = Field(
        None, alias="Monthly Charges"
    )
    total_charges: Optional[float] = Field(
        None, alias="Total Charges"
    )

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int


@app.get("/health")
def health_check() -> JSONResponse:
    return JSONResponse(
        content={"status": "ok"}, status_code=200
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(
    data: CustomerData = CustomerData(),
) -> JSONResponse:
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
        raise HTTPException(status_code=500, detail=str(e))
