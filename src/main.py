from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")


class CustomerData(BaseModel):
    feature: float
    # TODO: replace with actual feature columns from the trained model


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int


@app.get("/health")
def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, status_code=200)


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerData) -> JSONResponse:
    try:
        # TODO: replace mock with real model inference
        # input_tensor = torch.tensor([[data.feature, ...]])
        # with torch.no_grad():
        #     prob = model(input_tensor).item()
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
