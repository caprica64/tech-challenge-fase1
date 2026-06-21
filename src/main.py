from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from src.neural_net import ChurnMLP

app = FastAPI(title="Churn Prediction API")

class CustomerData(BaseModel):
    feature: float
    # adicionar as features 

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int

# Instanciar modelo 
# model = ChurnMLP(input_dim=...)
# model.load_state_dict(torch.load("models/mlp_weights.pth"))
# model.eval()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerData):
    try:
        # Converter dados de entrada para tensor
        # input_tensor = torch.tensor([[data.feature,]])
        
        # Fazer predição
        # with torch.no_grad():
        #     prob = model(input_tensor).item()
        
        # só um mock de retorno, depois remover
        prob = 0.85 
        return {"churn_probability": prob, "churn_prediction": 1 if prob > 0.5 else 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))