"""Avaliacao de modelos de churn."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from model_trainer import ChurnModel, predict, predict_proba


def evaluate_model(
    model: ChurnModel,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Avalia o modelo treinado no conjunto de teste."""
    probabilities = predict_proba(model=model, scaler=scaler, X=X_test)
    predictions = predict(model=model, scaler=scaler, X=X_test, threshold=threshold)
    y_true = y_test.to_numpy()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }

