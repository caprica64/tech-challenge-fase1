"""Avaliacao de modelos de churn."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from model_trainer import ChurnModel, predict_proba


logger = logging.getLogger(__name__)


def evaluate_model(
    model: ChurnModel,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Avalia o modelo treinado usando PR-AUC."""
    logger.info(
        "Iniciando avaliacao PR-AUC do modelo com %s amostras",
        len(X_test),
    )

    probabilities = predict_proba(model=model, scaler=scaler, X=X_test)
    y_true = y_test.to_numpy()

    metrics = {
        "pr_auc": average_precision_score(y_true, probabilities),
    }

    logger.info("Avaliacao concluida: %s", metrics)
    return metrics
