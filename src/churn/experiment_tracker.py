"""Registro de experimentos de churn com MLflow."""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model_trainer import ChurnModel


def log_experiment(
    model: ChurnModel,
    scaler: StandardScaler,
    metrics: dict[str, float],
    losses: list[float],
    params: dict[str, Any] | None = None,
    experiment_name: str = "churn_pytorch",
    run_name: str = "mlp_churn",
    tracking_uri: str | None = None,
) -> str:
    """Registra parametros, metricas, perdas, modelo e scaler no MLflow."""
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(_serialize_params(params))

        mlflow.log_metrics(metrics)

        for epoch, loss in enumerate(losses, start=1):
            mlflow.log_metric("train_loss", loss, step=epoch)

        mlflow.pytorch.log_model(model, "model")
        mlflow.sklearn.log_model(scaler, "scaler")

        return run.info.run_id


def get_best_run(
    experiment_name: str = "churn_pytorch",
    metric: str = "roc_auc",
    greater_is_better: bool = True,
    tracking_uri: str | None = None,
) -> pd.Series:
    """Retorna o melhor run registrado no MLflow pela metrica escolhida."""
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento nao encontrado no MLflow: {experiment_name}")

    order_direction = "DESC" if greater_is_better else "ASC"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order_direction}"],
    )

    if runs.empty:
        raise ValueError(f"Nenhum run encontrado no experimento: {experiment_name}")

    return runs.iloc[0]


def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
    serialized = {}

    for key, value in params.items():
        if isinstance(value, (list, tuple, dict)):
            serialized[key] = str(value)
        else:
            serialized[key] = value

    return serialized
