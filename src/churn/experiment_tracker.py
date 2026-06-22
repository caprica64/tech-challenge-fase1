"""Registro de experimentos de churn com MLflow."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model_trainer import ChurnModel


logger = logging.getLogger(__name__)


def log_experiment(
    model: ChurnModel | None,
    scaler: StandardScaler | None,
    metrics: dict[str, float],
    losses: list[float],
    params: dict[str, Any] | None = None,
    experiment_name: str = "churn_pytorch",
    run_name: str = "mlp_churn",
    tracking_uri: str | None = None,
) -> str:
    """Registra parametros, metricas, perdas e artefatos opcionais no MLflow."""
    logger.info(
        "Iniciando registro no MLflow: experiment_name=%s, run_name=%s",
        experiment_name,
        run_name,
    )

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("Tracking URI configurado: %s", tracking_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            serialized_params = _serialize_params(params)
            mlflow.log_params(serialized_params)
            logger.info("Parametros registrados no MLflow: %s", serialized_params)

        mlflow.log_metrics(metrics)
        logger.info("Metricas registradas no MLflow: %s", metrics)

        for epoch, loss in enumerate(losses, start=1):
            mlflow.log_metric("train_loss", loss, step=epoch)
        logger.info("Historico de loss registrado com %s epocas", len(losses))

        if model is not None:
            mlflow.pytorch.log_model(model, "model")
            logger.info("Modelo registrado como artefato")

        if scaler is not None:
            mlflow.sklearn.log_model(scaler, "scaler")
            logger.info("Scaler registrado como artefato")

        run_id = run.info.run_id
        logger.info("Registro no MLflow concluido. run_id=%s", run_id)
        return run_id


def log_metrics_experiment(
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    experiment_name: str = "churn_pytorch",
    run_name: str = "metrics_only",
    tracking_uri: str | None = None,
) -> str:
    """Registra uma run no MLflow apenas com parametros e metricas."""
    logger.info(
        "Iniciando registro de metricas no MLflow: experiment_name=%s, run_name=%s",
        experiment_name,
        run_name,
    )

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("Tracking URI configurado: %s", tracking_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            serialized_params = _serialize_params(params)
            mlflow.log_params(serialized_params)
            logger.info("Parametros registrados no MLflow: %s", serialized_params)

        mlflow.log_metrics(metrics)
        logger.info("Metricas registradas no MLflow: %s", metrics)

        run_id = run.info.run_id
        logger.info("Registro de metricas no MLflow concluido. run_id=%s", run_id)
        return run_id


def get_best_run(
    experiment_name: str = "churn_pytorch",
    metric: str = "pr_auc",
    greater_is_better: bool = True,
    tracking_uri: str | None = None,
) -> pd.Series:
    """Retorna o melhor run registrado no MLflow pela metrica escolhida."""
    logger.info(
        "Buscando melhor run no MLflow: experiment_name=%s, metric=%s, "
        "greater_is_better=%s",
        experiment_name,
        metric,
        greater_is_better,
    )

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("Tracking URI configurado: %s", tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error("Experimento nao encontrado no MLflow: %s", experiment_name)
        raise ValueError(f"Experimento nao encontrado no MLflow: {experiment_name}")

    order_direction = "DESC" if greater_is_better else "ASC"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order_direction}"],
    )

    if runs.empty:
        logger.error("Nenhum run encontrado no experimento: %s", experiment_name)
        raise ValueError(f"Nenhum run encontrado no experimento: {experiment_name}")

    best_run = runs.iloc[0]
    logger.info(
        "Melhor run encontrado: run_id=%s, %s=%s",
        best_run.get("run_id"),
        metric,
        best_run.get(f"metrics.{metric}"),
    )
    return best_run


def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
    serialized = {}

    for key, value in params.items():
        if isinstance(value, (list, tuple, dict)):
            serialized[key] = str(value)
        else:
            serialized[key] = value

    return serialized
