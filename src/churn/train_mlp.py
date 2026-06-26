"""
train_mlp.py — Construção, treinamento e avaliação de MLP com PyTorch.

Executa:
1. Carrega dados pré-processados
2. Treina MLP com validação cruzada estratificada
3. Compara com baselines (Logistic Regression, Random Forest)
4. Avalia no conjunto de teste reservado (holdout)
5. Registra tudo no MLflow
6. Persiste o melhor modelo

Uso:
    python -m src.churn.train_mlp
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.churn.model_trainer import (
    ChurnModel,
    cross_validate_model,
    fit_model,
    predict,
    predict_proba,
)

# ── Config ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "pre-processed"
    / "Telco_customer_churn_preprocessed.csv"
)

RANDOM_SEED = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "churn_mlp_pytorch"

# Features selecionadas no notebook 02 (SelectKBest top 30)
SELECTED_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Churn Score",
    "CLTV",
    "Dependents_Yes",
    "Internet Service_Fiber optic",
    "Internet Service_No",
    "Online Security_No internet service",
    "Online Backup_No internet service",
    "Device Protection_No internet service",
    "Tech Support_No internet service",
    "Streaming TV_No internet service",
    "Streaming Movies_No internet service",
    "Contract_One year",
    "Contract_Two year",
    "Paperless Billing_Yes",
    "Payment Method_Electronic check",
]

TARGET_COLUMN = "target"


# ── Métricas ───────────────────────────────────────────────────────
def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calcula todas as métricas de avaliação."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


# ── Análise de custo FP vs FN ─────────────────────────────────────
# Custo de negócio:
#   FN (churner não detectado) = perda de MRR ~R$100/cliente
#   FP (não-churner recebe oferta) = custo de retenção ~R$20/cliente
COST_FN = 100.0  # custo de perder um cliente
COST_FP = 20.0   # custo de oferta desnecessária


def compute_cost_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Analisa trade-off de custo para diferentes thresholds.

    Retorna tabela com custo total, FP, FN para cada threshold,
    permitindo escolher o ponto ótimo de operação.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        total_cost = fn * cost_fn + fp * cost_fp
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)

        results.append({
            "threshold": round(threshold, 2),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "cost_FN": fn * cost_fn,
            "cost_FP": fp * cost_fp,
            "total_cost": total_cost,
        })

    return pd.DataFrame(results)


def find_optimal_threshold(
    cost_df: pd.DataFrame,
) -> dict:
    """Retorna o threshold que minimiza o custo total."""
    best_row = cost_df.loc[
        cost_df["total_cost"].idxmin()
    ]
    return best_row.to_dict()


# ── Baselines ──────────────────────────────────────────────────────
def train_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Treina baselines e retorna tabela comparativa."""
    baselines = {
        "DummyClassifier (stratified)": DummyClassifier(
            strategy="stratified", random_state=RANDOM_SEED
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        ),
    }

    results = []
    for name, clf in baselines.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = (
            clf.predict_proba(X_test)[:, 1]
            if hasattr(clf, "predict_proba")
            else np.zeros_like(y_pred, dtype=float)
        )
        metrics = compute_all_metrics(y_test, y_pred, y_proba)
        metrics["model"] = name
        results.append(metrics)

        # Log no MLflow (excluindo 'model' que é string)
        with mlflow.start_run(
            run_name=name.lower().replace(" ", "_"),
            nested=True,
        ):
            mlflow.log_params({"model": name})
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if k != "model"}
            )
            mlflow.sklearn.log_model(clf, "model")

    return pd.DataFrame(results)


# ── MLP Training ───────────────────────────────────────────────────
def train_and_evaluate_mlp(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Treina a MLP, avalia, e retorna métricas + artefatos."""

    # ── Hiperparâmetros da MLP ─────────────────────────────────
    mlp_params = {
        "hidden_layers": (128, 64, 32),
        "dropout": 0.3,
        "activation": "relu",
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 64,
        "use_pos_weight": True,
    }

    logger.info("=== Validação cruzada da MLP ===")
    cv_results, cv_summary = cross_validate_model(
        X=X_train,
        y=y_train,
        n_splits=5,
        random_state=RANDOM_SEED,
        **mlp_params,
    )
    logger.info("CV Summary:\n%s", cv_summary)

    # ── Treino final no conjunto completo de treino ────────────
    logger.info("=== Treino final da MLP ===")
    model, scaler, losses = fit_model(
        X_train=X_train,
        y_train=y_train,
        random_state=RANDOM_SEED,
        **mlp_params,
    )

    # ── Avaliação no holdout ───────────────────────────────────
    y_proba_test = predict_proba(
        model=model, scaler=scaler, X=X_test
    )
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    test_metrics = compute_all_metrics(
        y_test.to_numpy(), y_pred_test, y_proba_test
    )

    # ── Métricas de treino (para detectar overfitting) ─────────
    y_proba_train = predict_proba(
        model=model, scaler=scaler, X=X_train
    )
    y_pred_train = (y_proba_train >= 0.5).astype(int)
    train_metrics = compute_all_metrics(
        y_train.to_numpy(), y_pred_train, y_proba_train
    )

    return {
        "model": model,
        "scaler": scaler,
        "losses": losses,
        "params": mlp_params,
        "cv_results": cv_results,
        "cv_summary": cv_summary,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


# ── Main ───────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 60)
    logger.info("INÍCIO — Treinamento MLP para Churn Prediction")
    logger.info("=" * 60)

    # ── 1. Carrega dados ───────────────────────────────────────
    if not DATA_PATH.exists():
        logger.error("Dataset não encontrado: %s", DATA_PATH)
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset carregado: %s", df.shape)

    # Garante coluna target
    assert TARGET_COLUMN in df.columns, (
        f"Coluna '{TARGET_COLUMN}' ausente"
    )

    # Filtra features disponíveis
    available = [
        f for f in SELECTED_FEATURES if f in df.columns
    ]
    if len(available) < len(SELECTED_FEATURES):
        missing = set(SELECTED_FEATURES) - set(available)
        logger.warning(
            "Features ausentes (ignoradas): %s", missing
        )

    X = df[available]
    y = df[TARGET_COLUMN]

    logger.info(
        "Features: %d | Target: %s | Positivos: %.1f%%",
        X.shape[1],
        TARGET_COLUMN,
        y.mean() * 100,
    )

    # ── 2. Split treino/teste ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    logger.info(
        "Split: treino=%d, teste=%d", len(X_train), len(X_test)
    )

    # ── 3. MLflow experiment ───────────────────────────────────
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="mlp_full_pipeline"):
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param(
            "features", str(available[:10]) + "..."
        )

        # ── 4. Baselines ───────────────────────────────────────
        logger.info("=== Treinando baselines ===")
        scaler_bl = StandardScaler()
        X_train_scaled = scaler_bl.fit_transform(X_train)
        X_test_scaled = scaler_bl.transform(X_test)

        baseline_df = train_baselines(
            X_train_scaled, X_test_scaled,
            y_train.to_numpy(), y_test.to_numpy(),
        )
        logger.info(
            "Baselines:\n%s",
            baseline_df.to_string(index=False),
        )

        # ── 5. MLP ────────────────────────────────────────────
        logger.info("=== Treinando MLP PyTorch ===")
        mlp_result = train_and_evaluate_mlp(
            X_train, X_test, y_train, y_test
        )

        # Log MLP params e metrics
        for k, v in mlp_result["params"].items():
            mlflow.log_param(f"mlp_{k}", str(v))

        for k, v in mlp_result["test_metrics"].items():
            mlflow.log_metric(f"test_{k}", v)
        for k, v in mlp_result["train_metrics"].items():
            mlflow.log_metric(f"train_{k}", v)

        # CV metrics
        cv_mean = mlp_result["cv_summary"].loc["mean"]
        for k, v in cv_mean.items():
            mlflow.log_metric(f"cv_mean_{k}", v)

        # Loss history
        for epoch, loss in enumerate(
            mlp_result["losses"], start=1
        ):
            mlflow.log_metric("train_loss", loss, step=epoch)

        # Log model (usa formato pickle clássico)
        mlflow.pytorch.log_model(
            mlp_result["model"],
            "mlp_model",
            serialization_format="pickle",
        )

        # ── 6. Tabela comparativa ─────────────────────────────
        mlp_row = {
            "model": "MLP PyTorch",
            **mlp_result["test_metrics"],
        }
        comparison = pd.concat(
            [baseline_df, pd.DataFrame([mlp_row])],
            ignore_index=True,
        )
        comparison = comparison.sort_values(
            "pr_auc", ascending=False
        )

        logger.info("=" * 60)
        logger.info("COMPARAÇÃO FINAL DE MODELOS")
        logger.info("=" * 60)
        logger.info("\n%s", comparison.to_string(index=False))

        # Overfitting check
        gap = (
            mlp_result["train_metrics"]["pr_auc"]
            - mlp_result["test_metrics"]["pr_auc"]
        )
        mlflow.log_metric("overfitting_gap_pr_auc", gap)
        logger.info("Overfitting gap (PR-AUC): %.4f", gap)

        # ── 7. Análise de trade-off de custo (FP vs FN) ───────
        logger.info("=" * 60)
        logger.info(
            "ANÁLISE DE TRADE-OFF DE CUSTO "
            "(FN=R$%.0f, FP=R$%.0f)",
            COST_FN, COST_FP,
        )
        logger.info("=" * 60)

        # Probabilidades do MLP no teste
        y_proba_test = predict_proba(
            model=mlp_result["model"],
            scaler=mlp_result["scaler"],
            X=X_test,
        )

        cost_df = compute_cost_analysis(
            y_true=y_test.to_numpy(),
            y_proba=y_proba_test,
        )
        logger.info(
            "Tabela de custos por threshold:\n%s",
            cost_df.to_string(index=False),
        )

        optimal = find_optimal_threshold(cost_df)
        logger.info(
            "Threshold ótimo: %.2f | "
            "Custo total: R$%.0f | "
            "Recall: %.2f | Precision: %.2f | "
            "FN: %d | FP: %d",
            optimal["threshold"],
            optimal["total_cost"],
            optimal["recall"],
            optimal["precision"],
            optimal["FN"],
            optimal["FP"],
        )

        # Comparação: threshold padrão (0.5) vs ótimo
        default_row = cost_df[
            cost_df["threshold"] == 0.5
        ].iloc[0]
        savings = (
            default_row["total_cost"] - optimal["total_cost"]
        )
        logger.info(
            "Economia vs threshold 0.5: R$%.0f "
            "(%.1f%% de redução)",
            savings,
            savings / max(default_row["total_cost"], 1) * 100,
        )

        # Log no MLflow
        mlflow.log_metric(
            "optimal_threshold", optimal["threshold"]
        )
        mlflow.log_metric(
            "cost_at_optimal", optimal["total_cost"]
        )
        mlflow.log_metric(
            "cost_at_default_05",
            default_row["total_cost"],
        )
        mlflow.log_metric("cost_savings", savings)
        mlflow.log_metric(
            "optimal_recall", optimal["recall"]
        )
        mlflow.log_metric(
            "optimal_precision", optimal["precision"]
        )

        # Salva tabela de custos
        cost_path = MODELS_DIR / "cost_analysis.csv"
        cost_df.to_csv(cost_path, index=False)
        mlflow.log_artifact(str(cost_path))

        # ── 8. Persistência ────────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Salva modelo PyTorch
        torch_path = MODELS_DIR / "mlp_churn.pt"
        torch.save(mlp_result["model"].state_dict(), torch_path)
        logger.info("Modelo MLP salvo: %s", torch_path)

        # Salva scaler
        scaler_path = MODELS_DIR / "mlp_scaler.joblib"
        joblib.dump(mlp_result["scaler"], scaler_path)
        logger.info("Scaler salvo: %s", scaler_path)

        # Salva comparação
        comparison_path = MODELS_DIR / "model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        mlflow.log_artifact(str(comparison_path))

        logger.info("=" * 60)
        logger.info("CONCLUÍDO — Artefatos em %s", MODELS_DIR)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
