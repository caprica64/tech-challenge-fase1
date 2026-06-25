"""Treinamento simples de rede neural para churn com PyTorch."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


logger = logging.getLogger(__name__)


class ChurnModel(nn.Module):
    """Rede neural MLP para classificacao binaria de churn."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] | tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(_get_activation_layer(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hidden_layers: list[int] | tuple[int, ...] = (64, 32),
    dropout: float = 0.0,
    activation: str = "relu",
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 64,
    random_state: int = 42,
    use_pos_weight: bool = False,
) -> tuple[ChurnModel, StandardScaler, list[float]]:
    """Treina a rede neural usando os conjuntos vindos do DataIngestion."""
    logger.info(
        "Iniciando treino do modelo: amostras=%s, features=%s, hidden_layers=%s, "
        "dropout=%.2f, activation=%s, learning_rate=%s, epochs=%s, batch_size=%s, "
        "use_pos_weight=%s",
        len(X_train),
        X_train.shape[1],
        hidden_layers,
        dropout,
        activation,
        learning_rate,
        epochs,
        batch_size,
        use_pos_weight,
    )

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    logger.debug("Seeds configurados com random_state=%s", random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logger.debug("StandardScaler ajustado no conjunto de treino")

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = ChurnModel(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout=dropout,
        activation=activation,
    )
    loss_function = _build_loss_function(
        y_train_tensor=y_train_tensor,
        use_pos_weight=use_pos_weight,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        losses.append(float(np.mean(epoch_losses)))

        if (epoch + 1) == 1 or (epoch + 1) == epochs or (epoch + 1) % 10 == 0:
            logger.info(
                "Epoca %s/%s finalizada com loss %.6f",
                epoch + 1,
                epochs,
                losses[-1],
            )

    logger.info("Treino concluido. Loss final: %.6f", losses[-1])
    return model, scaler, losses


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    hidden_layers: list[int] | tuple[int, ...] = (64, 32),
    dropout: float = 0.0,
    activation: str = "relu",
    learning_rate: float = 0.001,
    epochs: int = 50,
    batch_size: int = 64,
    random_state: int = 42,
    use_pos_weight: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa validacao cruzada estratificada para a rede neural."""
    logger.info(
        "Iniciando validacao cruzada estratificada: amostras=%s, features=%s, "
        "n_splits=%s, hidden_layers=%s, dropout=%.2f, learning_rate=%s, "
        "epochs=%s, use_pos_weight=%s",
        len(X),
        X.shape[1],
        n_splits,
        hidden_layers,
        dropout,
        learning_rate,
        epochs,
        use_pos_weight,
    )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_results = []

    for fold, (train_index, valid_index) in enumerate(splitter.split(X, y), start=1):
        logger.info("Iniciando fold %s/%s", fold, n_splits)

        X_train_fold = X.iloc[train_index]
        X_valid_fold = X.iloc[valid_index]
        y_train_fold = y.iloc[train_index]
        y_valid_fold = y.iloc[valid_index]

        model, scaler, losses = fit_model(
            X_train=X_train_fold,
            y_train=y_train_fold,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state + fold,
            use_pos_weight=use_pos_weight,
        )

        probabilities = predict_proba(model=model, scaler=scaler, X=X_valid_fold)
        metrics = _calculate_pr_auc(
            y_true=y_valid_fold.to_numpy(),
            probabilities=probabilities,
        )

        fold_result = {
            "fold": fold,
            "train_size": len(X_train_fold),
            "valid_size": len(X_valid_fold),
            "final_train_loss": losses[-1],
            **metrics,
        }
        fold_results.append(fold_result)

        logger.info("Fold %s/%s concluido: %s", fold, n_splits, fold_result)

    cv_results = pd.DataFrame(fold_results)
    summary = cv_results.drop(columns=["fold", "train_size", "valid_size"]).agg(
        ["mean", "std"]
    )

    logger.info("Validacao cruzada concluida. Resumo: %s", summary.to_dict())
    return cv_results, summary


def predict_proba(
    model: ChurnModel,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """Retorna a probabilidade de churn para cada amostra."""
    logger.info("Gerando probabilidades para %s amostras", len(X))

    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).numpy()

    logger.debug("Probabilidades geradas com shape=%s", probabilities.shape)
    return probabilities


def predict(
    model: ChurnModel,
    scaler: StandardScaler,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> np.ndarray:
    """Retorna as predicoes binarias do modelo."""
    logger.info(
        "Gerando predicoes binarias para %s amostras com threshold %.2f",
        len(X),
        threshold,
    )

    probabilities = predict_proba(model=model, scaler=scaler, X=X)
    predictions = (probabilities >= threshold).astype(int)

    logger.debug("Predicoes geradas com shape=%s", predictions.shape)
    return predictions


def _calculate_pr_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def _build_loss_function(
    y_train_tensor: torch.Tensor,
    use_pos_weight: bool,
) -> nn.BCEWithLogitsLoss:
    if not use_pos_weight:
        return nn.BCEWithLogitsLoss()

    positive_count = y_train_tensor.sum()
    negative_count = len(y_train_tensor) - positive_count

    if positive_count.item() == 0:
        logger.warning(
            "use_pos_weight=True foi solicitado, mas nao ha classe positiva no treino."
        )
        return nn.BCEWithLogitsLoss()

    pos_weight = negative_count / positive_count
    logger.info("Usando pos_weight=%.6f no BCEWithLogitsLoss", pos_weight.item())

    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight.item(), dtype=torch.float32)
    )


def _get_activation_layer(activation: str) -> nn.Module:
    activation_layers = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
    }

    if activation not in activation_layers:
        valid_options = ", ".join(activation_layers)
        logger.error("Ativacao invalida recebida: %s", activation)
        raise ValueError(f"Ativacao invalida: {activation}. Opcoes: {valid_options}")

    return activation_layers[activation]()
