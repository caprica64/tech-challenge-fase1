"""Treinamento simples de rede neural para churn com PyTorch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
) -> tuple[ChurnModel, StandardScaler, list[float]]:
    """Treina a rede neural usando os conjuntos vindos do DataIngestion."""
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

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
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for _ in range(epochs):
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

    return model, scaler, losses


def predict_proba(
    model: ChurnModel,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """Retorna a probabilidade de churn para cada amostra."""
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).numpy()

    return probabilities


def predict(
    model: ChurnModel,
    scaler: StandardScaler,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> np.ndarray:
    """Retorna as predicoes binarias do modelo."""
    probabilities = predict_proba(model=model, scaler=scaler, X=X)
    return (probabilities >= threshold).astype(int)


def _get_activation_layer(activation: str) -> nn.Module:
    activation_layers = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
    }

    if activation not in activation_layers:
        valid_options = ", ".join(activation_layers)
        raise ValueError(f"Ativacao invalida: {activation}. Opcoes: {valid_options}")

    return activation_layers[activation]()
