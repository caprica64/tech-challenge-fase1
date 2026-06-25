"""Funcoes de carga e tratamento dos dados de churn usados no projeto."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "data_telcom_customer_churn_ibm.csv"
)
MODEL_FEATURES = [
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "is_new_customer",
    "phone_services",
    "multiples_lines",
    "internet_dsl",
    "internet_fiber",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract_month_to_month",
    "contract_one_year",
    "contract_two_year",
    "paperless_billing",
    "payment_method_mailed_check",
    "payment_method_electronic_check",
    "payment_method_bank_transfer_automatic",
    "payment_method_credit_card_automatic",
    "monthly_charges",
    "total_charges",
    "avg_ticket",
    "high_risk_profile",
]
TARGET_COLUMN = "target"


logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Classe responsavel pela ingestao e tratamento dos dados de churn.

    Exemplo de uso:
    df = DataIngestion().treat_data()
    """

    def __init__(self, data_path: Path = PREPROCESSED_DATA_PATH):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Carrega os dados pre-processados de churn."""
        logger.info("Carregando dados de churn em: %s", self.data_path)

        if not self.data_path.exists():
            logger.error("Arquivo de dados nao encontrado: %s", self.data_path)
            raise FileNotFoundError(
                f"Arquivo de dados nao encontrado: {self.data_path}"
            )

        df = pd.read_csv(self.data_path)
        logger.info("Dados carregados com shape=%s", df.shape)
        return df

    def treat_data(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Realiza o tratamento de dados, incluindo limpeza, transformacoes
        e criacao de novas features.
        """
        if df is None:
            df = self.load_data()

        logger.info("Iniciando tratamento dos dados com shape=%s", df.shape)
        df = df.copy()
        self._validate_required_columns(df)

        num_cols = ["monthly_charges", "total_charges", "tenure_months"]
        for col in num_cols:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace(["", " ", "nan", "None"], np.nan)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        total_charges_mode = df["total_charges"].mode()[0]
        df["total_charges"] = df["total_charges"].fillna(total_charges_mode)

        df["gender_male"] = np.where(df["gender"] == "Male", 1, 0)
        df["senior_citizen"] = self._yes_no_to_binary(df["senior_citizen"])
        df["partner"] = self._yes_no_to_binary(df["partner"])
        df["dependents"] = self._yes_no_to_binary(df["dependents"])
        df["phone_services"] = self._yes_no_to_binary(df["phone_service"])
        df["paperless_billing"] = self._yes_no_to_binary(df["paperless_billing"])

        df["is_new_customer"] = np.where(df["tenure_months"].isin([1, 2]), 1, 0)
        df["multiples_lines"] = np.where(
            df["multiple_lines"].isin(["No", "No phone service"]),
            0,
            np.where(df["multiple_lines"] == "Yes", 1, np.nan),
        )

        df["internet_dsl"] = np.where(df["internet_service"] == "DSL", 1, 0)
        df["internet_fiber"] = np.where(df["internet_service"] == "Fiber optic", 1, 0)
        df["internet_none"] = np.where(df["internet_service"] == "No", 1, 0)

        cols_services = [
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ]
        for col in cols_services:
            df[col] = np.where(
                df[col] == "Yes",
                1,
                np.where(df[col].isin(["No", "No internet service"]), 0, np.nan),
            )

        df["contract_month_to_month"] = np.where(
            df["contract"] == "Month-to-month", 1, 0
        )
        df["contract_one_year"] = np.where(df["contract"] == "One year", 1, 0)
        df["contract_two_year"] = np.where(df["contract"] == "Two year", 1, 0)

        df["payment_method_mailed_check"] = np.where(
            df["payment_method"] == "Mailed check", 1, 0
        )
        df["payment_method_electronic_check"] = np.where(
            df["payment_method"] == "Electronic check", 1, 0
        )
        df["payment_method_bank_transfer_automatic"] = np.where(
            df["payment_method"] == "Bank transfer (automatic)", 1, 0
        )
        df["payment_method_credit_card_automatic"] = np.where(
            df["payment_method"] == "Credit card (automatic)", 1, 0
        )

        df["target"] = df["churn_value"]
        df["high_risk_profile"] = np.where(
            (df["is_new_customer"] == 1)
            & (df["contract_month_to_month"] == 1)
            & (df["internet_fiber"] == 1),
            1,
            0,
        )

        df["avg_ticket"] = np.where(
            df["tenure_months"] > 0,
            df["total_charges"] / df["tenure_months"],
            0,
        )
        df["avg_ticket"] = np.where(
            np.isinf(df["avg_ticket"]),
            np.nan,
            df["avg_ticket"],
        )
        df["avg_ticket"] = df["avg_ticket"].fillna(0)

        binary_cols = [
            "senior_citizen",
            "partner",
            "dependents",
            "is_new_customer",
            "phone_services",
            "multiples_lines",
            "internet_dsl",
            "internet_fiber",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
            "contract_month_to_month",
            "contract_one_year",
            "contract_two_year",
            "paperless_billing",
            "payment_method_mailed_check",
            "payment_method_electronic_check",
            "payment_method_bank_transfer_automatic",
            "payment_method_credit_card_automatic",
            "high_risk_profile",
        ]

        df["tenure_months"] = df["tenure_months"].fillna(0).astype(int)
        for col in binary_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        logger.info("Tratamento dos dados concluido com shape=%s", df.shape)
        return df

    def split_train_test(
        self,
        df: pd.DataFrame | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ):
        """Divide os dados em treino e teste"""
        logger.info(
            "Iniciando divisao treino/teste: test_size=%s, random_state=%s, "
            "stratify=%s",
            test_size,
            random_state,
            stratify,
        )

        self._validate_modeling_columns(df)

        X = df[MODEL_FEATURES]
        y = df[TARGET_COLUMN]
        stratify_values = y if stratify else None

        split = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_values,
        )
        X_train, X_test, y_train, y_test = split

        logger.info(
            "Divisao concluida: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
        )
        return split

    @staticmethod
    def _yes_no_to_binary(series: pd.Series) -> pd.Series:
        return np.where(series == "Yes", 1, np.where(series == "No", 0, np.nan))

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame) -> None:
        required_columns = [
            "gender",
            "senior_citizen",
            "partner",
            "dependents",
            "tenure_months",
            "phone_service",
            "multiple_lines",
            "internet_service",
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
            "contract",
            "paperless_billing",
            "payment_method",
            "monthly_charges",
            "total_charges",
            "churn_value",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error("Colunas obrigatorias ausentes: %s", missing_columns)
            raise ValueError(f"Colunas obrigatorias ausentes: {missing_columns}")

    @staticmethod
    def _validate_modeling_columns(df: pd.DataFrame) -> None:
        required_columns = [*MODEL_FEATURES, TARGET_COLUMN]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error("Colunas de modelagem ausentes: %s", missing_columns)
            raise ValueError(f"Colunas de modelagem ausentes: {missing_columns}")
