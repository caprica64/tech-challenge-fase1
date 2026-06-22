import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

def train_baselines(X_train, X_test, y_train, y_test):
    mlflow.set_experiment("Churn_Baselines")

    # Dummy
    with mlflow.start_run(run_name="Dummy_Classifier"):
        dummy = DummyClassifier(strategy="stratified", random_state=42) # Seed
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_test)
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.sklearn.log_model(dummy, "model")

    # Regressão Logística
    with mlflow.start_run(run_name="Logistic_Regression"):
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.log_metric("auc_roc", roc_auc_score(y_test, y_prob))
        mlflow.sklearn.log_model(lr, "model")