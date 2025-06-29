#!/usr/bin/env python3
import os
import json
import pickle

import numpy as np
import pandas as pd
import joblib
import dagshub
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Error loading data from {filepath}: {e}")


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=["Potability"], axis=1)
        y = data["Potability"]
        return X, y
    except Exception as e:
        raise RuntimeError(f"Error preparing data: {e}")


def load_model(filepath: str):
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {filepath}: {e}")


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
    except Exception as e:
        raise RuntimeError(f"Error evaluating model: {e}")


def save_metrics(metrics: dict, path: str) -> None:
    try:
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"Error saving metrics to {path}: {e}")


def main():
    # 1. Initialize DagsHub (configures MLflow for this repo)
    dagshub.init(repo_owner="githubshaurya", repo_name="mlops_full", mlflow=True)

    # 2. Set MLflow tracking URI before experiments
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

    # 3. Create or set the experiment on the remote server
    mlflow.set_experiment("Final experiment 1")

    # Paths
    test_data_path = "./data/processed/test_processed_mean.csv"
    model_path = "model.pkl"
    metrics_path = "metrics.json"
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    run_info_path = os.path.join(reports_dir, "run_info.json")

    # Load data and model
    data = load_data(test_data_path)
    X_test, y_test = prepare_data(data)
    model = load_model(model_path)

    # Compute and save metrics locally
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, metrics_path)

    # Start MLflow run and log everything
    with mlflow.start_run() as run:
        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        # Plot and log confusion matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        # Log code file
        mlflow.log_artifact(__file__)

        # Save and log trained model
        trained_model_path = "trained_model.pkl"
        joblib.dump(model, trained_model_path)
        mlflow.log_artifact(trained_model_path)

        # Save run info
        run_info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "model_name": "Best Model",
        }
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=4)
        mlflow.log_artifact(run_info_path)

        # Set tags
        mlflow.set_tag("author", "Shaurya")
        mlflow.set_tag("model", "RandomForest")

        # Console output
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print("Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
