#!/usr/bin/env python3

import os
import json
import pickle

import pandas as pd
import joblib
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

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


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = df.drop(columns=["Potability"])
        y = df["Potability"]
        return X, y
    except Exception as e:
        raise RuntimeError(f"Error preparing data: {e}")


def load_model(filepath: str):
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {filepath}: {e}")


def evaluate_model(model, X, y) -> dict:
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
    }


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ─── 1. Initialize DagsHub & MLflow ───────────────────────────────────────────
dagshub.init(repo_owner="githubshaurya", repo_name="mlops_final", mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/githubshaurya/mlops_final.mlflow')
mlflow.set_experiment("Final experiment 1")

# Fix the path to correctly reference the reports directory from project root
reports_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "run_info.json")
with open(reports_path, 'r') as file:
    run_info = json.load(file)

model_name = run_info['model_name']  # Fetch model name from the JSON file

# Create an MLflow client
client = MlflowClient()

# Use the local model file instead of downloading from remote server
# The model should be available locally as 'trained_model.pkl' or 'model.pkl'
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
local_model_path = os.path.join(project_root, "trained_model.pkl")

# Check if trained_model.pkl exists, otherwise try model.pkl
if not os.path.exists(local_model_path):
    local_model_path = os.path.join(project_root, "model.pkl")
    if not os.path.exists(local_model_path):
        raise FileNotFoundError("No model file found. Please ensure 'trained_model.pkl' or 'model.pkl' exists in the project root.")

# Load the model from the local pickle file
model = joblib.load(local_model_path)

# Now log the model as an MLflow model in a new run using the older API
with mlflow.start_run() as new_run:
    # Log the model as an MLflow model using the older API with artifact_path
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"  # Use artifact_path instead of name for compatibility
    )
    
    # Get the model URI from the new run
    model_uri = f"runs:/{new_run.info.run_id}/model"
    
    # Register the model using the traditional approach
    reg = mlflow.register_model(model_uri, model_name)
    
    # Get the model version
    model_version = reg.version
    
    # Transition the model version to Staging
    new_stage = "Staging"
    
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=True
    )
    
    print(f"Model {model_name} version {model_version} transitioned to {new_stage} stage.")