#!/usr/bin/env python3

import os
import json
import joblib
import dagshub
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Initialize DagsHub
dagshub.init(repo_owner="githubshaurya", repo_name="mlops_final", mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/githubshaurya/mlops_final.mlflow')
mlflow.set_experiment("Model Registration")

# Load run info
reports_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "run_info.json")
with open(reports_path, 'r') as file:
    run_info = json.load(file)

model_name = run_info['model_name']

# Load the local model
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
local_model_path = os.path.join(project_root, "trained_model.pkl")

if not os.path.exists(local_model_path):
    local_model_path = os.path.join(project_root, "model.pkl")
    if not os.path.exists(local_model_path):
        raise FileNotFoundError("No model file found.")

model = joblib.load(local_model_path)

# Simple approach: Just log the model and let DagsHub handle registration
with mlflow.start_run() as run:
    print(f"Starting new run: {run.info.run_id}")
    
    # Log the model using the most basic approach
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )
    
    # Log some basic info
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("registration_run", "true")
    
    print(f"Model logged successfully in run: {run.info.run_id}")
    print(f"Model URI: runs:/{run.info.run_id}/model")
    print(f"Model name: {model_name}")
    
    # Try to register manually
    try:
        model_uri = f"runs:/{run.info.run_id}/model"
        reg = mlflow.register_model(model_uri, model_name)
        print(f"Model registered successfully! Version: {reg.version}")
        
        # Try to transition to staging
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=reg.version,
            stage="Staging",
            archive_existing_versions=True
        )
        print(f"Model transitioned to Staging stage")
        
    except Exception as e:
        print(f"Registration failed: {e}")
        print("You can manually register the model from the DagsHub UI using:")
        print(f"Model URI: runs:/{run.info.run_id}/model")
        print(f"Model name: {model_name}")

print("Script completed. Check DagsHub UI for results.") 