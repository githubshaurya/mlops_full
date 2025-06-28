# Import necessary libraries
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='githubshaurya', repo_name='mlops_final', mlflow=True)
mlflow.set_experiment("Experiment 2")  # Name of the experiment in MLflow
mlflow.set_tracking_uri('https://dagshub.com/githubshaurya/mlops_final.mlflow')

# Load and split data
data = pd.read_csv(r"/mnt/c/NLP_practice/python_learn/MLOps/mlops_final/mlflow_final/water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# Fixed function to fill missing values
def fill_missing_with_median(df):
    return df.fillna(df.median(numeric_only=True))

# Preprocess data
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

# Prepare features and target
X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"], axis=1)
y_test = test_processed_data["Potability"]

# Scale features for algorithms that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=42, solver='liblinear'), True),  # Use scaled data
    "Random Forest": (RandomForestClassifier(n_estimators=250, random_state=42), False),
    "Support Vector Classifier": (SVC(random_state=42), True),  # Use scaled data
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "K-Nearest Neighbors": (KNeighborsClassifier(), True),  # Use scaled data
    "XG Boost": (XGBClassifier(random_state=42, eval_metric='logloss'), False)
}

# Start parent MLflow run
with mlflow.start_run(run_name="Water Potability Models Experiment"):
    for model_name, (model, use_scaled) in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            # Choose scaled or original data
            X_train_use = X_train_scaled if use_scaled else X_train
            X_test_use = X_test_scaled if use_scaled else X_test
            
            # Train model
            model.fit(X_train_use, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            
            # Calculate metrics with zero_division handling
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log whether scaling was used
            mlflow.log_param("feature_scaling", use_scaled)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            
            # Save and log confusion matrix
            cm_filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
            plt.savefig(cm_filename, dpi=100, bbox_inches='tight')
            plt.close()  # Important: close the plot to free memory
            mlflow.log_artifact(cm_filename)
            
            # Save and log model
            model_filename = f"{model_name.replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_filename)
            mlflow.log_artifact(model_filename)
            
            # Log scaler if used
            if use_scaled:
                scaler_filename = f"{model_name.replace(' ', '_')}_scaler.pkl"
                joblib.dump(scaler, scaler_filename)
                mlflow.log_artifact(scaler_filename)
            
            # Set tags
            mlflow.set_tag("author", "Shaurya")
            mlflow.set_tag("model", model_name)
            
            print(f"{model_name} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

print("All models have been trained and logged successfully.")
