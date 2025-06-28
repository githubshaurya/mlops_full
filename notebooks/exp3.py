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
import pickle
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='githubshaurya', repo_name='mlops_final', mlflow=True)
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment("Experiment 3")

# Load and split data
data = pd.read_csv(r"/mnt/c/NLP_practice/python_learn/MLOps/mlops_final/mlflow_final/water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# FIXED: Proper function to handle missing values
def fill_missing_with_mean(df):
    # Create a copy to avoid the FutureWarning
    df_copy = df.copy()
    for column in df_copy.columns:
        if df_copy[column].isnull().any():
            mean_value = df_copy[column].mean()
            # Use this approach instead of inplace=True
            df_copy[column] = df_copy[column].fillna(mean_value)
    return df_copy

# Fill missing values with mean
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

# Prepare features and target
X_train_original = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]
X_test_original = test_processed_data.drop(columns=["Potability"], axis=1)
y_test = test_processed_data["Potability"]

# FIXED: Create scaled versions separately
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_original)
X_test_scaled = scaler.transform(X_test_original)

# Check class distribution
print("Class distribution in training data:")
print(y_train.value_counts(normalize=True))

# FIXED: Define models with proper scaling logic and class balancing
models = {
    "Logistic Regression": (LogisticRegression(
        max_iter=5000, 
        random_state=42, 
        solver='liblinear',
        class_weight='balanced'  # ADDED: Handle class imbalance
    ), True),
    "Random Forest": (RandomForestClassifier(
        n_estimators=250, 
        random_state=42,
        class_weight='balanced'  # ADDED: Handle class imbalance
    ), False),
    "Support Vector Classifier": (SVC(
        random_state=42,
        class_weight='balanced'  # ADDED: Handle class imbalance
    ), True),
    "Decision Tree": (DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'  # ADDED: Handle class imbalance
    ), False),
    "K-Nearest Neighbors": (KNeighborsClassifier(), True),
    "XG Boost": (XGBClassifier(
        random_state=42, 
        eval_metric='logloss',
        scale_pos_weight=1  # Will be calculated below
    ), False)
}

# Calculate scale_pos_weight for XGBoost
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
models["XG Boost"] = (XGBClassifier(
    random_state=42, 
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
), False)

# Start MLflow experiment
with mlflow.start_run(run_name="Water Potability Models Experiment"):
    for model_name, (model, use_scaled) in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            
            # FIXED: Use appropriate data based on use_scaled flag
            if use_scaled:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
                mlflow.log_param("feature_scaling", True)
            else:
                X_train_use = X_train_original
                X_test_use = X_test_original
                mlflow.log_param("feature_scaling", False)
            
            # Train model
            model.fit(X_train_use, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_use)
            
            # ADDED: Check what the model is predicting
            unique_predictions = np.unique(y_pred)
            print(f"{model_name} - Unique predictions: {unique_predictions}")
            
            # FIXED: Calculate metrics with zero_division handling
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log class distribution info
            mlflow.log_param("class_balance_used", True)
            
            # FIXED: Confusion matrix with proper memory management
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            
            # Save confusion matrix
            cm_filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
            plt.savefig(cm_filename, dpi=100, bbox_inches='tight')
            plt.close()  # ADDED: Close plot to prevent memory leak
            
            # Log confusion matrix
            mlflow.log_artifact(cm_filename)
            
            # Save and log model
            model_filename = f"{model_name.replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_filename)
            mlflow.log_artifact(model_filename)
            
            # FIXED: Only log scaler if it was used
            if use_scaled:
                scaler_filename = f"{model_name.replace(' ', '_')}_scaler.pkl"
                joblib.dump(scaler, scaler_filename)
                mlflow.log_artifact(scaler_filename)
            
            # Set tags
            mlflow.set_tag("author", "Shaurya")
            mlflow.set_tag("model", model_name)
            
            # Print results
            print(f"{model_name} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

print("All models have been trained and logged successfully.")
