import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import dagshub
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='githubshaurya', repo_name='mlops_final', mlflow=True)
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment("Experiment 4")

# Load and split data
data = pd.read_csv(r"/mnt/c/NLP_practice/python_learn/MLOps/mlops_final/mlflow_final/water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# FIXED: Proper function to handle missing values
def fill_missing_with_mean(df):
    df_copy = df.copy()
    for column in df_copy.columns:
        if df_copy[column].isnull().any():
            mean_value = df_copy[column].mean()
            df_copy[column] = df_copy[column].fillna(mean_value)
    return df_copy

# Process data
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

# Prepare features and target
X_train = train_processed_data.drop(columns=["Potability"], axis=1)
y_train = train_processed_data["Potability"]
X_test = test_processed_data.drop(columns=["Potability"], axis=1)
y_test = test_processed_data["Potability"]

# Check class distribution
print("Class distribution:")
print(y_train.value_counts(normalize=True))

# FIXED: Add class balancing to Random Forest
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

param_dist = {
    'n_estimators': [100, 200, 300, 500, 1000],  # Different values of n_estimators to try
    'max_depth': [None, 4, 5, 6, 10],  # Different max_depth values to explore
}

# Perform RandomizedSearchCV to find the best hyperparameters for the Random Forest model
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)

# Start MLflow run
with mlflow.start_run(run_name="Random Forest Hyperparameter Tuning") as parent_run:
    
    # Fit RandomizedSearchCV
    print("Starting hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    
    # IMPROVED: Log only top 5 combinations instead of all 20
    # Get indices of top 5 performing combinations
    top_indices = np.argsort(random_search.cv_results_['mean_test_score'])[-5:]
    
    for idx, i in enumerate(top_indices):
        with mlflow.start_run(run_name=f"Top_{idx+1}_Combination", nested=True):
            mlflow.log_params(random_search.cv_results_['params'][i])
            mlflow.log_metric("cv_f1_score", random_search.cv_results_['mean_test_score'][i])
            mlflow.log_metric("cv_std", random_search.cv_results_['std_test_score'][i])

    # Log best parameters
    print("Best parameters found:", random_search.best_params_)
    print("Best CV F1 score:", random_search.best_score_)
    
    mlflow.log_params(random_search.best_params_)
    mlflow.log_metric("best_cv_f1_score", random_search.best_score_)
    
    # Get best model
    best_rf = random_search.best_estimator_
    
    # Make predictions on test set
    y_pred = best_rf.predict(X_test)
    
    # FIXED: Calculate metrics with zero_division handling
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Log test metrics
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    
    # FIXED: Use artifact-based model logging (DagsHub compatible)
    model_filename = "best_random_forest_model.pkl"
    joblib.dump(best_rf, model_filename)
    mlflow.log_artifact(model_filename)
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    
    # REMOVED: These lines cause DagsHub errors
    # mlflow.log_input() - can cause large storage usage
    # infer_signature() and mlflow.sklearn.log_model() - not supported by DagsHub
    
    # Log hyperparameter tuning details
    mlflow.log_param("tuning_method", "RandomizedSearchCV")
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("n_iter", 20)
    mlflow.log_param("scoring_metric", "f1")
    
    # Set tags
    mlflow.set_tag("author", "Shaurya")
    mlflow.set_tag("model_type", "RandomForest_Tuned")
    mlflow.set_tag("purpose", "hyperparameter_optimization")
    
    # Log source code
    mlflow.log_artifact(__file__)
    
    # Print results
    print(f"\nBest Model Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    print(f"\nTop 3 Most Important Features:")
    for i in range(min(3, len(feature_importance))):
        feat = feature_importance.iloc[i]
        print(f"{feat['feature']}: {feat['importance']:.4f}")

print("Hyperparameter tuning completed successfully!")