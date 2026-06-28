
# ==========================================================
# Import Libraries and model training
# ==========================================================

import os
import joblib
import pandas as pd
import xgboost as xgb
import mlflow

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_DIR = os.path.join(BASE_DIR, "..", "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:{os.path.abspath(MLFLOW_DIR)}")
mlflow.set_experiment("Tourism Package Prediction")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is missing.")

api = HfApi(token=HF_TOKEN)

train_df = pd.read_csv("hf://datasets/hiteshsharma/tourism-dataset/train.csv")
test_df = pd.read_csv("hf://datasets/hiteshsharma/tourism-dataset/test.csv")

target = "ProdTaken"
Xtrain = train_df.drop(columns=[target])
ytrain = train_df[target]
Xtest = test_df.drop(columns=[target])
ytest = test_df[target]

numeric_features = Xtrain.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=["object"]).columns.tolist()

scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

pipeline = make_pipeline(
    preprocessor,
    xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
)

param_grid = {
    "xgbclassifier__n_estimators":[100,200],
    "xgbclassifier__max_depth":[3,5,7],
    "xgbclassifier__learning_rate":[0.01,0.05,0.1],
    "xgbclassifier__subsample":[0.8,1.0],
    "xgbclassifier__colsample_bytree":[0.8,1.0]
}

grid = GridSearchCV(pipeline,param_grid=param_grid,cv=5,scoring="recall",n_jobs=-1,verbose=2)

with mlflow.start_run(run_name="XGBoost GridSearch"):

    model_name = "best_tourism_prediction_model_v1.joblib"

    # Train model
    grid.fit(Xtrain, ytrain)

    print("\nBest Parameters")
    print(grid.best_params_)

    best_model = grid.best_estimator_
    mlflow.log_params(grid.best_params_)
    # Predictions
    preds = best_model.predict(Xtest)
    train_preds = best_model.predict(Xtrain)

    # ============================
    # Calculate Evaluation Metrics
    # ============================

    accuracy = accuracy_score(ytest, preds)
    precision = precision_score(ytest, preds)
    recall = recall_score(ytest, preds)
    f1 = f1_score(ytest, preds)
    roc = roc_auc_score(ytest, preds)

    # ============================
    # Log Parameters
    # ============================

    mlflow.log_params(grid.best_params_)

    # ============================
    # Log Metrics
    # ============================

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)
    mlflow.log_metric("ROC AUC", roc)

    # ============================
    # Save Model
    # ============================

    joblib.dump(best_model, model_name)

    # ============================
    # Log Model Artifact
    # ============================

    mlflow.log_artifact(model_name, artifact_path="model")

print("\nTraining Report")
print(classification_report(ytrain, train_preds))

print("\nTesting Report")
print(classification_report(ytest, preds))

print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)
print("ROC AUC :", roc)
print("Best Parameters")
print(grid.best_params_)

print(classification_report(ytest,preds))
train_preds = best_model.predict(Xtrain)

print(classification_report(ytrain, train_preds))
repo_id="hiteshsharma/tourism-prediction-model"
try:
    api.repo_info(repo_id=repo_id,repo_type="model")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id,repo_type="model",private=False)

api.upload_file(
    path_or_fileobj=model_name,
    path_in_repo=model_name,
    repo_id=repo_id,
    repo_type="model"
)

print("Training completed successfully.")
print("Model uploaded to Hugging Face successfully.")
