
# ==========================================================
# Import Libraries
# ==========================================================

import os
import joblib
import pandas as pd
import xgboost as xgb

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from huggingface_hub import (
    HfApi,
    create_repo
)

from huggingface_hub.utils import RepositoryNotFoundError

# ==========================================================
# Hugging Face API
# ==========================================================

api = HfApi(token=os.getenv("HF_TOKEN"))

# ==========================================================
# Load Train/Test Dataset
# ==========================================================

train_path = "/content/train.csv"
test_path = "/content/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Training Shape :", train_df.shape)
print("Testing Shape  :", test_df.shape)

# ==========================================================
# Split Features and Target
# ==========================================================

target_col = "ProdTaken"

Xtrain = train_df.drop(columns=[target_col])
ytrain = train_df[target_col]

Xtest = test_df.drop(columns=[target_col])
ytest = test_df[target_col]

# ==========================================================
# Feature Lists
# ==========================================================

numeric_features = Xtrain.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = Xtrain.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric Features")
print(numeric_features)

print("\nCategorical Features")
print(categorical_features)

# ==========================================================
# Class Weight
# ==========================================================

class_weight = (
    ytrain.value_counts()[0]
    /
    ytrain.value_counts()[1]
)

print("Scale Pos Weight :", class_weight)

# ==========================================================
# Preprocessing Pipeline
# ==========================================================

preprocessor = make_column_transformer(

    (StandardScaler(), numeric_features),

    (
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )

)

# ==========================================================
# XGBoost Model
# ==========================================================

xgb_model = xgb.XGBClassifier(

    random_state=42,

    scale_pos_weight=class_weight,

    eval_metric="logloss"

)

# ==========================================================
# Hyperparameter Grid
# ==========================================================

param_grid = {

    "xgbclassifier__n_estimators":[100,200],

    "xgbclassifier__max_depth":[3,5,7],

    "xgbclassifier__learning_rate":[0.01,0.05,0.1],

    "xgbclassifier__subsample":[0.8,1.0],

    "xgbclassifier__colsample_bytree":[0.8,1.0]

}

# ==========================================================
# Pipeline
# ==========================================================

model_pipeline = make_pipeline(

    preprocessor,

    xgb_model

)

# ==========================================================
# Grid Search
# ==========================================================

grid_search = GridSearchCV(

    estimator=model_pipeline,

    param_grid=param_grid,

    cv=5,

    scoring="recall",

    n_jobs=-1,

    verbose=2

)

grid_search.fit(Xtrain, ytrain)

print("\nBest Parameters")
print(grid_search.best_params_)

# ==========================================================
# Best Model
# ==========================================================

best_model = grid_search.best_estimator_

# ==========================================================
# Prediction
# ==========================================================

y_pred_train = best_model.predict(Xtrain)

y_pred_test = best_model.predict(Xtest)

# ==========================================================
# Evaluation
# ==========================================================

print("\nTraining Report")

print(

    classification_report(

        ytrain,

        y_pred_train

    )

)

print("\nTesting Report")

print(

    classification_report(

        ytest,

        y_pred_test

    )

)

print("Accuracy :", accuracy_score(ytest,y_pred_test))

print("Precision :", precision_score(ytest,y_pred_test))

print("Recall :", recall_score(ytest,y_pred_test))

print("F1 Score :", f1_score(ytest,y_pred_test))

print("ROC AUC :", roc_auc_score(ytest,y_pred_test))

# ==========================================================
# Save Model
# ==========================================================

model_name = "best_tourism_prediction_model_v1.joblib"

joblib.dump(

    best_model,

    model_name

)

print("Model Saved Successfully")

# ==========================================================
# Upload Model to Hugging Face
# ==========================================================

repo_id = "hiteshsharma/tourism-prediction-model"

try:

    api.repo_info(

        repo_id=repo_id,

        repo_type="model"

    )

    print("Model Repository Exists")

except RepositoryNotFoundError:

    print("Creating Model Repository...")

    create_repo(

        repo_id=repo_id,

        repo_type="model",

        private=False

    )

api.upload_file(

    path_or_fileobj=model_name,

    path_in_repo=model_name,

    repo_id=repo_id,

    repo_type="model"

)

print("Model Uploaded Successfully")
