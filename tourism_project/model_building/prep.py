
# ==========================================================
# Import Libraries
# ==========================================================
import os
import pandas as pd

from sklearn.model_selection import train_test_split

from huggingface_hub import HfApi

# ==========================================================
# Hugging Face Configuration
# ==========================================================

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/hiteshsharma/tourism-dataset/tourism.csv"
# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Dataset Shape :", df.shape)

# ==========================================================
# Basic Cleaning
# ==========================================================

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Drop unwanted columns
drop_cols = []

if "Unnamed: 0" in df.columns:
    drop_cols.append("Unnamed: 0")

if "CustomerID" in df.columns:
    drop_cols.append("CustomerID")

df.drop(columns=drop_cols, inplace=True)

print("Columns after dropping IDs:")
print(df.columns.tolist())

# ==========================================================
# Data Cleaning
# ==========================================================

# Gender

df["Gender"] = (
    df["Gender"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["Gender"] = df["Gender"].replace({
    "fe male": "female"
})

df["Gender"] = df["Gender"].str.title()

# Marital Status

df["MaritalStatus"] = (
    df["MaritalStatus"]
    .astype(str)
    .str.strip()
    .str.title()
)

df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Unmarried":"Single"
})

# ==========================================================
# Missing Values
# ==========================================================

num_cols = df.select_dtypes(include=["int64","float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# ==========================================================
# Define Target
# ==========================================================

target_col = "ProdTaken"

X = df.drop(columns=[target_col])

y = df[target_col]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ==========================================================
# Create Final CSV Files
# ==========================================================

train_df = X_train.copy()
train_df[target_col] = y_train

test_df = X_test.copy()
test_df[target_col] = y_test

# ==========================================================
# Save Files
# ==========================================================

train_file = "tourism_project/data/train.csv"
test_file = "tourism_project/data/test.csv"

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print("Train.csv Saved")
print("Test.csv Saved")

# ==========================================================
# Upload to Hugging Face Dataset Repository
# ==========================================================

repo_id = "hiteshsharma/tourism-dataset" # Changed repo_id to match data_registry.py

for file in [train_file, test_file]:

    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=repo_id,
        repo_type="dataset"
    )

print("Train and Test files uploaded successfully.")
