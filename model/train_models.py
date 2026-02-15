# model/train_models.py
import pandas as pd
import numpy as np

import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# -------------------------
# 1) Load the dataset
# -------------------------
df = pd.read_csv("bank-marketing.csv", sep=";")

# -------------------------
# 2) Target variable
# -------------------------

target_col = "y"

# Convert target to 0/1
df[target_col] = df[target_col].map({"no": 0, "yes": 1})

# -------------------------
# 3) Separate X and y
# -------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------
# 4) Identify numeric and categorical columns
# -------------------------
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

# -------------------------
# 5) Preprocessing pipelines
# -------------------------

# For numeric data:
# - fill missing values with median
# - scale features using StandardScaler
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# For categorical data:
# - fill missing values with most_frequent
# - one-hot encode
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine numeric + categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# -------------------------
# 6) Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape, y_train.shape)
print("Test Shape :", X_test.shape, y_test.shape)

# -------------------------
# 7) Fit-transform preprocessing
# -------------------------
# This produces final ML-ready numeric arrays
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Train Shape:", X_train_processed.shape)
print("Processed Test Shape :", X_test_processed.shape)

print("\n✅ Step 2 completed successfully.")

# -------------------------
# STEP 2 (Load + Preprocess)
# -------------------------
df = pd.read_csv("bank-marketing.csv", sep=";")

# Target conversion yes/no -> 1/0
df["y"] = df["y"].map({"no": 0, "yes": 1})

X = df.drop(columns=["y"])
y = df["y"]

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# For Naive Bayes GaussianNB (needs dense arrays)
X_train_dense = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
X_test_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

print("✅ Data preprocessing completed.")
print("Train processed shape:", X_train_processed.shape)
print("Test processed shape :", X_test_processed.shape)

# -------------------------
# STEP 3: Define 6 Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10,random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
}

# -------------------------
# STEP 4: Metrics function
# -------------------------
def evaluate_model(model_name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)

    # Predictions
    y_pred = model.predict(Xte)

    # Probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(Xte)[:, 1]
    elif hasattr(model, "decision_function"):
        # convert decision scores to AUC usable values
        y_prob = model.decision_function(Xte)
    else:
        y_prob = y_pred  # fallback

    # Metrics
    acc = accuracy_score(yte, y_pred)
    auc = roc_auc_score(yte, y_prob)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec = recall_score(yte, y_pred, zero_division=0)
    f1 = f1_score(yte, y_pred, zero_division=0)
    mcc = matthews_corrcoef(yte, y_pred)

    # Confusion Matrix + Report
    cm = confusion_matrix(yte, y_pred)

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(yte, y_pred, zero_division=0))

    return {
        "ML Model": model_name,
        "Accuracy": round(acc, 4),
        "AUC Score": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4)
    }

# -------------------------
# Train + evaluate all models
# -------------------------
results = []

for name, model in models.items():
    # Use dense only for Naive Bayes
    if name == "Naive Bayes":
        row = evaluate_model(name, model, X_train_dense, X_test_dense, y_train, y_test)
    else:
        row = evaluate_model(name, model, X_train_processed, X_test_processed, y_train, y_test)
    results.append(row)

# Final comparison table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1 Score", ascending=False)

print("\n\n✅ FINAL MODEL COMPARISON TABLE")
print(results_df)

# Save table to CSV (optional)
results_df.to_csv("metrics.csv", index=False)
print("\n✅ Results saved as model_comparison_results.csv")

# Create model folder
os.makedirs("model", exist_ok=True)

# Save preprocessor
joblib.dump(preprocessor, "model/preprocessor.pkl")
print("✅ Saved: model/preprocessor.pkl")

# Save each trained model
joblib.dump(models["Logistic Regression"], "model/logistic_model.pkl")
joblib.dump(models["Decision Tree"], "model/decision_tree_model.pkl")
joblib.dump(models["KNN"], "model/knn_model.pkl")
joblib.dump(models["Naive Bayes"], "model/naive_bayes_model.pkl")
joblib.dump(models["Random Forest"], "model/random_forest_model.pkl")
joblib.dump(models["XGBoost"], "model/xgboost_model.pkl")

print("✅ All 6 models saved in /model folder")

# Load original dataset
df = pd.read_csv("bank-marketing.csv", sep=";")

# Take a small random sample for testing (example: 500 rows)
test_sample = df.sample(n=500, random_state=42)

# Save as CSV for Streamlit testing
test_sample.to_csv("test_sample.csv", index=False)

print("✅ Created test_sample.csv successfully!")
print(test_sample.head())

