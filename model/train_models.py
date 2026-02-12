# model/train_models.py
import os
import json
import argparse
import numpy as np
import pandas as pd

from joblib import dump
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

# XGBoost
from xgboost import XGBClassifier


REQUIRED_COLS = [
    "age","job","marital","education","default","balance","housing","loan",
    "contact","day","month","duration","campaign","pdays","previous","poutcome","y"
]
FEATURE_COLS = REQUIRED_COLS[:-1]
TARGET_COL = "y"

NUMERIC_COLS = ["age","balance","day","duration","campaign","pdays","previous"]
CATEGORICAL_COLS = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]


def detect_delimiter(sample_bytes: bytes) -> str:
    head = sample_bytes[:2048].decode("utf-8", errors="ignore")
    return ";" if head.count(";") > head.count(",") else ","


def load_dataset(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        raw = f.read()
    sep = detect_delimiter(raw)
    df = pd.read_csv(StringIO(raw.decode("utf-8", errors="ignore")), sep=sep)
    # Normalize header/values (strip quotes/spaces)
    df.columns = df.columns.str.strip().str.replace('"', '')
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().str.replace('"', '')
    return df


def prepare_xy(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y_raw = df[TARGET_COL].astype(str).str.lower().str.strip()
    # Map target: yes→1, no→0
    y = y_raw.map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Target column 'y' must contain only 'yes' or 'no'.")
    return X, y


def build_preprocessor(dense_output: bool = True) -> ColumnTransformer:
    cat_enc = OneHotEncoder(handle_unknown="ignore", sparse=not dense_output)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", cat_enc)
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_COLS),
            ("cat", cat_pipe, CATEGORICAL_COLS)
        ],
        remainder="drop"
    )
    return pre


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # Score for AUC if available
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score) if y_score is not None else np.nan
    }
    return metrics


def main(data_path: str):
    os.makedirs("model", exist_ok=True)

    df = load_dataset(data_path)
    X, y = prepare_xy(df)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save feature names for Streamlit validation
    with open("model/feature_names.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    # Preprocessors
    pre_dense = build_preprocessor(dense_output=True)   # for GaussianNB
    pre_sparse = build_preprocessor(dense_output=False) # for others

    models = {
        "Logistic Regression": Pipeline([
            ("pre", pre_sparse),
            ("clf", LogisticRegression(max_iter=5000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("pre", pre_sparse),
            ("clf", DecisionTreeClassifier(random_state=42))
        ]),
        "KNN": Pipeline([
            ("pre", pre_sparse),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": Pipeline([
            ("pre", pre_dense),   # dense for GaussianNB
            ("clf", GaussianNB())
        ]),
        "Random Forest": Pipeline([
            ("pre", pre_sparse),
            ("clf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
        ]),
        "XGBoost": Pipeline([
            ("pre", pre_sparse),
            ("clf", XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=5,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, n_jobs=-1, eval_metric="logloss"
            ))
        ])
    }

    results = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        m = evaluate(pipe, X_test, y_test)
        m["Model"] = name
        results.append(m)
        # Save model
        fname = name.lower().replace(" ", "_")
        dump(pipe, f"model/{fname}.joblib")

    # Save metrics
    metrics_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ].sort_values(by="AUC", ascending=False)
    metrics_df.to_csv("model/metrics.csv", index=False)

    # Save a small test CSV for Streamlit uploads (keeps free tier happy)
    test_out = X_test.copy()
    test_out["y"] = y_test.values
    test_out.sample(n=min(200, len(test_out)), random_state=42).to_csv(
        "model/test_sample.csv", index=False
    )

    print("\n=== Evaluation Results (sorted by AUC) ===")
    print(metrics_df.to_string(index=False))
    print("\nSaved models/metrics to ./model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/bank.csv",
                        help="Path to CSV with required columns.")
    args = parser.parse_args()
    main(args.data_path)
