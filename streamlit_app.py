# streamlit_app.py
import json
import io
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os

from joblib import load
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

st.set_page_config(page_title="Bank Marketing — ML Models", layout="wide")
st.title("🏦 Bank Marketing — Classification Models Dashboard")

st.caption("Upload a small **test CSV** → choose model → view metrics, confusion matrix & classification report. "
           "CSV can be comma **or** semicolon separated (the app auto-detects).")

# ---------- Show precomputed metrics ----------
st.subheader("✅ Precomputed Evaluation Metrics (on holdout split)")
try:
    metrics_df = pd.read_csv("model/metrics.csv")
    st.dataframe(metrics_df, use_container_width=True)
except Exception:
    st.warning("metrics.csv not found — run: `python model/train_models.py --data_path data/bank-marketing.csv`")
    st.stop()

# ---------- Load required feature list ----------
try:
    with open("model/feature_names.json", "r") as f:
        required_features = json.load(f)
except Exception:
    st.error("feature_names.json not found. Train models to generate it.")
    st.stop()

# ---------- CSV upload ----------
st.subheader("✅ Upload Test CSV")
st.info("Tip: Use the sample at **model/test_sample.csv** for a quick demo. ")

uploaded = st.file_uploader("Upload CSV (comma or semicolon separated)", type=["csv"])

def detect_delimiter(b: bytes) -> str:
    head = b[:2048].decode("utf-8", errors="ignore")
    return ";" if head.count(";") > head.count(",") else ","

def read_clean_csv(file) -> pd.DataFrame:
    data = file.read()
    sep = detect_delimiter(data)
    df = pd.read_csv(io.BytesIO(data), sep=sep)
    # Clean headers and string values
    df.columns = df.columns.str.strip().str.replace('"', '')
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().str.replace('"', '')
    return df

def normalize_target(y_series: pd.Series) -> pd.Series:
    y = y_series.astype(str).str.lower().str.strip()
    mapping = {"yes": 1, "no": 0, "1":1, "0":0}
    if not y.isin(mapping).all():
        raise ValueError("Target 'y' must contain only 'yes' or 'no' or 1 or 0.")
    return y.map(mapping).astype(int)

def compute_metrics(y_true, y_pred, y_score=None):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    out["AUC"] = roc_auc_score(y_true, y_score) if y_score is not None else np.nan
    return out

if uploaded is None:
    st.stop()

df_up = read_clean_csv(uploaded)
st.write("**Preview (top 10 rows):**")
st.dataframe(df_up.head(10), use_container_width=True)

# ---------- Model selection ----------
st.subheader("✅ Model Selection")

model_map = {
    "Logistic Regression": BASE_DIR / "model" / "logistic_regression.pkl",
    "Decision Tree":       BASE_DIR / "model" / "decision_tree.joblib",
    "KNN":                 BASE_DIR / "model" / "knn.joblib",
    "Naive Bayes":         BASE_DIR / "model" / "naive_bayes.joblib",
    "Random Forest":       BASE_DIR / "model" / "random_forest.joblib",
    "XGBoost":             BASE_DIR / "model" / "xgboost.joblib",
}

model_options = ["Select"] + list(model_map.keys())
model_name = st.selectbox("Choose a model", model_options)

if model_name == "Select":
    st.warning("Please select a model to proceed.")
    st.stop()
    
try:
    model = load(model_map[model_name])
except Exception:
    st.error("Saved model not found. Please train models first.")
    st.stop()

# Validate columns
missing = [c for c in required_features if c not in df_up.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

X = df_up[required_features].copy()

# Predict
y_pred = model.predict(X)
y_score = None
if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X)[:, 1]
elif hasattr(model, "decision_function"):
    y_score = model.decision_function(X)

st.subheader("🔮 Predictions (first 20)")
st.dataframe(pd.DataFrame({"prediction (1=yes)": y_pred}).head(20), use_container_width=True)

# Metrics if y provided
if "y" in df_up.columns:
    try:
        y_true = normalize_target(df_up["y"])
        st.subheader("✅ Metrics on Uploaded Test Data")
        met = compute_metrics(y_true, y_pred, y_score)
        st.json(met)

        st.subheader("✅ Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    
        cm_df = pd.DataFrame(
        cm,
        index=["True: no (0)", "True: yes (1)"],
        columns=["Pred: no (0)", "Pred: yes (1)"]
        )
        st.table(cm_df)

        st.subheader("✅ Classification Report")
        report = classification_report(y_true, y_pred, target_names=["no (0)", "yes (1)"], zero_division=0)
        st.text(report)
    except Exception as e:
        st.warning(f"Could not compute metrics because of the target column: {e}")
else:
    st.info("No target column 'y' found in uploaded CSV → showing predictions only.")
