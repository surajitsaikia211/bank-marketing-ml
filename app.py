import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Bank Marketing — ML Models", layout="wide")
st.title("🏦 Bank Marketing — Classification Models Dashboard")

st.caption("Upload a small **test CSV** → choose model → view metrics, confusion matrix & classification report. "
           "CSV can be comma **or** semicolon separated (the app auto-detects).")

# ----------------------------
# Load preprocessor + models
# ----------------------------
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("model/preprocessor.pkl")

    models = {
        "Logistic Regression": joblib.load("model/logistic_model.pkl"),
        "Decision Tree": joblib.load("model/decision_tree_model.pkl"),
        "KNN": joblib.load("model/knn_model.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes_model.pkl"),
        "Random Forest": joblib.load("model/random_forest_model.pkl"),
        "XGBoost": joblib.load("model/xgboost_model.pkl"),
    }
    return preprocessor, models

preprocessor, models = load_assets()

# ----------------------------
# Upload CSV (test data)
# ----------------------------

st.subheader("✅ Upload Test CSV")
st.info("Tip: Use the sample at **model/test_sample.csv** for a quick demo. ")

uploaded_file = st.file_uploader("Upload CSV (comma or semicolon separated)", type=["csv"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Dataset:")
    st.dataframe(df_test.head())

    if "y" not in df_test.columns:
        st.error("❌ Target column 'y' is missing in uploaded file. Please include 'y' column.")
        st.stop()

    # Convert target y (if yes/no)
    if df_test["y"].dtype == "object":
        df_test["y"] = df_test["y"].map({"no": 0, "yes": 1})

    X_test = df_test.drop(columns=["y"])
    y_test = df_test["y"]

    # ----------------------------
    # Select Model
    # ----------------------------
    st.subheader("✅ Model Selection")

    model_options = ["Select"] + list(models.keys())

    selected_model_name = st.selectbox(
    "Choose a classifier model:",
    model_options,
    index=0
    )

    if selected_model_name == "Select":
       st.warning("Please choose a model to continue.")
       st.stop()

    model = models[selected_model_name]

    # ----------------------------
    # Predict + Metrics
    # ----------------------------
    st.subheader("✅ Results & Performance Metrics")
    expected_cols = preprocessor.feature_names_in_
    missing = set(expected_cols) - set(X_test.columns)
    if missing:
       st.error(f"Missing columns: {missing}")
       st.stop()
    # Ensure uploaded data matches training columns

    # Match training schema exactly
    # Expected columns from training dataset
    expected_cols = [
       "age","job","marital","education","default","balance",
       "housing","loan","contact","day","month","duration",
       "campaign","pdays","previous","poutcome"
     ]

    missing = set(expected_cols) - set(X_test.columns)
    if missing:
       st.error(f"Missing columns: {missing}")
       st.stop()

 
    # Reorder columns exactly
    X_test = X_test[expected_cols]

    # Force numeric columns
    numeric_cols = [
        "age", "balance", "day", "duration",
        "campaign", "pdays", "previous"
    ]

    for col in numeric_cols:
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Preprocess
    X_test_processed = preprocessor.transform(X_test)

    # Naive Bayes requires dense array
    if selected_model_name == "Naive Bayes":
       if hasattr(X_test_processed, "toarray"):
          X_test_processed = X_test_processed.toarray()

    # Predictions
    y_pred = model.predict(X_test_processed)

    # Probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_processed)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test_processed)
    else:
        y_prob = y_pred

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Precision", f"{prec:.4f}")
    col5.metric("Recall", f"{rec:.4f}")
    col6.metric("F1 Score", f"{f1:.4f}")

    # Confusion matrix and report
    st.subheader("✅ Confusion Matrix & Classification Report")

    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.dataframe(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]))

    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    # Show predictions
    st.subheader("✅ Predictions Output")
    output_df = X_test.copy()
    output_df["Actual_y"] = y_test.values
    output_df["Predicted_y"] = y_pred
    st.dataframe(output_df.head(20))

    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions_output.csv",
        mime="text/csv"
    )
