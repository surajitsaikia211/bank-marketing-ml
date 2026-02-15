
# Bank Marketing – Multi-Model Classification (Streamlit + Deployment)

> End-to-end ML workflow: data preparation, six classification models, evaluation metrics, Streamlit app, and deployment on Streamlit Community Cloud.

---

## a. Problem Statement

The objective is to build, evaluate, and deploy multiple machine learning **classification models** to predict whether a client will subscribe to a term deposit (`y` ∈ {no, yes}) using the **UCI Bank Marketing dataset**.  
This assignment covers the complete ML lifecycle: modeling, evaluation, UI development with Streamlit, and deployment.

---

## b. Dataset Description

- **Dataset:** Bank Marketing Data Set (UCI Machine Learning Repository)  
- **Task:** Binary classification (`yes` / `no`)  
- **Target Variable:** `y`  
- **Features Used (minimum 12 features satisfied):**
  - **Numerical:** `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`
  - **Categorical:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
- **File Format:** CSV (the original UCI file is semicolon `;` separated; the project supports both `;` and `,`)

> **Data location in repo:** `data/bank-marketing.csv`

---

## c. Models Used 

The following **six** models were implemented on the **same dataset**:
1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbour (kNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Comparison Table (Generated results here)

> After running `python train_models.py`, saved metrics results at `model/metrics.csv`.  

| ML Model Name             | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
|---------------------------|----------|------|-----------|--------|------|------|
| Logistic Regression       |          |      |           |        |      |      |
| Decision Tree             |          |      |           |        |      |      |
| kNN                       |          |      |           |        |      |      |
| Naive Bayes               |          |      |           |        |      |      |
| Random Forest (Ensemble)  |          |      |           |        |      |      |
| XGBoost (Ensemble)        |          |      |           |        |      |      |

---

## Observations about Model Performance 

> Add dataset-specific insights for each model based on your results. You may use the following as a starting point and refine after you see your metrics.

| ML Model Name             | Observation about model performance |
|---------------------------|-------------------------------------|
| **Logistic Regression**   | Strong baseline; robust with one-hot encoded categorical variables; good AUC; fast to train; may miss nonlinear interactions. |
| **Decision Tree**         | Interpretable; can overfit without pruning; performance depends on depth and split criteria. |
| **kNN**                   | Sensitive to feature scaling and dimensionality; performance varies with `k` and distance metric. |
| **Naive Bayes**           | Very fast; assumes conditional independence; can perform competitively on high-dimensional OHE features. |
| **Random Forest (Ensemble)** | Typically strong across metrics; reduces overfitting via averaging; stable and robust; useful feature importances. |
| **XGBoost (Ensemble)**    | Often the best AUC/F1 with tuning; handles complex interactions and imbalance (via `scale_pos_weight`); may require careful hyperparameters. |

---

