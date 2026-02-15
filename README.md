
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

| ML Model Name             | Accuracy | AUC      | Precision | Recall    | F1       | MCC      |
|---------------------------|----------|----------|-----------|-----------|----------|----------|
| Logistic Regression       | 0.901250 | 0.905574 | 0.644483  | 0.347826  | 0.451811 | 0.426058 |
| Decision Tree             | 0.872830 | 0.700865 | 0.458182  | 0.476371  | 0.467099 | 0.395027 |
| kNN                       | 0.898596 | 0.850027 | 0.625668  | 0.331758  | 0.433601 | 0.407007 |         
| Naive Bayes               | 0.854805 | 0.810095 | 0.405904  | 0.519849  | 0.455864 | 0.377358 |          
| Random Forest (Ensemble)  | 0.908106 | 0.929519 | 0.676516  | 0.411153  | 0.511464 | 0.481630 |          
| XGBoost (Ensemble)        | 0.910649 | 0.934451 | 0.658629  | 0.490548  | 0.562297 | 0.520645 | 

---

## Observations about Model Performance 

> Added dataset-specific insights for each model based on metrics results. 

| ML Model Name             | Observation about model performance |
|---------------------------|-------------------------------------|
| **Logistic Regression**   | Strong baseline with good AUC (0.9056) and decent precision (0.6445). Recall is modest (0.3478), indicating it misses several positives in this imbalanced dataset. |
| **Decision Tree**         | nterpretable but comparatively weaker AUC (0.7009). Balanced precision/recall (~0.46–0.48) but overall lags behind ensembles. Potential overfitting without pruning/tuning. |
| **kNN**                   | Reasonable precision (0.6257) but the lowest recall (0.3318) among non‑NB models; sensitive to feature scaling and high dimensionality. |
| **Naive Bayes**           | Best recall (0.5198) among non‑boosted models, but precision is low (0.4059), leading to more false positives; overall accuracy and AUC are modest. |
| **Random Forest (Ensemble)** | Strong overall—high AUC (0.9295) and best precision (0.6765); recall remains moderate (0.4112). Very good MCC (0.4816), indicating robust balanced performance. |
| **XGBoost (Ensemble)**    | Best overall—highest accuracy (0.9106), AUC (0.9345), F1 (0.5623), and MCC (0.5206). Recall (0.4905) improves over LR/KNN/RF while keeping solid precision. With threshold tuning or class weighting, it can likely recover even more recall. |

---

