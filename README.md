
# Bank Marketing – Multi-Model Binary Classification (Streamlit + Deployment)

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

| ML Model Name             | Accuracy | AUC      | Precision | Recall    | F1       | MCC     |
|---------------------------|----------|----------|-----------|-----------|----------|---------|
| Logistic Regression       | 0.9012   | 0.9056   | 0.6445    | 0.3478    | 0.4518   | 0.4261  |  
| Decision Tree             | 0.8728   | 0.7009	  | 0.4582	  | 0.4764	  | 0.4671	 | 0.395   |
| kNN                       | 0.8986   | 0.85	    | 0.6257	  | 0.3318	  | 0.4336	 | 0.407   |          
| Naive Bayes               | 0.8548   | 0.8101	  | 0.4059	  | 0.5198	  | 0.4559	 | 0.3774  |          
| Random Forest (Ensemble)  | 0.8976   | 0.9207	  | 0.7157	  | 0.207	    | 0.3211	 | 0.3486  |          
| XGBoost (Ensemble)        | 0.9093   | 0.9348	  | 0.6591	  | 0.466 	  | 0.546	   | 0.5065  | 

---

## Observations about Model Performance 

> Added dataset-specific insights for each model based on metrics results. 

| ML Model Name             | Observation about model performance |
|---------------------------|-------------------------------------|
| **Logistic Regression**   | Strong baseline with good AUC (0.9056) and decent precision (0.6445). Recall is modest (0.3478), indicating it misses several positives in this imbalanced dataset. |
| **Decision Tree**         | Interpretable but comparatively weaker AUC (0.7009). Balanced precision/recall (~0.46–0.48) but overall lags behind ensembles. Potential overfitting without pruning/tuning. |
| **kNN**                   | Reasonable precision (0.6257) but the lowest recall (0.3318) among non‑NB models; sensitive to feature scaling and high dimensionality. |
| **Naive Bayes**           | Best recall (0.5198) among non‑boosted models, but precision is low (0.4059), leading to more false positives; overall accuracy and AUC are modest. |
| **Random Forest (Ensemble)** | Strong overall—high AUC (0.9207) and best precision (0.7157); recall remains moderate (0.207). Very good MCC (0.3486), indicating robust balanced performance. |
| **XGBoost (Ensemble)**    | Best overall—highest accuracy (0.9093), AUC (0.9348), F1 (0.546), and MCC (0.5065). Recall (0.466) improves over LR/KNN/RF while keeping solid precision. With threshold tuning or class weighting, it can likely recover even more recall. |

---

