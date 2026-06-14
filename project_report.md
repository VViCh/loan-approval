# Loan Approval Prediction

## 1. Project Overview

Loan approval is a critical decision-making process in the financial industry. Incorrectly approving high-risk applicants leads to loan defaults and financial loss, while incorrectly rejecting creditworthy applicants results in missed business opportunities. This project develops and compares multiple machine learning classifiers to predict loan approval outcomes, aiming to produce a model that is both accurate and practical for deployment.

The work is based on the research proposal *"A Comparative Study of Classification Algorithms for Loan Approval Prediction"* and follows its outlined methodology with additional refinements documented below.

## 2. Dataset

The dataset is the **Loan Approval Classification Data** from Kaggle (Taweilo, n.d.), consisting of binary-labeled loan applications (`loan_status`: 1 = approved, 0 = rejected).

**Feature summary:**

| Category | Features |
|---|---|
| Numerical | `person_age`, `person_income`, `person_emp_exp`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score` |
| Categorical | `person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file` |

**Known data challenges:**

- **Feature scale disparity:** `person_income` (tens of thousands) vs `loan_int_rate` (single-digit decimals). This directly affects distance-based algorithms (KNN, SVM), making standardization a prerequisite.
- **Potential outliers:** Extreme values observed in `person_age` and `person_income` during EDA. Tree-based models are robust to these, while the `StandardScaler` in the pipeline mitigates their impact on distance-based models.
- **Class imbalance:** The dataset exhibits an uneven distribution of approved vs rejected loans, addressed through SMOTE (see Section 3.2).

## 3. Methodology

### 3.1 Preprocessing

A Scikit-Learn `ColumnTransformer` encapsulates all transformations in a single pipeline to prevent data leakage during cross-validation:

- **Numerical features:** Median imputation → `StandardScaler`
- **Ordinal features (`person_education`):** Mode imputation → `OrdinalEncoder` with the hierarchy High School < Associate < Bachelor < Master < Doctorate
- **Nominal features:** Mode imputation → `OneHotEncoder` (first category dropped)

### 3.2 Class Imbalance Handling

SMOTE (Synthetic Minority Over-sampling Technique) is applied within the `imblearn` pipeline to generate synthetic examples of the minority class. Critically, SMOTE is only applied to training data — never to test or validation folds — preventing information leakage.

**Justification:** Brown & Mues (2012) demonstrate that resampling techniques significantly improve classifier performance on imbalanced credit scoring datasets, which directly parallels this study's domain.

**Empirical validation:** The notebook includes a direct comparison of Logistic Regression trained with and without SMOTE, demonstrating the recall improvement on the minority class.

### 3.3 Classifiers

Five algorithms, as specified in the proposal:

| Model | Role | Notes |
|---|---|---|
| Logistic Regression | Baseline | Simple, interpretable reference point |
| Decision Tree | Proposed | Captures non-linear relationships |
| Random Forest | Proposed | Ensemble method; reduces overfitting |
| SVM | Proposed | Effective with clear margin separation |
| KNN | Proposed | Instance-based; sensitive to feature scale |

### 3.4 Hyperparameter Tuning

`GridSearchCV` with Stratified 5-Fold cross-validation, scored on **ROC-AUC**. ROC-AUC is preferred over accuracy because accuracy can be misleading under class imbalance (Lessmann et al., 2015).

| Model | Tuned Parameters |
|---|---|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| SVM | `C`, `kernel` |
| KNN | `n_neighbors`, `weights` |

Cross-validation results are reported as **mean ± standard deviation** across folds.

### 3.5 Evaluation Metrics

As defined in the proposal, with the addition of ROC-AUC:

- **Accuracy** — overall correctness
- **Precision** — proportion of predicted approvals that are correct
- **Recall** — proportion of actual approvals that are captured
- **F1-Score** — harmonic mean of precision and recall
- **ROC-AUC** — discrimination ability across all thresholds
- **Confusion Matrix** — detailed breakdown of true/false positives and negatives

ROC-AUC was added because in loan approval, the relative cost of false positives (approving a default) vs false negatives (rejecting a viable loan) makes threshold-independent metrics essential.

## 4. Deployment

### 4.1 Architecture

The proposal originally specified Google Colab with Flask or Streamlit. The implementation uses a **single Flask application** that serves both the prediction API and the web frontend. This was chosen over the original plan because:

- A single-process architecture simplifies deployment and testing
- Flask is sufficient for the scope of this project and is widely understood in academic settings
- No additional frontend tooling (Node.js, npm) is required

### 4.2 Model Serving

All trained pipelines are serialized to `models/` using `joblib`. A `model_info.json` file records each model's evaluation metrics and identifies the best performer. The Flask app loads all models at startup and allows users to switch between them via a dropdown, with the best model selected by default.

### 4.3 User Interface

The web interface provides a loan application form with fields matching the dataset features. On submission, the selected model returns an approval/rejection prediction with the associated probability. The interface also displays the model's ROC-AUC score for transparency.

## 5. References

- Brown, I., & Mues, C. (2012). An experimental comparison of classification algorithms for imbalanced credit scoring data sets. *Expert Systems with Applications*, 39(3), 3446–3453.
- Goyal, A., & Kaur, R. (2020). Loan Approval Prediction Based on Machine Learning Approach. *2020 10th International Conference on Cloud Computing, Data Science & Engineering (Confluence)*, 521–524.
- Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124–136.
- Sadok, H., Sakka, F., & El Maknouzi, M. E. H. (2022). Artificial intelligence and bank credit analysis: A review. *Cogent Economics & Finance*, 10(1).
- Taweilo. (n.d.). Loan approval classification data [Data set]. Kaggle. https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
- Viswanatha V, Ramachandra A.C, Vishwas K N, & Adithya G. (2023). Prediction of Loan Approval in Banks using Machine Learning Approach. *International Journal of Engineering and Management Research*, 13(4), 7–19.
