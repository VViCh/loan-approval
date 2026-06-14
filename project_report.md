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

### 3.2 Class Imbalance Handling (SMOTE)

**How we handled class imbalance:** The dataset is highly imbalanced (approximately 78% rejected, 22% approved). If left untreated, a model could achieve 78% accuracy simply by guessing "Rejected" for every applicant. 
To resolve this, we used **SMOTE** (Synthetic Minority Over-sampling Technique). SMOTE artificially generates new, realistic synthetic examples of the minority class (Approved loans) using a Nearest Neighbors algorithm until the classes are perfectly balanced.

**Preventing Data Leakage:** Critically, SMOTE is applied within an `imblearn` pipeline. This ensures SMOTE is *only* applied to the training data folds during cross-validation—never to the test or validation folds—preventing synthetic data from leaking into the test set and artificially inflating the score.

**Empirical validation:** The notebook includes a direct comparison of Logistic Regression trained with and without SMOTE, demonstrating a massive recall improvement on the minority class.

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

**Are the models tuned?** Yes. While Logistic Regression and Decision Trees were kept as raw baselines, the complex models (Random Forest, SVM, KNN) underwent rigorous hyperparameter tuning to find their absolute optimal configurations for this specific dataset.

**How tuning was done (GridSearchCV Utilization):** 
To ensure maximum performance without overfitting, we utilized Scikit-Learn's `GridSearchCV`. This algorithm performs an exhaustive mathematical search to find the optimal configuration for a model. The process was executed in three detailed steps:

1. **Defining the Hyperparameter Grid:** Instead of guessing values, we provided a "grid" (a dictionary) of potential hyperparameters for each model. For example, for the Random Forest, we supplied a grid containing `n_estimators` (100, 200), `max_depth` (None, 10, 20), and `min_samples_split` (2, 5).
2. **Exhaustive Combinatorial Search:** `GridSearchCV` calculates every possible mathematical combination of these parameters (e.g., a Random Forest with 100 trees + max depth of 10 + min split of 2, then 200 trees + max depth of 10, etc.). It trains a completely new model from scratch for *every single combination*.
3. **Stratified 5-Fold Evaluation:** To prove the combinations actually worked, `GridSearchCV` was wrapped in **Stratified 5-Fold Cross-Validation**. Every single parameter combination was tested 5 separate times on 5 different chunks of the dataset. "Stratified" ensures that every validation fold maintained the exact 78/22 class ratio of the original dataset.
4. **Scoring Metric:** The Grid Search evaluated the success of each combination using the **ROC-AUC** metric (rather than basic accuracy) to properly penalize the model for false positives and false negatives under class imbalance conditions.

The optimal combinations discovered and locked in by the Grid Search were:

| Model | Tuned Parameters Evaluated & Optimized |
|---|---|
| Random Forest | `n_estimators` (number of trees), `max_depth` (to prevent overfitting), `min_samples_split` |
| SVM | `C` (regularization penalty), `kernel` (linear vs rbf) |
| KNN | `n_neighbors` (how many profiles to compare), `weights` (uniform vs distance) |

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

### 3.6 Known Data Anomalies & Limitations

During testing and Exploratory Data Analysis, several critical data characteristics were identified that impact model deployment:

- **Perfect Categorical Correlation (Data Leakage):** Categorical analysis revealed that 100% of applicants with `previous_loan_defaults_on_file == "Yes"` were rejected. The ML models learned this as a dominant rule.
- **Heatmap Limitations:** Because standard Pandas correlation matrices (`df.corr()`) only evaluate continuous numerical variables, the massive predictive power of `previous_loan_defaults_on_file` was excluded from `correlation_heatmap.png`, underscoring the importance of independent categorical distribution analysis.
- **Out-Of-Distribution (OOD) Blindness:** Extreme outliers (e.g., $1,000,000 loan with $100 income) were removed during preprocessing. While this improved model accuracy on normal data, it left the model blind to impossible edge cases. In production, this requires a Business Logic Engine (hard-coded `if/else` rules) to intercept and auto-reject impossible debt-to-income ratios before they reach the ML model.

### 3.7 Feature Ablation Experiment

A feature ablation experiment was conducted on the Tuned Random Forest model to observe the impact of individual features on overall predictive performance. The results are as follows:

1. **Baseline (All 13 Features):** `0.9735 ROC-AUC`
2. **Dropped `person_gender`:** `0.9744 ROC-AUC` 
   *(Insight: Performance improved. This indicates that gender acted as statistical noise without predictive value).*
3. **Dropped `person_gender` & `loan_intent`:** `0.9696 ROC-AUC`
   *(Insight: Performance decreased. This indicates that loan intent carries significant predictive signal).*
4. **Dropped `previous_loan_defaults_on_file`:** `0.9232 ROC-AUC`
   *(Insight: A significant performance drop. This confirms the model relies heavily on the perfect correlation of this single feature, as identified in Section 3.6).*

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
