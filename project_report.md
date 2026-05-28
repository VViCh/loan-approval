# Loan Approval Prediction

## 1. Project Overview
Loan approval is a critical process in the financial industry. Incorrectly approving high-risk loans leads to financial loss (defaults), while incorrectly rejecting viable loans results in lost business opportunities. This project aims to build a machine learning system capable of classifying loan approval status based on applicant features such as income, employment history, and credit history. The goal is to make the decision-making process faster, more objective, and highly accurate.

## 2. Dataset Description
The dataset used for this project is the **Loan Approval Classification Data** from Kaggle. It is a binary classification problem where the target variable is `loan_status` (1 = approved, 0 = rejected).

**Key Features Include:**
- `person_age`, `person_income`, `person_emp_exp`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score` (Numerical)
- `person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file` (Categorical)

**Data Challenges Identified:**
- **Vastly Different Scales:** Features like `person_income` are in the tens or hundreds of thousands, while `loan_int_rate` is a small decimal. This severely impacts distance-based algorithms like **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**, making Feature Scaling (`StandardScaler`) a mandatory preprocessing step.
- **Class Imbalance:** Typically, loan approval datasets have more approved loans than rejected ones, or vice versa. This requires handling to prevent the model from becoming biased towards the majority class.

---

## 3. Methodology & Execution Plan

### Phase 1: Data Preparation & Exploratory Data Analysis (EDA)
- **Data Cleaning:** Impute missing values (e.g., `loan_int_rate`, `person_emp_exp`).
- **EDA:** Visualize feature distributions and target class imbalance.
- **Automated Preprocessing Pipeline (Recommended Approach):** Instead of manual step-by-step cleaning, we will utilize Scikit-Learn's `Pipeline` and `ColumnTransformer`.
  - *Numerical Pipeline:* Imputation followed by `StandardScaler`.
  - *Categorical Pipeline:* Imputation followed by `OrdinalEncoder` (for hierarchical data like education) and `OneHotEncoder` (for nominal data like loan intent).
  - *Why Pipeline?* It prevents data leakage during cross-validation, streamlines the code, and makes saving the final model for production effortless.

### Phase 2: Baseline Modeling & Imbalance Handling
- **Data Splitting:** 80% Training, 20% Testing.
- **Imbalance Handling:** Apply **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively on the training set to generate synthetic examples of the minority class. Note: When using `Pipeline`, we will use `imblearn.pipeline.Pipeline` to ensure SMOTE is only applied during training folds in Cross-Validation.
- **Baseline Model:** Logistic Regression.
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### Phase 3: Advanced Modeling & Hyperparameter Tuning
- **Models:** Decision Tree, Random Forest, SVM, and KNN.
- **Tuning:** Use `GridSearchCV` with **Stratified K-Fold Cross-Validation** (to maintain the class distribution across folds).
  - *Random Forest:* `n_estimators`, `max_depth`.
  - *SVM:* `C`, `kernel`.
  - *KNN:* `n_neighbors`.
- **Comparison:** Evaluate optimized models using Confusion Matrices and ROC-AUC scores.

### Phase 4: System Architecture & Deployment
- **Model Export:** The complete `Pipeline` (including scaler, encoders, and the best-performing model) will be saved using `joblib`.
- **Backend API:** **FastAPI** will be used to serve the model. It provides high performance and automatic interactive API documentation (Swagger UI).
- **Frontend Interface:** A professional, responsive web application built with **Next.js** and **Tailwind CSS** to interact with the FastAPI backend.

### Phase 5: Evaluation & Documentation
- **User Testing:** Input edge-case scenarios into the web interface to ensure logical predictions.
- **Final Reporting:** Compile methodology, parameters, and results into the final academic report.
