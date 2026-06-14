# Loan Approval Prediction

A comparative study of classification algorithms for loan approval prediction.

**Team:**
Aulia Aca Azzahra · Juan Jonathan Suparmo · Samuel Rafin Djunaidi · Vincent Vic Chow

---

## How to Run

### Prerequisites

- Python 3.10+
- pip

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Open `loan_approval.v2.ipynb` in VS Code or Jupyter and **Run All** cells.

This will:
- Run the full EDA and generate charts in `charts/`
- Train all 5 classifiers + tune RF, SVM, and KNN
- Export all model pipelines to `models/`
- Write `models/model_info.json` with evaluation metrics

> **Note:** The first run takes a few minutes due to SVM training and GridSearchCV.
> After running, you should see `.pkl` files inside `models/`.

### 3. Start the web app

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in browser.

The web interface can do:
- Fill in a loan application form
- Select which model to use (dropdown at the top, defaults to the best one)
- See the prediction result with an approval probability gauge

---

## Pipeline Overview

```
Raw CSV -> EDA -> Preprocessing -> SMOTE -> Model Training -> Evaluation -> Export
```

### Step 1 — Exploratory Data Analysis

- Class distribution check (approved vs rejected)
- Numerical feature histograms (age, income, loan amount, interest rate, etc.)
- Outlier detection via boxplots (extreme values in `person_age` and `person_income`)
- Correlation heatmap to identify feature relationships
- Categorical feature breakdown by loan status

### Step 2 — Preprocessing (`ColumnTransformer`)

All transformations happen inside a single Scikit-Learn pipeline to prevent
data leakage during cross-validation.

| Feature type | Transformation |
|---|---|
| Numerical (8 features) | Median imputation → StandardScaler |
| Ordinal (`person_education`) | Mode imputation → OrdinalEncoder (High School < Associate < Bachelor < Master < Doctorate) |
| Nominal (4 features) | Mode imputation → OneHotEncoder (drop first) |

### Step 3 — Class Imbalance Handling (SMOTE)

The dataset has an uneven class distribution. SMOTE generates synthetic
minority-class samples during training only (never on test data). This is
handled by using `imblearn.pipeline.Pipeline` instead of the standard
sklearn pipeline.

The notebooks include a direct comparison of Logistic Regression with and
without SMOTE to demonstrate the recall improvement.

### Step 4 — Model Training & Comparison

Five classifiers from the proposal, all trained with SMOTE:

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 86.22% | 74.61% | 95.62% |
| Decision Tree | 88.92% | 76.42% | 86.02% |
| Random Forest | 91.99% | 82.10% | 97.30% |
| SVM | 88.24% | 77.61% | 96.16% |
| KNN | 84.53% | 71.20% | 91.53% |

### Step 5 — Hyperparameter Tuning (GridSearchCV + Stratified 5-Fold CV)

| Model | Parameters Tuned | Best Test ROC-AUC |
|---|---|---|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` | **97.36%** |
| SVM | `C`, `kernel` | 96.16% |
| KNN | `n_neighbors`, `weights` | 93.74% |

### Step 6 — Export & Deployment

All pipelines (including preprocessing + model) are saved as `.pkl` files.
The Flask app loads them at startup and exposes:
- `GET /` — Web interface
- `GET /api/models` — Available models with their metrics
- `POST /predict` — Accepts a JSON loan application, returns prediction + probability

## References

- Brown, I., & Mues, C. (2012). An experimental comparison of classification algorithms for imbalanced credit scoring data sets. *Expert Systems with Applications*, 39(3), 3446–3453.
- Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring. *European Journal of Operational Research*, 247(1), 124–136.
- Taweilo. (n.d.). Loan approval classification data. Kaggle. https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
