import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

print("Loading data...")
df = pd.read_csv('data/loan_data.csv')
y = df['loan_status'].astype(int)
X_all = df.drop(columns=['loan_status'])

all_num = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
all_ord = ['person_education']
all_nom = ['person_gender', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']

def train_evaluate(drop_cols, description):
    X = X_all.drop(columns=drop_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    num_f = [c for c in all_num if c not in drop_cols]
    ord_f = [c for c in all_ord if c not in drop_cols]
    nom_f = [c for c in all_nom if c not in drop_cols]
    
    transformers = []
    if num_f:
        transformers.append(('num', ImbPipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num_f))
    if ord_f:
        transformers.append(('ord', ImbPipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OrdinalEncoder(categories=[['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']], handle_unknown='use_encoded_value', unknown_value=-1))]), ord_f))
    if nom_f:
        transformers.append(('nom', ImbPipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))]), nom_f))
        
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42, n_jobs=-1)
    
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', rf)
    ])
    
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"[{description}] ROC-AUC: {auc:.4f} (Dropped: {drop_cols if drop_cols else 'None'})")
    return auc

print("Starting Feature Ablation Experiment...")
train_evaluate([], "Baseline (All 13 Features)")
train_evaluate(['person_gender'], "Dropped Gender")
train_evaluate(['person_gender', 'loan_intent'], "Dropped Gender & Intent")
train_evaluate(['previous_loan_defaults_on_file'], "Dropped Dominant Feature (Defaults)")

print("Experiment Complete.")
