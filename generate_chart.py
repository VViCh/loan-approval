import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

print("Loading data...")
df = pd.read_csv('data/loan_data.csv')
df_encoded = df.copy()

print("Encoding categoricals...")
le = LabelEncoder()
cat_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']

for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

df_encoded = df_encoded.astype(float)

print("Calculating correlation...")
fig, ax = plt.subplots(figsize=(14, 12))
corr = df_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.5, ax=ax, square=True, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})

ax.set_title('Complete Feature Correlation Heatmap', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig('charts/correlation_heatmap_complete.png')
print('SUCCESS_CHART_GENERATED')
