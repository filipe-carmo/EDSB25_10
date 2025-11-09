import pandas as pd
import os
import numpy as np

# ==== Paths ====
RAW_PATH = 'data/raw/HR_Attrition_Dataset.csv'
PROCESSED_PATH = 'data/processed/'
os.makedirs(PROCESSED_PATH, exist_ok=True)

# ==== 1. Load Data ====
df = pd.read_csv(RAW_PATH)

# ==== 2. Drop Uninformative Columns ====
drop_cols = ['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# ==== 3. Encode Target Variable ====
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# ==== 3. Column Types ====
ordinal_cols = [
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance'
]
nominal_cols = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]
numeric_cols = [
    col for col in df.columns
    if df[col].dtype in ['int64', 'float64'] and col not in ordinal_cols + ['Attrition']
]
scale_cols = numeric_cols + ordinal_cols  # to be scaled for LR/SVM

# ==== 3a. Outlier Handling ====
'''
for col in numeric_cols:
    # Apply IQR capping
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Cap values to lower/upper whisker
    df[col] = np.clip(df[col], lower, upper)
'''
# ==== 3b. Multicollinearity Check ====
'''
corr_matrix = df[numeric_cols].corr().abs()
# Mark highly correlated pairs (threshold > 0.9)
high_corr = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns if i != j and corr_matrix.loc[i, j] > 0.9]
print("Highly correlated features (threshold > 0.9):", high_corr)
# (You should decide which to drop based on domain and modeling, not automatically here.)
'''
# ==== 3c. Rare Category Handling ====
'''
for col in nominal_cols:
    freq = df[col].value_counts(normalize=True)
    rare_labels = freq[freq < 0.01].index  # define 'rare' as <1% freq (adjustable)
    df[col] = df[col].replace({lbl: 'Other' for lbl in rare_labels})
'''
# ==== 4. Train/Test/Val Split ====
from sklearn.model_selection import train_test_split

y = df['Attrition']
X = df.drop(columns=['Attrition'])

# First create test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
# Then split the rest into train/val (60/20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42
)

# ==== 5. Fill Missing Values ====
# Numeric & ordinal (median), nominal (Missing)
for df_ in [X_train, X_val, X_test]:
    df_[scale_cols] = df_[scale_cols].fillna(X_train[scale_cols].median())
    df_[nominal_cols] = df_[nominal_cols].fillna('Missing')

# ==== 6. Save splits ====
X_train.to_csv(os.path.join(PROCESSED_PATH, 'X_train.csv'), index=False)
X_val.to_csv(os.path.join(PROCESSED_PATH, 'X_val.csv'), index=False)
X_test.to_csv(os.path.join(PROCESSED_PATH, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(PROCESSED_PATH, 'y_train.csv'), index=False)
y_val.to_csv(os.path.join(PROCESSED_PATH, 'y_val.csv'), index=False)
y_test.to_csv(os.path.join(PROCESSED_PATH, 'y_test.csv'), index=False)
