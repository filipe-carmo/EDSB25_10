import pandas as pd
import numpy as np
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ==== Set output folder ====
PROCESSED_PATH = 'data/processed/'
os.makedirs(PROCESSED_PATH, exist_ok=True)

# ==== Feature Definitions ====
ordinalcols = [
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance'
]
drop_first_vars = ['BusinessTravel', 'Gender', 'MaritalStatus', 'OverTime']
numericcols = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]
scale_cols = numericcols + ordinalcols

selected_features = [
    'JobSatisfaction', 'JobInvolvement', 'EnvironmentSatisfaction',
    'JobRole_Research Director', 'JobRole_Sales Representative', 'BusinessTravel_Travel_Frequently',
    'WorkLifeBalance', 'Department_Research & Development', 'YearsAtCompanyLog',
    'TotalWorkingYearsLog', 'MaritalStatus_Single', 'OverTime_Yes', 'JobRole_Laboratory Technician',
    'JobRole_Manufacturing Director', 'MaritalStatus_Married', 'JobRole_Manager',
    'NumCompaniesWorked', 'JobRole_Healthcare Representative', 'Job_happiness_score'
]

# ==== Custom Transformers ====

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col, log_col in [('TotalWorkingYears', 'TotalWorkingYearsLog'),
                             ('YearsAtCompany', 'YearsAtCompanyLog')]:
            if col in X.columns:
                # Ensure numeric, forcefill, clip negatives (shouldn’t exist in HR, but just in case)
                series = pd.to_numeric(X[col], errors='coerce')
                series = series.fillna(series.median())
                series = series.clip(lower=0)
                X[log_col] = np.log1p(series)
            else:
                X[log_col] = np.nan
        return X


class CustomCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        # Dept: don't drop any
        dummies = pd.get_dummies(X['Department'], prefix='Department', drop_first=False)
        X = X.drop('Department', axis=1)
        X[dummies.columns] = dummies
        # EducationField: drop ONLY 'Other'
        edu_dummies = pd.get_dummies(X['EducationField'], prefix='EducationField')
        if 'EducationField_Other' in edu_dummies.columns:
            edu_dummies = edu_dummies.drop('EducationField_Other', axis=1)
        X = X.drop('EducationField', axis=1)
        X[edu_dummies.columns] = edu_dummies
        # JobRole: don't drop any
        jobrole_dummies = pd.get_dummies(X['JobRole'], prefix='JobRole', drop_first=False)
        X = X.drop('JobRole', axis=1)
        X[jobrole_dummies.columns] = jobrole_dummies
        # Drop-first for remaining categoricals
        for col in drop_first_vars:
            col_dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = X.drop(col, axis=1)
            X[col_dummies.columns] = col_dummies
        return X

class ColumnSynchronizer(BaseEstimator, TransformerMixin):
    def __init__(self, ref_columns):
        self.ref_columns = ref_columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        missing_cols = set(self.ref_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[self.ref_columns].reset_index(drop=True)
        return X

class JobHappinessAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['Job_happiness_score'] = (
            X['JobInvolvement'] +
            X['JobSatisfaction'] +
            X['YearsAtCompanyLog'] +
            X['EnvironmentSatisfaction'] +
            X['WorkLifeBalance'] -
            X.get('OverTime_Yes', 0)
        )
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        return X[self.features]

# ==== Preprocessing / Export Function ====

def preprocess(data_path, target_name='Attrition', test_size=0.2, val_size=0.25, random_state=42):
    
    # Load data and drop uninformative columns
    df = pd.read_csv(data_path)
    dropcols = ['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18']
    df = df.drop(columns=[col for col in dropcols if col in df.columns], errors='ignore')
    df[target_name] = df[target_name].map({'Yes': 1, 'No': 0})

    # Split data
    y = df[target_name]
    X = df.drop(columns=[target_name])
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=val_size, random_state=random_state)

    # --- Impute numeric columns (fit/train, transform all) and ensure numeric dtype
    num_imputer = SimpleImputer(strategy='median')
    for df_ in [X_train, X_val, X_test]:
        for col in numericcols:
            df_[col] = pd.to_numeric(df_[col], errors='coerce')
    X_train[numericcols] = num_imputer.fit_transform(X_train[numericcols])
    X_val[numericcols] = num_imputer.transform(X_val[numericcols])
    X_test[numericcols] = num_imputer.transform(X_test[numericcols])

    # --- Feature engineering (robust log features, job happiness score)
    engineer = CustomFeatureEngineer()  # Uses robust log transform code
    X_train = engineer.transform(X_train)
    X_val = engineer.transform(X_val)
    X_test = engineer.transform(X_test)

    # --- Scaling for numeric + ordinal columns
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    # --- Custom categorical encoding (as your class defines)
    encoder = CustomCategoricalEncoder()
    X_train = encoder.transform(X_train)
    X_val = encoder.transform(X_val)
    X_test = encoder.transform(X_test)

    # --- Column synchronization
    all_columns = sorted(set(X_train.columns) | set(X_val.columns) | set(X_test.columns))
    synchronizer = ColumnSynchronizer(ref_columns=all_columns)
    X_train = synchronizer.transform(X_train)
    X_val = synchronizer.transform(X_val)
    X_test = synchronizer.transform(X_test)

    # --- Add derived feature (job happiness)
    happiness = JobHappinessAdder()
    X_train = happiness.transform(X_train)
    X_val = happiness.transform(X_val)
    X_test = happiness.transform(X_test)

    # --- Impute engineered features (robust forced fill, just in case)
    for df_ in [X_train, X_val, X_test]:
        for col in ['YearsAtCompanyLog', 'TotalWorkingYearsLog', 'Job_happiness_score']:
            df_[col] = pd.to_numeric(df_[col], errors='coerce')
            df_[col] = df_[col].fillna(df_[col].median())

    # --- Feature selection (now done last)
    selector = FeatureSelector(features=selected_features)
    X_train = selector.transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)

    # --- Output to CSV
    PROCESSED_PATH = 'data/processed/'
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(PROCESSED_PATH, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_PATH, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(PROCESSED_PATH, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_PATH, "y_test.csv"), index=False)

    return X_train, X_val, X_test, y_train, y_val, y_test
