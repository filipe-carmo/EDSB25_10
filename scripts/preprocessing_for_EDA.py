import pandas as pd
import os

RAW_PATH = 'data/raw/HR_Attrition_Dataset.csv'
PROCESSED_PATH = 'data/processed/'
os.makedirs(PROCESSED_PATH, exist_ok=True)

df = pd.read_csv(RAW_PATH)

drop_cols = ['EmployeeCount', 'StandardHours', 'EmployeeNumber', 'Over18']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Fill missing values (use original labels/categories)
df = df.fillna('Missing')

df.to_csv(os.path.join(PROCESSED_PATH, 'cleaned_for_eda.csv'), index=False)
