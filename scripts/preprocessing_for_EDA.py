# preprocessing_for_EDA.py

import pandas as pd
from pathlib import Path

# Get and load the dataset
base_path = Path.cwd()
data_path = base_path / "data" / "raw" / "HR_Attrition_Dataset.csv"
HR = pd.read_csv(data_path)

# Set the index to EmployeeNumber (if the numbers are unique the first condition is meet, otherwise the second condition is meet)
if HR['EmployeeNumber'].is_unique:
    # Use EmployeeNumber directly as index
    HR.set_index('EmployeeNumber', inplace=True)
else:
    # If not unique, create a new index based on the order of appearance
    HR.reset_index(drop=True, inplace=True)
    HR.index.name = 'EmployeeNumber'

# Hanfle missing values by filling them with 'Missing' label
HR = HR.fillna('Missing')    

# Remove duplicate rows (if any)
HR = HR.drop_duplicates()

# Remove unvaluable features
cleaned_for_eda = HR.drop(columns=['EmployeeCount', 'StandardHours', 'Over18'])

# Export the dataset after initial preprocessing

# Build a portable path to data/processed relative to the notebook location 
processed_dir = base_path / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Export cleaned_for_eda DataFrame to CSV
cleaned_for_eda.to_csv(processed_dir / "cleaned_for_eda.csv", index=True)
