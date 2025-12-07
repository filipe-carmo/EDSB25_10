HR Attrition Project

I. Introduction and Methodology

The objective of this capstone project is to synthesize and apply the diverse skills and knowledge acquired throughout the coursework by developing an end-to-end data science solution focused on predicting employee attrition. The project integrates data exploration, feature engineering, machine learning, evaluation, and deployment, while emphasizing clear communication of results for business stakeholders.

This repository implements a full pipeline from business problem framing to deployment: data ingestion and cleaning, exploratory data analysis, feature selection, model training (Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Networks), threshold optimization with business-focused recall targets, and a Gradio app (via app.py and space.yaml) for serving predictions.

II. The Business Problem

High turnover rates are costly and disruptive, making it essential for HR to anticipate which employees are likely to leave. We are tasked with assisting a multinational consultancy firm in predicting employee attrition.

The primary goal is to predict whether an employee will leave the company, based on the data provided. Secondary goals include identifying key factors influencing attrition and recommending strategies to retain valuable employees.

The HR department and executive management will be the primary consumers of the insights, and they expect actionable recommendations based on the analysis.

III. Organization

Repository structure
.github/ – GitHub configuration for Git Actions

data/ – Raw and intermediate datasets used for analysis and modeling

models/ – Saved model artifacts (joblib) and metadata (JSON) used by the notebooks and the app

notebooks/ – All Jupyter notebooks for business framing, EDA, feature engineering, modeling, and final evaluation

old files/ – Legacy or exploratory files kept for reference, not part of the main pipeline

project_brief/ – Original project description, requirements, and briefing documents

reports/ – Final presentation

scripts/ – Auxiliary Python scripts

venv/ – Local virtual environment (ignored by Git)

.gitignore – Excludes unnecessary files from version control

app.py – Gradio-based application for interactive inference and deployment (local or HF Space)

preprocessing.py – Main preprocessing and feature engineering module used across models and the app

README.md – Project overview and instructions

README_HF.txt – Hugging Face Space–specific description and usage notes

requirements-dev.txt – Full development dependencies (notebooks, plotting, modeling)

requirements.txt – Minimal runtime dependencies for the app and core pipeline

space.yaml – Hugging Face Space configuration

Notebooks structure
All notebooks are located in notebooks/ and follow the pipeline order:

1.-Business-Problem-Data-Collection-and-Initial-Processing.ipynb

Defines the business context, documents assumptions, and loads the raw HR dataset from data/ into a clean tabular format.

2.-Exploratory-Data-Analysis.ipynb

Performs descriptive statistics, visual EDA, and class imbalance inspection to understand attrition patterns and potential data quality issues.

3.-Feature-Eng-and-Feature-Selection.ipynb

Runs correlation and statistical analyses, engineers domain features (e.g., log-transformed tenure, satisfaction-related variables), and defines the final 18-feature set used by all models.

4.-Logistic_Regression.ipynb

Trains a Logistic Regression model on the shared 18-feature dataset.

Applies SMOTE to rebalance the training data and uses StratifiedKFold + GridSearchCV to tune regularization, solver, and class weights.

Implements dual threshold optimization: F1-oriented and recall-oriented, with manual adjustment for robustness on the test set.

Saves the best model and metadata into models/.

5.-Decision-Tree-model.ipynb

Builds a Decision Tree classifier with strong interpretability constraints (limited depth, leaves, and pruning).

Uses class weighting to handle imbalance and a large hyperparameter grid with StratifiedKFold + GridSearchCV.

Applies dual threshold optimization and exports the model and feature importances to models/.

6.-Random-Forest.ipynb

Trains a Random Forest classifier using the same preprocessing pipeline.

Tunes tree depth, number of estimators, and other parameters, computes ROC/AUC and F1, and analyzes feature importance.

Saves the selected model and metadata into models/.

7.-XGBOOST.ipynb

Experiments with XGBoost to improve minority-class performance, tuning learning rate, depth, and class imbalance parameters.

8.-Neural-Networks.ipynb

Trains one or more feedforward neural networks (MLP) on the 18-feature dataset and compares performance to tree-based and linear models.

9.-Evaluation.ipynb

Consolidates test-set metrics across all models (accuracy, precision, recall, F1, ROC/AUC, confusion matrices).

Highlights trade-offs and proposes recommended configurations for different HR use cases (high recall vs. balanced).

IV. Git and Documentation Practices

README.md provides a high-level overview, repository map, and execution instructions.

Code in preprocessing.py, app.py, and key scripts/notebooks includes comments and (where applicable) docstrings to clarify data transformations and model logic.

Dependencies are defined in requirements.txt (runtime) and requirements-dev.txt (development / notebooks).

The repository is organized into clear folders (data/, notebooks/, models/, scripts/, reports/, etc.) rather than mixing everything in the root.

.gitignore excludes non-essential files such as venv/, .ipynb_checkpoints/, large artifacts, and OS-specific files.

Commit messages and file names are written to be meaningful and consistent with the notebook numbering and project phases.

V. How to Run the Project

1. Prerequisites
Python 3.10+ 

Git installed

Recommended: virtual environment (venv or conda)

2. Setup
Clone the repository:

git clone <your-repo-url>

cd <your-repo-folder>

(Optional if you are not using the existing venv/) Create and activate a virtual environment:

python -m venv .venv

source .venv/bin/activate (Linux/Mac)

.venv\Scripts\activate (Windows)

Install dependencies:

Development (notebooks + modeling): pip install -r requirements-dev.txt

Runtime / app only: pip install -r requirements.txt

3. Reproduce analysis and models
Ensure the HR attrition dataset is available under data/ in the expected path used in the notebooks.

Open the notebooks in notebooks/ and run them in numerical order (1 to 9).

Each modeling notebook imports preprocessing.py, trains the model, optimizes thresholds, and saves outputs into models/.

4. Run the Gradio app locally
After installing runtime dependencies and ensuring a model + metadata exist in models/, run:

python app.py

Open the URL printed in the console (for local runs, typically http://127.0.0.1:7860) to access the interface and score new employee records.

The app can also be used on https://huggingface.co/spaces/filipe-carmo/hr-attrition-prediction

VI. Deliverables

Jupyter Notebooks / Python Scripts

Full workflow from business problem framing and EDA to preprocessing, modeling, evaluation, and deployment artifacts.

Final Presentation

Professional slide deck in reports/ summarizing problem, approach, model comparison, and recommendations for HR and management.

Backup Slides

Extra material (detailed metrics, threshold analyses, feature importance plots, ablation notes) to support discussion and questions.

GitHub Repository

Organized, reproducible repository including:

notebooks/, scripts/, data/, models/, reports/, project_brief/

README.md and README_HF.txt

requirements*.txt, .gitignore, space.yaml, and app.py for deployment