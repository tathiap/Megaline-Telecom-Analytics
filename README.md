# Predicting User Subscription Behavior — Megaline Analysis & Machine Learning

Customer behavior analytics for a fictional telecom (Megaline).
This repo combines SQL/Python revenue analysis, A/B testing, and ML classification to predict Ultra plan subscriptions.

---
What's inside

* Revenue & A/B Testing (EDA) - Monthly usage aggregation, revenue calculator, Surf vs. Ultimate comparison, regional hypothesis tests.

* Machine Learning - Clean ML dataset, stratified splits, pipelines (scaling where needed), model selection, metrics & plots.

* Reproducible Data Pipeline - CSV → (optional) SQLite → processed features & ML-ready splits.

--- 

### Repo Structure 


.
├─ dataset/
│  ├─ users_behavior.csv                # base ML dataset (minutes, messages, mb_used, is_ultra)
│  └─ processed/                        # generated artifacts
│     ├─ features_user_month.csv        # monthly usage + plan economics + revenue
│     ├─ features_user_month.parquet
│     ├─ X_train.csv  y_train.csv       # ML splits (optional)
│     ├─ X_val.csv    y_val.csv
│     └─ X_test.csv   y_test.csv
├─ megaline2.db                         # SQLite database (single copy at repo root)
├─ Notebook/
│  ├─ Prepare_ML_Data.ipynb             # data cleaning, light features, stratified split
│  └─ (other analysis notebooks)
├─ Py. Scripts/                         # command-line runnable scripts
│  ├─ load_megaline_data.py             # (optional) CSV → SQLite tables
│  ├─ megaline_ml_pipeline.py           # builds monthly features & revenue
│  └─ Predict Ultra Plan.py             # trains & evaluates models
└─ sql/
   └─ megaline_analysis_views.sql       # optional reusable CREATE VIEWs (SQLite-style)

   
---

### Quick Start 

#### 1.Environment 

python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows (powershell)
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt

#### Requirements.txt

pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
jupyter

#### 2.Inputs 
Place CSVs in dataset/:

   * Required (for ML): users_behavior.csv
     Columns: minutes, messages, mb_used, is_ultra

   * Optional (for pipeline/EDA):
     megaline_calls.csv, megaline_internet.csv, megaline_messages.csv, megaline_users.csv, megaline_plans.csv

#### 3. Run from the command line 

# Build SQLite tables from CSVs
python "Py. Scripts/load_megaline_data.py"

# Build monthly usage features + revenue and save to dataset/processed/
python "Py. Scripts/megaline_ml_pipeline.py"

# Train & evaluate classification models for Ultra subscription
python "Py. Scripts/Predict Ultra Plan.py"

---

