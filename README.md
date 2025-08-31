# Predicting User Subscription Behavior — Megaline Analysis & Machine Learning

Customer behavior analytics for a fictional telecom (Megaline).
This repo combines SQL/Python revenue analysis, A/B testing, and ML classification to predict Ultra plan subscriptions.

---
***What's inside***

* Revenue & A/B Testing (EDA) - Monthly usage aggregation, revenue calculator, Surf vs. Ultimate comparison, regional hypothesis tests.

* Machine Learning - Clean ML dataset, stratified splits, pipelines (scaling where needed), model selection, metrics & plots.

* Reproducible Data Pipeline - CSV → (optional) SQLite → processed features & ML-ready splits.

--- 
### Repo Structure 

.
├─ dataset/
│  ├─ users_behavior.csv                 # base ML dataset (minutes, messages, mb_used, is_ultra)
│  └─ processed/                         # generated artifacts
│     ├─ features_user_month.csv         # monthly usage + plan economics + revenue
│     ├─ features_user_month.parquet
│     ├─ X_train.csv  y_train.csv        # ML splits (optional)
│     ├─ X_val.csv    y_val.csv
│     └─ X_test.csv   y_test.csv
├─ megaline2.db                          # SQLite database (single copy at repo root)
├─ Notebook/
│  ├─ Prepare_ML_Data.ipynb              # data cleaning, light features, stratified split
│  └─ (other analysis notebooks)
├─ Py. Scripts/                          # command-line runnable scripts (kept with space)
│  ├─ load_megaline_data.py              # (optional) CSV → SQLite tables
│  ├─ megaline_ml_pipeline.py            # builds monthly features & revenue
│  └─ Predict Ultra Plan.py              # trains & evaluates models
└─ sql/
   └─ megaline_analysis_views.sql        # optional reusable CREATE VIEWs (SQLite-style)

   
---

### Quick Start 

***1.Environment*** 

python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows (powershell)
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt

****Requirements.txt**** 

pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
jupyter

***2.Inputs***
Place CSVs in dataset/:

   * Required (for ML): users_behavior.csv
     Columns: minutes, messages, mb_used, is_ultra

   * Optional (for pipeline/EDA):
     megaline_calls.csv, megaline_internet.csv, megaline_messages.csv, megaline_users.csv, megaline_plans.csv



---
### What each component does

load_megaline_data.py (optional)
Loads dataset/megaline_*.csv into megaline2.db (SQLite). Safe to re-run.

megaline_ml_pipeline.py
Parses dates → month (YYYY-MM), aggregates calls/messages/internet by user-month, joins plan economics, computes monthly_revenue (base fee + overage), and saves:

   * dataset/processed/features_user_month.csv|.parquet

Predict Ultra Plan.py
Loads ML data, creates stratified 60/20/20 train/val/test, scales only models that need it (LR/SVM), tunes on validation F1 (Ultra=1) via GridSearchCV, evaluates on test, prints reports and plots (prediction histograms, confusion matrices, feature importance, ROC).

Notebook/Prepare_ML_Data.ipynb
Minimal cleaning & feature creation for users_behavior.csv, plus stratified splits (optional artifacts in dataset/processed/).

---
### Results (from latest run)

Model selection (validation F1 on Ultra):

   * Random Forest 0.64 (best)
   * Gradient Boosting 0.63
   * Decision Tree 0.58
   * Logistic Regression 0.51
   * SVM 0.49

Final test (Random Forest):

   * Accuracy: 0.80 (643 rows)
   * Ultra (class=1): Precision 0.73, Recall 0.56, F1 0.63
   * Confusion matrix (Test):
      * Actual 0 → Pred 0: 404, Pred 1: 42
      * Actual 1 → Pred 0: 86, Pred 1: 111

***Interpretation***: Tree-based models perform best. RF achieves strong precision on Ultra with moderate recall. If the business goal is to capture more Ultra prospects, increase recall via threshold tuning or class-weight/oversampling.

Revenue & A/B testing (typical outcomes):

* Ultimate has higher ARPU per user.
* Surf can drive more total revenue through scale and occasional overage.
* T-test (Surf vs. Ultimate) usually shows statistically significant mean revenue differences (p < 0.05).
* Regional comparisons (e.g., NY–NJ vs others) often differ meaningfully → targetable campaigns.
---
### Troubleshooting

* Path not found / FileNotFoundError
  Scripts use:

  from pathlib import Path
  HERE = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
  PROJECT_ROOT = HERE.parents[1]
  DATA_DIR = PROJECT_ROOT / "dataset"
  PROC_DIR = DATA_DIR / "processed"
  DB_PATH  = PROJECT_ROOT / "megaline2.db"

Ensure there is one megaline2.db at the repo root and the folder name is dataset/ (singular).

* Headless plotting (servers/CI):

  import matplotlib
  matplotlib.use("Agg")
---
### Roadmap
* Probability threshold tuning (maximize recall for Ultra with minimal precision drop)
* Robust time-series revenue forecasting by plan
* Parameterized reporting notebook
* (Optional) MySQL variant of SQL views

