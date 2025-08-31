#!/usr/bin/env python
# coding: utf-8

# Prepare_ML_Data
# This notebook/script loads and cleans the Megaline dataset, performs basic feature
# engineering, and creates stratified train/validation/test splits.
# The prepared datasets are saved to datasets/processed/ for use in modeling scripts.


import pandas as pd
import numpy as np
import sqlite3
import mysql.connector
import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Paths

PROJECT_ROOT = Path.cwd().parents[0] if (Path.cwd().name == "notebooks") else Path.cwd()
DATA_DIR = PROJECT_ROOT / "dataset"
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = PROJECT_ROOT / "megaline2.db"

# Load data (CSV preferred, fallback to SQLite)

csv_path = DATA_DIR / "users_behavior.csv"

if csv_path.exists():
    df = pd.read_csv(csv_path)
    src = f"CSV: {csv_path.relative_to(PROJECT_ROOT)}"
elif DB_PATH.exists():
    with sqlite3.connect(DB_PATH) as conn:
        # adjust table name if yours differs
        df = pd.read_sql_query("SELECT * FROM users_behavior;", conn)
    src = f"SQLite: {DB_PATH.name} (users_behavior)"
else:
    raise FileNotFoundError("users_behavior.csv not found and no users_behavior table in megaline2.db")

print("Loaded from:", src)
print("Shape:", df.shape)
display(df.head())



# Normalize columns & basic cleaning

# Expect at minimum: minutes, messages, mb_used, is_ultra
expected = {"minutes", "messages", "mb_used", "is_ultra"}

# If your file uses alternate names, map them here:
rename_map = {}
if "calls" in df.columns:   rename_map["calls"] = "minutes"
if "mb" in df.columns:      rename_map["mb"] = "mb_used"
if "ultra" in df.columns:   rename_map["ultra"] = "is_ultra"

if rename_map:
    df = df.rename(columns=rename_map)

missing = expected - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# Keep only needed columns + optional engineered features later
df = df.drop_duplicates().dropna(subset=["minutes","messages","mb_used","is_ultra"]).reset_index(drop=True)

# Ensure numeric
for c in ["minutes","messages","mb_used","is_ultra"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["minutes","messages","mb_used","is_ultra"]).reset_index(drop=True)



# Light feature engineering (optional, leakage-safe)

df["usage_total"] = df["minutes"] + df["messages"] + df["mb_used"]

# Build features/target & stratified split (60/20/20)

features = ["minutes", "messages", "mb_used", "usage_total"]
target = "is_ultra"

X = df[features].copy()
y = df[target].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Shapes:")
print("  Train:", X_train.shape, y_train.shape)
print("  Val  :", X_val.shape,   y_val.shape)
print("  Test :", X_test.shape,  y_test.shape)

# Save prepared splits for modeling

X_train.to_csv(PROC_DIR / "X_train.csv", index=False)
y_train.to_csv(PROC_DIR / "y_train.csv", index=False)
X_val.to_csv(  PROC_DIR / "X_val.csv",   index=False)
y_val.to_csv(  PROC_DIR / "y_val.csv",   index=False)
X_test.to_csv( PROC_DIR / "X_test.csv",  index=False)
y_test.to_csv( PROC_DIR / "y_test.csv",  index=False)

# Also save a single clean dataset if you prefer one-file modeling
clean_all = df[features + [target]]
clean_all.to_csv(PROC_DIR / "users_behavior_clean.csv", index=False)

print("\n Saved to datasets/processed/:")
for f in ["X_train.csv","y_train.csv","X_val.csv","y_val.csv","X_test.csv","y_test.csv","users_behavior_clean.csv"]:
    print("  -", f)

# Quick sanity checks

def show_balance(name, yy):
    pct = yy.value_counts(normalize=True).sort_index().round(3).to_dict()
    print(f"{name} class balance:", pct)

print()
show_balance("Train", y_train)
show_balance("Val  ", y_val)
show_balance("Test ", y_test)


# 

# In[ ]:




