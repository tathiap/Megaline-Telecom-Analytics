#!/usr/bin/env python
# coding: utf-8

"""
megaline_ml_pipeline.py

Builds model-ready features from Megaline raw data.
- Prefers reading raw tables from SQLite (megaline2.db)
- Falls back to CSVs in /datasets if DB is missing
- Aggregates to user month, joins plan info, computes monthly_revenue
- Writes processed outputs to /datasets/processed/
"""

from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

# Paths
# Resolve repo root from this script's location
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]

# Prefer 'datasets', fallback to 'dataset'
DATA_DIR = PROJECT_ROOT / "datasets"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "dataset"

# Output folder for processed artifacts
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# SQLite DB at repo root
DB_PATH = PROJECT_ROOT / "megaline2.db"

features_path_csv  = PROC_DIR / "features_user_month.csv"
features_path_parq = PROC_DIR / "features_user_month.parquet"


# Load raw tables: SQLite first, else CSVs

tables = ["calls", "internet", "messages", "users", "plans"]
dfs = {}

if DB_PATH.exists():
    print(f"Loading from SQLite: {DB_PATH.name}")
    conn = sqlite3.connect(DB_PATH)
    for t in tables:
        dfs[t] = pd.read_sql_query(f"SELECT * FROM {t};", conn)
        print(f"  - {t}: {dfs[t].shape}")
    conn.close()
else:
    print("SQLite DB not found; falling back to CSVs in /datasets")
    for t in tables:
        p = DATA_DIR / f"megaline_{t}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        dfs[t] = pd.read_csv(p)
        print(f"  - {t}: {dfs[t].shape}")

calls    = dfs["calls"]
internet = dfs["internet"]
messages = dfs["messages"]
users    = dfs["users"]
plans    = dfs["plans"]


# Normalize date 

def ensure_month(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df["month"] = df[c].dt.to_period("M").astype(str)
            return df
    raise KeyError(f"None of the date columns {date_cols} found in {df.columns.tolist()}")

calls    = ensure_month(calls,    ["call_date","date","timestamp"])
internet = ensure_month(internet, ["session_date","date","timestamp"])
messages = ensure_month(messages, ["message_date","date","timestamp"])


# Aggregate to user-month

# calls: sum minutes
min_col = "minutes" if "minutes" in calls.columns else ("duration" if "duration" in calls.columns else None)
if min_col is None:
    raise KeyError("Expected a call duration column named 'minutes' or 'duration'.")
calls_agg = (calls.groupby(["user_id","month"], as_index=False)[min_col]
                  .sum()
                  .rename(columns={min_col: "total_minutes"}))

# messages: either sum provided count or count rows
if "messages" in messages.columns:
    messages_agg = (messages.groupby(["user_id","month"], as_index=False)["messages"]
                           .sum()
                           .rename(columns={"messages":"messages_sent"}))
else:
    messages_agg = (messages.groupby(["user_id","month"], as_index=False)
                           .size()
                           .rename(columns={"size":"messages_sent"}))

# internet: sum MB
mb_col = "mb_used" if "mb_used" in internet.columns else ("mb" if "mb" in internet.columns else None)
if mb_col is None:
    raise KeyError("Expected an internet usage column named 'mb_used' or 'mb'.")
internet_agg = (internet.groupby(["user_id","month"], as_index=False)[mb_col]
                      .sum()
                      .rename(columns={mb_col:"data_volume_mb"}))


# Merge to monthly_usage

monthly_usage = (calls_agg
                 .merge(internet_agg, on=["user_id","month"], how="outer")
                 .merge(messages_agg, on=["user_id","month"], how="outer")
                 .fillna(0))

monthly_usage = monthly_usage.merge(users[["user_id","plan","city"]], on="user_id", how="left")
monthly_usage = monthly_usage.merge(plans, left_on="plan", right_on="plan_name", how="left")
monthly_usage.drop(columns=["plan_name"], inplace=True)


# Revenue calculator

def calculate_revenue(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "total_minutes","messages_sent","data_volume_mb",
        "minutes_included","messages_included","mb_per_month_included",
        "usd_monthly_pay","usd_per_minute","usd_per_message","usd_per_gb"
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns for revenue calc: {missing}")

    out[required] = out[required].apply(pd.to_numeric, errors="coerce").fillna(0)
    out["excess_minutes"]  = (out["total_minutes"]  - out["minutes_included"]).clip(lower=0)
    out["excess_messages"] = (out["messages_sent"]  - out["messages_included"]).clip(lower=0)
    out["excess_data_gb"]  = ((out["data_volume_mb"] - out["mb_per_month_included"]) / 1024).clip(lower=0)
    out["monthly_revenue"] = (
        out["usd_monthly_pay"]
        + out["excess_minutes"]  * out["usd_per_minute"]
        + out["excess_messages"] * out["usd_per_message"]
        + out["excess_data_gb"]  * out["usd_per_gb"]
    ).round(2)
    return out

monthly_usage = calculate_revenue(monthly_usage)


# Save processed outputs

features_cols = [
    "user_id","month","plan","city",
    "total_minutes","messages_sent","data_volume_mb",
    "minutes_included","messages_included","mb_per_month_included",
    "usd_monthly_pay","usd_per_minute","usd_per_message","usd_per_gb",
    "excess_minutes","excess_messages","excess_data_gb","monthly_revenue"
]
features_df = monthly_usage[features_cols].copy()

features_path_csv  = PROC_DIR / "features_user_month.csv"
features_path_parq = PROC_DIR / "features_user_month.parquet"

features_df.to_csv(features_path_csv, index=False)
try:
    features_df.to_parquet(features_path_parq, index=False)
except Exception:
    pass  # parquet may not be installed; CSV is enough

print("Saved:")
print("  -", features_path_csv)
print("  -", features_path_parq if features_path_parq.exists() else "(parquet not written)")
print("\nPreview:")
print(features_df.head())


