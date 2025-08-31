import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from pathlib import Path
import sqlite3

# Paths 
PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
DATA_DIR = PROJECT_ROOT / "datasets"          # change to "data" if that's your folder
DB_PATH  = PROJECT_ROOT / "megaline2.db"

# Load data (SQLite first, else CSVs) 
def load_data():
    tables = ["calls", "internet", "messages", "plans", "users"]
    dfs = {}

    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        try:
            for t in tables:
                dfs[t] = pd.read_sql_query(f"SELECT * FROM {t};", conn)
        finally:
            conn.close()
    else:
        for t in tables:
            p = DATA_DIR / f"megaline_{t}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing file: {p}")
            dfs[t] = pd.read_csv(p)

    return dfs["calls"], dfs["internet"], dfs["messages"], dfs["plans"], dfs["users"]

# Preprocess (dates → month YYYY-MM, sanitize durations) 
def preprocess_data(calls, internet, messages, users):
    # Calls
    if "call_date" not in calls.columns:
        raise KeyError("Expected 'call_date' in calls.")
    calls["call_date"] = pd.to_datetime(calls["call_date"], errors="coerce")

    # handle duration vs minutes
    dur_col = "minutes" if "minutes" in calls.columns else ("duration" if "duration" in calls.columns else None)
    if dur_col is None:
        raise KeyError("Calls needs 'minutes' or 'duration' column.")
    # ceil durations, min of 1 minute if you want to bill that way
    calls[dur_col] = pd.to_numeric(calls[dur_col], errors="coerce").fillna(0)
    calls[dur_col] = np.ceil(calls[dur_col]).clip(lower=1)

    # Internet
    if "session_date" not in internet.columns:
        raise KeyError("Expected 'session_date' in internet.")
    internet["session_date"] = pd.to_datetime(internet["session_date"], errors="coerce")

    # Messages
    if "message_date" not in messages.columns:
        raise KeyError("Expected 'message_date' in messages.")
    messages["message_date"] = pd.to_datetime(messages["message_date"], errors="coerce")

    # Month as YYYY-MM (prevents year collisions)
    calls["month"]    = calls["call_date"].dt.to_period("M").astype(str)
    internet["month"] = internet["session_date"].dt.to_period("M").astype(str)
    messages["month"] = messages["message_date"].dt.to_period("M").astype(str)

    return calls, internet, messages, users

#  Monthly Aggregation 
def aggregate_usage(calls, internet, messages):
    # calls total minutes
    dur_col = "minutes" if "minutes" in calls.columns else "duration"
    calls_monthly = (calls.groupby(["user_id","month"], as_index=False)[dur_col]
                          .sum()
                          .rename(columns={dur_col: "total_minutes"}))

    # messages: sum column if exists, else count rows
    if "messages" in messages.columns:
        messages_monthly = (messages.groupby(["user_id","month"], as_index=False)["messages"]
                                   .sum()
                                   .rename(columns={"messages":"messages_sent"}))
    else:
        messages_monthly = (messages.groupby(["user_id","month"], as_index=False)
                                   .size()
                                   .rename(columns={"size":"messages_sent"}))

    # internet total MB
    mb_col = "mb_used" if "mb_used" in internet.columns else ("mb" if "mb" in internet.columns else None)
    if mb_col is None:
        raise KeyError("Internet needs 'mb_used' or 'mb' column.")
    internet_monthly = (internet.groupby(["user_id","month"], as_index=False)[mb_col]
                              .sum()
                              .rename(columns={mb_col:"data_volume_mb"}))

    # Merge aggregates
    df = (calls_monthly
          .merge(internet_monthly, on=["user_id","month"], how="outer")
          .merge(messages_monthly, on=["user_id","month"], how="outer")
          .fillna(0))
    return df

# Merge with User and Plan Info 
def enrich_with_metadata(df, users, plans):
    if not {"user_id","plan","city"}.issubset(users.columns):
        missing = {"user_id","plan","city"} - set(users.columns)
        raise KeyError(f"Users missing columns: {missing}")
    if "plan_name" not in plans.columns:
        raise KeyError("Plans needs 'plan_name' column to join.")

    df = df.merge(users[["user_id","plan","city"]], on="user_id", how="left")
    df = df.merge(plans, left_on="plan", right_on="plan_name", how="left").drop(columns="plan_name")
    return df

# Revenue Calculation 
def calculate_revenue(df):
    req = [
        "total_minutes","messages_sent","data_volume_mb",
        "minutes_included","messages_included","mb_per_month_included",
        "usd_monthly_pay","usd_per_minute","usd_per_message","usd_per_gb"
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for revenue calc: {missing}")

    out = df.copy()
    out[req] = out[req].apply(pd.to_numeric, errors="coerce").fillna(0)

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

# A/B Tests
def perform_ab_testing(df):
    # plan names normalized to lower
    plan_col = df["plan"].str.lower()

    surf     = df.loc[plan_col == "surf",     "monthly_revenue"]
    ultimate = df.loc[plan_col == "ultimate", "monthly_revenue"]

    t_stat, p_val = ttest_ind(ultimate, surf, equal_var=False, nan_policy="omit")
    print(f"[A/B Test - Plan Revenue] T-statistic: {t_stat:.2f}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Reject H0: Significant difference in revenue between Surf and Ultimate plans.")
    else:
        print("Fail to reject H0: No significant difference in revenue.")

    # Region split (robust to NaN)
    city = df["city"].fillna("")
    ny   = df.loc[city.str.contains("NY-NJ", case=False, na=False), "monthly_revenue"]
    other= df.loc[~city.str.contains("NY-NJ", case=False, na=False), "monthly_revenue"]

    t_stat2, p_val2 = ttest_ind(ny, other, equal_var=False, nan_policy="omit")
    print(f"[A/B Test - Region Revenue] T-statistic: {t_stat2:.2f}, p-value: {p_val2:.4f}")
    if p_val2 < 0.05:
        print("Reject H0: Significant revenue difference between NY-NJ and other regions.")
    else:
        print("Fail to reject H0: No significant revenue difference by region.")

#  Plot 
def plot_revenue_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="monthly_revenue", hue="plan", kde=True, bins=30, alpha=0.7)
    plt.title("Revenue Distribution by Plan")
    plt.xlabel("Monthly Revenue ($)")
    plt.ylabel("Frequency")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()

# Main 
if __name__ == "__main__":
    calls, internet, messages, plans, users = load_data()
    calls, internet, messages, users = preprocess_data(calls, internet, messages, users)

    usage_df   = aggregate_usage(calls, internet, messages)
    enriched   = enrich_with_metadata(usage_df, users, plans)
    final_df   = calculate_revenue(enriched)

    print("\nAverage Revenue by Plan:")
    print(final_df.groupby("plan", as_index=False)["monthly_revenue"].agg(["mean","median","sum"]).round(2))

    perform_ab_testing(final_df)
    plot_revenue_distribution(final_df)