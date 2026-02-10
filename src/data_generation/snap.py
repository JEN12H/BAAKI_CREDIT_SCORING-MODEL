import pandas as pd
import numpy as np
import time

# =============================
# CONFIG
# =============================
LOOKBACK_MONTHS = 3
MAX_SAVE_RETRIES = 3

# =============================
# HELPER FUNCTIONS
# =============================
def count_consecutive_missed(series):
    """Count consecutive missed payments from most recent backwards."""
    streak = 0
    for val in reversed(series.values):
        if val == 1:
            streak += 1
        else:
            break
    return streak

def safe_divide(a, b, default=0):
    """Safe division handling zero and None"""
    if b is None or b == 0:
        return default
    return a / b

def categorize_debt_burden(pct):
    """Categorize debt burden percentage into risk tiers"""
    if pct < 10:
        return 0  # Low risk
    elif pct < 30:
        return 1  # Medium risk
    elif pct < 50:
        return 2  # High risk
    else:
        return 3  # Critical risk

def categorize_account_age(months):
    """Categorize account age into maturity buckets"""
    if months < 6:
        return 0  # New
    elif months < 24:
        return 1  # Established
    else:
        return 2  # Mature

def safe_save_csv(df, filename, retries=3):
    """Save CSV with retry logic for file lock issues"""
    for attempt in range(retries):
        try:
            df.to_csv(filename, index=False)
            return True
        except PermissionError:
            print(f"⚠️ File locked, retrying in 2s... ({attempt + 1}/{retries})")
            time.sleep(2)
    # Fallback to backup file
    backup_name = filename.replace('.csv', '_backup.csv')
    df.to_csv(backup_name, index=False)
    print(f"⚠️ Saved to {backup_name} instead")
    return False

# =============================
# LOAD DATA
# =============================
customers = pd.read_csv("data/customers.csv")
behavior = pd.read_csv("data/credit_behavior_monthly.csv")

# Sanity checks
assert "customer_id" in customers.columns
assert "customer_id" in behavior.columns
assert "month" in behavior.columns
assert "default_event" in behavior.columns

# Create customer lookup for faster access
customer_lookup = customers.set_index("customer_id").to_dict("index")

snapshots = []

# =============================
# CREATE SNAPSHOTS
# =============================
for cid in behavior["customer_id"].unique():

    user_df = (
        behavior[behavior["customer_id"] == cid]
        .sort_values("month")
        .reset_index(drop=True)
    )
    
    # Get customer profile data
    customer = customer_lookup.get(cid, {})
    monthly_income = customer.get("monthly_income", 0)
    account_age_months = customer.get("account_age_months", 0)

    # Track historical defaults for this customer
    historical_defaults = []

    # need at least 3 past months + 1 current month
    for t in range(LOOKBACK_MONTHS, len(user_df)):

        past = user_df.iloc[t - LOOKBACK_MONTHS : t]
        last = user_df.iloc[t - 1]
        current = user_df.iloc[t]
        
        # Track defaults up to this point
        prior_defaults = user_df.iloc[:t]["default_event"].tolist()

        # -------------------------
        # BASIC AGGREGATES
        # -------------------------
        util_avg_3m = past["credit_utilization"].mean()
        payment_ratio_avg_3m = past["payment_ratio"].mean()
        max_outstanding_3m = past["outstanding_balance"].max()
        avg_txn_amt_3m = past["avg_transaction_amount"].mean()
        avg_txn_count_3m = past["num_transactions"].mean()
        late_payments_3m = past["late_payment"].sum()
        missed_due_count_3m = past["missed_due_flag"].sum()

        # -------------------------
        # RECENCY / VELOCITY (KEY)
        # -------------------------
        missed_due_last_1m = int(last["missed_due_flag"])
        payment_ratio_last_1m = last["payment_ratio"]
        outstanding_delta_3m = (
            past["outstanding_balance"].iloc[-1]
            - past["outstanding_balance"].iloc[0]
        )
        bnpl_active_last_1m = int(last["num_transactions"] > 0)

        # -------------------------
        # NEW: RISK SIGNALS (HIGH RECALL)
        # -------------------------
        # Consecutive missed payments streak
        consecutive_missed_due = count_consecutive_missed(past["missed_due_flag"])
        
        # Worst-case scenarios (more predictive than averages)
        payment_ratio_min_3m = past["payment_ratio"].min()
        worst_util_3m = past["credit_utilization"].max()
        
        # Historical default behavior
        ever_defaulted = 1 if sum(prior_defaults) > 0 else 0
        default_count_history = sum(prior_defaults)
        
        # Months since last default (0 if never defaulted)
        months_since_last_default = 0
        if ever_defaulted:
            for i, d in enumerate(reversed(prior_defaults)):
                if d == 1:
                    months_since_last_default = i
                    break

        # -------------------------
        # NEW: AFFORDABILITY (PRECISION)
        # -------------------------
        # Debt burden relative to income
        outstanding_to_income_pct = safe_divide(
            current["outstanding_balance"] * 100, 
            monthly_income, 
            default=100  # High risk if no income data
        )
        
        # Income affordability score
        income_affordability_score = safe_divide(
            monthly_income, 
            max_outstanding_3m + 1,
            default=0
        )
        
        # Debt burden category (0=Low, 1=Med, 2=High, 3=Critical)
        debt_burden_category = categorize_debt_burden(outstanding_to_income_pct)

        # -------------------------
        # NEW: BEHAVIORAL TRENDS (AUC)
        # -------------------------
        # Payment ratio trend (positive = improving, negative = deteriorating)
        payment_ratio_trend = past["payment_ratio"].iloc[-1] - past["payment_ratio"].iloc[0]
        
        # Utilization trend
        utilization_trend = past["credit_utilization"].iloc[-1] - past["credit_utilization"].iloc[0]
        
        # Outstanding balance growth rate
        outstanding_growth_rate = safe_divide(
            outstanding_delta_3m,
            past["outstanding_balance"].iloc[0] + 1,
            default=0
        )
        
        # Combined deterioration flag
        is_deteriorating = int(payment_ratio_trend < 0 and utilization_trend > 0)

        # -------------------------
        # NEW: ENGAGEMENT & STABILITY
        # -------------------------
        # Active months in last 3
        active_months_3m = int((past["num_transactions"] > 0).sum())
        
        # Average utilization when active (more accurate)
        active_mask = past["num_transactions"] > 0
        if active_mask.any():
            avg_util_when_active = past.loc[active_mask, "credit_utilization"].mean()
        else:
            avg_util_when_active = 0.0
            
        # Account maturity bucket
        account_age_bucket = categorize_account_age(account_age_months)
        
        # -------------------------
        # NEW: COMPOSITE RISK SCORE
        # -------------------------
        # Weighted risk score (0-100 scale)
        risk_score = (
            missed_due_last_1m * 25 +  # Strongest signal
            consecutive_missed_due * 15 +
            (1 - payment_ratio_min_3m) * 20 +
            min(outstanding_to_income_pct / 100, 1) * 15 +
            is_deteriorating * 10 +
            (1 if payment_ratio_avg_3m < 0.7 else 0) * 10 +
            ever_defaulted * 5
        )
        risk_score = min(risk_score, 100)  # Cap at 100

        # -------------------------
        # TARGET (NEXT MONTH)
        # -------------------------
        default_next_1m = int(current["default_event"])

        snapshot = {
            # Identifiers
            "customer_id": cid,
            "snapshot_month": int(current["month"]),

            # Basic Aggregates
            "util_avg_3m": util_avg_3m,
            "payment_ratio_avg_3m": payment_ratio_avg_3m,
            "max_outstanding_3m": max_outstanding_3m,
            "avg_txn_amt_3m": avg_txn_amt_3m,
            "avg_txn_count_3m": avg_txn_count_3m,
            "late_payments_3m": late_payments_3m,
            "missed_due_count_3m": missed_due_count_3m,

            # Recency / Velocity
            "missed_due_last_1m": missed_due_last_1m,
            "payment_ratio_last_1m": payment_ratio_last_1m,
            "outstanding_delta_3m": outstanding_delta_3m,
            "bnpl_active_last_1m": bnpl_active_last_1m,

            # Risk Signals (NEW)
            "consecutive_missed_due": consecutive_missed_due,
            "payment_ratio_min_3m": payment_ratio_min_3m,
            "worst_util_3m": worst_util_3m,
            "ever_defaulted": ever_defaulted,
            "default_count_history": default_count_history,
            "months_since_last_default": months_since_last_default,

            # Affordability (NEW)
            "outstanding_to_income_pct": outstanding_to_income_pct,
            "income_affordability_score": income_affordability_score,
            "debt_burden_category": debt_burden_category,

            # Behavioral Trends (NEW)
            "payment_ratio_trend": payment_ratio_trend,
            "utilization_trend": utilization_trend,
            "outstanding_growth_rate": outstanding_growth_rate,
            "is_deteriorating": is_deteriorating,

            # Engagement & Stability (NEW)
            "active_months_3m": active_months_3m,
            "avg_util_when_active": avg_util_when_active,
            "account_age_bucket": account_age_bucket,

            # Composite Risk Score (NEW)
            "risk_score": risk_score,

            # Target
            "default_next_1m": default_next_1m
        }

        snapshots.append(snapshot)

# =============================
# BUILD SNAPSHOT DATAFRAME
# =============================
snapshots_df = pd.DataFrame(snapshots)

# =============================
# ENRICH WITH CUSTOMER PROFILE
# =============================
snapshots_df = snapshots_df.merge(
    customers,
    on="customer_id",
    how="left"
)

# =============================
# SANITY CHECKS
# =============================
assert snapshots_df["missed_due_count_3m"].max() <= LOOKBACK_MONTHS
assert snapshots_df["default_next_1m"].isin([0, 1]).all()
assert snapshots_df["consecutive_missed_due"].max() <= LOOKBACK_MONTHS
assert snapshots_df["active_months_3m"].max() <= LOOKBACK_MONTHS
assert snapshots_df["debt_burden_category"].isin([0, 1, 2, 3]).all()

# =============================
# SAVE
# =============================
if safe_save_csv(snapshots_df, "data/model_snapshots.csv"):
    print("✅ model_snapshots.csv created successfully")
else:
    print("⚠️ Saved to backup file due to permission issue")

print(f"Rows: {snapshots_df.shape[0]}")
print(f"Features: {snapshots_df.shape[1]}")
print(f"Default rate: {round(snapshots_df['default_next_1m'].mean(), 4)}")

# =============================
# FEATURE QUALITY REPORT
# =============================
print("\n📊 New Feature Statistics:")
print(f"  - Consecutive missed due rate: {(snapshots_df['consecutive_missed_due'] > 0).mean():.2%}")
print(f"  - Ever defaulted rate: {snapshots_df['ever_defaulted'].mean():.2%}")
print(f"  - Deteriorating behavior rate: {snapshots_df['is_deteriorating'].mean():.2%}")
print(f"  - Avg risk score: {snapshots_df['risk_score'].mean():.1f}")
print(f"  - High debt burden (>30%): {(snapshots_df['outstanding_to_income_pct'] > 30).mean():.2%}")
