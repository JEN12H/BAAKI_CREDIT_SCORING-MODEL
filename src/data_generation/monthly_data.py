import random
import pandas as pd

random.seed(42)

# =========================
# CONFIG
# =========================
N_CUSTOMERS = 10000
N_MONTHS = 12

rows = []

# =========================
# STEP 1: GENERATE MONTHLY BEHAVIOR
# =========================
for cid in range(1, N_CUSTOMERS + 1):

    outstanding_balance = 0

    for month in range(1, N_MONTHS + 1):

        active = random.random() > 0.25  # 75% chance active

        if not active:
            rows.append([
                cid,
                month,
                0.0,     # credit_utilization
                0,       # late_payment
                1.0,     # payment_ratio
                outstanding_balance,
                0,       # default_event (placeholder)
                0,       # num_transactions
                0,       # avg_transaction_amount
                0        # missed_due_flag
            ])
            continue

        bill_amount = random.randint(500, 10000)
        payment_ratio = round(random.uniform(0.4, 1.0), 2)

        paid_amount = int(bill_amount * payment_ratio)
        unpaid_amount = bill_amount - paid_amount
        outstanding_balance += unpaid_amount

        missed_due_flag = 1 if payment_ratio < 0.8 else 0
        late_payment = 1 if (0 < unpaid_amount < bill_amount) else 0

        credit_utilization = round(bill_amount / 10000, 2)
        num_transactions = random.randint(1, 5)
        avg_txn_amt = int(bill_amount / num_transactions)

        rows.append([
            cid,
            month,
            credit_utilization,
            late_payment,
            payment_ratio,
            outstanding_balance,
            0,  # default_event placeholder
            num_transactions,
            avg_txn_amt,
            missed_due_flag
        ])

df = pd.DataFrame(rows, columns=[
    "customer_id",
    "month",
    "credit_utilization",
    "late_payment",
    "payment_ratio",
    "outstanding_balance",
    "default_event",
    "num_transactions",
    "avg_transaction_amount",
    "missed_due_flag"
])

# =========================
# STEP 2: GENERATE DEFAULT EVENTS (CALIBRATED)
# =========================
df["default_event"] = 0

for cid in df.customer_id.unique():

    user_df = df[df.customer_id == cid].sort_values("month")

    for i in range(1, len(user_df)):

        row = user_df.iloc[i]
        prev = user_df.iloc[i - 1]

        # ---- CALIBRATED DEFAULT PROBABILITY ----
        p_default = (
            0.015                              # base risk
            + 0.18 * prev.missed_due_flag
            + 0.10 * (1 - prev.payment_ratio)
            + 0.06 * min(prev.outstanding_balance / 10000, 1)
            + 0.04 * prev.late_payment
        )

        # mild randomness (realism)
        p_default += random.uniform(-0.015, 0.015)

        # clamp probability
        p_default = max(0.005, min(p_default, 0.45))

        df.loc[row.name, "default_event"] = int(random.random() < p_default)

# =========================
# STEP 3: SAVE
# =========================
df.to_csv("data/credit_behavior_monthly.csv", index=False)

print("✅ credit_behavior_monthly.csv generated")
print("Default rate:", round(df["default_event"].mean(), 4))

# =========================
# QUICK SANITY CHECKS
# =========================
print("\nDefault rate by missed_due_flag:")
print(df.groupby("missed_due_flag")["default_event"].mean())
print("\nDefault rate by payment_ratio buckets:")