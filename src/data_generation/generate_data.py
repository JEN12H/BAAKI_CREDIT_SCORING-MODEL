import random
import pandas as pd

random.seed(42)

N_CUSTOMERS = 10000
customers = []

for cid in range(1, N_CUSTOMERS + 1):

    age = random.randint(21, 75)

    # -------- Employment
    if age >= 60:
        employment_status = "Retired"
    else:
        employment_status = random.choices(
            ["Salaried", "Self-Employed", "Business", "Daily Wage"],
            weights=[0.45, 0.25, 0.15, 0.15]
        )[0]

    # -------- Education (realistic)
    education_level = random.choices(
        [
            "No Formal Education",
            "Primary",
            "Secondary",
            "High School",
            "Graduate",
            "Postgraduate"
        ],
        weights=[0.08, 0.12, 0.22, 0.26, 0.22, 0.10]
    )[0]

    # -------- City tier (descriptive, weak signal)
    city_tier = random.choices(
        ["Tier-1", "Tier-2", "Tier-3"],
        weights=[0.25, 0.45, 0.30]
    )[0]

    dependents = random.randint(0, 4)
    residence_type = random.choice(["Owned", "Rented"])
    account_age_months = random.randint(1, 120)

    # -------- Income logic (LONG TAIL)
    income_bucket = random.random()

    if employment_status == "Daily Wage":
        monthly_income = random.randint(10000, 25000)

    elif employment_status == "Salaried":
        if income_bucket < 0.80:
            monthly_income = random.randint(20000, 60000)
        elif income_bucket < 0.97:
            monthly_income = random.randint(60000, 100000)
        else:
            monthly_income = random.randint(100000, 150000)

    elif employment_status == "Self-Employed":
        if income_bucket < 0.65:
            monthly_income = random.randint(25000, 60000)
        elif income_bucket < 0.90:
            monthly_income = random.randint(60000, 120000)
        else:
            monthly_income = random.randint(120000, 180000)

    elif employment_status == "Business":
        if income_bucket < 0.50:
            monthly_income = random.randint(30000, 70000)
        elif income_bucket < 0.80:
            monthly_income = random.randint(70000, 150000)
        else:
            monthly_income = random.randint(150000, 250000)

    else:  # Retired
        monthly_income = random.randint(15000, 50000)

    customers.append([
        cid,
        age,
        employment_status,
        education_level,
        monthly_income,
        city_tier,
        dependents,
        residence_type,
        account_age_months
    ])

pd.DataFrame(customers, columns=[
    "customer_id",
    "age",
    "employment_status",
    "education_level",
    "monthly_income",
    "city_tier",
    "dependents",
    "residence_type",
    "account_age_months"
]).to_csv("data/customers.csv", index=False)

print("✅ customers.csv generated (realistic income & education)")
