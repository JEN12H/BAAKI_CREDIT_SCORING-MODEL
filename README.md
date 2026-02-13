# 🛒 Intelligent Credit Scoring Engine for BNPL (Buy Now Pay Later)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![Scikit-Learn](https://img.shields.io/badge/sklearn-1.2%2B-yellow)

## 📖 Project Overview
This project is a real-time **Risk Assessment Engine** designed for **Buy Now Pay Later (BNPL)** applications.

The biggest challenge in BNPL is the **Cold Start Problem**: new users want to buy *instantly* at checkout, but platforms have zero transaction history for them. Traditional banks reject these users, leading to lost sales.

**Our Solution**: A **Dual-Model Architecture** that enables **Instant Approvals** for new shoppers while protecting the platform from fraud and default risk.

---

## 🚀 Key Features
1.  **Instant Checkout Decisions**:
    *   **Cold Start Model**: Instantly scores new users (0-3 months) using *only* checkout & demographic data (Age, Income, Education) to assign a safe initial spending limit.
    *   **Full Model**: Unlocks higher spending limits for repeat users (6+ months) based on repayment behavior (Payment Ratios, BNPL usage).
2.  **Safety Guardrails**: 
    *   **Tiered Spending Caps**: New users are capped at lower amounts (e.g., ₹5,000) until they prove reliability.
    *   **Risk Rules**: Automatically flags high-risk profiles (e.g., very young borrowers with low income) to prevent default.
3.  **End-to-End MLOps & Data Pipeline**:
    *   **Automated Data Pipeline**: One-click extraction and feature engineering from raw CSVs. 
    *   **Watch Mode**: Automatically recalculates credit scores as soon as you save data in Excel or CSV.
    *   **MLflow** for tracking experiment accuracy.
    *   **GitHub Actions** for automated testing (CI/CD).
    *   **FastAPI** for sub-second inference at checkout.

---

## 🛠️ Project Architecture

```
├── .github/workflows/    # CI/CD Pipeline (Automated Testing)
├── data/                 # Raw CSVs (customers, behavior) & Calculated Snapshots
├── models/               # Trained models (.pkl) & Feature Config
├── src/                  # Source Code
│   ├── analysis/         # Cold Start Research & Audits
│   ├── data_generation/  # Synthetic Data Engine
│   ├── app.py            # FastAPI Production Server
│   ├── pipeline.py       # DATA PIPELINE ORCHESTRATOR (Watch Mode)
│   ├── cold_start_handler.py # COLD START LOGIC & GUARDRAILS
│   ├── train_model.py    # Training Pipeline (with auto-sync)
│   └── test_model.py     # Evaluation Suite
├── params.yaml           # Centralized Hyperparameter Config
└── requirements.txt      # Project Dependencies
```

---

## 🧠 Solved: The Cold Start Problem in BNPL
### The Challenge
A user tries to buy a ₹15,000 phone on EMI but has never used the app before. A standard model sees "0 history" and feels "This person is a mystery" – it usually defaults to a **Reject**, causing the business to lose a potential loyal customer.

### Our Solution: The Two-Model Strategy
Instead of using one "all-or-nothing" model, we built a tiered system that handles a customer differently as they grow with us.

---

#### 🏗️ Model 1: The Cold Start Model (Demographics)
**"Approval Based on Who You Are"**

*   **When it's used**: During the very first checkout (0-3 months of tenure).
*   **What it looks at**: Static data points like your **Age, Employment stability, Education level, and Monthly Income.**
*   **The Logic**: It uses a conservative "Demographic Profile" to predict risk. Since we don't know your spending habits yet, it assumes a cautious stance.
*   **The Result**: It grants a **Safe Entry Limit** (e.g., ₹5,000 to ₹10,000). This allows the customer to join the platform instantly without the business taking a massive risk.

#### 📈 Model 2: The Full Behavioral Model (Transaction History)
**"Reward Based on How You Pay"**

*   **When it's used**: After a user has been with us for 6+ months.
*   **What it looks at**: Thousands of data points from their **actual app usage** – how quickly they pay bills, their repayment ratios, and missed payment streaks.
*   **The Logic**: Transaction data is **10x more accurate** than basic demographic data. This model ignores "who you are" and looks only at "how responsible you are."
*   **The Result**: If the user is reliable, the system **unlocks High Spending Limits** (up to ₹100,000).

---

### 🛡️ Why use two models?
1.  **Stop "Blind Rejections"**: Traditional systems reject new users because they have no data. Our Model 1 gives them a chance.
2.  **Precision Risk Management**: Behavioral data is the "Gold Standard" of credit. Using it for established users means we can give huge limits to trustworthy people while being very precise about who to stop.
3.  **Customer Growth Path**: It creates a "gamified" experience where users know that by paying on time, they are "leveling up" from Model 1 to Model 2, earning higher trust and limits.

---

### 🏛️ Tiered Decision Logic
We implemented a **Tiered Decision Logic** controlled by `ColdStartHandler`:

| Tier | Tenure | Strategy | Spending Limit |
| :--- | :--- | :--- | :--- |
| **1. New Shopper** | 0-3 Months | **Demographic Model + Guardrails** | ₹5,000 - ₹10,000 |
| **2. Building Trust** | 3-6 Months | **Blended Score** (40% Static / 60% Behavior) | ₹25,000 |
| **3. Power User** | 6+ Months | **Full Behavioral Model** | ₹100,000 |

---

## 📊 Understanding Credit Behavior (The Behavioral Features)
For established customers, our model stops looking at just "who you are" (Age/Income) and starts looking at **"how you handle money."** Here is a non-technical breakdown of the behavioral metrics our system calculates automatically:

### 🚩 1. The "Red Flags" (Risk Signals)
These are the most critical features that alert the system to potential defaults.
*   **Consecutive Missed Payments**: Does the user miss payments back-to-back? A "streak" of missed payments is the strongest predictor that a user won't pay the next bill.
*   **Recent Default History**: Has the user defaulted in the past? The system tracks how many months it has been since the last issue.
*   **Late Payment Count**: Simply counting how many times a user was late in the last 3 months.

### 💳 2. Financial Discipline (Repayment Habits)
These metrics measure if a user is living within their means.
*   **Repayment Ratio**: If a user spends ₹1,000, do they pay back the full ₹1,000 or only ₹400? Higher ratios mean better discipline.
*   **Credit Limit Usage (Utilization)**: Is the user constantly maxing out their limit? Using 90% of your limit is much riskier than using 20%.
*   **Payment Trends**: Is the user's behavior getting better or worse? If someone's repayment ratio is dropping every month, the system flags them as "Deteriorating."

### 💰 3. Financial Health (Affordability)
These metrics ensure we aren't lending more than a user can actually afford.
*   **Debt-to-Income Ratio**: The system compares total outstanding debt against the user's monthly salary.
*   **Income Affordability Score**: A safety score that checks if there is "breathing room" in the user's budget after paying their BNPL bills.
*   **Debt Growth Rate**: Is the user's debt growing faster than they can pay it off?

### 🛍️ 4. Shopping Habits (Engagement)
These help the model understand if the user is a regular, stable shopper.
*   **Active Months**: Has the user been shopping consistently over the last quarter, or was it a one-time spending spree?
*   **Average Bill Size**: Helps distinguish between someone buying groceries (stable) vs. someone buying expensive electronics (potential high-risk "hit and run").

### ⚖️ 5. The Internal "Risk Score"
Finally, our pipeline combines all of the above into a single **0 to 100 Risk Score**. 
*   **0-20**: Very Safe (Prime User)
*   **20-50**: Caution (Moderate Risk)
*   **50+**: High Alert (Extreme Risk)

---

## ⚡ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-repo/bnpl-credit-model.git
    cd bnpl-credit-model
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Data Pipeline**
    The system now automatically syncs your CSV data:
    ```bash
    # Option A: One-time calculation
    python src/pipeline.py
    
    # Option B: Automatic "Watch Mode" (Recalculates every time you save your CSV)
    python src/pipeline.py --watch
    ```

---

## 🏃‍♂️ How to Run

### 1. Training & Auto-Recalibration
The training scripts are now **pipeline-aware**. They automatically regenerate features from raw data before training:
```bash
python src/train_model.py
```
*This ensures you are always training on the latest version of your CSV data.*

### 2. Start the API Server
Launch the production-ready REST API for the checkout system:
```bash
uvicorn src.app:app --reload
```

### 3. Open Swagger UI
Visit **http://127.0.0.1:8000/docs** to test endpoints interactively.

---

## 🔌 API Usage Examples

### 1. Score a New User (At Checkout)
**Endpoint:** `POST /predict/cold-start`

*Input (User verifying profile at checkout):*
```json
{
  "age": 28,
  "employment_status": "Salaried",
  "monthly_income": 85000,
  "education_level": "Graduate",
  "city_tier": "Tier-1",
  "dependents": 0,
  "residence_type": "Rented",
  "account_age_months": 1
}
```

*Response:*
```json
{
  "model_type": "cold_start_guarded",
  "credit_score": 704,
  "decision": "Approve_Low_Limit",
  "max_credit_limit": 10000,
  "recommendation": "Cold start model has limited predictive power..."
}
```
*Result: User is approved for ₹10,000 buy-now-pay-later limit.*

### 2. Score a Power User (Limit Increase)
**Endpoint:** `POST /predict/full`

*Input (User requesting higher limit):*
```json
{
  "age": 45,
  "monthly_income": 150000,
  "account_age_months": 48,
  "util_avg_3m": 0.10,
  "payment_ratio_avg_3m": 1.0,
  "missed_due_count_3m": 0,
  ... (other behavioral features)
}
```

*Response:*
```json
{
  "model_type": "full_model",
  "credit_score": 815,
  "decision": "Approve",
  "max_credit_limit": 100000
}
```
*Result: User unlocked for ₹100,000 limit.*

---

## 📊 Performance & Results
*   **Full Model AUC:** 0.85+ (Excellent discrimination on established users)
*   **Cold Start Logic:** Successfully minimizes default rates.
    *   *High Risk New User* -> **Rejected** or **Capped at ₹5k**.
    *   *Safe New User* -> **Approved** but **Capped at ₹15k**.

## 🛡️ License
MIT License.
