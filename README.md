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
3.  **End-to-End MLOps**:
    *   **MLflow** for tracking experiment accuracy.
    *   **GitHub Actions** for automated testing (CI/CD).
    *   **FastAPI** for sub-second inference at checkout.

---

## 🛠️ Project Architecture

```
├── .github/workflows/    # CI/CD Pipeline (Automated Testing)
├── data/                 # Generated synthetic BNPL transaction data
├── models/               # Trained models (.pkl) & Feature Config
├── src/                  # Source Code
│   ├── analysis/         # Cold Start Research & Audits
│   ├── data_generation/  # Synthetic Data Engine (Realistic Simulation)
│   ├── app.py            # FastAPI Production Server
│   ├── cold_start_handler.py # COLD START LOGIC & GUARDRAILS
│   ├── train_model.py    # Training Pipeline
│   └── test_model.py     # Evaluation Suite
├── params.yaml           # Centralized Hyperparameter Config
└── requirements.txt      # Project Dependencies
```

---

## 🧠 Solved: The Cold Start Problem in BNPL
### The Challenge
A user tries to buy a ₹15,000 phone on EMI but has never used the app before. A standard model sees "0 history" and rejects them.

### Our Solution
We implemented a **Tiered Decision Logic** controlled by `ColdStartHandler`:

| Tier | Tenure | Strategy | Spending Limit |
| :--- | :--- | :--- | :--- |
| **1. New Shopper** | 0-3 Months | **Demographic Model + Guardrails** | ₹5,000 - ₹10,000 |
| **2. Building Trust** | 3-6 Months | **Blended Score** (40% Static / 60% Behavior) | ₹25,000 |
| **3. Power User** | 6+ Months | **Full Behavioral Model** | ₹100,000 |

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

3.  **Generate Fresh Data**
    ```bash
    python src/data_generation/generate_data.py
    python src/data_generation/monthly_data.py
    python src/data_generation/snap.py
    ```

---

## 🏃‍♂️ How to Run

### 1. Train the Models
Retrain both models using the configuration in `params.yaml`:
```bash
python src/train_cold_start_model.py
```
*Artifacts (models/metrics) will be saved to the `models/` folder.*

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
