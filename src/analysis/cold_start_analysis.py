"""
Cold Start Problem Analysis
============================
Analyzes how well the credit score model handles new customers with no transaction history.

The cold start problem in credit scoring occurs when:
1. New customers have no behavioral/transaction history
2. Features like payment_ratio, utilization, missed_dues are all zero/neutral
3. The model must rely primarily on demographic/static features
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("COLD START PROBLEM ANALYSIS")
print("=" * 70)

# =============================
# LOAD MODEL AND DATA
# =============================
print("\n📁 Loading model and training data...")
model = joblib.load("credit_score_model.pkl")
df = pd.read_csv("model_snapshots.csv").drop('customer_id', axis=1)

# Get the expected features from training data
X_train_sample = df.drop("default_next_1m", axis=1)
expected_features = list(X_train_sample.columns)

print(f"   Total features expected by model: {len(expected_features)}")

# =============================
# CATEGORIZE FEATURES
# =============================
print("\n📊 Feature Categorization...")

# Static features (available at onboarding)
static_features = [
    'age', 'employment_status', 'education_level', 'monthly_income',
    'city_tier', 'dependents', 'residence_type', 'account_age_months',
    'snapshot_month'  # Just a time indicator
]

# Behavioral features (require transaction history)
behavioral_features = [
    'util_avg_3m', 'payment_ratio_avg_3m', 'max_outstanding_3m',
    'avg_txn_amt_3m', 'avg_txn_count_3m', 'late_payments_3m',
    'missed_due_count_3m', 'missed_due_last_1m', 'payment_ratio_last_1m',
    'outstanding_delta_3m', 'bnpl_active_last_1m',
    # New features from snap.py
    'consecutive_missed_due', 'payment_ratio_min_3m', 'worst_util_3m',
    'ever_defaulted', 'default_count_history', 'months_since_last_default',
    'outstanding_to_income_pct', 'income_affordability_score', 'debt_burden_category',
    'payment_ratio_trend', 'utilization_trend', 'outstanding_growth_rate',
    'is_deteriorating', 'active_months_3m', 'avg_util_when_active',
    'account_age_bucket', 'risk_score'
]

print(f"\n   Static features (available at onboarding): {len(static_features)}")
for f in static_features:
    if f in expected_features:
        print(f"      ✅ {f}")
    else:
        print(f"      ❌ {f} (not in model)")

print(f"\n   Behavioral features (require history): {len(behavioral_features)}")
behavioral_in_model = [f for f in behavioral_features if f in expected_features]
print(f"      {len(behavioral_in_model)} behavioral features in model")

# =============================
# FEATURE IMPORTANCE BY CATEGORY
# =============================
print("\n🔍 Feature Importance Analysis...")

# Load feature importance
fi = pd.read_csv("feature_importance.csv")

# Categorize feature importance
static_importance = 0
behavioral_importance = 0

for _, row in fi.iterrows():
    feat = row['feature']
    imp = row['importance']
    
    # Check if it's a static feature (including one-hot encoded versions)
    is_static = any(feat.startswith(sf) for sf in static_features)
    is_behavioral = any(feat.startswith(bf) for bf in behavioral_features)
    
    if is_static:
        static_importance += imp
    elif is_behavioral:
        behavioral_importance += imp

total_importance = static_importance + behavioral_importance

print(f"\n   Total Feature Importance Breakdown:")
print(f"   ───────────────────────────────────")
print(f"   Static features:     {static_importance:.4f} ({static_importance/total_importance*100:.1f}%)")
print(f"   Behavioral features: {behavioral_importance:.4f} ({behavioral_importance/total_importance*100:.1f}%)")

if behavioral_importance > static_importance * 2:
    print(f"\n   ⚠️ WARNING: Model heavily relies on behavioral features!")
    print(f"      This may cause issues for cold start customers.")
else:
    print(f"\n   ✅ Model has reasonable balance between feature types.")

# =============================
# CREATE COLD START TEST CASES
# =============================
print("\n🧪 Testing Cold Start Scenarios...")

def prob_to_credit_score(p):
    """Convert default probability to credit score (300-900 scale)"""
    score = 900 - (p * 600)
    return int(max(300, min(score, 900)))

def final_decision(score):
    """Get lending decision based on credit score"""
    if score >= 750:
        return "Approve"
    elif score >= 650:
        return "Approve_Low_Limit"
    elif score >= 550:
        return "Conditional"
    else:
        return "Reject"

# Get default values for behavioral features (neutral/zero)
behavioral_defaults = {
    'util_avg_3m': 0.0,
    'payment_ratio_avg_3m': 1.0,  # Assume perfect payment if no history
    'max_outstanding_3m': 0.0,
    'avg_txn_amt_3m': 0.0,
    'avg_txn_count_3m': 0.0,
    'late_payments_3m': 0,
    'missed_due_count_3m': 0,
    'missed_due_last_1m': 0,
    'payment_ratio_last_1m': 1.0,
    'outstanding_delta_3m': 0.0,
    'bnpl_active_last_1m': 0,
    'snapshot_month': 4,  # Default snapshot month
    # New features from snap.py - cold start defaults
    'consecutive_missed_due': 0,
    'payment_ratio_min_3m': 1.0,  # Perfect payment
    'worst_util_3m': 0.0,
    'ever_defaulted': 0,
    'default_count_history': 0,
    'months_since_last_default': 0,
    'outstanding_to_income_pct': 0.0,  # No debt
    'income_affordability_score': 1.0,  # Will be recalculated
    'debt_burden_category': 0,  # Low risk
    'payment_ratio_trend': 0.0,  # No trend
    'utilization_trend': 0.0,
    'outstanding_growth_rate': 0.0,
    'is_deteriorating': 0,
    'active_months_3m': 0,
    'avg_util_when_active': 0.0,
    'account_age_bucket': 0,  # New account
    'risk_score': 0.0  # Low risk for new customers
}

# Cold start test cases - diverse new customers
cold_start_cases = [
    {
        "name": "Young Salaried Professional (Low Risk)",
        "age": 28, "employment_status": "Salaried", "education_level": "Graduate",
        "monthly_income": 60000, "city_tier": "Tier-1", "dependents": 0,
        "residence_type": "Rented", "account_age_months": 1
    },
    {
        "name": "Senior Executive (Very Low Risk)",
        "age": 45, "employment_status": "Salaried", "education_level": "Postgraduate",
        "monthly_income": 150000, "city_tier": "Tier-1", "dependents": 2,
        "residence_type": "Owned", "account_age_months": 0
    },
    {
        "name": "Self-Employed Business Owner (Medium Risk)",
        "age": 35, "employment_status": "Business", "education_level": "Graduate",
        "monthly_income": 80000, "city_tier": "Tier-2", "dependents": 1,
        "residence_type": "Owned", "account_age_months": 2
    },
    {
        "name": "Daily Wage Worker (Higher Risk)",
        "age": 32, "employment_status": "Daily Wage", "education_level": "Secondary",
        "monthly_income": 15000, "city_tier": "Tier-3", "dependents": 3,
        "residence_type": "Rented", "account_age_months": 0
    },
    {
        "name": "Fresh Graduate (Unknown Risk)",
        "age": 22, "employment_status": "Salaried", "education_level": "Graduate",
        "monthly_income": 25000, "city_tier": "Tier-2", "dependents": 0,
        "residence_type": "Rented", "account_age_months": 0
    },
    {
        "name": "Retired Professional (Medium-Low Risk)",
        "age": 62, "employment_status": "Retired", "education_level": "Graduate",
        "monthly_income": 40000, "city_tier": "Tier-2", "dependents": 0,
        "residence_type": "Owned", "account_age_months": 3
    },
    {
        "name": "Self-Employed Low Income (Higher Risk)",
        "age": 38, "employment_status": "Self-Employed", "education_level": "High School",
        "monthly_income": 18000, "city_tier": "Tier-3", "dependents": 2,
        "residence_type": "Rented", "account_age_months": 1
    },
    {
        "name": "No Formal Education Worker (High Risk)",
        "age": 40, "employment_status": "Daily Wage", "education_level": "No Formal Education",
        "monthly_income": 12000, "city_tier": "Tier-3", "dependents": 4,
        "residence_type": "Rented", "account_age_months": 0
    }
]

print("\n   Cold Start Customer Predictions:")
print("   " + "─" * 90)
print(f"   {'Customer Profile':<45} {'Prob':<8} {'Score':<8} {'Decision':<15}")
print("   " + "─" * 90)

cold_start_results = []

for case in cold_start_cases:
    # Build full feature set
    customer = behavioral_defaults.copy()
    
    # Add static features
    for key in ['age', 'employment_status', 'education_level', 'monthly_income',
                'city_tier', 'dependents', 'residence_type', 'account_age_months']:
        customer[key] = case.get(key)
    
    # Recalculate income_affordability_score based on income
    if customer['monthly_income'] > 0:
        customer['income_affordability_score'] = customer['monthly_income'] / 1.0  # Simplified
    
    # Create DataFrame with correct column order
    customer_df = pd.DataFrame([customer])[expected_features]
    
    # Predict
    try:
        prob = model.predict_proba(customer_df)[0, 1]
        score = prob_to_credit_score(prob)
        decision = final_decision(score)
        
        print(f"   {case['name']:<45} {prob:.4f}   {score:<8} {decision:<15}")
        
        cold_start_results.append({
            'profile': case['name'],
            'default_prob': prob,
            'credit_score': score,
            'decision': decision
        })
    except Exception as e:
        print(f"   {case['name']:<45} ERROR: {str(e)[:30]}")

print("   " + "─" * 90)

# =============================
# COMPARE WITH ESTABLISHED CUSTOMERS
# =============================
print("\n📈 Comparison: Cold Start vs Established Customers...")

# Get a sample of established customers from training data
established_sample = df.sample(1000, random_state=42)
X_established = established_sample.drop("default_next_1m", axis=1)
y_established = established_sample["default_next_1m"]

# Predict on established customers
proba_established = model.predict_proba(X_established)[:, 1]
scores_established = [prob_to_credit_score(p) for p in proba_established]

# Cold start probabilities
cold_start_probs = [r['default_prob'] for r in cold_start_results]
cold_start_scores = [r['credit_score'] for r in cold_start_results]

print(f"\n   Metric                    Cold Start    Established")
print(f"   ─────────────────────────────────────────────────────")
print(f"   Mean Default Probability  {np.mean(cold_start_probs):.4f}        {np.mean(proba_established):.4f}")
print(f"   Mean Credit Score         {np.mean(cold_start_scores):.1f}         {np.mean(scores_established):.1f}")
print(f"   Min Credit Score          {np.min(cold_start_scores):<5}         {np.min(scores_established):<5}")
print(f"   Max Credit Score          {np.max(cold_start_scores):<5}         {np.max(scores_established):<5}")

# =============================
# IDENTIFY COLD START ISSUES
# =============================
print("\n⚠️ Cold Start Problem Assessment:")
print("   " + "─" * 60)

issues = []

# Issue 1: All cold start customers get similar scores
score_variance = np.std(cold_start_scores)
if score_variance < 50:
    issues.append("LOW VARIANCE: Cold start scores have low variance - model may not differentiate well between new customer profiles")
    print(f"   ⚠️ Score variance for cold start: {score_variance:.1f} (low)")
else:
    print(f"   ✅ Score variance for cold start: {score_variance:.1f} (acceptable)")

# Issue 2: Cold start scores are unrealistically high/low
avg_cold_start_score = np.mean(cold_start_scores)
avg_established_score = np.mean(scores_established)
score_diff = abs(avg_cold_start_score - avg_established_score)

if score_diff > 100:
    issues.append(f"SCORE BIAS: Cold start scores differ significantly from established customers ({score_diff:.0f} points)")
    print(f"   ⚠️ Cold start vs established difference: {score_diff:.1f} points (significant)")
else:
    print(f"   ✅ Cold start vs established difference: {score_diff:.1f} points (acceptable)")

# Issue 3: High-risk profile gets approved
high_risk_approved = [r for r in cold_start_results if 'High Risk' in r['profile'] and r['decision'] == 'Approve']
if high_risk_approved:
    issues.append("FALSE APPROVALS: High-risk cold start profiles are being approved")
    print(f"   ⚠️ High-risk profiles approved: {len(high_risk_approved)}")
else:
    print(f"   ✅ High-risk profiles handled appropriately")

# Issue 4: Low-risk profile gets rejected
low_risk_rejected = [r for r in cold_start_results if 'Low Risk' in r['profile'] and r['decision'] == 'Reject']
if low_risk_rejected:
    issues.append("FALSE REJECTIONS: Low-risk cold start profiles are being rejected")
    print(f"   ⚠️ Low-risk profiles rejected: {len(low_risk_rejected)}")
else:
    print(f"   ✅ Low-risk profiles handled appropriately")

# Issue 5: Model relies too heavily on behavioral features
if behavioral_importance > total_importance * 0.7:
    issues.append("FEATURE IMBALANCE: Model relies >70% on behavioral features which are unavailable for cold start")

# =============================
# RECOMMENDATIONS
# =============================
print("\n💡 Recommendations for Production:")
print("   " + "─" * 60)

if len(issues) == 0:
    print("\n   ✅ Model handles cold start reasonably well!")
    print("\n   Suggested approach for new customers:")
    print("   1. Use model predictions with static features + neutral behavioral defaults")
    print("   2. Apply conservative limits for new customers (lower credit limits)")
    print("   3. Implement a 'probationary period' with progressive limit increases")
    print("   4. Consider using alternative data (bureau scores, bank statements) for verification")
else:
    print("\n   ⚠️ Issues identified that may affect cold start handling:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    print("\n   Recommended solutions:")
    print("   1. Create a separate 'new customer' model using only static features")
    print("   2. Implement a tiered approach: use demographic-only model for first 3 months")
    print("   3. Set conservative default limits for cold start customers")
    print("   4. Integrate external credit bureau data for new customer assessment")
    print("   5. Use rule-based system for initial decisions, switch to ML after history builds")

# =============================
# COLD START HANDLING STRATEGY
# =============================
print("\n📋 Suggested Cold Start Handling Strategy:")
print("   " + "─" * 60)
print("""
   PHASE 1: New Customer (0-3 months)
   ─────────────────────────────────
   • Use static features only for scoring
   • Apply maximum credit limit cap (e.g., ₹10,000)
   • Require higher income threshold
   • Monthly review of account behavior

   PHASE 2: Building History (3-6 months)  
   ──────────────────────────────────────
   • Blend static + emerging behavioral features
   • Gradually increase credit limits based on behavior
   • Weight behavioral features at 30%

   PHASE 3: Established Customer (6+ months)
   ─────────────────────────────────────────
   • Full model with all features
   • Standard credit limits apply
   • Regular periodic reviews
""")

# Save cold start results
cold_start_df = pd.DataFrame(cold_start_results)
cold_start_df.to_csv("cold_start_analysis_results.csv", index=False)
print("   📁 Results saved to cold_start_analysis_results.csv")

print("\n" + "=" * 70)
print("COLD START ANALYSIS COMPLETE")
print("=" * 70)
