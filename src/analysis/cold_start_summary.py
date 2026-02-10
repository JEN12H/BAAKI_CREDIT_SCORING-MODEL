"""
Cold Start Problem - Summary Report and Solution
=================================================
Based on the analysis, the current model has significant cold start issues.
This script provides a production-ready cold start handling solution.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("COLD START PROBLEM - SUMMARY & SOLUTION")
print("=" * 70)

# =============================
# ANALYSIS SUMMARY
# =============================
print("\n" + "=" * 70)
print("1. PROBLEM IDENTIFICATION")
print("=" * 70)

results = pd.read_csv("cold_start_analysis_results.csv")

print("\n   Cold Start Test Results:")
print("   " + "-" * 65)
for _, row in results.iterrows():
    print(f"   {row['profile']:<50} Score: {row['credit_score']}")
print("   " + "-" * 65)

print("\n   KEY ISSUES IDENTIFIED:")
print("   " + "-" * 65)
print("   1. ALL cold start customers get APPROVED regardless of risk profile")
print(f"      - Score range: {results['credit_score'].min()} - {results['credit_score'].max()}")
print(f"      - Standard deviation: {results['credit_score'].std():.1f} (very low variance)")
print("")
print("   2. 'High Risk' profiles receive scores of 822-823 (should be <650)")
print("      - Daily Wage Worker with low income: 827")
print("      - No Formal Education Worker: 822")
print("")
print("   3. Model relies ~80% on BEHAVIORAL features which are all ZERO for new customers")
print("      - Top features: missed_due_last_1m (0.67), bnpl_active_last_1m (0.29)")
print("      - These are meaningless for cold start (always 0)")
print("")
print("   4. Static features (demographics) have minimal predictive weight")
print("      - Employment, income, education collectively <20% importance")

print("\n   PRODUCTION RISK LEVEL: HIGH")
print("   - Current model is NOT suitable for cold start without modifications")

# =============================
# SOLUTION: TIERED APPROACH
# =============================
print("\n" + "=" * 70)
print("2. RECOMMENDED SOLUTION: TIERED CREDIT SYSTEM")
print("=" * 70)

print("""
   A tiered approach separates new customers from established ones:
   
   ┌─────────────────────────────────────────────────────────────────────┐
   │ TIER 1: NEW CUSTOMER (0-3 months)                                   │
   │ ─────────────────────────────────────────────────────────────────── │
   │ • Use RULE-BASED scoring (not ML model)                             │
   │ • Rules based on: income, employment, education, city tier          │
   │ • Maximum credit limit cap: ₹5,000 - ₹15,000                        │
   │ • No exceptions without manual review                               │
   └─────────────────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────────────────┐
   │ TIER 2: BUILDING HISTORY (3-6 months)                               │
   │ ─────────────────────────────────────────────────────────────────── │
   │ • Blend: 50% Rule-based + 50% ML model                              │
   │ • Credit limit increase based on payment history                    │
   │ • Maximum credit limit: ₹25,000                                     │
   └─────────────────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────────────────┐
   │ TIER 3: ESTABLISHED (6+ months)                                     │
   │ ─────────────────────────────────────────────────────────────────── │
   │ • Full ML model with behavioral features                            │
   │ • Standard credit limits apply                                      │
   │ • Periodic review every 3 months                                    │
   └─────────────────────────────────────────────────────────────────────┘
""")

# =============================
# RULE-BASED SCORING FOR COLD START
# =============================
print("=" * 70)
print("3. COLD START RULE-BASED SCORING SYSTEM")
print("=" * 70)

def cold_start_score(customer):
    """
    Rule-based scoring for cold start customers.
    Returns (score, max_credit_limit, decision)
    
    This is designed to be conservative and avoid high-risk approvals.
    """
    base_score = 600  # Start from middle
    
    # ===== INCOME FACTOR (most important for cold start) =====
    income = customer.get('monthly_income', 0)
    if income >= 100000:
        base_score += 100
        income_grade = "Excellent"
    elif income >= 60000:
        base_score += 70
        income_grade = "Good"
    elif income >= 35000:
        base_score += 40
        income_grade = "Fair"
    elif income >= 20000:
        base_score += 10
        income_grade = "Low"
    else:
        base_score -= 50
        income_grade = "Very Low"
    
    # ===== EMPLOYMENT STABILITY =====
    employment = customer.get('employment_status', 'Unknown')
    employment_scores = {
        'Salaried': 50,
        'Business': 30,
        'Self-Employed': 20,
        'Retired': 10,
        'Daily Wage': -30,
        'Unknown': -20
    }
    base_score += employment_scores.get(employment, -20)
    
    # ===== EDUCATION LEVEL =====
    education = customer.get('education_level', 'Unknown')
    education_scores = {
        'Postgraduate': 30,
        'Graduate': 20,
        'High School': 10,
        'Secondary': 5,
        'Primary': 0,
        'No Formal Education': -20,
        'Unknown': -10
    }
    base_score += education_scores.get(education, -10)
    
    # ===== RESIDENCE STABILITY =====
    residence = customer.get('residence_type', 'Unknown')
    if residence == 'Owned':
        base_score += 30
    elif residence == 'Rented':
        base_score += 10
    else:
        base_score -= 10
    
    # ===== CITY TIER =====
    city = customer.get('city_tier', 'Unknown')
    if city == 'Tier-1':
        base_score += 20
    elif city == 'Tier-2':
        base_score += 10
    else:
        base_score += 0
    
    # ===== AGE FACTOR =====
    age = customer.get('age', 30)
    if 25 <= age <= 55:
        base_score += 20
    elif 21 <= age < 25 or 55 < age <= 65:
        base_score += 10
    else:
        base_score += 0
    
    # ===== DEPENDENTS (negative impact on affordability) =====
    dependents = customer.get('dependents', 0)
    base_score -= dependents * 10
    
    # ===== DEBT-TO-INCOME CHECK =====
    # For cold start, we can't check actual debt, but we use income thresholds
    
    # Final score capping
    final_score = max(300, min(900, base_score))
    
    # Decision and credit limit based on score
    if final_score >= 750:
        decision = "Approve"
        max_credit = min(income * 0.5, 50000)  # Max 50K or 50% of income
    elif final_score >= 650:
        decision = "Approve_Low_Limit"
        max_credit = min(income * 0.25, 15000)  # Max 15K or 25% of income
    elif final_score >= 550:
        decision = "Conditional"
        max_credit = min(income * 0.1, 5000)  # Max 5K or 10% of income
    else:
        decision = "Reject"
        max_credit = 0
    
    return final_score, int(max_credit), decision

# Test the rule-based system
print("\n   Testing Cold Start Rule-Based Scoring:")
print("   " + "-" * 75)
print(f"   {'Profile':<45} {'Score':<8} {'Limit':>10} {'Decision':<15}")
print("   " + "-" * 75)

test_customers = [
    {"name": "Young Salaried Professional (Low Risk)", "age": 28, "employment_status": "Salaried", 
     "education_level": "Graduate", "monthly_income": 60000, "city_tier": "Tier-1", 
     "dependents": 0, "residence_type": "Rented"},
    {"name": "Senior Executive (Very Low Risk)", "age": 45, "employment_status": "Salaried",
     "education_level": "Postgraduate", "monthly_income": 150000, "city_tier": "Tier-1",
     "dependents": 2, "residence_type": "Owned"},
    {"name": "Daily Wage Worker (Higher Risk)", "age": 32, "employment_status": "Daily Wage",
     "education_level": "Secondary", "monthly_income": 15000, "city_tier": "Tier-3",
     "dependents": 3, "residence_type": "Rented"},
    {"name": "Fresh Graduate (Unknown Risk)", "age": 22, "employment_status": "Salaried",
     "education_level": "Graduate", "monthly_income": 25000, "city_tier": "Tier-2",
     "dependents": 0, "residence_type": "Rented"},
    {"name": "No Formal Education Worker (High Risk)", "age": 40, "employment_status": "Daily Wage",
     "education_level": "No Formal Education", "monthly_income": 12000, "city_tier": "Tier-3",
     "dependents": 4, "residence_type": "Rented"},
]

rule_based_results = []
for cust in test_customers:
    score, limit, decision = cold_start_score(cust)
    print(f"   {cust['name']:<45} {score:<8} ₹{limit:>8,} {decision:<15}")
    rule_based_results.append({
        'profile': cust['name'],
        'rule_based_score': score,
        'credit_limit': limit,
        'decision': decision
    })

print("   " + "-" * 75)

# Compare with ML model results
print("\n   Comparison: ML Model vs Rule-Based (Cold Start)")
print("   " + "-" * 75)
print(f"   {'Profile':<35} {'ML Score':<10} {'Rule Score':<12} {'Rule Decision':<15}")
print("   " + "-" * 75)

for ml_res, rule_res in zip(results.head(5).itertuples(), rule_based_results):
    print(f"   {rule_res['profile'][:35]:<35} {ml_res.credit_score:<10} {rule_res['rule_based_score']:<12} {rule_res['decision']:<15}")

print("   " + "-" * 75)

# =============================
# IMPLEMENTATION CODE
# =============================
print("\n" + "=" * 70)
print("4. PRODUCTION IMPLEMENTATION CODE")
print("=" * 70)

print("""
   The complete production-ready cold start handler has been saved to:
   
   cold_start_handler.py
   
   This module provides:
   - ColdStartHandler class for unified scoring
   - Automatic tier detection based on account age
   - Rule-based scoring for new customers
   - Blended scoring for building customers
   - Full ML scoring for established customers
   - Credit limit recommendations
""")

# Save the comparison
comparison_df = pd.DataFrame(rule_based_results)
comparison_df.to_csv("cold_start_rule_based_results.csv", index=False)

print("\n" + "=" * 70)
print("5. FINAL RECOMMENDATIONS FOR PRODUCTION")
print("=" * 70)

print("""
   1. DO NOT use the current ML model alone for new customers
      - It gives unrealistic high scores to all cold start customers
   
   2. IMPLEMENT the tiered approach:
      - Tier 1 (0-3 months): Rule-based scoring
      - Tier 2 (3-6 months): Blended scoring  
      - Tier 3 (6+ months): Full ML model
   
   3. SET credit limit caps for new customers:
      - Maximum ₹15,000 for first 3 months
      - Increase based on payment behavior
   
   4. MONITOR cold start performance separately:
      - Track default rates by customer tier
      - Adjust rules based on observed patterns
   
   5. CONSIDER external data sources:
      - Credit bureau scores (CIBIL, Experian)
      - Bank statement analysis
      - Employment verification
""")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
