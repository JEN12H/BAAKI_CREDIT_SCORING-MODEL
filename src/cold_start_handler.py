"""
Cold Start Handler - Production Module (Updated)
=================================================
Uses the trained ML models for credit scoring with proper cold start handling.

Key Insight from Training:
- Static features alone (demographics) have AUC ~0.51 (barely better than random)
- This confirms the cold start problem: new customers can't be reliably scored
- Solution: Use conservative limits + rule-based guardrails + gradual limit increases

This module provides:
- ColdStartHandler class with trained ML models
- Tiered approach based on account age
- Conservative risk management for new customers
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ColdStartHandler:
    """
    Production-ready credit scoring handler that addresses cold start problem.
    
    Uses two trained models:
    1. cold_start_model.pkl - For new customers (static features only)
    2. credit_score_model.pkl - For established customers (all features)
    
    Implements tiered approach with conservative limits for new customers.
    """
    
    # Tier thresholds (in months)  
    TIER_1_MAX_MONTHS = 3   # New customer
    TIER_2_MAX_MONTHS = 6   # Building history
    
    # Credit limit caps by tier (conservative for cold start)
    MAX_LIMIT_TIER_1 = 10000   # Very conservative for new customers
    MAX_LIMIT_TIER_2 = 25000
    MAX_LIMIT_TIER_3 = 100000
    
    def __init__(self, 
                 cold_start_model_path: str = "cold_start_model.pkl",
                 full_model_path: str = "credit_score_model.pkl",
                 feature_config_path: str = "feature_config.pkl"):
        """Initialize with trained models."""
        
        # Load cold start model
        try:
            self.cold_start_model = joblib.load(cold_start_model_path)
            self.cold_start_loaded = True
            print(f"   Loaded: {cold_start_model_path}")
        except Exception as e:
            print(f"   Warning: Could not load cold start model: {e}")
            self.cold_start_model = None
            self.cold_start_loaded = False
        
        # Load full model
        try:
            self.full_model = joblib.load(full_model_path)
            self.full_model_loaded = True
            print(f"   Loaded: {full_model_path}")
        except Exception as e:
            print(f"   Warning: Could not load full model: {e}")
            self.full_model = None
            self.full_model_loaded = False
        
        # Load feature config
        try:
            self.feature_config = joblib.load(feature_config_path)
            print(f"   Loaded: {feature_config_path}")
        except:
            # Default feature lists
            self.feature_config = {
                'static_features': [
                    'age', 'employment_status', 'education_level', 'monthly_income',
                    'city_tier', 'dependents', 'residence_type', 'account_age_months'
                ],
                'all_features': None
            }
    
    def get_customer_tier(self, account_age_months: int) -> Tuple[int, str]:
        """Determine customer tier based on account age."""
        if account_age_months <= self.TIER_1_MAX_MONTHS:
            return 1, "New Customer (Cold Start)"
        elif account_age_months <= self.TIER_2_MAX_MONTHS:
            return 2, "Building History"
        else:
            return 3, "Established Customer"
    
    def _prob_to_score(self, prob: float) -> int:
        """Convert default probability to credit score (300-900 scale)."""
        score = 900 - (prob * 600)
        return int(max(300, min(900, score)))
    
    def _score_to_decision(self, score: int) -> str:
        """Convert credit score to lending decision."""
        if score >= 750:
            return "Approve"
        elif score >= 650:
            return "Approve_Low_Limit"
        elif score >= 550:
            return "Conditional"
        else:
            return "Reject"
    
    def _apply_risk_guardrails(self, customer: Dict, score: int, max_limit: int) -> Tuple[int, int, str, list]:
        """
        Apply rule-based guardrails for cold start customers.
        
        These rules add protection since ML model for cold start has low accuracy.
        Returns: (adjusted_score, adjusted_limit, adjusted_decision, warnings)
        """
        warnings = []
        adjusted_score = score
        adjusted_limit = max_limit
        
        income = customer.get('monthly_income', 0)
        employment = customer.get('employment_status', 'Unknown')
        dependents = customer.get('dependents', 0)
        age = customer.get('age', 30)
        
        # Rule 1: Low income check
        if income < 20000:
            adjusted_score = min(adjusted_score, 600)
            adjusted_limit = min(adjusted_limit, 5000)
            warnings.append("Low income - maximum limit capped")
        
        # Rule 2: High dependents relative to income
        if dependents > 0 and income / (dependents + 1) < 15000:
            adjusted_score = min(adjusted_score, 580)
            adjusted_limit = min(adjusted_limit, 5000)
            warnings.append("High dependency ratio - limit reduced")
        
        # Rule 3: Risky employment type
        risky_employment = ['Daily Wage', 'Unemployed']
        if employment in risky_employment:
            adjusted_score = min(adjusted_score, 550)
            adjusted_limit = min(adjusted_limit, 3000)
            warnings.append(f"High-risk employment ({employment}) - conservative limits applied")
        
        # Rule 4: Very young borrower
        if age < 23:
            adjusted_limit = min(adjusted_limit, 8000)
            warnings.append("Young borrower - limit capped")
        
        # Rule 5: Very old borrower
        if age > 60:
            adjusted_limit = min(adjusted_limit, income * 0.25)
            warnings.append("Senior borrower - limit adjusted to income")
        
        # Get final decision based on adjusted score
        adjusted_decision = self._score_to_decision(adjusted_score)
        
        return adjusted_score, int(adjusted_limit), adjusted_decision, warnings
    
    def score_cold_start(self, customer: Dict) -> Dict:
        """Score a new customer using cold start model + guardrails."""
        
        static_features = self.feature_config.get('static_features', [])
        
        # Prepare feature DataFrame
        customer_df = pd.DataFrame([{k: customer.get(k) for k in static_features}])
        
        if self.cold_start_loaded:
            try:
                prob = self.cold_start_model.predict_proba(customer_df)[0, 1]
                ml_score = self._prob_to_score(prob)
            except Exception as e:
                prob = 0.2  # Default probability
                ml_score = 780
        else:
            prob = 0.2
            ml_score = 780
        
        # Calculate initial limit based on income
        income = customer.get('monthly_income', 0)
        initial_limit = min(income * 0.3, self.MAX_LIMIT_TIER_1)
        
        # Apply guardrails (critical for cold start due to low model accuracy)
        final_score, final_limit, decision, warnings = self._apply_risk_guardrails(
            customer, ml_score, initial_limit
        )
        
        return {
            'ml_probability': prob,
            'ml_score': ml_score,
            'final_score': final_score,
            'decision': decision,
            'max_credit_limit': final_limit,
            'risk_warnings': warnings,
            'model_used': 'cold_start_model + guardrails',
            'note': 'Cold start model has limited predictive power. Conservative limits applied.'
        }
    
    def score_established(self, customer: Dict) -> Dict:
        """Score an established customer using full model."""
        
        # Prepare features with defaults for any missing values
        all_features = self.feature_config.get('all_features', [])
        
        # Default values for behavioral features
        defaults = {
            'snapshot_month': 6, 'util_avg_3m': 0.0, 'payment_ratio_avg_3m': 1.0,
            'max_outstanding_3m': 0.0, 'avg_txn_amt_3m': 0.0, 'avg_txn_count_3m': 0.0,
            'late_payments_3m': 0, 'missed_due_count_3m': 0, 'missed_due_last_1m': 0,
            'payment_ratio_last_1m': 1.0, 'outstanding_delta_3m': 0.0,
            'bnpl_active_last_1m': 0, 'consecutive_missed_due': 0,
            'payment_ratio_min_3m': 1.0, 'worst_util_3m': 0.0, 'ever_defaulted': 0,
            'default_count_history': 0, 'months_since_last_default': 0,
            'outstanding_to_income_pct': 0.0, 'income_affordability_score': 1.0,
            'debt_burden_category': 0, 'payment_ratio_trend': 0.0,
            'utilization_trend': 0.0, 'outstanding_growth_rate': 0.0,
            'is_deteriorating': 0, 'active_months_3m': 0, 'avg_util_when_active': 0.0,
            'account_age_bucket': 0, 'risk_score': 0.0
        }
        
        # Merge customer data with defaults
        feature_dict = defaults.copy()
        for key in customer:
            feature_dict[key] = customer[key]
        
        if all_features:
            customer_df = pd.DataFrame([feature_dict])[all_features]
        else:
            customer_df = pd.DataFrame([feature_dict])
        
        if self.full_model_loaded:
            try:
                prob = self.full_model.predict_proba(customer_df)[0, 1]
                score = self._prob_to_score(prob)
            except Exception as e:
                prob = 0.2
                score = 780
        else:
            prob = 0.2
            score = 780
        
        decision = self._score_to_decision(score)
        
        # Calculate limit based on score and income
        income = customer.get('monthly_income', 0)
        if score >= 750:
            max_limit = min(income * 1.0, self.MAX_LIMIT_TIER_3)
        elif score >= 650:
            max_limit = min(income * 0.5, self.MAX_LIMIT_TIER_3)
        elif score >= 550:
            max_limit = min(income * 0.25, 20000)
        else:
            max_limit = 0
        
        return {
            'ml_probability': prob,
            'final_score': score,
            'decision': decision,
            'max_credit_limit': int(max_limit),
            'model_used': 'full_model',
            'note': 'Full model with behavioral features provides reliable predictions.'
        }
    
    def score_customer(self, customer: Dict) -> Dict:
        """
        Main scoring function. Automatically selects appropriate model based on tier.
        
        Args:
            customer: Dictionary with customer features
            
        Returns:
            Complete scoring result with recommendation
        """
        account_age = customer.get('account_age_months', 0)
        tier, tier_desc = self.get_customer_tier(account_age)
        
        result = {
            'customer_tier': tier,
            'tier_description': tier_desc,
            'account_age_months': account_age
        }
        
        if tier == 1:
            # TIER 1: Cold Start - Use cold start model with guardrails
            scoring = self.score_cold_start(customer)
            result.update(scoring)
            result['recommendation'] = (
                "New customer - monitor closely. Consider increasing limit after "
                "3 months of good payment history."
            )
            
        elif tier == 2:
            # TIER 2: Building - Blend cold start and full model
            cold_result = self.score_cold_start(customer)
            full_result = self.score_established(customer)
            
            # Blend scores (40% cold start, 60% full model)
            blended_score = int(0.4 * cold_result['final_score'] + 0.6 * full_result['final_score'])
            blended_prob = 0.4 * cold_result['ml_probability'] + 0.6 * full_result['ml_probability']
            
            decision = self._score_to_decision(blended_score)
            income = customer.get('monthly_income', 0)
            max_limit = min(income * 0.5, self.MAX_LIMIT_TIER_2)
            
            result.update({
                'ml_probability': blended_prob,
                'final_score': blended_score,
                'decision': decision,
                'max_credit_limit': int(max_limit),
                'model_used': 'blended (40% cold_start + 60% full)',
                'recommendation': (
                    "Building credit history. Good payment behavior will unlock higher limits."
                )
            })
            
        else:
            # TIER 3: Established - Use full model
            scoring = self.score_established(customer)
            result.update(scoring)
            result['recommendation'] = "Established customer - standard credit policies apply."
        
        return result


# =============================
# DEMONSTRATION
# =============================
if __name__ == "__main__":
    print("=" * 70)
    print("COLD START HANDLER - PRODUCTION DEMO")
    print("=" * 70)
    print("\n   Loading models...")
    
    handler = ColdStartHandler()
    
    print("\n" + "=" * 70)
    print("TEST SCENARIOS")
    print("=" * 70)
    
    # Test various customer profiles
    test_cases = [
        # TIER 1: New Customers
        {
            "name": "New - High Income Professional",
            "age": 35, "account_age_months": 1, "monthly_income": 80000,
            "employment_status": "Salaried", "education_level": "Graduate",
            "city_tier": "Tier-1", "dependents": 1, "residence_type": "Owned"
        },
        {
            "name": "New - Low Income Daily Wage",
            "age": 40, "account_age_months": 0, "monthly_income": 15000,
            "employment_status": "Daily Wage", "education_level": "Primary",
            "city_tier": "Tier-3", "dependents": 4, "residence_type": "Rented"
        },
        {
            "name": "New - Fresh Graduate",
            "age": 22, "account_age_months": 0, "monthly_income": 25000,
            "employment_status": "Salaried", "education_level": "Graduate",
            "city_tier": "Tier-2", "dependents": 0, "residence_type": "Rented"
        },
        # TIER 2: Building History
        {
            "name": "Building - Good Payment History",
            "age": 30, "account_age_months": 5, "monthly_income": 50000,
            "employment_status": "Salaried", "education_level": "Graduate",
            "city_tier": "Tier-2", "dependents": 0, "residence_type": "Rented",
            "payment_ratio_avg_3m": 0.98, "missed_due_count_3m": 0
        },
        # TIER 3: Established
        {
            "name": "Established - With History",
            "age": 45, "account_age_months": 12, "monthly_income": 100000,
            "employment_status": "Business", "education_level": "Postgraduate",
            "city_tier": "Tier-1", "dependents": 2, "residence_type": "Owned",
            "payment_ratio_avg_3m": 0.90, "missed_due_count_3m": 1
        },
        {
            "name": "Established - Poor History",
            "age": 38, "account_age_months": 18, "monthly_income": 60000,
            "employment_status": "Salaried", "education_level": "Graduate",
            "city_tier": "Tier-1", "dependents": 1, "residence_type": "Rented",
            "payment_ratio_avg_3m": 0.5, "missed_due_count_3m": 3,
            "late_payments_3m": 2, "missed_due_last_1m": 1
        }
    ]
    
    print("\n   Results:")
    print("   " + "-" * 85)
    
    for case in test_cases:
        name = case.pop('name')
        result = handler.score_customer(case)
        
        print(f"\n   {name}")
        print(f"   Tier: {result['customer_tier']} ({result['tier_description']})")
        print(f"   Model: {result.get('model_used', 'N/A')}")
        print(f"   Score: {result['final_score']} | Decision: {result['decision']}")
        print(f"   Credit Limit: Rs. {result['max_credit_limit']:,}")
        
        if 'risk_warnings' in result and result['risk_warnings']:
            print(f"   Warnings: {', '.join(result['risk_warnings'])}")
    
    print("\n   " + "-" * 85)
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
