from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os
import uvicorn

from .cold_start_handler import ColdStartHandler

# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="API for predicting credit default risk using Cold Start (new users) and Full (existing users) models with intelligent routing.",
    version="2.0.0"
)

# Initialize Handler
try:
    handler = ColdStartHandler(
        cold_start_model_path="models/cold_start_model.pkl",
        full_model_path="models/credit_score_model.pkl",
        feature_config_path="models/feature_config.pkl"
    )
    print("✅ ColdStartHandler initialized successfully")
except Exception as e:
    print(f"❌ Error initializing handler: {e}")
    handler = None

# Helper Functions (moved to Handler)
# ... removed ...

# Pydantic Schemas
class ColdStartInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    employment_status: str = Field(..., description="Employment type (Salaried, Self-Employed, etc.)")
    education_level: str = Field(..., description="Highest education level")
    monthly_income: float = Field(..., ge=0, description="Monthly income")
    city_tier: str = Field(..., description="City tier classification (Tier-1, Tier-2, Tier-3)")
    dependents: int = Field(..., ge=0, description="Number of dependents")
    residence_type: str = Field(..., description="Residence type (Owned, Rented)")
    account_age_months: int = Field(..., ge=0, description="Age of account in months (0 for new)")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "employment_status": "Salaried",
                "education_level": "Graduate",
                "monthly_income": 50000.0,
                "city_tier": "Tier-1",
                "dependents": 1,
                "residence_type": "Rented",
                "account_age_months": 0
            }
        }

class FullModelInput(ColdStartInput):
    # Behavioral Features
    util_avg_3m: float = Field(..., description="Average utilization over last 3 months")
    payment_ratio_avg_3m: float = Field(..., description="Average payment ratio over last 3 months")
    max_outstanding_3m: float = Field(..., description="Maximum outstanding balance over last 3 months")
    avg_txn_amt_3m: float = Field(..., description="Average transaction amount over last 3 months")
    avg_txn_count_3m: float = Field(..., description="Average transaction count over last 3 months")
    late_payments_3m: int = Field(..., ge=0, description="Number of late payments in last 3 months")
    missed_due_count_3m: int = Field(..., ge=0, description="Number of missed due dates in last 3 months")
    missed_due_last_1m: int = Field(..., ge=0, le=1, description="Missed due date in last 1 month (0 or 1)")
    payment_ratio_last_1m: float = Field(..., description="Payment ratio in last 1 month")
    outstanding_delta_3m: float = Field(..., description="Change in outstanding balance over 3 months")
    bnpl_active_last_1m: int = Field(..., ge=0, le=1, description="BNPL active in last 1 month (0 or 1)")
    
    # Derived Signals
    consecutive_missed_due: int = Field(..., ge=0, description="Consecutive missed due dates")
    payment_ratio_min_3m: float = Field(..., description="Minimum payment ratio in last 3 months")
    worst_util_3m: float = Field(..., description="Maximum utilization in last 3 months")
    ever_defaulted: int = Field(..., ge=0, le=1, description="Has ever defaulted (0 or 1)")
    default_count_history: int = Field(..., ge=0, description="Total history of defaults")
    months_since_last_default: int = Field(..., ge=0, description="Months since last default")
    outstanding_to_income_pct: float = Field(..., description="Outstanding balance as % of income")
    income_affordability_score: float = Field(..., description="Income to debt ratio score")
    debt_burden_category: int = Field(..., ge=0, le=3, description="Debt burden category (0-3)")
    payment_ratio_trend: float = Field(..., description="Trend in payment ratio")
    utilization_trend: float = Field(..., description="Trend in utilization")
    outstanding_growth_rate: float = Field(..., description="Rate of outstanding balance growth")
    is_deteriorating: int = Field(..., ge=0, le=1, description="Is credit behavior deteriorating (0 or 1)")
    active_months_3m: int = Field(..., ge=0, le=3, description="Months active in last 3 months")
    avg_util_when_active: float = Field(..., description="Average utilization when active")
    account_age_bucket: int = Field(..., ge=0, le=2, description="Account age bucket (0-2)")
    risk_score: float = Field(..., ge=0, le=100, description="Internal risk score (0-100)")
    snapshot_month: int = Field(..., description="Month of snapshot")

    class Config:
        schema_extra = {
            "example": {
                # Inherited static fields
                "age": 35,
                "employment_status": "Salaried",
                "education_level": "Graduate",
                "monthly_income": 80000.0,
                "city_tier": "Tier-1",
                "dependents": 2,
                "residence_type": "Owned",
                "account_age_months": 24,
                
                # Behavioral fields
                "util_avg_3m": 0.35,
                "payment_ratio_avg_3m": 0.95,
                "max_outstanding_3m": 15000.0,
                "avg_txn_amt_3m": 2500.0,
                "avg_txn_count_3m": 5.0,
                "late_payments_3m": 0,
                "missed_due_count_3m": 0,
                "missed_due_last_1m": 0,
                "payment_ratio_last_1m": 1.0,
                "outstanding_delta_3m": 200.0,
                "bnpl_active_last_1m": 1,
                "consecutive_missed_due": 0,
                "payment_ratio_min_3m": 0.9,
                "worst_util_3m": 0.4,
                "ever_defaulted": 0,
                "default_count_history": 0,
                "months_since_last_default": 0,
                "outstanding_to_income_pct": 18.5,
                "income_affordability_score": 5.3,
                "debt_burden_category": 1,
                "payment_ratio_trend": 0.05,
                "utilization_trend": -0.02,
                "outstanding_growth_rate": 0.01,
                "is_deteriorating": 0,
                "active_months_3m": 3,
                "avg_util_when_active": 0.35,
                "account_age_bucket": 1,
                "risk_score": 15.0,
                "snapshot_month": 6
            }
        }

# API Endpoints
@app.get("/")
def root():
    status = {
        "status": "online",
        "handler_initialized": handler is not None,
        "models": {
            "cold_start": "Loaded" if handler and handler.cold_start_loaded else "Not Loaded",
            "full_model": "Loaded" if handler and handler.full_model_loaded else "Not Loaded"
        }
    }
    return status

@app.get("/health")
def health_check():
    if not handler or not handler.cold_start_loaded or not handler.full_model_loaded:
        raise HTTPException(status_code=503, detail="Models not fully loaded")
    return {"status": "healthy"}

@app.post("/predict/cold-start")
def predict_cold_start(data: ColdStartInput):
    if not handler:
        raise HTTPException(status_code=503, detail="Scoring handler not initialized")
    
    try:
        # Use handler's logic which includes guardrails
        result = handler.score_cold_start(data.dict())
        
        return {
            "model_type": "cold_start_guarded",
            "default_probability": round(result['ml_probability'], 4),
            "credit_score": result['final_score'],
            "decision": result['decision'],
            "max_credit_limit": result['max_credit_limit'],
            "risk_warnings": result.get('risk_warnings', []),
            "recommendation": result.get('note', '')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/full")
def predict_full_model(data: FullModelInput):
    if not handler:
        raise HTTPException(status_code=503, detail="Scoring handler not initialized")
        
    try:
        # Use handler's logic
        result = handler.score_established(data.dict())
        
        return {
            "model_type": "full_model",
            "default_probability": round(result['ml_probability'], 4),
            "credit_score": result['final_score'],
            "decision": result['decision'],
            "max_credit_limit": result['max_credit_limit'],
            "risk_factors": {
                "utilization": data.util_avg_3m,
                "missed_payments": data.missed_due_count_3m,
                "risk_score": data.risk_score
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
