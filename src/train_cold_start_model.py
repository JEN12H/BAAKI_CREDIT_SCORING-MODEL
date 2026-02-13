"""
Cold Start Model Training
==========================
Train two models:
1. Cold Start Model - Uses ONLY static/demographic features (for new customers)
2. Full Model - Uses all features including behavioral (for established customers)

This addresses the banking cold start problem properly with ML models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)
import joblib
import warnings
import logging
import mlflow
import mlflow.sklearn
import yaml
from datetime import datetime
import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generation.snap import run_snap_pipeline

# Configure warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cold_start_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure MLflow
mlflow.set_experiment("Cold_Start_Credit_Model")

logger.info("=" * 70)
logger.info("COLD START MODEL TRAINING (Enhanced + MLflow)")
logger.info("=" * 70)

# =============================
# LOAD DATA
# =============================
# =============================
# LOAD CONFIGURATION
# =============================
def load_params():
    try:
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("params.yaml not found, using defaults")
        return {
            'models': {
                'logistic_regression': {'max_iter': 1000, 'class_weight': 'balanced', 'solver': 'lbfgs'},
                'random_forest': {'n_estimators': 100, 'class_weight': 'balanced', 'n_jobs': -1},
                'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
            },
            'data': {'random_state': 42}
        }

params = load_params()

# =============================
# LOAD DATA
# =============================
logger.info("[1/7] Loading data...")

# --- AUTOMATIC DATA RECALIBRATION ---
logger.info("Recalibrating features from raw data...")
success = run_snap_pipeline()
if not success:
    logger.warning("Pipeline failed, using existing data/model_snapshots.csv")
# ------------------------------------

df = pd.read_csv("data/model_snapshots.csv")
df = df.drop('customer_id', axis=1)

logger.info(f"Total samples: {df.shape[0]}")
logger.info(f"Total features: {df.shape[1]}")
logger.info(f"Default rate: {df['default_next_1m'].mean():.4f} ({df['default_next_1m'].sum()} defaults)")

# =============================
# DEFINE FEATURE SETS
# =============================
logger.info("[2/7] Defining feature sets...")

# STATIC FEATURES - Available at customer onboarding (cold start)
static_features = [
    'age',
    'employment_status', 
    'education_level',
    'monthly_income',
    'city_tier',
    'dependents', 
    'residence_type',
    'account_age_months'
]

# BEHAVIORAL FEATURES - Require transaction history
behavioral_features = [
    'util_avg_3m', 'payment_ratio_avg_3m', 'max_outstanding_3m',
    'avg_txn_amt_3m', 'avg_txn_count_3m', 'late_payments_3m',
    'missed_due_count_3m', 'missed_due_last_1m', 'payment_ratio_last_1m',
    'outstanding_delta_3m', 'bnpl_active_last_1m',
    'consecutive_missed_due', 'payment_ratio_min_3m', 'worst_util_3m',
    'ever_defaulted', 'default_count_history', 'months_since_last_default',
    'outstanding_to_income_pct', 'income_affordability_score', 'debt_burden_category',
    'payment_ratio_trend', 'utilization_trend', 'outstanding_growth_rate',
    'is_deteriorating', 'active_months_3m', 'avg_util_when_active',
    'account_age_bucket', 'risk_score', 'snapshot_month'
]

# Verify features exist
static_in_data = [f for f in static_features if f in df.columns]
behavioral_in_data = [f for f in behavioral_features if f in df.columns]

logger.info(f"Static features: {len(static_in_data)}")
logger.info(f"Behavioral features: {len(behavioral_in_data)}")
logger.info(f"Total features for full model: {len(static_in_data) + len(behavioral_in_data)}")

# =============================
# PREPARE DATA
# =============================
logger.info("[3/7] Preparing data...")

# Target variable
y = df["default_next_1m"]

# Features for cold start model (static only)
X_static = df[static_in_data].copy()

# Features for full model (all features)
all_features = static_in_data + behavioral_in_data
X_full = df[all_features].copy()

# Identify categorical and numerical columns for each feature set
def get_feature_types(X):
    categorical = [col for col in X.columns if X[col].dtype == 'object']
    numerical = [col for col in X.columns if X[col].dtype != 'object']
    return categorical, numerical

static_cat, static_num = get_feature_types(X_static)
full_cat, full_num = get_feature_types(X_full)

logger.info("Cold Start Model:")
logger.info(f"   Categorical: {static_cat}")
logger.info(f"   Numerical: {len(static_num)} columns")
logger.info("Full Model:")
logger.info(f"   Categorical: {full_cat}")
logger.info(f"   Numerical: {len(full_num)} columns")

# Train/test split (same split for both models for fair comparison)
X_static_train, X_static_test, y_train, y_test = train_test_split(
    X_static, y, test_size=0.2, random_state=42, stratify=y
)
X_full_train, X_full_test, _, _ = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

logger.info(f"Training set: {len(X_static_train)} samples")
logger.info(f"Test set: {len(X_static_test)} samples")

# =============================
# BUILD PREPROCESSING PIPELINES
# =============================
logger.info("[4/7] Building preprocessing pipelines...")

def build_preprocessor(numerical_features, categorical_features):
    """Build a preprocessing pipeline for given feature sets."""
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

# Preprocessors for each model
static_preprocessor = build_preprocessor(static_num, static_cat)
full_preprocessor = build_preprocessor(full_num, full_cat)

# =============================
# TRAIN COLD START MODEL
# =============================

# Start MLflow Run
with mlflow.start_run(run_name=f"Cold_Start_Train_{datetime.now().strftime('%Y%m%d_%H%M')}"):

    # Log global params
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("static_feature_count", len(static_in_data))
    mlflow.log_param("full_feature_count", len(all_features))

    logger.info("[5/7] Training COLD START model (static features only)...")

# Try multiple algorithms for cold start model
    cold_start_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=params['models']['logistic_regression']['max_iter'],
            random_state=params['data']['random_state'], 
            class_weight=params['models']['logistic_regression']['class_weight']
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=params['models']['random_forest']['n_estimators'],
            random_state=params['data']['random_state'],
            class_weight=params['models']['random_forest']['class_weight'],
            n_jobs=params['models']['random_forest']['n_jobs']
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=params['models']['gradient_boosting']['n_estimators'],
            random_state=params['data']['random_state'],
            learning_rate=params['models']['gradient_boosting']['learning_rate'],
            max_depth=params['models']['gradient_boosting']['max_depth']
        )
    }
    
    cold_start_results = {}
    best_cold_start_model = None
    best_cold_start_auc = 0
    best_cold_start_name = ""
    
    for name, classifier in cold_start_models.items():
        logger.info(f"Training {name}...")
    
    model = Pipeline(steps=[
        ('preprocessor', static_preprocessor),
        ('classifier', classifier)
    ])
    
    model.fit(X_static_train, y_train)
    
    y_pred = model.predict(X_static_test)
    y_pred_proba = model.predict_proba(X_static_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cold_start_results[name] = {
        'model': model,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    logger.info(f"      AUC: {auc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    if auc > best_cold_start_auc:
        best_cold_start_auc = auc
        best_cold_start_model = model
        best_cold_start_name = name

    logger.info(f"Best Cold Start Model: {best_cold_start_name} (AUC: {best_cold_start_auc:.4f})")
    
    # Log best cold start metrics
    mlflow.log_param("best_cold_start_algorithm", best_cold_start_name)
    mlflow.log_metric("cold_start_auc", best_cold_start_auc)
    mlflow.log_metric("cold_start_recall", cold_start_results[best_cold_start_name]['recall'])

    # =============================
    # TRAIN FULL MODEL
    # =============================
    logger.info("[6/7] Training FULL model (all features)...")

    full_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=params['models']['logistic_regression']['max_iter'],
            random_state=params['data']['random_state'], 
            class_weight=params['models']['logistic_regression']['class_weight']
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=params['models']['random_forest']['n_estimators'],
            random_state=params['data']['random_state'],
            class_weight=params['models']['random_forest']['class_weight'],
            n_jobs=params['models']['random_forest']['n_jobs']
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=params['models']['gradient_boosting']['n_estimators'],
            random_state=params['data']['random_state'],
            learning_rate=params['models']['gradient_boosting']['learning_rate'],
            max_depth=params['models']['gradient_boosting']['max_depth']
        )
    }
    
    full_results = {}
    best_full_model = None
    best_full_auc = 0
    best_full_name = ""
    
    for name, classifier in full_models.items():
        logger.info(f"Training {name}...")
    
    model = Pipeline(steps=[
        ('preprocessor', full_preprocessor),
        ('classifier', classifier)
    ])
    
    model.fit(X_full_train, y_train)
    
    y_pred = model.predict(X_full_test)
    y_pred_proba = model.predict_proba(X_full_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    full_results[name] = {
        'model': model,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    logger.info(f"      AUC: {auc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    if auc > best_full_auc:
        best_full_auc = auc
        best_full_model = model
        best_full_name = name

    logger.info(f"Best Full Model: {best_full_name} (AUC: {best_full_auc:.4f})")
    
    # Log best full model metrics
    mlflow.log_param("best_full_model_algorithm", best_full_name)
    mlflow.log_metric("full_model_auc", best_full_auc)
    mlflow.log_metric("full_model_recall", full_results[best_full_name]['recall'])

    # =============================
    # SAVE MODELS
    # =============================
    logger.info("[7/7] Saving models and results...")

    # Save cold start model
    joblib.dump(best_cold_start_model, 'models/cold_start_model.pkl')
    logger.info(f"Saved: models/cold_start_model.pkl ({best_cold_start_name})")
    mlflow.sklearn.log_model(best_cold_start_model, "cold_start_model")
    
    # Save full model
    joblib.dump(best_full_model, 'models/credit_score_model.pkl')
    logger.info(f"Saved: models/credit_score_model.pkl ({best_full_name})")
    mlflow.sklearn.log_model(best_full_model, "full_model")

    # Save feature lists for reference
    feature_config = {
        'static_features': static_in_data,
        'behavioral_features': behavioral_in_data,
        'all_features': all_features,
        'static_categorical': static_cat,
        'static_numerical': static_num,
        'full_categorical': full_cat,
        'full_numerical': full_num
    }
    joblib.dump(feature_config, 'models/feature_config.pkl')
    logger.info("Saved: models/feature_config.pkl")
    mlflow.log_artifact("models/feature_config.pkl")

    # =============================
    # DETAILED RESULTS
    # =============================
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 70)

    logger.info("COLD START MODEL (Static Features Only)")
    logger.info(f"Best Algorithm: {best_cold_start_name}")
    logger.info(f"Features Used: {len(static_in_data)} (demographics only)")
    
    res = cold_start_results[best_cold_start_name]
    logger.info(f"Accuracy:  {res['accuracy']:.4f}")
    logger.info(f"Precision: {res['precision']:.4f}")
    logger.info(f"Recall:    {res['recall']:.4f}")
    logger.info(f"F1 Score:  {res['f1']:.4f}")
    logger.info(f"ROC-AUC:   {res['auc']:.4f}")
    
    logger.info("Classification Report:\n" + classification_report(y_test, res['y_pred'], target_names=['Non-Default', 'Default']))

    cm = confusion_matrix(y_test, res['y_pred'])
    logger.info(f"Confusion Matrix:\n{cm}")

    logger.info("-" * 70)
    logger.info("FULL MODEL (All Features)")
    logger.info(f"Best Algorithm: {best_full_name}")
    logger.info(f"Features Used: {len(all_features)} (demographics + behavioral)")
    
    res = full_results[best_full_name]
    logger.info(f"Accuracy:  {res['accuracy']:.4f}")
    logger.info(f"Precision: {res['precision']:.4f}")
    logger.info(f"Recall:    {res['recall']:.4f}")
    logger.info(f"F1 Score:  {res['f1']:.4f}")
    logger.info(f"ROC-AUC:   {res['auc']:.4f}")
    
    logger.info("Classification Report:\n" + classification_report(y_test, res['y_pred'], target_names=['Non-Default', 'Default']))

    cm = confusion_matrix(y_test, res['y_pred'])
    logger.info(f"Confusion Matrix:\n{cm}")

    # =============================
    # MODEL COMPARISON TABLE
    # =============================
    logger.info("=" * 70)
    logger.info("SUMMARY - MODEL COMPARISON")
    logger.info("=" * 70)

    logger.info("-" * 65)
    logger.info(f"{'Model':<25} {'Type':<15} {'AUC':<10} {'Recall':<10} {'F1':<10}")
    logger.info("-" * 65)

    # Cold start models
    for name, res in cold_start_results.items():
        logger.info(f"{name:<25} {'Cold Start':<15} {res['auc']:<10.4f} {res['recall']:<10.4f} {res['f1']:<10.4f}")

    logger.info("-" * 65)

    # Full models
    for name, res in full_results.items():
        logger.info(f"{name:<25} {'Full':<15} {res['auc']:<10.4f} {res['recall']:<10.4f} {res['f1']:<10.4f}")

    logger.info("-" * 65)

    # =============================
    # GENERATE ROC CURVES
    # =============================
    logger.info("Generating ROC curves...")
    
    plt.figure(figsize=(12, 5))
    
    # Cold Start Models
    plt.subplot(1, 2, 1)
    for name, res in cold_start_results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={res["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cold Start Model - ROC Curves\n(Static Features Only)')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Full Models
    plt.subplot(1, 2, 2)
    for name, res in full_results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={res["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Full Model - ROC Curves\n(All Features)')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved: model_comparison_roc.png")
    mlflow.log_artifact("model_comparison_roc.png")

    logger.info("TRAINING COMPLETE! Models and artifacts logged to MLflow.")

# =============================
# TEST COLD START SCENARIOS
# =============================
print("\n" + "=" * 70)
print("COLD START MODEL - TEST SCENARIOS")
print("=" * 70)

def prob_to_score(p):
    """Convert probability to credit score (300-900)"""
    return int(max(300, min(900, 900 - (p * 600))))

def get_decision(score):
    """Get lending decision based on score"""
    if score >= 750: return "Approve"
    elif score >= 650: return "Approve_Low_Limit"
    elif score >= 550: return "Conditional"
    else: return "Reject"

# Test cases
test_customers = [
    {"name": "High Income Salaried", "age": 35, "employment_status": "Salaried", 
     "education_level": "Graduate", "monthly_income": 80000, "city_tier": "Tier-1",
     "dependents": 1, "residence_type": "Owned", "account_age_months": 1},
    
    {"name": "Very High Income Executive", "age": 45, "employment_status": "Salaried",
     "education_level": "Postgraduate", "monthly_income": 150000, "city_tier": "Tier-1",
     "dependents": 2, "residence_type": "Owned", "account_age_months": 0},
    
    {"name": "Middle Income Self-Employed", "age": 38, "employment_status": "Self-Employed",
     "education_level": "Graduate", "monthly_income": 45000, "city_tier": "Tier-2",
     "dependents": 2, "residence_type": "Rented", "account_age_months": 2},
    
    {"name": "Low Income Daily Wage", "age": 32, "employment_status": "Daily Wage",
     "education_level": "Secondary", "monthly_income": 15000, "city_tier": "Tier-3",
     "dependents": 3, "residence_type": "Rented", "account_age_months": 0},
    
    {"name": "Fresh Graduate", "age": 22, "employment_status": "Salaried",
     "education_level": "Graduate", "monthly_income": 25000, "city_tier": "Tier-2",
     "dependents": 0, "residence_type": "Rented", "account_age_months": 0},
    
    {"name": "No Education Worker", "age": 40, "employment_status": "Daily Wage",
     "education_level": "No Formal Education", "monthly_income": 12000, "city_tier": "Tier-3",
     "dependents": 4, "residence_type": "Rented", "account_age_months": 1},
]

print("\n   Predictions using COLD START MODEL:")
print("   " + "-" * 75)
print(f"   {'Customer Profile':<35} {'Prob':<8} {'Score':<8} {'Decision':<15}")
print("   " + "-" * 75)

for cust in test_customers:
    name = cust.pop('name')
    cust_df = pd.DataFrame([cust])
    
    prob = best_cold_start_model.predict_proba(cust_df)[0, 1]
    score = prob_to_score(prob)
    decision = get_decision(score)
    
    print(f"   {name:<35} {prob:.4f}   {score:<8} {decision:<15}")

print("   " + "-" * 75)

# =============================
# FINAL SUMMARY
# =============================
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)

print(f"""
   Files Generated:
   ────────────────
   1. cold_start_model.pkl      - For NEW customers (0-3 months)
   2. credit_score_model.pkl    - For ESTABLISHED customers (6+ months)
   3. feature_config.pkl        - Feature lists for each model
   4. model_comparison_roc.png  - ROC curve comparison

   Model Performance Summary:
   ──────────────────────────
   Cold Start Model ({best_cold_start_name}):
      - Uses {len(static_in_data)} static/demographic features
      - AUC: {best_cold_start_auc:.4f}
      - Recall: {cold_start_results[best_cold_start_name]['recall']:.4f}

   Full Model ({best_full_name}):
      - Uses {len(all_features)} features (static + behavioral)
      - AUC: {best_full_auc:.4f}
      - Recall: {full_results[best_full_name]['recall']:.4f}

   Production Usage:
   ─────────────────
   from cold_start_handler import ColdStartHandler
   handler = ColdStartHandler()
   result = handler.score_customer(customer_data)
""")
