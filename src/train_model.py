"""
Model Training Script (Enhanced)
=================================
Retrain the credit score prediction model using the new enhanced feature set
from snap.py with improved handling for class imbalance.

Features include:
- Risk signals (consecutive missed payments, worst utilization, historical defaults)
- Affordability metrics (outstanding to income percentage, income affordability score)
- Behavioral trends (payment ratio trend, utilization trend)
- Engagement & stability metrics
- Composite risk score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Configure warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure MLflow
mlflow.set_experiment("Credit_Score_Production_Model")

logger.info("=" * 60)
logger.info("CREDIT SCORE MODEL TRAINING (Enhanced + MLflow)")
logger.info("=" * 60)

# =============================
# LOAD CONFIGURATION
# =============================
def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

params = load_params()

# =============================
# LOAD DATA
# =============================
logger.info("Loading data from data/model_snapshots.csv...")

try:
    df = pd.read_csv("data/model_snapshots.csv")
    df = df.drop('customer_id', axis=1)
    
    logger.info(f"Rows: {df.shape[0]}")
    logger.info(f"Columns: {df.shape[1]}")
    logger.info(f"Default rate: {df['default_next_1m'].mean():.4f}")

except FileNotFoundError:
    logger.error("model_snapshots.csv not found! Please ensure data is generated.")
    raise

# =============================
# FEATURE ANALYSIS
# =============================
logger.info("Analyzing features...")

# Identify categorical and numerical columns
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != 'default_next_1m']

logger.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
logger.info(f"Numerical features ({len(numerical_cols)}): {len(numerical_cols)} columns")

# =============================
# PREPARE FEATURES AND TARGET
# =============================
logger.info("Preparing features and target...")
X = df.drop("default_next_1m", axis=1)
y = df["default_next_1m"]

# =============================
# TRAIN/TEST SPLIT
# =============================
logger.info("Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logger.info(f"Training set: {X_train.shape[0]} samples")
logger.info(f"Test set: {X_test.shape[0]} samples")
logger.info(f"Train default rate: {y_train.mean():.4f}")
logger.info(f"Test default rate: {y_test.mean():.4f}")

# =============================
# BUILD PREPROCESSING PIPELINE
# =============================
print("\n🏗️ Building preprocessing pipeline...")

# Preprocessing for numerical features
numerical_features = [col for col in X.columns if X[col].dtype != 'object']
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Start MLflow Run
with mlflow.start_run(run_name=f"Full_Model_Train_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    
    # Log run params
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("feature_count", X.shape[1])
    
    # =============================
    # TRAIN MULTIPLE MODELS
    # =============================
    logger.info("Training models with cross-validation...")

    models = {
        'Logistic Regression (Balanced)': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                max_iter=params['models']['logistic_regression']['max_iter'],
                random_state=params['data']['random_state'], 
                class_weight=params['models']['logistic_regression']['class_weight'],
                solver=params['models']['logistic_regression']['solver']
            ))
        ]),
        'Random Forest (Balanced)': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=params['models']['random_forest']['n_estimators'],
                random_state=params['data']['random_state'],
                class_weight=params['models']['random_forest']['class_weight'],
                n_jobs=params['models']['random_forest']['n_jobs']
            ))
        ]),
        'Gradient Boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=params['models']['gradient_boosting']['n_estimators'],
                random_state=params['data']['random_state'],
                learning_rate=params['models']['gradient_boosting']['learning_rate'],
                max_depth=params['models']['gradient_boosting']['max_depth']
            ))
        ])
    }
    
    results = {}
    best_model = None
    best_auc = 0

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"   Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
        
        # Log metrics for each candidate model (optional, usually enable for hyperparameter tuning)
        # mlflow.log_metric(f"{name}_auc", roc_auc)
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model = name

    logger.info(f"🏆 Best Model: {best_model} (AUC: {best_auc:.4f})")
    
    # Log best model parameters
    mlflow.log_param("best_model_algorithm", best_model)
    mlflow.log_metric("best_auc", best_auc)

    # =============================
    # DETAILED EVALUATION OF BEST MODEL
    # =============================
    logger.info(f"Detailed Evaluation of {best_model}...")

    model = results[best_model]['model']
    y_pred = results[best_model]['y_pred']
    y_pred_proba = results[best_model]['y_pred_proba']

    # Log specific metrics for the best model run
    mlflow.log_metric("accuracy", results[best_model]['accuracy'])
    mlflow.log_metric("precision", results[best_model]['precision'])
    mlflow.log_metric("recall", results[best_model]['recall'])
    mlflow.log_metric("f1_score", results[best_model]['f1'])
    mlflow.log_metric("roc_auc", results[best_model]['roc_auc'])

    logger.info("Model Performance Metrics:")
    logger.info(f"Accuracy:  {results[best_model]['accuracy']:.4f}")
    logger.info(f"Precision: {results[best_model]['precision']:.4f}")
    logger.info(f"Recall:    {results[best_model]['recall']:.4f}")
    logger.info(f"F1 Score:  {results[best_model]['f1']:.4f}")
    logger.info(f"ROC-AUC:   {results[best_model]['roc_auc']:.4f}")

    logger.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # =============================
    # FEATURE IMPORTANCE
    # =============================
    logger.info("Analyzing feature importance...")

    # Get feature names after preprocessing
    feature_names = numerical_features.copy()
    if categorical_features:
        encoder = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = encoder.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_feature_names)

    # Get feature importance based on model type
    classifier = model.named_steps['classifier']
    if hasattr(classifier, 'coef_'):
        # Logistic Regression
        importance = np.abs(classifier.coef_[0])
        importance_type = 'Absolute Coefficients'
    elif hasattr(classifier, 'feature_importances_'):
        # Tree-based models
        importance = classifier.feature_importances_
        importance_type = 'Feature Importance'
    else:
        importance = np.zeros(len(feature_names))
        importance_type = 'N/A'

    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    logger.info(f"Top 15 Most Important Features ({importance_type}):")
    for i, row in feature_importance.head(15).iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")

    # =============================
    # SAVE BEST MODEL & ARTIFACTS
    # =============================
    logger.info("Saving best model and artifacts...")
    
    # Save Model
    joblib.dump(model, 'models/credit_score_model.pkl')
    logger.info("Model saved to models/credit_score_model.pkl")
    
    # Log Model to MLflow
    mlflow.sklearn.log_model(model, "model")
    logger.info("Model logged to MLflow")

# =============================
# CREDIT SCORE UTILITIES
# =============================
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

    # =============================
    # TEST ON SAMPLE DATA
    # =============================
    logger.info("Testing on sample predictions...")

    # Get sample predictions from test set
    sample_indices = np.random.choice(len(X_test), size=10, replace=False)
    X_sample = X_test.iloc[sample_indices]
    y_sample_actual = y_test.iloc[sample_indices]
    proba_sample = model.predict_proba(X_sample)[:, 1]

    # Pretty print sample
    # (Since this is interactive check, we can keep print for immediate visibility or use logger)
    logger.info(f"{'Actual':<8} {'Prob':<8} {'Score':<8} {'Decision':<18}")
    for i, (actual, prob) in enumerate(zip(y_sample_actual, proba_sample)):
        score = prob_to_credit_score(prob)
        decision = final_decision(score)
        logger.info(f"{actual:<8} {prob:.4f}   {score:<8} {decision:<18}")

    # =============================
    # GENERATE ROC CURVES FOR ALL MODELS
    # =============================
    logger.info("Generating ROC curves...")
    plt.figure(figsize=(10, 6))

    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {result["roc_auc"]:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves - Model Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("ROC curves saved to roc_curve.png")
    mlflow.log_artifact("roc_curve.png")

    # =============================
    # SAVE FEATURE IMPORTANCE
    # =============================
    feature_importance.to_csv('feature_importance.csv', index=False)
    logger.info("Feature importance saved to feature_importance.csv")
    mlflow.log_artifact("feature_importance.csv")

    # =============================
    # SUMMARY
    # =============================
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Files Generated: credit_score_model.pkl, roc_curve.png, feature_importance.csv, training.log")
    
    logger.info(f"Best Model Performance ({best_model}):")
    logger.info(f"Accuracy:  {results[best_model]['accuracy']:.4f}")
    logger.info(f"Precision: {results[best_model]['precision']:.4f}")
    logger.info(f"Recall:    {results[best_model]['recall']:.4f}")
    logger.info(f"F1 Score:  {results[best_model]['f1']:.4f}")
    logger.info(f"ROC-AUC:   {results[best_model]['roc_auc']:.4f}")
