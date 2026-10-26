"""
================================================================================
AI-POWERED FINANCIAL FRAUD DETECTION SYSTEM (PROTOTYPE)
Central Configuration & Hyperparameter Registry
================================================================================
"""

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.cache', 'matplotlib'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SRC_DIR = os.path.join(BASE_DIR, 'src')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
TESTS_DIR = os.path.join(BASE_DIR, 'tests')
WORKSPACE_DB_PATH = os.path.join(BASE_DIR, 'storage', 'workspaces.sqlite3')
DB_PATH = WORKSPACE_DB_PATH

# Ensure all structural directories exist
for folder in [DATA_DIR, SRC_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, DOCS_DIR, TESTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Streaming & Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TRANSACTIONS_TOPIC = os.environ.get('KAFKA_TRANSACTIONS_TOPIC', 'financial.transactions')
KAFKA_ALERTS_TOPIC = os.environ.get('KAFKA_ALERTS_TOPIC', 'financial.fraud-alerts')
KAFKA_DLQ_TOPIC = os.environ.get('KAFKA_DLQ_TOPIC', 'financial.transactions.dlq')
KAFKA_CONSUMER_GROUP = os.environ.get('KAFKA_CONSUMER_GROUP', 'fraudguard-inference-group')

# Experiment Reproducibility
RANDOM_STATE = 42

# Dataset & Feature Specifications
DATASET_PATH = os.path.join(DATA_DIR, 'transactions.csv')
SAMPLE_TEST_PATH = os.path.join(DATA_DIR, 'sample_test.csv')
LOG_FILE_PATH = os.path.join(LOGS_DIR, 'pipeline.log')

TARGET_COLUMN = 'isFraud'
TEST_SPLIT_SIZE = 0.20
VALIDATION_SPLIT_SIZE = 0.20
TRANSACTION_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

RAW_FEATURE_COLUMNS = [
    'step',
    'type',
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest'
]

# Model Artifact Paths
RF_MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.joblib')
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, 'feature_names.joblib')
MODEL_FILES = {
    'Random Forest': 'random_forest_model.joblib',
    'XGBoost': 'xgboost_model.joblib',
    'Deep Neural Network': 'deep_neural_network.joblib',
    'Deep Autoencoder': 'deep_autoencoder.joblib',
}
MODEL_FAMILIES = {
    'Random Forest': 'Machine Learning', 'XGBoost': 'Machine Learning',
    'Deep Neural Network': 'Deep Learning', 'Deep Autoencoder': 'Deep Learning',
}
METADATA_PATH = os.path.join(RESULTS_DIR, 'experiment_metadata.json')

# Result Artifact Paths
COMPARISON_CSV_PATH = os.path.join(RESULTS_DIR, 'model_comparison.csv')
COMPARISON_CHART_PATH = os.path.join(RESULTS_DIR, 'model_comparison_chart.png')
CONFUSION_MATRICES_PATH = os.path.join(RESULTS_DIR, 'confusion_matrices.png')
ROC_CURVES_PATH = os.path.join(RESULTS_DIR, 'roc_curves.png')
PR_CURVES_PATH = os.path.join(RESULTS_DIR, 'precision_recall_curves.png')
SMOTE_PLOT_PATH = os.path.join(RESULTS_DIR, 'class_distribution_before_after_smote.png')
SHAP_SUMMARY_PATH = os.path.join(RESULTS_DIR, 'shap_summary_plot.png')
SHAP_IMPORTANCE_PATH = os.path.join(RESULTS_DIR, 'shap_feature_importance.png')

# Multi-Tier Fraud Risk Assessment Engine (Advisory Compliance Thresholds)
RISK_THRESHOLDS = {
    'LOW': 0.30,
    'MEDIUM': 0.60,
    'HIGH': 0.80,
    'CRITICAL': 1.00
}

RISK_ACTIONS = {
    'LOW': 'Routine review recommended. This application does not clear or execute transactions.',
    'MEDIUM': 'Moderate Risk: Recommend Step-up 2FA / OTP secondary user re-authentication.',
    'HIGH': 'Elevated Risk: Recommend temporary transaction hold and secondary verification review.',
    'CRITICAL': 'Urgent manual fraud review recommended; consider a temporary transaction hold.'
}

# API Gateway Security & Limits: Require environment-provided secrets in deployment
API_KEY_ENABLED = os.environ.get('FRAUDGUARD_API_KEY_ENABLED', 'True').lower() in ('true', '1')
ENV_API_KEY = os.environ.get('FRAUDGUARD_API_KEY')
_raw_keys = os.environ.get('FRAUDGUARD_API_KEYS', '')
VALID_API_KEYS = {k.strip() for k in _raw_keys.split(',') if k.strip()}
if ENV_API_KEY:
    VALID_API_KEYS.add(ENV_API_KEY.strip())

# Test secret strictly used by automated test harnesses via explicit test injection
TEST_API_KEY = 'test-analyst-secret-offline'

MAX_BATCH_SIZE = int(os.environ.get('FRAUDGUARD_MAX_BATCH_SIZE', '5000'))
RATE_LIMIT_PER_MINUTE = int(os.environ.get('FRAUDGUARD_RATE_LIMIT', '120'))
