"""
Configuration settings for pump maintenance prediction
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data configuration
FEATURE_COLUMNS = [
    'temperature',
    'pressure', 
    'vibration',
    'flow_rate',
    'motor_current',
    'bearing_temperature',
    'oil_level',
    'power_consumption',
    'efficiency',
    'operating_hours',
    'load_factor',
    'ambient_temperature',
    'humidity'
]

TARGET_COLUMN = 'needs_maintenance'


# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models")
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Model configurations
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    },
    'lightgbm': {
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 5,
        'min_split_gain': 0.01,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }
}

# Hyperparameter tuning configurations
TUNING_CONFIG = {
    'random_forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
}

# API configuration
API_CONFIG = {
    'host': os.getenv("API_HOST", "0.0.0.0"),
    'port': int(os.getenv("API_PORT", 8000)),
    'reload': True,
    'log_level': os.getenv("LOG_LEVEL", "info").lower()
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    'port': int(os.getenv("STREAMLIT_PORT", 8501)),
    'server_headless': True,
    'browser_gather_usage_stats': False
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'scale_features': True,
    'handle_missing': True,
    'feature_selection': False
}

# Logging configuration
LOGGING_CONFIG = {
    'level': os.getenv("LOG_LEVEL", "INFO"),
    'format': "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    'rotation': "10 MB",
    'retention': "1 week"
}

MAINTENANCE_THRESHOLDS = {
    'temperature': 80.0,      # °C
    'pressure': 120.0,        # bar (minimum)
    'vibration': 4.0,         # Hz
    'flow_rate': 220.0,       # L/min (minimum)
    'motor_current': 20.0,    # A
    'bearing_temperature': 85.0,  # °C
    'oil_level': 60.0,        # % (minimum)
    'efficiency': 0.75        # minimum efficiency
}


@dataclass
class FeatureConfig:
    """Feature configuration for the model"""
    numerical_features: list
    categorical_features: list
    target_column: str
    feature_names: list

# Default feature configuration (adjust based on your data)
DEFAULT_FEATURES = FeatureConfig(
    numerical_features=['temperature', 'pressure', 'vibration', 'flow_rate'],
    categorical_features=[],
    target_column='maintenance_needed',
    feature_names=['temperature', 'pressure', 'vibration', 'flow_rate']
)