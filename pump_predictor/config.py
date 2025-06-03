"""
Configuration settings for the pump predictor
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Model parameters
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

# Feature configuration
FEATURE_COLUMNS = [
    'pressure',
    'temperature',
    'speed',
    'vibration',
    'oil_level',
    'runtime_hours'
]

TARGET_COLUMN = 'needs_maintenance'

# Training parameters
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42