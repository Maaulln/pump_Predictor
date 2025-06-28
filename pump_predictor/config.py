"""
Configuration settings for pump maintenance prediction with environment support
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConfigValidationError(Exception):
    """Configuration validation error"""
    pass

class EnvironmentConfig:
    """Environment-specific configuration"""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.testing = os.getenv("TESTING", "false").lower() == "true"
        
        # Validate environment
        valid_envs = ["development", "staging", "production"]
        if self.environment not in valid_envs:
            raise ConfigValidationError(f"Invalid environment: {self.environment}. Must be one of {valid_envs}")
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_staging(self) -> bool:
        return self.environment == "staging"

class APIConfig:
    """API configuration"""
    
    def __init__(self):
        self.host = os.getenv("API_HOST", "localhost")
        self.port = int(os.getenv("API_PORT", "8000"))
        self.workers = int(os.getenv("API_WORKERS", "1"))
        self.reload = os.getenv("API_RELOAD", "false").lower() == "true"
        self.secret_key = os.getenv("SECRET_KEY")
        
        # Validate required settings for production
        env_config = EnvironmentConfig()
        if env_config.is_production and not self.secret_key:
            raise ConfigValidationError("SECRET_KEY is required in production")

class SecurityConfig:
    """Security configuration"""
    
    def __init__(self):
        self.api_key_admin = os.getenv("API_KEY_ADMIN")
        self.api_key_user = os.getenv("API_KEY_USER")
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))
        
        # Validate for production
        env_config = EnvironmentConfig()
        if env_config.is_production:
            if not self.api_key_admin or self.api_key_admin == "admin_key_123_change_in_production":
                raise ConfigValidationError("API_KEY_ADMIN must be changed in production")
            if not self.api_key_user or self.api_key_user == "user_key_456_change_in_production":
                raise ConfigValidationError("API_KEY_USER must be changed in production")

class LoggingConfig:
    """Logging configuration"""
    
    def __init__(self):
        self.level = os.getenv("LOG_LEVEL", "INFO")
        self.format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ConfigValidationError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")

# Initialize configuration
ENV_CONFIG = EnvironmentConfig()
API_CONFIG = APIConfig()
SECURITY_CONFIG = SecurityConfig()
LOGGING_CONFIG = LoggingConfig()
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