"""
Data preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

from pump_predictor.config import FEATURE_COLUMNS, TARGET_COLUMN
from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded data from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features"""
        df = df.copy()
        
        # Calculate rolling averages
        df['pressure_rolling_mean'] = df['pressure'].rolling(window=24, min_periods=1).mean()
        df['vibration_rolling_std'] = df['vibration'].rolling(window=24, min_periods=1).std()
        
        # Create interaction features
        df['pressure_temp_interaction'] = df['pressure'] * df['temperature']
        df['efficiency_score'] = df['speed'] / (df['temperature'] + 1)
        
        return df
        
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, ...]:
        """Prepare data for training"""
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Split features and target
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test