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
        self.scaler = None
        self.feature_names = None
        self.target_column = None
        self.fitted = False
        
    def load_data(self, data_source) -> pd.DataFrame:
        """Load data from CSV file or DataFrame"""
        try:
            if isinstance(data_source, pd.DataFrame):
                logger.info("Data loaded from DataFrame")
                return data_source.copy()
            elif isinstance(data_source, str):
                df = pd.read_csv(data_source)
                logger.info(f"Successfully loaded data from {data_source}")
                return df
            else:
                raise ValueError("data_source must be a file path string or pandas DataFrame")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features"""
        df = df.copy()
        
        # Calculate rolling averages if columns exist
        if 'pressure' in df.columns:
            df['pressure_rolling_mean'] = df['pressure'].rolling(window=24, min_periods=1).mean()
        if 'vibration' in df.columns:
            df['vibration_rolling_std'] = df['vibration'].rolling(window=24, min_periods=1).std()
        
        # Create interaction features if columns exist
        if 'pressure' in df.columns and 'temperature' in df.columns:
            df['pressure_temp_interaction'] = df['pressure'] * df['temperature']
        if 'speed' in df.columns and 'temperature' in df.columns:
            df['efficiency_score'] = df['speed'] / (df['temperature'] + 1)
        
        # Add test-expected features
        if 'temperature' in df.columns and 'pressure' in df.columns:
            df['temp_pressure_ratio'] = df['temperature'] / (df['pressure'] + 1)
        if 'vibration' in df.columns and 'flow_rate' in df.columns:
            df['vibration_flow_ratio'] = df['vibration'] / (df['flow_rate'] + 1)
        
        return df
        
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, target_column: str = None) -> Tuple[np.ndarray, ...]:
        """Prepare data for training"""
        if df.empty:
            raise ValueError("Data is empty")
            
        # Use provided target column or default
        target_col = target_column or TARGET_COLUMN
        
        # Handle missing values only for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Split features and target - use only numeric columns for features
        # Exclude non-numeric columns like timestamp, pump_id
        all_numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in all_numeric_cols if col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Store feature names for later use
        self.feature_names = list(X.columns)
        self.target_column = target_col
        
        # Scale features
        X_scaled = self.scale_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values in the dataframe"""
        numeric_cols = df.select_dtypes(include=[np.number])
        non_numeric_cols = df.select_dtypes(exclude=[np.number])
        
        if strategy == 'mean':
            df_filled = df.copy()
            df_filled[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())
            return df_filled
        elif strategy == 'median':
            df_filled = df.copy()
            df_filled[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.median())
            return df_filled
        elif strategy == 'drop':
            return df.dropna()
        else:
            logger.warning(f"Unknown missing value strategy: {strategy}. Using mean.")
            df_filled = df.copy()
            df_filled[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())
            return df_filled

    def detect_outliers(self, df: pd.DataFrame, z_thresh: float = 3.0) -> pd.Series:
        """Detect outliers using z-score method"""
        try:
            from scipy.stats import zscore
        except ImportError:
            # Fallback to manual z-score calculation
            def zscore(x):
                if x.std() == 0:
                    return pd.Series(np.zeros(len(x)), index=x.index)
                return (x - x.mean()) / x.std()
        
        numeric_cols = df.select_dtypes(include=[np.number])
        if numeric_cols.empty:
            return pd.Series(np.zeros(len(df)), index=df.index, dtype=bool)
            
        z_scores = numeric_cols.apply(zscore)
        outliers = (z_scores.abs() > z_thresh).any(axis=1)
        return outliers

    def remove_outliers(self, df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
        """Remove outliers using z-score method"""
        try:
            from scipy.stats import zscore
        except ImportError:
            # Fallback to manual z-score calculation
            def zscore(x):
                if x.std() == 0:
                    return pd.Series(np.zeros(len(x)), index=x.index)
                return (x - x.mean()) / x.std()
        
        numeric_cols = df.select_dtypes(include=[np.number])
        if numeric_cols.empty:
            return df.copy()
            
        z_scores = numeric_cols.apply(zscore)
        filtered_df = df[(z_scores.abs() <= z_thresh).all(axis=1)]
        return filtered_df

    def split_features_target(self, df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataframe into features and target"""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler"""
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        if not self.fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        # Return as DataFrame with original column names
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler"""
        if not self.fitted or self.scaler is None:
            raise ValueError("Scaler not fitted yet")
        
        # Use stored feature names if available, otherwise use all columns
        if self.feature_names:
            X = df[self.feature_names]
        else:
            X = df
            
        return self.scaler.transform(X)

    def get_feature_names(self) -> Optional[list]:
        """Get feature names"""
        return self.feature_names

    def create_synthetic_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Create synthetic data for testing"""
        np.random.seed(42)
        
        data = {
            'temperature': np.random.uniform(40, 120, n_samples),
            'pressure': np.random.uniform(100, 300, n_samples),
            'vibration': np.random.uniform(0, 10, n_samples),
            'flow_rate': np.random.uniform(150, 400, n_samples),
            'maintenance_needed': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        return pd.DataFrame(data)

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the data"""
        summary = {
            'shape': df.shape,
            'missing_values': {
                'total': df.isnull().sum().sum(),
                'by_column': df.isnull().sum().to_dict()
            },
            'data_types': df.dtypes.to_dict(),
            'numerical_summary': df.describe().to_dict()
        }
        return summary

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data for required columns and value ranges"""
        # For test data, check if it has the minimum required columns
        required_cols = ['temperature', 'pressure', 'vibration', 'flow_rate']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns in data: {missing_cols}")
            return False
            
        # Check for invalid values (negative where they shouldn't be)
        if 'pressure' in df.columns and (df['pressure'] < 0).any():
            logger.error("Invalid negative pressure values found")
            return False
            
        if 'temperature' in df.columns and (df['temperature'] < -50).any():
            logger.error("Invalid extremely low temperature values found")
            return False
            
        return True

    def analyze_correlations(self, df: pd.DataFrame) -> dict:
        """Analyze correlations between features"""
        # Only use numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No numeric columns found for correlation analysis")
            return {
                'correlation_matrix': pd.DataFrame(),
                'high_correlations': [],
                'target_correlations': {}
            }
            
        correlation_matrix = numeric_df.corr()
        
        # Create target correlations if target column exists
        target_correlations = {}
        if self.target_column and self.target_column in correlation_matrix.columns:
            target_correlations = correlation_matrix[self.target_column].to_dict()
        elif 'maintenance_needed' in correlation_matrix.columns:
            target_correlations = correlation_matrix['maintenance_needed'].to_dict()
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_correlations': self._find_high_correlations(correlation_matrix),
            'target_correlations': target_correlations
        }
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> list:
        """Find highly correlated feature pairs"""
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if not np.isnan(corr_value) and abs(corr_value) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        return high_corr_pairs

    def save_preprocessor(self, filepath):
        """Save preprocessor to file"""
        try:
            import joblib
        except ImportError:
            logger.error("joblib not available for saving preprocessor")
            return
            
        preprocessor_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'fitted': self.fitted
        }
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load preprocessor from file"""
        try:
            import joblib
        except ImportError:
            logger.error("joblib not available for loading preprocessor")
            return
            
        preprocessor_data = joblib.load(filepath)
        self.scaler = preprocessor_data['scaler']
        self.feature_names = preprocessor_data['feature_names']
        self.target_column = preprocessor_data['target_column']
        self.fitted = preprocessor_data['fitted']
        logger.info(f"Preprocessor loaded from {filepath}")
