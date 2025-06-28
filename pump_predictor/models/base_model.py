"""
Base model class for pump maintenance prediction
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from pathlib import Path
from sklearn.exceptions import NotFittedError

from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    def __init__(self, model_params: Dict[str, Any]):
        self.model_params = model_params
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model"""
        pass
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        try:
            return self.model.predict(X)
        except NotFittedError:
            raise ValueError("Model not trained")
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        if hasattr(self.model, 'predict_proba'):
            try:
                return self.model.predict_proba(X)
            except NotFittedError:
                raise ValueError("Model not trained")
        else:
            raise NotImplementedError("Model doesn't support probability prediction")
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
    def get_classification_report(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """Get detailed classification report"""
        y_pred = self.predict(X_test)
        return classification_report(y_test, y_pred)
        
    @abstractmethod
    def feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores"""
        pass
        
    def save_model(self, filepath: Path) -> None:
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'model_params': self.model_params,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, filepath: Path) -> None:
        """Load model from disk"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.model_params = model_data['model_params']
            self.feature_names = model_data.get('feature_names')
            self.is_trained = model_data.get('is_trained', True)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def get_model_size(self) -> str:
        """Get model size in memory"""
        import sys
        if self.model is not None:
            size_bytes = sys.getsizeof(self.model)
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.2f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.2f} MB"
        return "0 B"
