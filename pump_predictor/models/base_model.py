"""
Base model class for pump maintenance prediction
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    def __init__(self, model_params: Dict[str, Any]):
        self.model_params = model_params
        self.model = None
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model"""
        pass
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
    def save_model(self, filepath: str) -> None:
        """Save model to disk"""
        try:
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, filepath: str) -> None:
        """Load model from disk"""
        try:
            self.model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise