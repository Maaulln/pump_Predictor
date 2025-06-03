"""
Random Forest implementation for pump maintenance prediction
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Dict, Any

from pump_predictor.models.base_model import BaseModel
from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

class RandomForestModel(BaseModel):
    def __init__(self, model_params: Dict[str, Any]):
        super().__init__(model_params)
        self.model = RandomForestClassifier(**model_params)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train Random Forest model"""
        try:
            logger.info("Training Random Forest model...")
            self.model.fit(X_train, y_train)
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        return dict(zip(
            self.model.feature_names_in_,
            self.model.feature_importances_
        ))