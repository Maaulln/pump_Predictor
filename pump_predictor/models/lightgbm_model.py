"""
LightGBM model implementation
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from .base_model import BaseModel
from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

class LightGBMModel(BaseModel):
    def __init__(self, model_params):
        super().__init__(model_params)
        self.model = lgb.LGBMClassifier(**model_params)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train LightGBM model"""
        logger.info("Training LightGBM model...")
        
        # Store feature names if available
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        
        # Perform cross-validation before training
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info("LightGBM training completed")
        
    def feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.feature_importances_
        
    def get_feature_importance_dict(self) -> dict:
        """Get feature importance as dictionary"""
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance()))]
        else:
            feature_names = self.feature_names
            
        return dict(zip(feature_names, self.feature_importance()))