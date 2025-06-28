"""
Ensemble model implementation
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from typing import Dict, List
from .base_model import BaseModel
from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

class EnsembleModel(BaseModel):
    def __init__(self, models_dict: Dict[str, BaseModel], voting='soft', weights=None):
        # Create model params from individual models
        model_params = {
            'voting': voting,
            'weights': weights,
            'models': {name: model.model_params for name, model in models_dict.items()}
        }
        super().__init__(model_params)
        
        self.base_models = models_dict
        self.voting = voting
        self.weights = weights
        
        # Validate that all models are trained
        for name, model in models_dict.items():
            if not model.is_trained:
                raise ValueError(f"Model {name} is not trained yet")
        
        # Create estimators list
        estimators = [(name, model.model) for name, model in models_dict.items()]
        
        self.model = VotingClassifier(
            estimators=estimators, 
            voting=voting,
            weights=weights
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train ensemble model (base models should already be trained)"""
        logger.info("Creating ensemble model...")
        
        # Store feature names if available
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        
        # Fit the voting classifier
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info("Ensemble model created successfully")
        
    def feature_importance(self) -> np.ndarray:
        """Get averaged feature importance from base models"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        importances = []
        weights = self.weights or [1.0] * len(self.base_models)
        
        for i, (name, model) in enumerate(self.base_models.items()):
            try:
                importance = model.feature_importance()
                if importance is not None:
                    # Weight the importance
                    weighted_importance = importance * weights[i]
                    importances.append(weighted_importance)
            except:
                logger.warning(f"Could not get feature importance from {name}")
        
        if not importances:
            return None
            
        # Weighted average of importances
        total_weights = sum(weights[:len(importances)])
        return np.sum(importances, axis=0) / total_weights
        
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from individual models"""
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X)
        return predictions
    
    def get_model_probabilities(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get probabilities from individual models"""
        probabilities = {}
        for name, model in self.base_models.items():
            try:
                probabilities[name] = model.predict_proba(X)
            except:
                logger.warning(f"Could not get probabilities from {name}")
        return probabilities