"""
XGBoost model implementation
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from .base_model import BaseModel
from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

class XGBoostModel(BaseModel):
    def __init__(self, model_params):
        super().__init__(model_params)
        self.model = xgb.XGBClassifier(**model_params)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
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
        
        logger.info("XGBoost training completed")
        
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
    
    def get_model_params(self) -> dict:
        """Get current model parameters"""
        return self.model.get_params()
    
    def plot_tree(self, tree_index: int = 0, save_path: str = None):
        """Plot a specific tree"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(20, 10))
            xgb.plot_tree(self.model, num_trees=tree_index, ax=ax)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Tree plot saved to {save_path}")
            
            plt.show()
        except ImportError:
            logger.warning("Matplotlib not available for tree plotting")