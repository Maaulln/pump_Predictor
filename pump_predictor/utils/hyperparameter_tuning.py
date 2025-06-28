"""
Advanced hyperparameter tuning utilities
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna
from pump_predictor.utils.logger import get_logger
from pump_predictor.config import TUNING_CONFIG

logger = get_logger(__name__)

class HyperparameterTuner:
    def __init__(self, model_type: str, cv_folds: int = 5, scoring: str = 'f1'):
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.scoring = scoring
        
    def get_param_grid(self) -> Dict[str, Any]:
        """Get parameter grid for different models"""
        return TUNING_CONFIG.get(self.model_type, {})
    
    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray, 
                   param_grid: Dict[str, Any] = None) -> Tuple[Dict[str, Any], float]:
        """Perform grid search"""
        if param_grid is None:
            param_grid = self.get_param_grid()
            
        base_model = self._create_base_model()
        
        logger.info(f"Starting grid search for {self.model_type}...")
        logger.info(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Grid search completed. Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_params_, grid_search.best_score_
    
    def random_search(self, X_train: np.ndarray, y_train: np.ndarray,
                     param_distributions: Dict[str, Any] = None,
                     n_iter: int = 100) -> Tuple[Dict[str, Any], float]:
        """Perform randomized search"""
        if param_distributions is None:
            param_distributions = self.get_param_grid()
            
        base_model = self._create_base_model()
        
        logger.info(f"Starting random search for {self.model_type} with {n_iter} iterations...")
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"Random search completed. Best score: {random_search.best_score_:.4f}")
        logger.info(f"Best parameters: {random_search.best_params_}")
        
        return random_search.best_params_, random_search.best_score_
    
    def optuna_search(self, X_train: np.ndarray, y_train: np.ndarray,
                     n_trials: int = 100) -> Tuple[Dict[str, Any], float]:
        """Perform Optuna optimization"""
        logger.info(f"Starting Optuna optimization for {self.model_type} with {n_trials} trials...")
        
        def objective(trial):
            params = self._suggest_params(trial)
            model = self._create_base_model(params)
            
            # Cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring=self.scoring)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Optuna optimization completed. Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params, study.best_value
    
    def _create_base_model(self, params: Dict[str, Any] = None):
        """Create base model"""
        if params is None:
            params = {}
            
        if self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=42, **params)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(random_state=42, **params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(random_state=42, **params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _suggest_params(self, trial):
        """Suggest parameters for Optuna trial"""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }

class AutoTuner:
    """Automated hyperparameter tuning for multiple models"""
    
    def __init__(self, cv_folds: int = 5, scoring: str = 'f1'):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.results = {}
    
    def tune_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                       method: str = 'optuna', n_trials: int = 50) -> Dict[str, Dict[str, Any]]:
        """Tune all available models"""
        models = ['random_forest', 'xgboost', 'lightgbm']
        
        for model_type in models:
            logger.info(f"Tuning {model_type}...")
            tuner = HyperparameterTuner(model_type, self.cv_folds, self.scoring)
            
            try:
                if method == 'optuna':
                    best_params, best_score = tuner.optuna_search(X_train, y_train, n_trials)
                elif method == 'random':
                    best_params, best_score = tuner.random_search(X_train, y_train, n_iter=n_trials)
                elif method == 'grid':
                    best_params, best_score = tuner.grid_search(X_train, y_train)
                else:
                    raise ValueError(f"Unknown tuning method: {method}")
                
                self.results[model_type] = {
                    'best_params': best_params,
                    'best_score': best_score
                }
                
            except Exception as e:
                logger.error(f"Error tuning {model_type}: {str(e)}")
                
        return self.results
    
    def get_best_model_config(self) -> Tuple[str, Dict[str, Any]]:
        """Get the best model configuration"""
        if not self.results:
            raise ValueError("No tuning results available")
            
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['best_score'])
        return best_model, self.results[best_model]['best_params']