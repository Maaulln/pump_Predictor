"""
Tests for model implementations
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pump_predictor.models.random_forest_model import RandomForestModel
from pump_predictor.models.xgboost_model import XGBoostModel
from pump_predictor.models.lightgbm_model import LightGBMModel
from pump_predictor.models.ensemble_model import EnsembleModel
from pump_predictor.config import MODEL_CONFIG

class TestBaseModelFunctionality:
    """Test base model functionality across all model types"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training and test data"""
        np.random.seed(42)
        n_samples = 200
        n_features = 4
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        
        # Split into train/test
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return (X_train, X_test, y_train, y_test)
    
    @pytest.fixture(params=['random_forest', 'xgboost', 'lightgbm'])
    def model(self, request):
        """Parametrized fixture to test all model types"""
        model_type = request.param
        
        if model_type == 'random_forest':
            return RandomForestModel(MODEL_CONFIG['random_forest'])
        elif model_type == 'xgboost':
            return XGBoostModel(MODEL_CONFIG['xgboost'])
        elif model_type == 'lightgbm':
            return LightGBMModel(MODEL_CONFIG['lightgbm'])
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model.model is not None
        assert model.is_trained == False
        assert model.feature_names is None
        assert hasattr(model, 'model_params')
    
    def test_model_training(self, model, sample_data):
        """Test model training"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Before training
        assert model.is_trained == False
        
        # Train model
        model.train(X_train, y_train)
        
        # After training
        assert model.is_trained == True
    
    def test_model_prediction(self, model, sample_data):
        """Test model prediction"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model first
        model.train(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert set(np.unique(predictions)).issubset({0, 1})
    
    def test_model_predict_proba(self, model, sample_data):
        """Test model probability prediction"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model first
        model.train(X_train, y_train)
        
        # Get probabilities
        probabilities = model.predict_proba(X_test)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X_test), 2)  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert (probabilities >= 0).all() and (probabilities <= 1).all()  # Valid probabilities
    
    def test_model_evaluation(self, model, sample_data):
        """Test model evaluation"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model first
        model.train(X_train, y_train)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_feature_importance(self, model, sample_data):
        """Test feature importance extraction"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model first
        model.train(X_train, y_train)
        
        # Get feature importance
        importance = model.feature_importance()
        
        assert importance is not None
        assert isinstance(importance, np.ndarray)
        assert len(importance) == X_train.shape[1]
        assert (importance >= 0).all()  # Importance should be non-negative
    
    def test_model_save_load(self, model, sample_data, tmp_path):
        """Test model saving and loading"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model
        model.train(X_train, y_train)
        
        # Save model
        save_path = tmp_path / "test_model.joblib"
        model.save_model(save_path)
        
        assert save_path.exists()
        
        # Create new model instance and load
        if isinstance(model, RandomForestModel):
            new_model = RandomForestModel(MODEL_CONFIG['random_forest'])
        elif isinstance(model, XGBoostModel):
            new_model = XGBoostModel(MODEL_CONFIG['xgboost'])
        elif isinstance(model, LightGBMModel):
            new_model = LightGBMModel(MODEL_CONFIG['lightgbm'])
        
        new_model.load_model(save_path)
        
        # Test that loaded model works
        assert new_model.is_trained == True
        
        # Compare predictions
        original_pred = model.predict(X_test)
        loaded_pred = new_model.predict(X_test)
        
        assert np.array_equal(original_pred, loaded_pred)

class TestRandomForestModel:
    """Specific tests for Random Forest model"""
    
    @pytest.fixture
    def rf_model(self):
        return RandomForestModel(MODEL_CONFIG['random_forest'])
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 4)
        y_train = np.random.choice([0, 1], 100)
        X_test = np.random.randn(20, 4)
        y_test = np.random.choice([0, 1], 20)
        return X_train, X_test, y_train, y_test
    
    def test_rf_specific_methods(self, rf_model, sample_data):
        """Test Random Forest specific methods"""
        X_train, X_test, y_train, y_test = sample_data
        
        rf_model.train(X_train, y_train)
        
        # Test feature importance dictionary
        importance_dict = rf_model.get_feature_importance_dict()
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) == X_train.shape[1]
        
        # Test model parameters
        params = rf_model.get_model_params()
        assert isinstance(params, dict)
        assert 'n_estimators' in params
        
        # Test trees info
        trees_info = rf_model.get_trees_info()
        assert isinstance(trees_info, dict)
        assert 'n_trees' in trees_info
        assert 'max_depth' in trees_info

class TestXGBoostModel:
    """Specific tests for XGBoost model"""
    
    @pytest.fixture
    def xgb_model(self):
        return XGBoostModel(MODEL_CONFIG['xgboost'])
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 4)
        y_train = np.random.choice([0, 1], 100)
        X_test = np.random.randn(20, 4)
        y_test = np.random.choice([0, 1], 20)
        return X_train, X_test, y_train, y_test
    
    def test_xgb_specific_methods(self, xgb_model, sample_data):
        """Test XGBoost specific methods"""
        X_train, X_test, y_train, y_test = sample_data
        
        xgb_model.train(X_train, y_train)
        
        # Test feature importance dictionary
        importance_dict = xgb_model.get_feature_importance_dict()
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) == X_train.shape[1]
        
        # Test model parameters
        params = xgb_model.get_model_params()
        assert isinstance(params, dict)
        assert 'n_estimators' in params

class TestLightGBMModel:
    """Specific tests for LightGBM model"""
    
    @pytest.fixture
    def lgb_model(self):
        return LightGBMModel(MODEL_CONFIG['lightgbm'])
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 4)
        y_train = np.random.choice([0, 1], 100)
        X_test = np.random.randn(20, 4)
        y_test = np.random.choice([0, 1], 20)
        return X_train, X_test, y_train, y_test
    
    def test_lgb_specific_methods(self, lgb_model, sample_data):
        """Test LightGBM specific methods"""
        X_train, X_test, y_train, y_test = sample_data
        
        lgb_model.train(X_train, y_train)
        
        # Test feature importance dictionary
        importance_dict = lgb_model.get_feature_importance_dict()
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) == X_train.shape[1]

class TestEnsembleModel:
    """Tests for Ensemble model"""
    
    @pytest.fixture
    def trained_models(self):
        """Create trained base models for ensemble"""
        np.random.seed(42)
        X_train = np.random.randn(100, 4)
        y_train = np.random.choice([0, 1], 100)
        
        # Create and train base models
        models = {
            'rf': RandomForestModel(MODEL_CONFIG['random_forest']),
            'xgb': XGBoostModel(MODEL_CONFIG['xgboost']),
            'lgb': LightGBMModel(MODEL_CONFIG['lightgbm'])
        }
        
        for model in models.values():
            model.train(X_train, y_train)
        
        return models, X_train, y_train
    
    def test_ensemble_creation(self, trained_models):
        """Test ensemble model creation"""
        models, X_train, y_train = trained_models
        
        # Create ensemble
        ensemble = EnsembleModel(models, voting='soft')
        ensemble.train(X_train, y_train)
        
        assert ensemble.is_trained == True
        assert len(ensemble.base_models) == 3
    
    def test_ensemble_predictions(self, trained_models):
        """Test ensemble predictions"""
        models, X_train, y_train = trained_models
        X_test = np.random.randn(20, 4)
        
        # Create and train ensemble
        ensemble = EnsembleModel(models, voting='soft')
        ensemble.train(X_train, y_train)
        
        # Test predictions
        predictions = ensemble.predict(X_test)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        
        # Test individual model predictions
        individual_preds = ensemble.get_model_predictions(X_test)
        assert isinstance(individual_preds, dict)
        assert len(individual_preds) == 3
        
        # Test probabilities
        individual_probs = ensemble.get_model_probabilities(X_test)
        assert isinstance(individual_probs, dict)
    
    def test_ensemble_feature_importance(self, trained_models):
        """Test ensemble feature importance"""
        models, X_train, y_train = trained_models
        
        # Create and train ensemble
        ensemble = EnsembleModel(models, voting='soft')
        ensemble.train(X_train, y_train)
        
        # Test weighted feature importance
        importance = ensemble.feature_importance()
        assert importance is not None
        assert isinstance(importance, np.ndarray)
        assert len(importance) == X_train.shape[1]
    
    def test_ensemble_with_weights(self, trained_models):
        """Test ensemble with custom weights"""
        models, X_train, y_train = trained_models
        
        # Create ensemble with weights
        weights = [0.5, 0.3, 0.2]
        ensemble = EnsembleModel(models, voting='soft', weights=weights)
        ensemble.train(X_train, y_train)
        
        assert ensemble.weights == weights
        assert ensemble.is_trained == True

class TestModelErrorHandling:
    """Test error handling across models"""
    
    def test_prediction_without_training(self):
        """Test prediction without training raises error"""
        model = RandomForestModel(MODEL_CONFIG['random_forest'])
        X_test = np.random.randn(10, 4)
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(X_test)
    
    def test_evaluation_without_training(self):
        """Test evaluation without training raises error"""
        model = RandomForestModel(MODEL_CONFIG['random_forest'])
        X_test = np.random.randn(10, 4)
        y_test = np.random.choice([0, 1], 10)
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.evaluate(X_test, y_test)
    
    def test_feature_importance_without_training(self):
        """Test feature importance without training raises error"""
        model = RandomForestModel(MODEL_CONFIG['random_forest'])
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.feature_importance()
    
    def test_ensemble_with_untrained_models(self):
        """Test ensemble creation with untrained models"""
        models = {
            'rf': RandomForestModel(MODEL_CONFIG['random_forest']),
            'xgb': XGBoostModel(MODEL_CONFIG['xgboost'])
        }
        
        with pytest.raises(ValueError, match="not trained"):
            EnsembleModel(models)

class TestModelPerformance:
    """Performance and stress tests for models"""
    
    def test_large_dataset_training(self):
        """Test training on larger dataset"""
        # Create larger dataset
        np.random.seed(42)
        X_train = np.random.randn(1000, 10)
        y_train = np.random.choice([0, 1], 1000)
        
        model = RandomForestModel({'n_estimators': 10, 'random_state': 42})
        
        # Should complete without error
        model.train(X_train, y_train)
        assert model.is_trained == True
    
    def test_memory_efficiency(self):
        """Test that models don't consume excessive memory"""
        model = RandomForestModel({'n_estimators': 5, 'random_state': 42})
        
        X_train = np.random.randn(100, 4)
        y_train = np.random.choice([0, 1], 100)
        
        model.train(X_train, y_train)
        
        # Model size should be reasonable
        model_size = model.get_model_size()
        assert isinstance(model_size, str)
        # Basic check that size is reported
        assert any(unit in model_size for unit in ['B', 'KB', 'MB'])

    @pytest.fixture
    def sample_data(self):
        """Generate sample training and test data"""
        np.random.seed(42)
        n_samples = 200
        n_features = 4
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        
        # Split into train/test
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return (X_train, X_test, y_train, y_test)
    
    @pytest.fixture(params=['random_forest', 'xgboost', 'lightgbm'])
    def model(self, request):
        """Parametrized fixture to test all model types"""
        model_type = request.param
        
        if model_type == 'random_forest':
            return RandomForestModel(MODEL_CONFIG['random_forest'])
        elif model_type == 'xgboost':
            return XGBoostModel(MODEL_CONFIG['xgboost'])
        elif model_type == 'lightgbm':
            return LightGBMModel(MODEL_CONFIG['lightgbm'])
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model.model is not None
        assert model.is_trained == False
        assert model.feature_names is None
        assert hasattr(model, 'model_params')
    
    def test_model_training(self, model, sample_data):
        """Test model training"""
        X_train, X_test, y_train, y_test = sample_data
        
        # Before training
        assert model.is_trained == False
        
        # Train model
        model.train(X_train, y_train)
        
        # After training
        assert model.is_trained == True