"""
API tests for FastAPI endpoints
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import joblib
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pump_predictor.api.main import app, model_manager
from pump_predictor.models.random_forest_model import RandomForestModel
from pump_predictor.config import MODEL_CONFIG

# Test client
client = TestClient(app)

@pytest.fixture
def mock_model():
    """Create a mock trained model"""
    model = RandomForestModel(MODEL_CONFIG['random_forest'])
    
    # Create some fake training data
    np.random.seed(42)
    X = np.random.rand(100, 4)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    
    model.train(X, y)
    return model

@pytest.fixture
def setup_model_manager(mock_model):
    """Setup model manager with mock model"""
    original_model = model_manager.model
    original_trained = model_manager.is_model_loaded()
    
    # Set mock model
    model_manager.model = mock_model
    model_manager.metadata = {
        'model_type': 'RandomForestModel',
        'performance': {
            'RandomForest': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.75,
                'f1': 0.77
            }
        }
    }
    
    yield
    
    # Restore original state
    model_manager.model = original_model

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data["service"] == "Pump Maintenance Prediction API"
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "uptime_seconds" in data
        assert "total_predictions" in data
        assert "total_errors" in data
        assert "error_rate" in data

class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    def test_single_prediction_success(self, setup_model_manager):
        """Test successful single prediction"""
        test_data = {
            "temperature": 75.5,
            "pressure": 150.0,
            "vibration": 2.5,
            "flow_rate": 250.0
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "needs_maintenance" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert "model_type" in data
        assert "timestamp" in data
        
        # Validate data types
        assert isinstance(data["needs_maintenance"], bool)
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    
    def test_single_prediction_validation_error(self, setup_model_manager):
        """Test prediction with invalid data"""
        test_data = {
            "temperature": 300.0,  # Invalid temperature
            "pressure": 150.0,
            "vibration": 2.5,
            "flow_rate": 250.0
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_single_prediction_missing_field(self, setup_model_manager):
        """Test prediction with missing field"""
        test_data = {
            "temperature": 75.5,
            "pressure": 150.0,
            "vibration": 2.5
            # Missing flow_rate
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422
    
    def test_batch_prediction_success(self, setup_model_manager):
        """Test successful batch prediction"""
        test_data = {
            "data": [
                {
                    "temperature": 75.5,
                    "pressure": 150.0,
                    "vibration": 2.5,
                    "flow_rate": 250.0
                },
                {
                    "temperature": 85.0,
                    "pressure": 200.0,
                    "vibration": 4.0,
                    "flow_rate": 300.0
                }
            ],
            "include_details": True
        }
        
        response = client.post("/predict/batch", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "summary" in data
        assert "total_count" in data
        assert "maintenance_needed_count" in data
        assert "processing_time" in data
        
        # Check predictions
        predictions = data["predictions"]
        assert len(predictions) == 2
        
        for pred in predictions:
            assert "needs_maintenance" in pred
            assert "confidence" in pred
            assert "risk_level" in pred
    
    def test_batch_prediction_empty_data(self, setup_model_manager):
        """Test batch prediction with empty data"""
        test_data = {
            "data": []
        }
        
        response = client.post("/predict/batch", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_explain_prediction(self, setup_model_manager):
        """Test prediction explanation endpoint"""
        test_data = {
            "temperature": 75.5,
            "pressure": 150.0,
            "vibration": 2.5,
            "flow_rate": 250.0
        }
        
        response = client.post("/predict/explain", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "feature_contributions" in data
        assert "explanation_text" in data
        
        # Check prediction structure
        prediction = data["prediction"]
        assert "needs_maintenance" in prediction
        assert "confidence" in prediction

class TestModelInfoEndpoints:
    """Test model information endpoints"""
    
    def test_model_info(self, setup_model_manager):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_type" in data
        assert "version" in data
        assert "features" in data
        assert "performance_metrics" in data
        
        # Check features
        features = data["features"]
        assert isinstance(features, list)
        assert len(features) > 0
    
    def test_model_performance(self, setup_model_manager):
        """Test model performance endpoint"""
        response = client.get("/model/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)
    
    def test_feature_importance(self, setup_model_manager):
        """Test feature importance endpoint"""
        response = client.get("/model/feature-importance")
        assert response.status_code == 200
        
        data = response.json()
        assert "features" in data
        assert "model_type" in data
        assert "total_features" in data
        
        features = data["features"]
        assert isinstance(features, dict)
        assert len(features) > 0

class TestAPIErrorHandling:
    """Test API error handling"""
    
    def test_prediction_without_model(self):
        """Test prediction when model is not loaded"""
        # Ensure model is not loaded
        original_model = model_manager.model
        model_manager.model = None
        
        test_data = {
            "temperature": 75.5,
            "pressure": 150.0,
            "vibration": 2.5,
            "flow_rate": 250.0
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 503  # Service unavailable
        
        # Restore model
        model_manager.model = original_model
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint"""
        response = client.get("/invalid_endpoint")
        assert response.status_code == 404
    
    def test_invalid_method(self, setup_model_manager):
        """Test invalid HTTP method"""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405  # Method not allowed

class TestDataValidation:
    """Test data validation"""
    
    @pytest.mark.parametrize("temperature,expected_status", [
        (-60, 422),  # Too low
        (250, 422),  # Too high
        (75, 200),   # Valid
    ])
    def test_temperature_validation(self, setup_model_manager, temperature, expected_status):
        """Test temperature validation"""
        test_data = {
            "temperature": temperature,
            "pressure": 150.0,
            "vibration": 2.5,
            "flow_rate": 250.0
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == expected_status
    
    @pytest.mark.parametrize("pressure,expected_status", [
        (-10, 422),  # Negative
        (0, 200),    # Zero (valid)
        (500, 200),  # High but valid
    ])
    def test_pressure_validation(self, setup_model_manager, pressure, expected_status):
        """Test pressure validation"""
        test_data = {
            "temperature": 75.0,
            "pressure": pressure,
            "vibration": 2.5,
            "flow_rate": 250.0
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == expected_status

class TestAPIPerformance:
    """Test API performance characteristics"""
    
    def test_prediction_response_time(self, setup_model_manager):
        """Test that prediction response time is reasonable"""
        import time
        
        test_data = {
            "temperature": 75.5,
            "pressure": 150.0,
            "vibration": 2.5,
            "flow_rate": 250.0
        }
        
        start_time = time.time()
        response = client.post("/predict", json=test_data)
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be under 1 second
        response_time = end_time - start_time
        assert response_time < 1.0
    
    def test_batch_prediction_performance(self, setup_model_manager):
        """Test batch prediction performance"""
        # Create batch of 10 predictions
        test_data = {
            "data": [
                {
                    "temperature": 75.5 + i,
                    "pressure": 150.0 + i * 5,
                    "vibration": 2.5 + i * 0.1,
                    "flow_rate": 250.0 + i * 2
                }
                for i in range(10)
            ]
        }
        
        import time
        start_time = time.time()
        response = client.post("/predict/batch", json=test_data)
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Batch processing should be efficient
        response_time = end_time - start_time
        assert response_time < 2.0  # Should complete within 2 seconds
        
        # Check that processing time is reported
        data = response.json()
        assert "processing_time" in data
        assert data["processing_time"] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])