"""
Pydantic schemas for FastAPI endpoints
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime

class PumpData(BaseModel):
    """Schema for individual pump sensor data"""
    temperature: float = Field(..., description="Temperature reading in Celsius", ge=-50, le=200)
    pressure: float = Field(..., description="Pressure reading in PSI", ge=0, le=1000)
    vibration: float = Field(..., description="Vibration reading in Hz", ge=0, le=100)
    flow_rate: float = Field(..., description="Flow rate in L/min", ge=0, le=1000)
    # Add more features as needed based on your actual data
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not -50 <= v <= 200:
            raise ValueError('Temperature must be between -50 and 200 Celsius')
        return v
    
    @validator('pressure')
    def validate_pressure(cls, v):
        if v < 0:
            raise ValueError('Pressure cannot be negative')
        return v
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model prediction"""
        return np.array([self.temperature, self.pressure, self.vibration, self.flow_rate])
    
    class Config:
        schema_extra = {
            "example": {
                "temperature": 75.5,
                "pressure": 150.0,
                "vibration": 2.5,
                "flow_rate": 250.0
            }
        }

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    needs_maintenance: bool = Field(..., description="Whether maintenance is needed")
    confidence: float = Field(..., description="Prediction confidence (0-1)", ge=0, le=1)
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    model_type: str = Field(..., description="Type of model used for prediction")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    prediction_id: Optional[str] = Field(None, description="Unique prediction identifier")
    
    @validator('confidence')
    def round_confidence(cls, v):
        return round(v, 4)
    
    class Config:
        schema_extra = {
            "example": {
                "needs_maintenance": True,
                "confidence": 0.8543,
                "risk_level": "HIGH",
                "model_type": "RandomForestModel",
                "timestamp": "2023-12-01T10:30:00",
                "prediction_id": "pred_12345"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    data: List[PumpData] = Field(..., description="List of pump data for batch prediction")
    include_details: bool = Field(default=False, description="Include detailed prediction info")
    
    @validator('data')
    def validate_data_length(cls, v):
        if len(v) == 0:
            raise ValueError('Data list cannot be empty')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 records per batch')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "temperature": 75.5,
                        "pressure": 150.0,
                        "vibration": 2.5,
                        "flow_rate": 250.0
                    }
                ],
                "include_details": False
            }
        }

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, Any] = Field(..., description="Batch prediction summary")
    total_count: int = Field(..., description="Total number of predictions")
    maintenance_needed_count: int = Field(..., description="Number of pumps needing maintenance")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "needs_maintenance": True,
                        "confidence": 0.8543,
                        "risk_level": "HIGH",
                        "model_type": "RandomForestModel",
                        "timestamp": "2023-12-01T10:30:00"
                    }
                ],
                "summary": {
                    "high_risk": 15,
                    "medium_risk": 25,
                    "low_risk": 60
                },
                "total_count": 100,
                "maintenance_needed_count": 15,
                "processing_time": 0.156
            }
        }

class ModelInfo(BaseModel):
    """Schema for model information"""
    model_type: str = Field(..., description="Type of model")
    version: str = Field(..., description="Model version")
    features: List[str] = Field(..., description="List of feature names")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    training_date: Optional[str] = Field(None, description="Model training date")
    model_size: Optional[str] = Field(None, description="Model size in memory")
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "RandomForestModel",
                "version": "1.0.0",
                "features": ["temperature", "pressure", "vibration", "flow_rate"],
                "performance_metrics": {
                    "accuracy": 0.8765,
                    "precision": 0.8543,
                    "recall": 0.8321,
                    "f1": 0.8431
                },
                "training_date": "2023-12-01T08:00:00",
                "model_size": "15.2 MB"
            }
        }

class ModelPerformance(BaseModel):
    """Schema for detailed model performance"""
    model_name: str = Field(..., description="Name of the model")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "RandomForest",
                "accuracy": 0.8765,
                "precision": 0.8543,
                "recall": 0.8321,
                "f1_score": 0.8431,
                "confusion_matrix": [[85, 15], [12, 88]],
                "feature_importance": {
                    "temperature": 0.3245,
                    "pressure": 0.2876,
                    "vibration": 0.2234,
                    "flow_rate": 0.1645
                }
            }
        }

class HealthCheck(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2023-12-01T10:30:00",
                "model_loaded": True,
                "version": "1.0.0",
                "uptime": 3600.5
            }
        }

class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: Optional[str] = Field(None, description="Request identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid input data",
                "error_code": "VALIDATION_ERROR",
                "timestamp": "2023-12-01T10:30:00",
                "request_id": "req_12345"
            }
        }

class FeatureImportance(BaseModel):
    """Schema for feature importance response"""
    features: Dict[str, float] = Field(..., description="Feature importance scores")
    model_type: str = Field(..., description="Model type")
    total_features: int = Field(..., description="Total number of features")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "temperature": 0.3245,
                    "pressure": 0.2876,
                    "vibration": 0.2234,
                    "flow_rate": 0.1645
                },
                "model_type": "RandomForestModel",
                "total_features": 4
            }
        }

class PredictionExplanation(BaseModel):
    """Schema for prediction explanation"""
    prediction: PredictionResponse = Field(..., description="The prediction")
    feature_contributions: Dict[str, float] = Field(..., description="Feature contributions to prediction")
    explanation_text: str = Field(..., description="Human-readable explanation")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": {
                    "needs_maintenance": True,
                    "confidence": 0.8543,
                    "risk_level": "HIGH",
                    "model_type": "RandomForestModel",
                    "timestamp": "2023-12-01T10:30:00"
                },
                "feature_contributions": {
                    "temperature": 0.25,
                    "pressure": 0.35,
                    "vibration": 0.30,
                    "flow_rate": 0.10
                },
                "explanation_text": "High pressure and vibration levels indicate immediate maintenance required."
            }
        }