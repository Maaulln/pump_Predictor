"""
FastAPI application for pump maintenance prediction
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import sys
import uuid
import time
from typing import List, Dict, Any, Optional
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pump_predictor.api.schemas import (
    PumpData, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, ModelInfo, ModelPerformance,
    HealthCheck, ErrorResponse, FeatureImportance,
    PredictionExplanation
)
from pump_predictor.utils.logger import get_logger
from pump_predictor.config import API_CONFIG

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🔧 Pump Maintenance Prediction API",
    description="Advanced ML-powered API for predicting pump maintenance needs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_metadata = None
start_time = datetime.now()
prediction_count = 0
error_count = 0

class ModelManager:
    """Manages model loading and caching"""
    
    def __init__(self):
        self.model = None
        self.metadata = None
        self.load_timestamp = None
    
    def load_model(self, model_path: Path = None, metadata_path: Path = None):
        """Load model and metadata"""
        try:
            if model_path is None:
                model_path = Path("models/best_model.joblib")
            if metadata_path is None:
                metadata_path = Path("models/model_metadata.joblib")
            
            if model_path.exists():
                model_data = joblib.load(model_path)
                self.model = model_data['model'] if isinstance(model_data, dict) else model_data
                self.load_timestamp = datetime.now()
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
                
            if metadata_path.exists():
                self.metadata = joblib.load(metadata_path)
                logger.info(f"Model metadata loaded from {metadata_path}")
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_model_loaded():
            return {}
        
        info = {
            'model_type': type(self.model).__name__,
            'load_timestamp': self.load_timestamp.isoformat() if self.load_timestamp else None,
            'is_trained': getattr(self.model, 'is_trained', True)
        }
        
        if self.metadata:
            info.update(self.metadata)
        
        return info

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global prediction_count, error_count
    
    logger.info("🚀 Starting Pump Maintenance Prediction API...")
    
    try:
        model_manager.load_model()
        prediction_count = 0
        error_count = 0
        logger.info("✅ API startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ API startup failed: {str(e)}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"📝 [{request_id}] {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"✅ [{request_id}] Completed in {process_time:.3f}s - Status: {response.status_code}")
    
    return response

def get_model():
    """Dependency to get loaded model"""
    if not model_manager.is_model_loaded():
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please ensure model is trained and loaded."
        )
    return model_manager.model

def calculate_risk_level(confidence: float, prediction: bool) -> str:
    """Calculate risk level based on prediction and confidence"""
    if not prediction:
        return "LOW"
    elif confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.6:
        return "MEDIUM"
    else:
        return "LOW"

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Pump Maintenance Prediction API",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "performance": "/model/performance"
        }
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check endpoint"""
    global prediction_count, error_count, start_time
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthCheck(
        status="healthy" if model_manager.is_model_loaded() else "degraded",
        model_loaded=model_manager.is_model_loaded(),
        version="1.0.0",
        uptime=uptime
    )

@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    global prediction_count, error_count, start_time
    
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "uptime_seconds": uptime,
        "total_predictions": prediction_count,
        "total_errors": error_count,
        "error_rate": error_count / max(prediction_count, 1),
        "predictions_per_minute": (prediction_count / max(uptime / 60, 1)),
        "model_loaded": model_manager.is_model_loaded()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_maintenance(data: PumpData, current_model=Depends(get_model)):
    """Single prediction endpoint with enhanced features"""
    global prediction_count, error_count
    
    try:
        prediction_count += 1
        
        # Convert to array and reshape for prediction
        features = data.to_array().reshape(1, -1)
        
        # Make prediction
        prediction = current_model.predict(features)[0]
        
        # Get probability if available
        try:
            probabilities = current_model.predict_proba(features)[0]
            confidence = probabilities.max()
        except:
            confidence = 0.7 if prediction else 0.3  # Default confidence
        
        # Calculate risk level
        risk_level = calculate_risk_level(confidence, bool(prediction))
        
        # Generate prediction ID
        prediction_id = f"pred_{uuid.uuid4().hex[:8]}"
        
        return PredictionResponse(
            needs_maintenance=bool(prediction),
            confidence=confidence,
            risk_level=risk_level,
            model_type=type(current_model).__name__,
            prediction_id=prediction_id
        )
        
    except Exception as e:
        error_count += 1
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict_maintenance(request: BatchPredictionRequest, 
                                  background_tasks: BackgroundTasks,
                                  current_model=Depends(get_model)):
    """Batch prediction endpoint with summary statistics"""
    global prediction_count, error_count
    
    try:
        start_time = time.time()
        predictions = []
        risk_summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        maintenance_count = 0
        
        for i, pump_data in enumerate(request.data):
            prediction_count += 1
            
            # Convert to array and reshape for prediction
            features = pump_data.to_array().reshape(1, -1)
            
            # Make prediction
            prediction = current_model.predict(features)[0]
            
            # Get probability if available
            try:
                probabilities = current_model.predict_proba(features)[0]
                confidence = probabilities.max()
            except:
                confidence = 0.7 if prediction else 0.3
            
            # Calculate risk level
            risk_level = calculate_risk_level(confidence, bool(prediction))
            risk_summary[risk_level] += 1
            
            if prediction:
                maintenance_count += 1
            
            # Create prediction response
            pred_response = PredictionResponse(
                needs_maintenance=bool(prediction),
                confidence=confidence,
                risk_level=risk_level,
                model_type=type(current_model).__name__,
                prediction_id=f"batch_pred_{i+1}_{uuid.uuid4().hex[:6]}"
            )
            
            predictions.append(pred_response)
        
        processing_time = time.time() - start_time
        
        # Create summary
        summary = {
            "risk_distribution": risk_summary,
            "maintenance_rate": maintenance_count / len(request.data),
            "average_confidence": np.mean([p.confidence for p in predictions]),
            "processing_time_per_item": processing_time / len(request.data)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            total_count=len(request.data),
            maintenance_needed_count=maintenance_count,
            processing_time=processing_time
        )
        
    except Exception as e:
        error_count += 1
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(current_model=Depends(get_model)):
    """Get comprehensive model information"""
    try:
        model_info = model_manager.get_model_info()
        
        # Get feature names
        feature_names = []
        if hasattr(current_model, 'feature_names') and current_model.feature_names:
            feature_names = current_model.feature_names
        else:
            # Default feature names based on schema
            feature_names = ["temperature", "pressure", "vibration", "flow_rate"]
        
        # Get performance metrics
        performance_metrics = {}
        if model_manager.metadata and 'performance' in model_manager.metadata:
            # Get best model performance
            best_model_performance = model_manager.metadata['performance']
            if isinstance(best_model_performance, dict):
                for model_name, metrics in best_model_performance.items():
                    if isinstance(metrics, dict):
                        performance_metrics = metrics
                        break
        
        # Get model size
        model_size = getattr(current_model, 'get_model_size', lambda: "Unknown")()
        
        return ModelInfo(
            model_type=type(current_model).__name__,
            version="1.0.0",
            features=feature_names,
            performance_metrics=performance_metrics,
            training_date=model_info.get('training_timestamp'),
            model_size=model_size
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/model/performance")
async def get_model_performance():
    """Get detailed model performance metrics"""
    if model_manager.metadata and 'performance' in model_manager.metadata:
        return model_manager.metadata['performance']
    else:
        raise HTTPException(status_code=404, detail="Performance metrics not available")

@app.get("/model/feature-importance", response_model=FeatureImportance)
async def get_feature_importance(current_model=Depends(get_model)):
    """Get model feature importance"""
    try:
        importance = current_model.feature_importance()
        if importance is None:
            raise HTTPException(status_code=404, detail="Feature importance not available for this model")
        
        # Get feature names
        if hasattr(current_model, 'feature_names') and current_model.feature_names:
            feature_names = current_model.feature_names
        else:
            feature_names = ["temperature", "pressure", "vibration", "flow_rate"]
        
        # Create feature importance dictionary
        features_dict = dict(zip(feature_names, importance.tolist()))
        
        return FeatureImportance(
            features=features_dict,
            model_type=type(current_model).__name__,
            total_features=len(feature_names)
        )
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

@app.post("/predict/explain", response_model=PredictionExplanation)
async def explain_prediction(data: PumpData, current_model=Depends(get_model)):
    """Get prediction with detailed explanation"""
    try:
        # Get prediction first
        features = data.to_array().reshape(1, -1)
        prediction = current_model.predict(features)[0]
        
        try:
            probabilities = current_model.predict_proba(features)[0]
            confidence = probabilities.max()
        except:
            confidence = 0.7 if prediction else 0.3
        
        risk_level = calculate_risk_level(confidence, bool(prediction))
        
        # Create prediction response
        prediction_response = PredictionResponse(
            needs_maintenance=bool(prediction),
            confidence=confidence,
            risk_level=risk_level,
            model_type=type(current_model).__name__,
            prediction_id=f"explain_{uuid.uuid4().hex[:8]}"
        )
        
        # Get feature importance for explanation
        feature_contributions = {}
        try:
            importance = current_model.feature_importance()
            if importance is not None:
                feature_names = getattr(current_model, 'feature_names', 
                                      ["temperature", "pressure", "vibration", "flow_rate"])
                feature_values = data.to_array()
                
                # Simple contribution calculation (can be enhanced with SHAP)
                normalized_importance = importance / importance.sum()
                normalized_values = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-8)
                
                for i, (name, imp, val) in enumerate(zip(feature_names, normalized_importance, normalized_values)):
                    feature_contributions[name] = float(imp * val)
        except:
            feature_contributions = {}
        
        # Generate explanation text
        explanation_text = f"Based on the {type(current_model).__name__} model, "
        if prediction:
            explanation_text += f"maintenance is recommended with {confidence:.1%} confidence. "
            if risk_level == "HIGH":
                explanation_text += "High risk indicators detected."
            elif risk_level == "MEDIUM":
                explanation_text += "Moderate risk indicators present."
        else:
            explanation_text += f"normal operation predicted with {confidence:.1%} confidence."
        
        return PredictionExplanation(
            prediction=prediction_response,
            feature_contributions=feature_contributions,
            explanation_text=explanation_text
        )
        
    except Exception as e:
        logger.error(f"Error explaining prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to explain prediction: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    global error_count
    error_count += 1
    
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app, 
        host=API_CONFIG['host'], 
        port=API_CONFIG['port'],
        reload=API_CONFIG['reload'],
        log_level=API_CONFIG['log_level']
    )