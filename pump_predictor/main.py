"""
Main execution module for pump maintenance prediction
"""
import pandas as pd
from pathlib import Path

from pump_predictor.config import MODEL_CONFIG, DATA_DIR, MODEL_DIR
from pump_predictor.data.preprocessing import DataPreprocessor
from pump_predictor.models.random_forest_model import RandomForestModel
from pump_predictor.models.xgboost_model import XGBoostModel
from pump_predictor.utils.visualization import plot_feature_importance, plot_confusion_matrix
from pump_predictor.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    data = preprocessor.load_data(DATA_DIR / "pump_data.csv")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data)
    
    # Train and evaluate Random Forest model
    rf_model = RandomForestModel(MODEL_CONFIG['random_forest'])
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    
    logger.info("Random Forest Performance:")
    for metric, value in rf_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Train and evaluate XGBoost model
    xgb_model = XGBoostModel(MODEL_CONFIG['xgboost'])
    xgb_model.train(X_train, y_train)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    
    logger.info("XGBoost Performance:")
    for metric, value in xgb_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Visualize results
    plot_feature_importance(
        rf_model.feature_importance(),
        "Random Forest Feature Importance"
    )
    plot_feature_importance(
        xgb_model.feature_importance(),
        "XGBoost Feature Importance"
    )
    
    # Save best model
    best_model = rf_model if rf_metrics['f1'] > xgb_metrics['f1'] else xgb_model
    best_model.save_model(MODEL_DIR / "best_model.joblib")

if __name__ == "__main__":
    main()