"""
Enhanced main execution module for pump maintenance prediction
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import time
from datetime import datetime
from typing import Dict, Any, Tuple

from pump_predictor.config import MODEL_CONFIG, DATA_DIR, MODEL_DIR, PREPROCESSING_CONFIG
from pump_predictor.data.preprocessing import DataPreprocessor
from pump_predictor.models.random_forest_model import RandomForestModel
from pump_predictor.models.xgboost_model import XGBoostModel
from pump_predictor.models.lightgbm_model import LightGBMModel
from pump_predictor.models.ensemble_model import EnsembleModel
from pump_predictor.utils.visualization import ModelVisualizer, InteractiveVisualizer
from pump_predictor.utils.hyperparameter_tuning import HyperparameterTuner, AutoTuner
from pump_predictor.utils.model_explainer import ModelExplainer
from pump_predictor.utils.report_generator import ReportGenerator
from pump_predictor.utils.logger import get_logger
import joblib

logger = get_logger(__name__)

class PumpPredictorPipeline:
    """Complete machine learning pipeline for pump maintenance prediction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.training_history = {}
        
        # Initialize components
        self.visualizer = ModelVisualizer()
        self.interactive_viz = InteractiveVisualizer()
        self.report_generator = ReportGenerator()
        
        # Ensure directories exist
        DATA_DIR.mkdir(exist_ok=True)
        MODEL_DIR.mkdir(exist_ok=True)
    
    def load_and_prepare_data(self, data_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare data for training"""
        logger.info("Loading and preparing data...")
        
        if data_path is None:
            data_path = DATA_DIR / "pump_data.csv"
        else:
            # Fix path if it is the old hardcoded absolute path
            old_path = '/Users/Maulana/Downloads/KULIAH/JURNAL DOC/code/ml_KTI/data/pump_data.csv'
            if str(data_path) == old_path:
                logger.warning(f"Old data path detected: {old_path}. Updating to project data directory.")
                data_path = DATA_DIR / "pump_data.csv"
            # Convert to string if Path object
            if not isinstance(data_path, (str, bytes)):
                data_path = str(data_path)
        
        # Load data
        if not isinstance(data_path, (str, bytes)):
            # Convert Path or other types to string
            data_path_str = str(data_path)
        else:
            data_path_str = data_path

        data = self.preprocessor.load_data(data_path_str)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Generate data report
        self.report_generator.create_data_report(data)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_data(data)
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self, model_types: list = None) -> None:
        """Initialize models based on configuration"""
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'lightgbm']
        
        logger.info(f"Initializing models: {model_types}")
        
        for model_type in model_types:
            if model_type == 'random_forest':
                self.models['RandomForest'] = RandomForestModel(MODEL_CONFIG['random_forest'])
            elif model_type == 'xgboost':
                self.models['XGBoost'] = XGBoostModel(MODEL_CONFIG['xgboost'])
            elif model_type == 'lightgbm':
                self.models['LightGBM'] = LightGBMModel(MODEL_CONFIG['lightgbm'])
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                           method: str = 'optuna', n_trials: int = 50) -> Dict[str, Dict[str, Any]]:
        """Perform hyperparameter tuning for all models"""
        logger.info(f"Starting hyperparameter tuning using {method}...")
        
        auto_tuner = AutoTuner()
        tuning_results = auto_tuner.tune_all_models(X_train, y_train, method, n_trials)
        
        # Update model configurations with best parameters
        for model_name in self.models.keys():
            model_type = model_name.lower().replace('forest', '_forest')
            if model_type in tuning_results:
                best_params = tuning_results[model_type]['best_params']
                logger.info(f"Updating {model_name} with best parameters: {best_params}")
                
                # Reinitialize model with best parameters
                if model_name == 'RandomForest':
                    self.models[model_name] = RandomForestModel(best_params)
                elif model_name == 'XGBoost':
                    self.models[model_name] = XGBoostModel(best_params)
                elif model_name == 'LightGBM':
                    self.models[model_name] = LightGBMModel(best_params)
        
        return tuning_results
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Train all initialized models"""
        logger.info("Starting model training...")
        start_time = time.time()
        
        results = {}
        trained_models = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            model_start_time = time.time()
            
            try:
                # Train model
                model.train(X_train, y_train)
                trained_models[model_name] = model
                
                # Calculate training time
                training_time = time.time() - model_start_time
                logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.models = trained_models
        total_training_time = time.time() - start_time
        
        logger.info(f"All models trained in {total_training_time:.2f} seconds")
        
        # Store training history
        self.training_history = {
            'total_time': f"{total_training_time:.2f} seconds",
            'models_trained': list(self.models.keys()),
            'training_timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        logger.info("Evaluating models...")
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            try:
                metrics = model.evaluate(X_test, y_test)
                results[model_name] = metrics
                
                logger.info(f"{model_name} Performance:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def create_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Create ensemble model from trained models"""
        if len(self.models) < 2:
            logger.warning("Need at least 2 models to create ensemble")
            return
        
        logger.info("Creating ensemble model...")
        
        try:
            ensemble = EnsembleModel(self.models, voting='soft')
            ensemble.train(X_train, y_train)
            self.models['Ensemble'] = ensemble
            
            logger.info("Ensemble model created successfully")
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {str(e)}")
    
    def select_best_model(self, metric: str = 'f1') -> str:
        """Select best model based on specified metric"""
        if not self.results:
            raise ValueError("No evaluation results available")
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x][metric])
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} ({metric}: {self.results[best_model_name][metric]:.4f})")
        
        # Update training history
        self.training_history['best_model'] = best_model_name
        self.training_history['best_score'] = self.results[best_model_name][metric]
        
        return best_model_name
    
    def generate_visualizations(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Generate all visualizations"""
        logger.info("Generating visualizations...")
        
        # Model comparison
        self.visualizer.plot_model_comparison(self.results)
        self.interactive_viz.create_model_comparison_dashboard(self.results)
        
        # Individual model visualizations
        for model_name, model in self.models.items():
            try:
                # Feature importance
                importance = model.feature_importance()
                if importance is not None:
                    feature_names = model.feature_names or [f"feature_{i}" for i in range(len(importance))]
                    importance_dict = dict(zip(feature_names, importance))
                    
                    self.visualizer.plot_feature_importance(importance_dict, f"{model_name} Feature Importance")
                    self.interactive_viz.create_feature_importance_plot(importance_dict, f"{model_name} Feature Importance")
                
                # Confusion matrix
                y_pred = model.predict(X_test)
                self.visualizer.plot_confusion_matrix(y_test, y_pred, title=f"{model_name} Confusion Matrix")
                
            except Exception as e:
                logger.warning(f"Could not generate visualizations for {model_name}: {str(e)}")
    
    def generate_model_explanations(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Generate model explanations and interpretability reports"""
        logger.info("Generating model explanations...")
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Generating explanations for {model_name}...")
                
                explainer = ModelExplainer(model, model.feature_names)
                explainer.setup_shap_explainer(X_test[:100])
                explainer.calculate_shap_values(X_test[:50])
                explainer.generate_explanation_report(X_test[:100], y_test[:100])
                
            except Exception as e:
                logger.warning(f"Could not generate explanations for {model_name}: {str(e)}")
    
    def save_models(self) -> None:
        """Save all trained models"""
        logger.info("Saving models...")
        
        for model_name, model in self.models.items():
            try:
                model_path = MODEL_DIR / f"{model_name.lower()}_model.joblib"
                model.save_model(model_path)
                logger.info(f"{model_name} saved to {model_path}")
                
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
        
        # Save best model separately
        if self.best_model:
            best_model_path = MODEL_DIR / "best_model.joblib"
            self.best_model.save_model(best_model_path)
            
            # Save metadata
            metadata = {
                'model_type': type(self.best_model).__name__,
                'performance': self.results,
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_path = MODEL_DIR / "model_metadata.joblib"
            joblib.dump(metadata, metadata_path)
            
            logger.info(f"Best model and metadata saved")
    
    def generate_reports(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Generate comprehensive reports"""
        logger.info("Generating reports...")
        
        # Model performance report
        model_report_path = self.report_generator.create_model_report(
            self.models, 
            (X_test, y_test)
        )
        logger.info(f"Model report generated: {model_report_path}")
        
        # Training report
        training_report_path = self.report_generator.create_training_report(self.training_history)
        logger.info(f"Training report generated: {training_report_path}")
        
        # Export model summary
        summary_path = self.report_generator.export_model_summary(self.models)
        logger.info(f"Model summary exported: {summary_path}")
    
    def run_complete_pipeline(self, data_path: str = None, tune_hyperparams: bool = False,
                            model_types: list = None, create_ensemble: bool = True) -> Dict[str, Any]:
        """Run the complete ML pipeline"""
        logger.info("🚀 Starting complete pump maintenance prediction pipeline...")
        
        pipeline_start_time = time.time()
        
        try:
            # 1. Load and prepare data
            X_train, X_test, y_train, y_test = self.load_and_prepare_data(data_path)
            
            # 2. Initialize models
            self.initialize_models(model_types)
            
            # 3. Hyperparameter tuning (optional)
            if tune_hyperparams:
                tuning_results = self.tune_hyperparameters(X_train, y_train)
                logger.info("Hyperparameter tuning completed")
            
            # 4. Train models
            self.train_models(X_train, y_train)
            
            # 5. Evaluate models
            self.evaluate_models(X_test, y_test)
            
            # 6. Create ensemble (optional)
            if create_ensemble:
                self.create_ensemble_model(X_train, y_train)
                # Re-evaluate with ensemble
                self.evaluate_models(X_test, y_test)
            
            # 7. Select best model
            best_model_name = self.select_best_model()
            
            # 8. Generate visualizations
            self.generate_visualizations(X_test, y_test)
            
            # 9. Generate model explanations
            self.generate_model_explanations(X_test, y_test)
            
            # 10. Save models
            self.save_models()
            
            # 11. Generate reports
            self.generate_reports(X_test, y_test)
            
            pipeline_time = time.time() - pipeline_start_time
            logger.info(f"✅ Pipeline completed successfully in {pipeline_time:.2f} seconds")
            
            return {
                'success': True,
                'best_model': best_model_name,
                'results': self.results,
                'pipeline_time': pipeline_time,
                'models_trained': list(self.models.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'pipeline_time': time.time() - pipeline_start_time
            }

def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description='Pump Maintenance Prediction Pipeline')
    
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--models', nargs='+', default=['random_forest', 'xgboost', 'lightgbm'],
                       help='Models to train')
    parser.add_argument('--no-ensemble', action='store_true', help='Skip ensemble creation')
    parser.add_argument('--quick', action='store_true', help='Quick run (no tuning, basic models)')
    
    args = parser.parse_args()

    # Fix model_types if passed as comma-separated string
    model_types = args.models
    if len(model_types) == 1 and ',' in model_types[0]:
        model_types = model_types[0].split(',')

    # Configure based on arguments
    if args.quick:
        model_types = ['random_forest']
        tune_hyperparams = False
        create_ensemble = False
    else:
        tune_hyperparams = args.tune
        create_ensemble = not args.no_ensemble
    
    # Initialize and run pipeline
    pipeline = PumpPredictorPipeline()
    
    results = pipeline.run_complete_pipeline(
        data_path=args.data,
        tune_hyperparams=tune_hyperparams,
        model_types=model_types,
        create_ensemble=create_ensemble
    )
    
    # Print summary
    if results['success']:
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⭐ Best Model: {results['best_model']}")
        print(f"⏱️  Total Time: {results['pipeline_time']:.2f} seconds")
        print(f"🤖 Models Trained: {', '.join(results['models_trained'])}")
        print("\n📊 Performance Summary:")
        
        for model_name, metrics in results['results'].items():
            print(f"  {model_name}:")
            print(f"    - Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    - F1 Score:  {metrics['f1']:.4f}")
        
        print(f"\n📁 Check reports/ directory for detailed analysis")
        print(f"📁 Check models/ directory for saved models")
        
    else:
        print("\n" + "="*60)
        print("❌ PIPELINE FAILED!")
        print("="*60)
        print(f"Error: {results['error']}")
        print(f"Time: {results['pipeline_time']:.2f} seconds")

if __name__ == "__main__":
    main()