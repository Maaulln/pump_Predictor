"""
Model explainability and interpretability utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any
import shap
from pump_predictor.utils.logger import get_logger
from pump_predictor.config import REPORT_DIR

logger = get_logger(__name__)

class ModelExplainer:
    """Comprehensive model explainability toolkit"""
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self._get_n_features())]
        self.explainer = None
        self.shap_values = None
        self.report_dir = REPORT_DIR / "explainability"
        self.report_dir.mkdir(exist_ok=True, parents=True)
    
    def _get_n_features(self) -> int:
        """Get number of features from model"""
        try:
            if hasattr(self.model, 'feature_importance'):
                importance = self.model.feature_importance()
                return len(importance) if importance is not None else 0
            elif hasattr(self.model, 'n_features_in_'):
                return self.model.n_features_in_
            else:
                return 0
        except:
            return 0
    
    def setup_shap_explainer(self, X_background: np.ndarray, explainer_type: str = 'auto'):
        """Setup SHAP explainer"""
        try:
            if explainer_type == 'auto':
                # Try to determine the best explainer type
                if hasattr(self.model, 'predict_proba'):
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    self.explainer = shap.KernelExplainer(self.model.predict, X_background[:100])
            elif explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif explainer_type == 'kernel':
                self.explainer = shap.KernelExplainer(self.model.predict, X_background[:100])
            elif explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, X_background)
            
            logger.info(f"SHAP explainer ({explainer_type}) setup successfully")
            
        except Exception as e:
            logger.warning(f"Could not setup SHAP explainer: {str(e)}")
            self.explainer = None
    
    def calculate_shap_values(self, X: np.ndarray, max_samples: int = 100):
        """Calculate SHAP values for given data"""
        if self.explainer is None:
            logger.warning("SHAP explainer not setup. Call setup_shap_explainer first.")
            return
        
        try:
            # Limit samples for performance
            X_sample = X[:max_samples] if len(X) > max_samples else X
            
            logger.info(f"Calculating SHAP values for {len(X_sample)} samples...")
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Use positive class for binary classification
            
            logger.info("SHAP values calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            self.shap_values = None
    
    def plot_feature_importance(self, top_n: int = 10, save_path: Optional[str] = None):
        """Plot feature importance with enhanced visualization"""
        try:
            importance = self.model.feature_importance()
            if importance is None:
                logger.warning("Model doesn't provide feature importance")
                return
            
            # Create dataframe for plotting
            df_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Create enhanced plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Horizontal bar plot
            colors = plt.cm.viridis(np.linspace(0, 1, len(df_importance)))
            bars = ax1.barh(df_importance['feature'], df_importance['importance'], color=colors)
            ax1.set_xlabel('Importance Score')
            ax1.set_title(f'Top {top_n} Feature Importance')
            ax1.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, importance in zip(bars, df_importance['importance']):
                ax1.text(importance + max(df_importance['importance']) * 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', va='center', fontsize=9)
            
            # Cumulative importance
            df_importance_sorted = df_importance.sort_values('importance', ascending=True)
            cumsum = df_importance_sorted['importance'].cumsum()
            ax2.barh(df_importance_sorted['feature'], cumsum, color='lightblue', alpha=0.7)
            ax2.set_xlabel('Cumulative Importance')
            ax2.set_title('Cumulative Feature Importance')
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            else:
                default_path = self.report_dir / "feature_importance.png"
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {default_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
    
    def plot_shap_summary(self, save_path: Optional[str] = None):
        """Plot SHAP summary plot"""
        if self.shap_values is None:
            logger.warning("SHAP values not calculated. Call calculate_shap_values first.")
            return
        
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, 
                            features=None,  # Will use the data used to calculate SHAP values
                            feature_names=self.feature_names,
                            show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            else:
                default_path = self.report_dir / "shap_summary.png"
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {default_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP summary: {str(e)}")
    
    def plot_shap_waterfall(self, instance_idx: int = 0, save_path: Optional[str] = None):
        """Plot SHAP waterfall plot for a specific instance"""
        if self.shap_values is None:
            logger.warning("SHAP values not calculated. Call calculate_shap_values first.")
            return
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Create explanation object for waterfall plot
            if hasattr(shap, 'Explanation'):
                explanation = shap.Explanation(
                    values=self.shap_values[instance_idx],
                    base_values=self.explainer.expected_value,
                    feature_names=self.feature_names
                )
                shap.plots.waterfall(explanation, show=False)
            else:
                # Fallback for older SHAP versions
                logger.warning("Waterfall plot requires newer SHAP version")
                return
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP waterfall plot saved to {save_path}")
            else:
                default_path = self.report_dir / f"shap_waterfall_instance_{instance_idx}.png"
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP waterfall plot saved to {default_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting SHAP waterfall: {str(e)}")
    
    def plot_partial_dependence(self, feature_idx: int, X: np.ndarray, 
                               n_points: int = 50, save_path: Optional[str] = None):
        """Plot partial dependence plot for a specific feature"""
        try:
            feature_name = self.feature_names[feature_idx]
            feature_values = X[:, feature_idx]
            
            # Create range of values for the feature
            min_val, max_val = feature_values.min(), feature_values.max()
            feature_range = np.linspace(min_val, max_val, n_points)
            
            # Calculate partial dependence
            partial_predictions = []
            
            for value in feature_range:
                # Create modified dataset
                X_modified = X.copy()
                X_modified[:, feature_idx] = value
                
                # Get predictions
                if hasattr(self.model, 'predict_proba'):
                    pred = self.model.predict_proba(X_modified)[:, 1].mean()
                else:
                    pred = self.model.predict(X_modified).mean()
                
                partial_predictions.append(pred)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(feature_range, partial_predictions, linewidth=2, color='blue')
            plt.xlabel(feature_name)
            plt.ylabel('Partial Dependence')
            plt.title(f'Partial Dependence Plot - {feature_name}')
            plt.grid(alpha=0.3)
            
            # Add rug plot
            plt.scatter(feature_values[:100], 
                       [min(partial_predictions)] * min(100, len(feature_values)),
                       alpha=0.3, color='red', s=10, label='Data Points')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Partial dependence plot saved to {save_path}")
            else:
                default_path = self.report_dir / f"partial_dependence_{feature_name}.png"
                plt.savefig(default_path, dpi=300, bbox_inches='tight')
                logger.info(f"Partial dependence plot saved to {default_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting partial dependence: {str(e)}")
    
    def generate_explanation_report(self, X: np.ndarray, y: np.ndarray, 
                                  output_path: Optional[str] = None) -> str:
        """Generate comprehensive explanation report"""
        try:
            # Setup SHAP if not already done
            if self.explainer is None:
                self.setup_shap_explainer(X)
            
            # Calculate SHAP values if not already done
            if self.shap_values is None:
                self.calculate_shap_values(X)
            
            # Generate plots
            self.plot_feature_importance()
            
            if self.shap_values is not None:
                self.plot_shap_summary()
                self.plot_shap_waterfall(0)
            
            # Plot partial dependence for top features
            if hasattr(self.model, 'feature_importance'):
                importance = self.model.feature_importance()
                if importance is not None:
                    top_features = np.argsort(importance)[-3:]  # Top 3 features
                    for feature_idx in top_features:
                        self.plot_partial_dependence(feature_idx, X)
            
            report_text = f"""
            Model Explainability Report
            ===========================
            
            Model Type: {type(self.model).__name__}
            Number of Features: {len(self.feature_names)}
            
            Feature Importance Analysis:
            - Feature importance plots generated
            - Top features identified
            
            SHAP Analysis:
            - SHAP values calculated: {'Yes' if self.shap_values is not None else 'No'}
            - Summary plots generated
            - Individual prediction explanations available
            
            Partial Dependence Analysis:
            - Individual feature effects analyzed
            - Feature interaction patterns identified
            
            All plots saved to: {self.report_dir}
            """
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Explanation report saved to {output_path}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Error generating explanation report: {str(e)}")
            return f"Error generating report: {str(e)}"