"""
Comprehensive report generation utilities
"""
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from jinja2 import Template
from pump_predictor.utils.logger import get_logger
from pump_predictor.utils.visualization import ModelVisualizer, InteractiveVisualizer
from pump_predictor.utils.model_explainer import ModelExplainer
from pump_predictor.config import REPORT_DIR

logger = get_logger(__name__)

class ReportGenerator:
    """Comprehensive model performance and analysis report generator"""
    
    def __init__(self):
        self.report_dir = REPORT_DIR
        self.report_dir.mkdir(exist_ok=True, parents=True)
        self.visualizer = ModelVisualizer(save_plots=True, show_plots=False)
        self.interactive_viz = InteractiveVisualizer()
    
    def create_model_report(self, models: Dict[str, Any], test_data: Tuple[np.ndarray, np.ndarray],
                          output_path: Optional[str] = None) -> str:
        """Create comprehensive model performance report"""
        X_test, y_test = test_data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_path = self.report_dir / f"model_report_{timestamp}.html"
        
        logger.info("Generating comprehensive model report...")
        
        # Collect model performance data
        results = {}
        model_details = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Basic metrics
            metrics = model.evaluate(X_test, y_test)
            results[name] = metrics
            
            # Model details
            details = {
                'type': type(model).__name__,
                'parameters': model.model_params,
                'is_trained': model.is_trained,
                'feature_names': model.feature_names
            }
            
            # Feature importance if available
            try:
                importance = model.feature_importance()
                if importance is not None:
                    details['feature_importance'] = dict(zip(
                        model.feature_names or [f"feature_{i}" for i in range(len(importance))],
                        importance.tolist()
                    ))
            except:
                details['feature_importance'] = None
            
            model_details[name] = details
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        self._generate_report_visualizations(models, results, X_test, y_test)
        
        # Create HTML report
        html_content = self._create_html_report(results, model_details, timestamp)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Model report saved to {output_path}")
        return str(output_path)
    
    def create_data_report(self, data: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """Create data analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_path = self.report_dir / f"data_report_{timestamp}.html"
        
        logger.info("Generating data analysis report...")
        
        # Data statistics
        data_stats = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'duplicates': data.duplicated().sum(),
            'memory_usage': data.memory_usage(deep=True).sum(),
        }
        
        # Numerical columns statistics
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            data_stats['numerical_stats'] = data[numerical_cols].describe().to_dict()
        
        # Categorical columns statistics
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            data_stats['categorical_stats'] = {
                col: data[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            }
        
        # Generate data visualizations
        self._generate_data_visualizations(data)
        
        # Create HTML report
        html_content = self._create_data_html_report(data_stats, timestamp)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Data report saved to {output_path}")
        return str(output_path)
    
    def create_training_report(self, training_history: Dict[str, Any], 
                             output_path: Optional[str] = None) -> str:
        """Create training process report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_path = self.report_dir / f"training_report_{timestamp}.html"
        
        logger.info("Generating training report...")
        
        # Create HTML report
        html_content = self._create_training_html_report(training_history, timestamp)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Training report saved to {output_path}")
        return str(output_path)
    
    def _generate_report_visualizations(self, models: Dict[str, Any], results: Dict[str, Dict[str, float]],
                                      X_test: np.ndarray, y_test: np.ndarray):
        """Generate all visualizations for the report"""
        
        # Model comparison
        self.visualizer.plot_model_comparison(results)
        
        # Interactive comparison
        self.interactive_viz.create_model_comparison_dashboard(results)
        
        # Individual model visualizations
        for name, model in models.items():
            try:
                # Predictions
                y_pred = model.predict(X_test)
                
                # Confusion matrix
                self.visualizer.plot_confusion_matrix(y_test, y_pred, title=f"{name} Confusion Matrix")
                
                # Feature importance
                try:
                    importance = model.feature_importance()
                    if importance is not None:
                        feature_names = model.feature_names or [f"feature_{i}" for i in range(len(importance))]
                        importance_dict = dict(zip(feature_names, importance))
                        self.visualizer.plot_feature_importance(importance_dict, f"{name} Feature Importance")
                        self.interactive_viz.create_feature_importance_plot(importance_dict, f"{name} Feature Importance")
                except:
                    logger.warning(f"Could not generate feature importance for {name}")
                
                # Model explainability
                try:
                    explainer = ModelExplainer(model, model.feature_names)
                    explainer.setup_shap_explainer(X_test[:100])
                    explainer.calculate_shap_values(X_test[:50])
                    explainer.plot_feature_importance()
                except Exception as e:
                    logger.warning(f"Could not generate explainability plots for {name}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error generating visualizations for {name}: {str(e)}")
    
    def _generate_data_visualizations(self, data: pd.DataFrame):
        """Generate data analysis visualizations"""
        try:
            # Data overview
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Missing values heatmap
            if data.isnull().sum().sum() > 0:
                sns.heatmap(data.isnull(), cbar=True, ax=axes[0, 0])
                axes[0, 0].set_title('Missing Values Heatmap')
            else:
                axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
                axes[0, 0].set_title('Missing Values Status')
            
            # Data types distribution
            dtype_counts = data.dtypes.value_counts()
            axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Data Types Distribution')
            
            # Numerical columns correlation
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                correlation_matrix = data[numerical_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
                axes[1, 0].set_title('Correlation Matrix')
            else:
                axes[1, 0].text(0.5, 0.5, 'Insufficient numerical columns', ha='center', va='center')
                axes[1, 0].set_title('Correlation Analysis')
            
            # Target distribution (if exists)
            if 'target' in data.columns or 'maintenance_needed' in data.columns:
                target_col = 'target' if 'target' in data.columns else 'maintenance_needed'
                data[target_col].value_counts().plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Target Distribution')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'No target column found', ha='center', va='center')
                axes[1, 1].set_title('Target Analysis')
            
            plt.tight_layout()
            save_path = self.report_dir / "plots" / "data_overview.png"
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating data visualizations: {str(e)}")
    
    def _create_html_report(self, results: Dict[str, Dict[str, float]], 
                          model_details: Dict[str, Dict], timestamp: str) -> str:
        """Create HTML report content"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pump Maintenance Prediction - Model Report</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin: 30px 0; }
                .metric-table { border-collapse: collapse; width: 100%; }
                .metric-table th, .metric-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .metric-table th { background-color: #f2f2f2; }
                .best-score { background-color: #d4edda; }
                .model-details { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .timestamp { color: #666; font-size: 0.9em; }
                .summary-box { background-color: #e7f3ff; padding: 20px; border-left: 4px solid #007bff; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔧 Pump Maintenance Prediction Model Report</h1>
                <p class="timestamp">Generated on: {{ timestamp }}</p>
            </div>
            
            <div class="section">
                <h2>📊 Model Performance Summary</h2>
                <div class="summary-box">
                    <h3>Best Performing Model: {{ best_model }}</h3>
                    <p>F1 Score: {{ best_f1 }}</p>
                    <p>Accuracy: {{ best_accuracy }}</p>
                </div>
                
                <table class="metric-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for model, metrics in results.items() %}
                        <tr {% if model == best_model %}class="best-score"{% endif %}>
                            <td>{{ model }}</td>
                            <td>{{ "%.4f"|format(metrics.accuracy) }}</td>
                            <td>{{ "%.4f"|format(metrics.precision) }}</td>
                            <td>{{ "%.4f"|format(metrics.recall) }}</td>
                            <td>{{ "%.4f"|format(metrics.f1) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>🔍 Model Details</h2>
                {% for model_name, details in model_details.items() %}
                <div class="model-details">
                    <h3>{{ model_name }}</h3>
                    <p><strong>Type:</strong> {{ details.type }}</p>
                    <p><strong>Trained:</strong> {{ details.is_trained }}</p>
                    <p><strong>Features:</strong> {{ details.feature_names|length if details.feature_names else 'N/A' }}</p>
                    {% if details.feature_importance %}
                    <h4>Top Features:</h4>
                    <ul>
                        {% for feature, importance in (details.feature_importance.items()|list)[:5] %}
                        <li>{{ feature }}: {{ "%.4f"|format(importance) }}</li>
                        {% endfor %}
                    </ul>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>📈 Visualizations</h2>
                <p>Generated plots can be found in the reports/plots directory:</p>
                <ul>
                    <li>Model comparison charts</li>
                    <li>Confusion matrices</li>
                    <li>Feature importance plots</li>
                    <li>ROC curves</li>
                    <li>Interactive dashboards</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                <ul>
                    <li>Best model for deployment: <strong>{{ best_model }}</strong></li>
                    <li>Consider ensemble methods if performance gaps are small</li>
                    <li>Monitor model performance in production</li>
                    <li>Retrain periodically with new data</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        best_f1 = results[best_model]['f1']
        best_accuracy = results[best_model]['accuracy']
        
        template = Template(html_template)
        return template.render(
            timestamp=timestamp,
            results=results,
            model_details=model_details,
            best_model=best_model,
            best_f1=f"{best_f1:.4f}",
            best_accuracy=f"{best_accuracy:.4f}"
        )
    
    def _create_data_html_report(self, data_stats: Dict[str, Any], timestamp: str) -> str:
        """Create data analysis HTML report"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pump Maintenance Prediction - Data Report</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin: 30px 0; }
                .stats-table { border-collapse: collapse; width: 100%; }
                .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .stats-table th { background-color: #f2f2f2; }
                .info-box { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .timestamp { color: #666; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Data Analysis Report</h1>
                <p class="timestamp">Generated on: {{ timestamp }}</p>
            </div>
            
            <div class="section">
                <h2>📋 Dataset Overview</h2>
                <div class="info-box">
                    <p><strong>Shape:</strong> {{ data_stats.shape[0] }} rows × {{ data_stats.shape[1] }} columns</p>
                    <p><strong>Memory Usage:</strong> {{ "%.2f"|format(data_stats.memory_usage / 1024 / 1024) }} MB</p>
                    <p><strong>Missing Values:</strong> {{ data_stats.missing_values.values()|sum }}</p>
                    <p><strong>Duplicate Rows:</strong> {{ data_stats.duplicates }}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>📝 Column Information</h2>
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Data Type</th>
                            <th>Missing Values</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for col in data_stats.columns %}
                        <tr>
                            <td>{{ col }}</td>
                            <td>{{ data_stats.dtypes[col] }}</td>
                            <td>{{ data_stats.missing_values[col] }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            {% if data_stats.numerical_stats %}
            <div class="section">
                <h2>🔢 Numerical Statistics</h2>
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Statistic</th>
                            {% for col in data_stats.numerical_stats.keys() %}
                            <th>{{ col }}</th>
                            {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                        {% for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'] %}
                        <tr>
                            <td>{{ stat }}</td>
                            {% for col in data_stats.numerical_stats.keys() %}
                            <td>{{ "%.4f"|format(data_stats.numerical_stats[col][stat]) if data_stats.numerical_stats[col][stat] is not none else 'N/A' }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
            
            <div class="section">
                <h2>📈 Data Quality Assessment</h2>
                <div class="info-box">
                    <h3>Quality Score: {{ quality_score }}/10</h3>
                    <p>{{ quality_assessment }}</p>
                </div>
            </div>
            
        </body>
        </html>
        """
        
        # Calculate data quality score
        quality_score = 10
        quality_issues = []
        
        # Check for missing values
        missing_ratio = sum(data_stats['missing_values'].values()) / (data_stats['shape'][0] * data_stats['shape'][1])
        if missing_ratio > 0.1:
            quality_score -= 2
            quality_issues.append("High missing value ratio")
        elif missing_ratio > 0.05:
            quality_score -= 1
            quality_issues.append("Moderate missing value ratio")
        
        # Check for duplicates
        duplicate_ratio = data_stats['duplicates'] / data_stats['shape'][0]
        if duplicate_ratio > 0.1:
            quality_score -= 2
            quality_issues.append("High duplicate ratio")
        elif duplicate_ratio > 0.05:
            quality_score -= 1
            quality_issues.append("Moderate duplicate ratio")
        
        quality_assessment = "Good data quality" if quality_score >= 8 else f"Issues found: {', '.join(quality_issues)}"
        
        template = Template(html_template)
        return template.render(
            timestamp=timestamp,
            data_stats=data_stats,
            quality_score=quality_score,
            quality_assessment=quality_assessment
        )
    
    def _create_training_html_report(self, training_history: Dict[str, Any], timestamp: str) -> str:
        """Create training process HTML report"""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Process Report</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin: 30px 0; }
                .info-box { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .timestamp { color: #666; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Training Process Report</h1>
                <p class="timestamp">Generated on: {{ timestamp }}</p>
            </div>
            
            <div class="section">
                <h2>⏱️ Training Summary</h2>
                <div class="info-box">
                    <p><strong>Total Training Time:</strong> {{ training_history.get('total_time', 'N/A') }}</p>
                    <p><strong>Models Trained:</strong> {{ training_history.get('models_trained', 'N/A') }}</p>
                    <p><strong>Best Model:</strong> {{ training_history.get('best_model', 'N/A') }}</p>
                </div>
            </div>
            
        </body>
        </html>
        """
        
        template = Template(html_template)
        return template.render(
            timestamp=timestamp,
            training_history=training_history
        )
    
    def export_model_summary(self, models: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Export model summary to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_path = self.report_dir / f"model_summary_{timestamp}.json"
        
        summary = {
            'timestamp': timestamp,
            'models': {}
        }
        
        for name, model in models.items():
            model_summary = {
                'type': type(model).__name__,
                'parameters': model.model_params,
                'is_trained': model.is_trained,
                'feature_names': model.feature_names
            }
            
            # Add feature importance if available
            try:
                importance = model.feature_importance()
                if importance is not None:
                    model_summary['feature_importance'] = dict(zip(
                        model.feature_names or [f"feature_{i}" for i in range(len(importance))],
                        importance.tolist()
                    ))
            except:
                model_summary['feature_importance'] = None
            
            summary['models'][name] = model_summary
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Model summary exported to {output_path}")
        return str(output_path)