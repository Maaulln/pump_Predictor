"""
Enhanced visualization utilities for pump maintenance prediction
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from pathlib import Path
import warnings
from pump_predictor.utils.logger import get_logger
from pump_predictor.config import REPORT_DIR

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

# Set enhanced styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class ModelVisualizer:
    """Comprehensive model visualization toolkit"""
    
    def __init__(self, save_plots: bool = True, show_plots: bool = True):
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.report_dir = REPORT_DIR / "plots"
        self.report_dir.mkdir(exist_ok=True, parents=True)
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], 
                              title: str = "Feature Importance", 
                              top_n: int = 10) -> None:
        """Plot feature importance with enhanced styling"""
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Horizontal bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        bars = ax1.barh(features, importances, color=colors)
        ax1.set_xlabel('Importance Score')
        ax1.set_title(f'{title} - Top {top_n}')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax1.text(importance + max(importances) * 0.01, i, f'{importance:.3f}', 
                    va='center', fontsize=9)
        
        # Pie chart for top features
        ax2.pie(importances, labels=features, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Feature Importance Distribution - Top {top_n}')
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.report_dir / f"feature_importance_{title.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: List[str] = None,
                            title: str = "Confusion Matrix") -> None:
        """Enhanced confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = ['No Maintenance', 'Maintenance Required']
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title(f'{title} - Counts')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_title(f'{title} - Percentages')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.report_dir / f"confusion_matrix_{title.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """Comprehensive model comparison visualization with enhanced metrics"""
        df = pd.DataFrame(results).T
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Create enhanced figure with subplots
        fig = plt.figure(figsize=(24, 16))
        
        # Color palette for models
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
        model_colors = dict(zip(df.index, colors))
        
        # 1. Individual metric comparison with error bars (if available)
        for i, metric in enumerate(metrics, 1):
            ax = plt.subplot(3, 4, i)
            bars = ax.bar(df.index, df[metric], color=[model_colors[model] for model in df.index],
                         alpha=0.8, edgecolor='black', linewidth=1)
            ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{metric.capitalize()} Score', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Highlight best performing model
            best_idx = df[metric].idxmax()
            best_bar_idx = list(df.index).index(best_idx)
            bars[best_bar_idx].set_edgecolor('gold')
            bars[best_bar_idx].set_linewidth(3)
        
        # 2. Side-by-side comparison
        ax = plt.subplot(3, 4, 5)
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + i*width, df[metric], width, label=metric.capitalize(),
                         alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Side-by-Side Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.1)
        
        # 3. Radar chart with enhanced styling
        ax = plt.subplot(3, 4, 6, projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for model_name in df.index:
            values = df.loc[model_name, metrics].values
            values = np.concatenate((values, [values[0]]))
            ax.plot(angles, values, 'o-', linewidth=3, label=model_name, 
                   markersize=8, alpha=0.8)
            ax.fill(angles, values, alpha=0.2)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        # 4. Ranking heatmap
        ax = plt.subplot(3, 4, 7)
        # Create ranking matrix (1 = best, len(df) = worst)
        ranking_df = df[metrics].rank(ascending=False)
        sns.heatmap(ranking_df, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Rank (1=Best)'}, ax=ax, 
                   linewidths=0.5, linecolor='white')
        ax.set_title('Model Ranking Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Models', fontweight='bold')
        
        # 5. Overall score comparison
        ax = plt.subplot(3, 4, 8)
        overall_scores = df[metrics].mean(axis=1)
        bars = ax.bar(overall_scores.index, overall_scores.values, 
                     color=[model_colors[model] for model in overall_scores.index],
                     alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_title('Overall Performance Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight best model
        best_model = overall_scores.idxmax()
        best_bar_idx = list(overall_scores.index).index(best_model)
        bars[best_bar_idx].set_edgecolor('gold')
        bars[best_bar_idx].set_linewidth(3)
        
        for bar, value in zip(bars, overall_scores.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Model efficiency plot (if training time available)
        if 'training_time' in df.columns:
            ax = plt.subplot(3, 4, 9)
            scatter = ax.scatter(df['training_time'], overall_scores, 
                               c=[model_colors[model] for model in df.index],
                               s=200, alpha=0.7, edgecolors='black', linewidth=1)
            
            for i, model in enumerate(df.index):
                ax.annotate(model, (df.loc[model, 'training_time'], overall_scores[model]),
                           xytext=(5, 5), textcoords='offset points', fontweight='bold')
            
            ax.set_xlabel('Training Time (seconds)', fontweight='bold')
            ax.set_ylabel('Overall Score', fontweight='bold')
            ax.set_title('Efficiency vs Performance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 7. Metric correlation heatmap
        ax = plt.subplot(3, 4, 10)
        metric_corr = df[metrics].T.corr()
        sns.heatmap(metric_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, linewidths=0.5, linecolor='white')
        ax.set_title('Model Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 8. Performance distribution
        ax = plt.subplot(3, 4, 11)
        df_melted = df[metrics].reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
        df_melted.rename(columns={'index': 'Model'}, inplace=True)
        
        box_plot = sns.boxplot(data=df_melted, x='Metric', y='Score', ax=ax, palette='Set2')
        sns.swarmplot(data=df_melted, x='Metric', y='Score', ax=ax, color='black', alpha=0.6, size=4)
        ax.set_title('Score Distribution by Metric', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 9. Winner summary
        ax = plt.subplot(3, 4, 12)
        ax.axis('off')
        
        # Calculate winners for each metric
        winners = {}
        for metric in metrics:
            winners[metric] = df[metric].idxmax()
        
        overall_winner = overall_scores.idxmax()
        
        summary_text = f"""
🏆 PERFORMANCE SUMMARY

Overall Winner: {overall_winner}
Score: {overall_scores[overall_winner]:.3f}

Metric Winners:
• Accuracy: {winners['accuracy']} ({df.loc[winners['accuracy'], 'accuracy']:.3f})
• Precision: {winners['precision']} ({df.loc[winners['precision'], 'precision']:.3f})
• Recall: {winners['recall']} ({df.loc[winners['recall'], 'recall']:.3f})
• F1-Score: {winners['f1']} ({df.loc[winners['f1'], 'f1']:.3f})

Recommendations:
• For balanced performance: {overall_winner}
• For minimizing false positives: {winners['precision']}
• For catching all maintenance cases: {winners['recall']}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Comprehensive Model Performance Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if self.save_plots:
            save_path = self.report_dir / "comprehensive_model_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Comprehensive model comparison plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curves(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, (y_true, y_prob) in models_data.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if self.save_plots:
            save_path = self.report_dir / "roc_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_precision_recall_curves(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """Plot Precision-Recall curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, (y_true, y_prob) in models_data.items():
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = auc(recall, precision)
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'{model_name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        if self.save_plots:
            save_path = self.report_dir / "precision_recall_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curves plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_algorithm_convergence(self, training_history: Dict[str, List[float]]) -> None:
        """Plot training convergence for algorithms that support it"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for idx, (model_name, history) in enumerate(training_history.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            epochs = range(1, len(history) + 1)
            
            ax.plot(epochs, history, color=colors[idx % len(colors)], 
                   linewidth=3, marker='o', markersize=4, alpha=0.8)
            ax.set_title(f'{model_name} - Training Convergence', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Iteration/Epoch', fontweight='bold')
            ax.set_ylabel('Loss/Error', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add trend line
            z = np.polyfit(epochs, history, 1)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), '--', color='red', alpha=0.7, linewidth=2)
            
            # Highlight final performance
            ax.annotate(f'Final: {history[-1]:.4f}', 
                       xy=(epochs[-1], history[-1]), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontweight='bold')
        
        # Remove empty subplots
        for idx in range(len(training_history), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Algorithm Convergence Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.report_dir / "algorithm_convergence.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Algorithm convergence plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_accuracy_progression(self, accuracy_data: Dict[str, Dict[str, List[float]]]) -> None:
        """Plot accuracy progression during training/validation"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training accuracy progression
        ax1 = axes[0]
        for model_name, data in accuracy_data.items():
            if 'train_accuracy' in data:
                epochs = range(1, len(data['train_accuracy']) + 1)
                ax1.plot(epochs, data['train_accuracy'], 
                        label=f'{model_name} (Train)', linewidth=3, alpha=0.8)
        
        ax1.set_title('Training Accuracy Progression', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch/Iteration', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(0, 1)
        
        # Validation accuracy progression
        ax2 = axes[1]
        for model_name, data in accuracy_data.items():
            if 'val_accuracy' in data:
                epochs = range(1, len(data['val_accuracy']) + 1)
                ax2.plot(epochs, data['val_accuracy'], 
                        label=f'{model_name} (Val)', linewidth=3, alpha=0.8)
        
        ax2.set_title('Validation Accuracy Progression', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch/Iteration', fontweight='bold')
        ax2.set_ylabel('Accuracy', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.report_dir / "accuracy_progression.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Accuracy progression plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_hyperparameter_sensitivity(self, param_results: Dict[str, Dict[Any, float]]) -> None:
        """Plot hyperparameter sensitivity analysis"""
        n_params = len(param_results)
        cols = min(3, n_params)
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_params == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        colors = plt.cm.viridis(np.linspace(0, 1, max([len(v) for v in param_results.values()])))
        
        for idx, (param_name, results) in enumerate(param_results.items()):
            row, col = idx // cols, idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            params = list(results.keys())
            scores = list(results.values())
            
            bars = ax.bar(range(len(params)), scores, 
                         color=colors[:len(params)], alpha=0.8, 
                         edgecolor='black', linewidth=1)
            
            ax.set_title(f'{param_name} Sensitivity', fontsize=12, fontweight='bold')
            ax.set_xlabel('Parameter Value', fontweight='bold')
            ax.set_ylabel('Accuracy Score', fontweight='bold')
            ax.set_xticks(range(len(params)))
            ax.set_xticklabels([str(p) for p in params], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Highlight best parameter
            best_idx = np.argmax(scores)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Remove empty subplots
        for idx in range(n_params, rows * cols):
            if rows > 1:
                fig.delaxes(axes[idx // cols, idx % cols])
            else:
                fig.delaxes(axes[idx])
        
        plt.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.report_dir / "hyperparameter_sensitivity.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Hyperparameter sensitivity plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_cross_validation_scores(self, cv_results: Dict[str, List[float]]) -> None:
        """Plot cross-validation scores for model comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot for CV scores
        models = list(cv_results.keys())
        scores = [cv_results[model] for model in models]
        
        box_plot = ax1.boxplot(scores, labels=models, patch_artist=True, 
                              notch=True, showmeans=True)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy Score', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        
        # Mean scores with confidence intervals
        means = [np.mean(scores) for scores in cv_results.values()]
        stds = [np.std(scores) for scores in cv_results.values()]
        
        bars = ax2.bar(models, means, yerr=stds, capsize=5, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax2.set_title('Mean CV Scores with Standard Deviation', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Mean Accuracy Score', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.report_dir / "cross_validation_scores.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Cross-validation scores plot saved to {save_path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()

class InteractiveVisualizer:
    """Enhanced Interactive Plotly visualizations"""
    
    def __init__(self):
        self.report_dir = REPORT_DIR / "interactive_plots"
        self.report_dir.mkdir(exist_ok=True, parents=True)
    
    def create_advanced_model_comparison_dashboard(self, results: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create comprehensive interactive model comparison dashboard"""
        df_results = pd.DataFrame(results).T.reset_index()
        df_results.rename(columns={'index': 'Model'}, inplace=True)
        
        # Create subplots with different types
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Accuracy Comparison', 'Precision vs Recall', 'F1 Score Distribution',
                'Overall Performance', 'Model Ranking', 'ROC Comparison',
                'Performance Radar', 'Score Correlation', 'Training Efficiency'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "scatterpolar"}, {"type": "heatmap"}, {"type": "scatter"}]
            ]
        )
        
        models = df_results['Model'].tolist()
        colors = px.colors.qualitative.Set3[:len(models)]
        
        # 1. Accuracy comparison
        fig.add_trace(
            go.Bar(
                x=models,
                y=df_results['accuracy'],
                name='Accuracy',
                marker_color=colors,
                text=df_results['accuracy'].round(3),
                textposition='auto',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Precision vs Recall scatter
        fig.add_trace(
            go.Scatter(
                x=df_results['precision'],
                y=df_results['recall'],
                mode='markers+text',
                text=models,
                textposition='top center',
                marker=dict(size=15, color=colors, line=dict(width=2, color='black')),
                name='Precision vs Recall',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. F1 Score distribution (simulating box plot with scatter)
        for i, model in enumerate(models):
            fig.add_trace(
                go.Box(
                    y=[df_results.loc[i, 'f1']],
                    name=model,
                    marker_color=colors[i],
                    showlegend=False
                ),
                row=1, col=3
            )
        
        # 4. Overall performance (mean of all metrics)
        overall_scores = df_results[['accuracy', 'precision', 'recall', 'f1']].mean(axis=1)
        fig.add_trace(
            go.Bar(
                x=models,
                y=overall_scores,
                name='Overall Score',
                marker_color=colors,
                text=overall_scores.round(3),
                textposition='auto',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 5. Model ranking heatmap
        ranking_data = df_results[['accuracy', 'precision', 'recall', 'f1']].rank(ascending=False)
        fig.add_trace(
            go.Heatmap(
                z=ranking_data.values,
                x=['Accuracy', 'Precision', 'Recall', 'F1'],
                y=models,
                colorscale='RdYlGn_r',
                text=ranking_data.values,
                texttemplate='%{text:.0f}',
                showscale=False
            ),
            row=2, col=2
        )
        
        # 6. ROC comparison (placeholder - would need actual ROC data)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Random Classifier',
                showlegend=False
            ),
            row=2, col=3
        )
        
        # 7. Performance radar chart
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for i, model in enumerate(models):
            values = df_results.loc[i, metrics].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=model,
                    line_color=colors[i],
                    fillcolor=colors[i],
                    opacity=0.6
                ),
                row=3, col=1
            )
        
        # 8. Score correlation matrix
        corr_matrix = df_results[['accuracy', 'precision', 'recall', 'f1']].corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                showscale=False
            ),
            row=3, col=2
        )
        
        # 9. Training efficiency (if training time data available)
        if 'training_time' in df_results.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_results['training_time'],
                    y=overall_scores,
                    mode='markers+text',
                    text=models,
                    textposition='top center',
                    marker=dict(size=15, color=colors, line=dict(width=2, color='black')),
                    name='Efficiency vs Performance',
                    showlegend=False
                ),
                row=3, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="🚀 Advanced Model Performance Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        
        fig.update_xaxes(title_text="Precision", row=1, col=2)
        fig.update_yaxes(title_text="Recall", row=1, col=2)
        
        fig.update_xaxes(title_text="Models", row=2, col=1)
        fig.update_yaxes(title_text="Overall Score", row=2, col=1)
        
        if 'training_time' in df_results.columns:
            fig.update_xaxes(title_text="Training Time (s)", row=3, col=3)
            fig.update_yaxes(title_text="Performance", row=3, col=3)
        
        # Save interactive plot
        save_path = self.report_dir / "advanced_model_dashboard.html"
        fig.write_html(save_path)
        logger.info(f"Advanced interactive dashboard saved to {save_path}")
        
        return fig
    
    def create_algorithm_performance_timeline(self, results_history: Dict[str, List[Dict[str, float]]]) -> go.Figure:
        """Create timeline showing algorithm performance over training iterations"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (model_name, history) in enumerate(results_history.items()):
            iterations = list(range(1, len(history) + 1))
            accuracies = [h.get('accuracy', 0) for h in history]
            
            fig.add_trace(go.Scatter(
                x=iterations,
                y=accuracies,
                mode='lines+markers',
                name=f'{model_name} Accuracy',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>{model_name}</b><br>Iteration: %{{x}}<br>Accuracy: %{{y:.3f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='🎯 Algorithm Performance Timeline',
            xaxis_title='Training Iteration',
            yaxis_title='Accuracy Score',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        # Save interactive plot
        save_path = self.report_dir / "algorithm_performance_timeline.html"
        fig.write_html(save_path)
        logger.info(f"Algorithm performance timeline saved to {save_path}")
        
        return fig
    
    def create_feature_importance_plot(self, importance_dict: Dict[str, float], 
                                     title: str = "Feature Importance") -> go.Figure:
        """Create interactive feature importance plot"""
        df_importance = pd.DataFrame(list(importance_dict.items()), 
                                   columns=['Feature', 'Importance'])
        df_importance = df_importance.sort_values('Importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=df_importance['Importance'],
            y=df_importance['Feature'],
            orientation='h',
            marker=dict(
                color=df_importance['Importance'],
                colorscale='Viridis',
                colorbar=dict(title="Importance Score")
            ),
            text=df_importance['Importance'].round(3),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Interactive {title}",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(df_importance) * 25)
        )
        
        # Save interactive plot
        save_path = self.report_dir / f"feature_importance_{title.lower().replace(' ', '_')}.html"
        fig.write_html(save_path)
        logger.info(f"Interactive feature importance plot saved to {save_path}")
        
        return fig

# Convenience functions for backward compatibility
def plot_feature_importance(importance_dict: Dict[str, float], title: str = "Feature Importance", 
                          top_n: int = 10, save_path: Optional[str] = None):
    """Plot feature importance (backward compatibility)"""
    visualizer = ModelVisualizer(save_plots=save_path is not None)
    visualizer.plot_feature_importance(importance_dict, title, top_n)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = "Confusion Matrix", save_path: Optional[str] = None):
    """Plot confusion matrix (backward compatibility)"""
    visualizer = ModelVisualizer(save_plots=save_path is not None)
    visualizer.plot_confusion_matrix(y_true, y_pred, title=title)

def plot_model_comparison(results: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
    """Plot model comparison (backward compatibility)"""
    visualizer = ModelVisualizer(save_plots=save_path is not None)
    visualizer.plot_model_comparison(results)

# New enhanced convenience functions
def create_comprehensive_analysis(results: Dict[str, Dict[str, float]], 
                                training_history: Optional[Dict[str, List[float]]] = None,
                                cv_results: Optional[Dict[str, List[float]]] = None,
                                param_sensitivity: Optional[Dict[str, Dict[Any, float]]] = None,
                                save_plots: bool = True, show_plots: bool = True) -> None:
    """Create comprehensive analysis with all available visualizations"""
    
    visualizer = ModelVisualizer(save_plots=save_plots, show_plots=show_plots)
    interactive_viz = InteractiveVisualizer()
    
    # Basic model comparison
    logger.info("Creating comprehensive model comparison...")
    visualizer.plot_model_comparison(results)
    
    # Interactive dashboard
    logger.info("Creating interactive model dashboard...")
    interactive_viz.create_advanced_model_comparison_dashboard(results)
    
    # Training history analysis
    if training_history:
        logger.info("Creating algorithm convergence analysis...")
        visualizer.plot_algorithm_convergence(training_history)
        interactive_viz.create_algorithm_performance_timeline(
            {model: [{'accuracy': acc} for acc in hist] for model, hist in training_history.items()}
        )
    
    # Cross-validation analysis
    if cv_results:
        logger.info("Creating cross-validation analysis...")
        visualizer.plot_cross_validation_scores(cv_results)
    
    # Hyperparameter sensitivity
    if param_sensitivity:
        logger.info("Creating hyperparameter sensitivity analysis...")
        visualizer.plot_hyperparameter_sensitivity(param_sensitivity)
    
    logger.info("✅ Comprehensive analysis complete!")

def compare_algorithms_detailed(results: Dict[str, Dict[str, float]], 
                              model_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
                              save_plots: bool = True, show_plots: bool = True) -> Dict[str, Any]:
    """Detailed algorithm comparison with statistical analysis"""
    
    visualizer = ModelVisualizer(save_plots=save_plots, show_plots=show_plots)
    
    # Create comprehensive comparison
    visualizer.plot_model_comparison(results)
    
    # ROC and Precision-Recall curves if model data available
    if model_data:
        visualizer.plot_roc_curves(model_data)
        visualizer.plot_precision_recall_curves(model_data)
    
    # Statistical analysis
    df = pd.DataFrame(results).T
    analysis = {
        'best_overall': df.mean(axis=1).idxmax(),
        'best_accuracy': df['accuracy'].idxmax(),
        'best_precision': df['precision'].idxmax(),
        'best_recall': df['recall'].idxmax(),
        'best_f1': df['f1'].idxmax(),
        'performance_std': df.std(axis=1).to_dict(),
        'performance_range': (df.max(axis=1) - df.min(axis=1)).to_dict(),
        'rankings': df.rank(ascending=False).to_dict()
    }
    
    logger.info(f"🏆 Best overall performer: {analysis['best_overall']}")
    logger.info(f"📊 Performance analysis complete!")
    
    return analysis