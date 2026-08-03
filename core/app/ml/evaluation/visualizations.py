"""
Visualization utilities for model evaluation.
"""

from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

from ..common.utils import ensure_directory
from ..common.exceptions import EvaluationError


class Visualizer:
    """
    Visualization utilities for model evaluation.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.style = 'seaborn-v0_8-darkgrid'
        plt.style.use(self.style)
        
    def plot_confusion_matrix(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        labels: Optional[List] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None,
        normalize: bool = False,
        **kwargs
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title: Plot title
            save_path: Path to save figure
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Matplotlib figure
        """
        try:
            from sklearn.metrics import confusion_matrix
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
            else:
                fmt = 'd'
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            sns.heatmap(
                cm,
                annot=True,
                fmt=fmt,
                cmap='Blues',
                xticklabels=labels or np.unique(y_true),
                yticklabels=labels or np.unique(y_true),
                ax=ax,
                cbar_kws={'label': 'Normalized' if normalize else 'Count'}
            )
            
            ax.set_title(title)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot confusion matrix: {str(e)}") from e
    
    def plot_roc_curve(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_proba: Union[np.ndarray, pd.Series],
        title: str = "ROC Curve",
        save_path: Optional[Path] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        try:
            from sklearn.metrics import roc_curve, auc
            
            # Convert to numpy arrays
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            if isinstance(y_proba, pd.Series):
                y_proba = y_proba.values
            
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Binary classification
            if y_proba.ndim == 1:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
            
            # Multiclass - one-vs-rest
            elif y_proba.ndim == 2:
                n_classes = y_proba.shape[1]
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(
                        (y_true == i).astype(int),
                        y_proba[:, i]
                    )
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(title)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"ROC curve saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot ROC curve: {str(e)}") from e
    
    def plot_precision_recall_curve(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_proba: Union[np.ndarray, pd.Series],
        title: str = "Precision-Recall Curve",
        save_path: Optional[Path] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            # Convert to numpy arrays
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            if isinstance(y_proba, pd.Series):
                y_proba = y_proba.values
            
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Binary classification
            if y_proba.ndim == 1:
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                ap_score = average_precision_score(y_true, y_proba)
                ax.plot(recall, precision, label=f'PR (AP = {ap_score:.3f})')
            
            # Multiclass - one-vs-rest
            elif y_proba.ndim == 2:
                n_classes = y_proba.shape[1]
                for i in range(n_classes):
                    precision, recall, _ = precision_recall_curve(
                        (y_true == i).astype(int),
                        y_proba[:, i]
                    )
                    ap_score = average_precision_score(
                        (y_true == i).astype(int),
                        y_proba[:, i]
                    )
                    ax.plot(recall, precision, label=f'Class {i} (AP = {ap_score:.3f})')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(title)
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Precision-Recall curve saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot precision-recall curve: {str(e)}") from e
    
    def plot_feature_importance(
        self,
        feature_importance: Union[np.ndarray, Dict[str, float]],
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[Path] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Feature importance scores
            feature_names: Feature names
            top_n: Number of top features to display
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        try:
            # Convert to dataframe
            if isinstance(feature_importance, np.ndarray):
                if feature_names is None:
                    feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                })
            elif isinstance(feature_importance, dict):
                importance_df = pd.DataFrame([
                    {'feature': k, 'importance': v}
                    for k, v in feature_importance.items()
                ])
            else:
                raise ValueError("feature_importance must be ndarray or dict")
            
            # Sort and take top N
            importance_df = importance_df.sort_values('importance', ascending=False)
            if len(importance_df) > top_n:
                importance_df = importance_df.head(top_n)
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            sns.barplot(
                data=importance_df,
                x='importance',
                y='feature',
                palette='viridis_r',
                ax=ax
            )
            
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot feature importance: {str(e)}") from e
    
    def plot_residuals(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        title: str = "Residual Analysis",
        save_path: Optional[Path] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot residual analysis for regression.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        try:
            residuals = np.array(y_true) - np.array(y_pred)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.dpi)
            
            # Residuals vs Predicted
            axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
            axes[0, 0].axhline(y=0, color='r', linestyle='--')
            axes[0, 0].set_xlabel('Predicted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist='norm', plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Histogram of residuals
            axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 0].axvline(x=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Histogram of Residuals')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Residuals order plot
            axes[1, 1].plot(residuals, 'o', alpha=0.5)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Observation Order')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals by Order')
            axes[1, 1].grid(True, alpha=0.3)
            
            fig.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Residual plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot residuals: {str(e)}") from e
    
    def plot_learning_curve(
        self,
        train_scores: Union[List[float], np.ndarray],
        val_scores: Union[List[float], np.ndarray],
        n_points: Optional[int] = None,
        title: str = "Learning Curve",
        save_path: Optional[Path] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot learning curve.
        
        Args:
            train_scores: Training scores
            val_scores: Validation scores
            n_points: Number of points to plot (for x-axis)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        try:
            if n_points is None:
                n_points = len(train_scores)
            
            x_axis = np.linspace(0.1, 1.0, n_points)
            
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            ax.plot(x_axis, train_scores, label='Training Score', linewidth=2)
            ax.plot(x_axis, val_scores, label='Validation Score', linewidth=2)
            
            ax.fill_between(
                x_axis,
                np.array(train_scores) - np.std(train_scores),
                np.array(train_scores) + np.std(train_scores),
                alpha=0.2
            )
            ax.fill_between(
                x_axis,
                np.array(val_scores) - np.std(val_scores),
                np.array(val_scores) + np.std(val_scores),
                alpha=0.2
            )
            
            ax.set_xlabel('Training Set Proportion')
            ax.set_ylabel('Score')
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Learning curve saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot learning curve: {str(e)}") from e
    
    def plot_actual_vs_predicted(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        title: str = "Actual vs Predicted",
        save_path: Optional[Path] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot actual vs predicted values for regression.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            ax.scatter(y_true, y_pred, alpha=0.5)
            
            # Perfect prediction line
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Actual vs predicted plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot actual vs predicted: {str(e)}") from e
    
    def plot_calibration_curve(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_proba: Union[np.ndarray, pd.Series],
        n_bins: int = 10,
        title: str = "Calibration Curve",
        save_path: Optional[Path] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot calibration curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        try:
            from sklearn.calibration import calibration_curve
            
            # Convert to numpy arrays
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            if isinstance(y_proba, pd.Series):
                y_proba = y_proba.values
            
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Binary classification
            if y_proba.ndim == 1:
                prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
                ax.plot(prob_pred, prob_true, marker='o', label='Model')
            
            # Multiclass - one-vs-rest
            elif y_proba.ndim == 2:
                n_classes = y_proba.shape[1]
                for i in range(n_classes):
                    prob_true, prob_pred = calibration_curve(
                        (y_true == i).astype(int),
                        y_proba[:, i],
                        n_bins=n_bins
                    )
                    ax.plot(prob_pred, prob_true, marker='o', label=f'Class {i}')
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Calibration curve saved to {save_path}")
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot calibration curve: {str(e)}") from e