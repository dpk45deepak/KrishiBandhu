"""
Classification metrics for model evaluation.
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    log_loss,
    brier_score_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
)
from loguru import logger

from ..common.exceptions import EvaluationError


class ClassificationMetrics:
    """
    Comprehensive classification metrics calculator.
    """
    
    @staticmethod
    def calculate(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_proba: Optional[Union[np.ndarray, pd.Series]] = None,
        average: str = 'weighted',
        labels: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            average: Averaging method for multiclass metrics
            labels: List of class labels
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Convert to numpy arrays
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            if isinstance(y_pred, pd.Series):
                y_pred = y_pred.values
            if y_proba is not None and isinstance(y_proba, pd.Series):
                y_proba = y_proba.values
            
            # Get unique classes
            unique_classes = labels or np.unique(y_true)
            n_classes = len(unique_classes)
            
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
            
            # Precision, Recall, F1
            metrics['precision'] = precision_score(
                y_true, y_pred, average=average, labels=labels
            )
            metrics['recall'] = recall_score(
                y_true, y_pred, average=average, labels=labels
            )
            metrics['f1'] = f1_score(
                y_true, y_pred, average=average, labels=labels
            )
            
            # Per-class metrics (if binary or small number of classes)
            if n_classes <= 10:
                precision_per_class = precision_score(
                    y_true, y_pred, average=None, labels=labels
                )
                recall_per_class = recall_score(
                    y_true, y_pred, average=None, labels=labels
                )
                f1_per_class = f1_score(
                    y_true, y_pred, average=None, labels=labels
                )
                
                for i, cls in enumerate(unique_classes):
                    metrics[f'precision_class_{cls}'] = precision_per_class[i]
                    metrics[f'recall_class_{cls}'] = recall_per_class[i]
                    metrics[f'f1_class_{cls}'] = f1_per_class[i]
            
            # Advanced metrics
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
            
            # Probability-based metrics
            if y_proba is not None:
                # Binary classification
                if n_classes == 2:
                    try:
                        # For binary, use positive class probabilities
                        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                            pos_proba = y_proba[:, 1]
                        else:
                            pos_proba = y_proba
                        
                        metrics['roc_auc'] = roc_auc_score(y_true, pos_proba)
                        metrics['average_precision'] = average_precision_score(
                            y_true, pos_proba
                        )
                        metrics['log_loss'] = log_loss(y_true, y_proba)
                        metrics['brier_score'] = brier_score_loss(y_true, pos_proba)
                    except Exception as e:
                        logger.warning(f"Could not calculate probability metrics: {e}")
                
                # Multiclass
                elif n_classes > 2 and y_proba.ndim == 2:
                    try:
                        metrics['roc_auc_ovr'] = roc_auc_score(
                            y_true, y_proba, multi_class='ovr'
                        )
                        metrics['roc_auc_ovo'] = roc_auc_score(
                            y_true, y_proba, multi_class='ovo'
                        )
                        metrics['log_loss'] = log_loss(y_true, y_proba)
                    except Exception as e:
                        logger.warning(f"Could not calculate multiclass probability metrics: {e}")
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            metrics['confusion_matrix'] = cm.tolist()  # Store as list for serialization
            
            # Derived metrics from confusion matrix
            if cm.shape == (2, 2):  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                metrics['true_negatives'] = int(tn)
                metrics['false_positives'] = int(fp)
                metrics['false_negatives'] = int(fn)
                metrics['true_positives'] = int(tp)
                
                # Additional binary metrics
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
                metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Classification report as dictionary
            report = classification_report(
                y_true, y_pred, labels=labels, output_dict=True
            )
            metrics['classification_report'] = report
            
            return metrics
            
        except Exception as e:
            raise EvaluationError(f"Failed to calculate classification metrics: {str(e)}") from e
    
    @staticmethod
    def get_roc_curve(
        y_true: Union[np.ndarray, pd.Series],
        y_proba: Union[np.ndarray, pd.Series]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate ROC curve data.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with fpr, tpr, thresholds
        """
        try:
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            if isinstance(y_proba, pd.Series):
                y_proba = y_proba.values
            
            # For binary classification
            if y_proba.ndim == 1:
                fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            else:
                # Multiclass - use one-vs-rest
                n_classes = y_proba.shape[1]
                fpr_dict = {}
                tpr_dict = {}
                thresholds_dict = {}
                
                for i in range(n_classes):
                    fpr, tpr, thresholds = roc_curve(
                        (y_true == i).astype(int),
                        y_proba[:, i]
                    )
                    fpr_dict[i] = fpr
                    tpr_dict[i] = tpr
                    thresholds_dict[i] = thresholds
                
                return {
                    'fpr': fpr_dict,
                    'tpr': tpr_dict,
                    'thresholds': thresholds_dict,
                    'multi_class': True
                }
            
            return {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'multi_class': False
            }
            
        except Exception as e:
            raise EvaluationError(f"Failed to calculate ROC curve: {str(e)}") from e
    
    @staticmethod
    def get_precision_recall_curve(
        y_true: Union[np.ndarray, pd.Series],
        y_proba: Union[np.ndarray, pd.Series]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate precision-recall curve data.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with precision, recall, thresholds
        """
        try:
            if isinstance(y_true, pd.Series):
                y_true = y_true.values
            if isinstance(y_proba, pd.Series):
                y_proba = y_proba.values
            
            # For binary classification
            if y_proba.ndim == 1:
                precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
                return {
                    'precision': precision,
                    'recall': recall,
                    'thresholds': thresholds
                }
            else:
                # Multiclass - use one-vs-rest
                n_classes = y_proba.shape[1]
                precision_dict = {}
                recall_dict = {}
                thresholds_dict = {}
                
                for i in range(n_classes):
                    precision, recall, thresholds = precision_recall_curve(
                        (y_true == i).astype(int),
                        y_proba[:, i]
                    )
                    precision_dict[i] = precision
                    recall_dict[i] = recall
                    thresholds_dict[i] = thresholds
                
                return {
                    'precision': precision_dict,
                    'recall': recall_dict,
                    'thresholds': thresholds_dict,
                    'multi_class': True
                }
            
        except Exception as e:
            raise EvaluationError(f"Failed to calculate precision-recall curve: {str(e)}") from e
    
    @staticmethod
    def get_classification_report(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        labels: Optional[List] = None,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate a classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of class labels
            target_names: List of class names
            
        Returns:
            Classification report as string
        """
        try:
            return classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=target_names
            )
        except Exception as e:
            raise EvaluationError(f"Failed to generate classification report: {str(e)}") from e
    
    @staticmethod
    def summary(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_proba: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete summary of classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Complete summary dictionary
        """
        metrics = ClassificationMetrics.calculate(y_true, y_pred, y_proba)
        
        # Add additional info
        unique_classes = np.unique(y_true)
        summary = {
            'metrics': metrics,
            'n_samples': len(y_true),
            'n_classes': len(unique_classes),
            'classes': unique_classes.tolist(),
            'class_counts': {str(c): int(np.sum(y_true == c)) for c in unique_classes}
        }
        
        return summary