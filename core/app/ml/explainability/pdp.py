"""
Partial Dependence Plot (PDP) explainer.
"""

from typing import Any, Optional, Union, Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from loguru import logger

from ..common.models import BaseMLModel
from ..common.exceptions import EvaluationError
from ..common.utils import ensure_directory


class PDPExplainer:
    """
    Partial Dependence Plot explainer.
    Shows the relationship between features and predictions.
    """
    
    def __init__(
        self,
        model: BaseMLModel,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize PDP explainer.
        
        Args:
            model: Trained model
            X: Training data
            feature_names: Feature names
            **kwargs: Additional arguments
        """
        self.model = model
        self.X = X
        self.feature_names = feature_names
        self.kwargs = kwargs
        
        logger.info("PDP explainer initialized")
    
    def calculate(
        self,
        feature: Union[str, int],
        grid_resolution: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate partial dependence for a feature.
        
        Args:
            feature: Feature name or index
            grid_resolution: Number of grid points
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with grid points and average predictions
        """
        try:
            # Convert to numpy array
            if isinstance(self.X, pd.DataFrame):
                X = self.X.values
                if self.feature_names is None:
                    self.feature_names = self.X.columns.tolist()
                
                # Get feature index
                if isinstance(feature, str):
                    feature_idx = self.feature_names.index(feature)
                else:
                    feature_idx = feature
                    feature = self.feature_names[feature_idx]
            else:
                X = self.X
                if isinstance(feature, str):
                    raise ValueError("Feature names not available. Use index instead.")
                feature_idx = feature
            
            # Get the underlying model
            model_obj = self.model.get_model() if hasattr(self.model, 'get_model') else self.model
            
            # Calculate partial dependence
            pdp = partial_dependence(
                model_obj,
                X,
                features=[feature_idx],
                grid_resolution=grid_resolution,
                kind='average',
                **kwargs
            )
            
            # Extract results
            grid_points = pdp['grid_values'][0]
            avg_predictions = pdp['average'][0]
            
            # For classification, get probabilities for each class
            if hasattr(self.model, 'get_n_classes'):
                n_classes = self.model.get_n_classes()
                if n_classes > 2 and avg_predictions.ndim > 1:
                    # Multi-class
                    results = {}
                    for i in range(n_classes):
                        results[f'Class_{i}'] = {
                            'grid': grid_points,
                            'values': avg_predictions[:, i] if avg_predictions.ndim > 1 else avg_predictions
                        }
                    return {
                        'feature': feature,
                        'results': results,
                        'multi_class': True,
                        'n_classes': n_classes
                    }
            
            return {
                'feature': feature,
                'grid': grid_points,
                'values': avg_predictions,
                'multi_class': False
            }
            
        except Exception as e:
            raise EvaluationError(f"Failed to calculate partial dependence: {str(e)}") from e
    
    def plot(
        self,
        feature: Union[str, int],
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Plot partial dependence for a feature.
        
        Args:
            feature: Feature name or index
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        pdp_result = self.calculate(feature, **kwargs)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if pdp_result.get('multi_class', False):
            # Multi-class plot
            for class_name, result in pdp_result['results'].items():
                ax.plot(
                    result['grid'],
                    result['values'],
                    label=class_name,
                    linewidth=2
                )
            ax.legend()
        else:
            # Single plot
            ax.plot(
                pdp_result['grid'],
                pdp_result['values'],
                linewidth=2,
                color='blue'
            )
            ax.fill_between(
                pdp_result['grid'],
                pdp_result['values'],
                alpha=0.3
            )
        
        ax.set_xlabel(f'Feature: {pdp_result["feature"]}')
        ax.set_ylabel('Average Prediction')
        ax.set_title(f'Partial Dependence Plot for {pdp_result["feature"]}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            ensure_directory(save_path.parent)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"PDP plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_multiple(
        self,
        features: List[Union[str, int]],
        cols: int = 2,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[Path] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Plot multiple partial dependence plots.
        
        Args:
            features: List of features
            cols: Number of columns
            figsize: Figure size
            save_path: Path to save figure
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        n_features = len(features)
        rows = (n_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows > 1 else [axes]
        
        for i, feature in enumerate(features):
            pdp_result = self.calculate(feature, **kwargs)
            
            ax = axes[i]
            
            if pdp_result.get('multi_class', False):
                for class_name, result in pdp_result['results'].items():
                    ax.plot(
                        result['grid'],
                        result['values'],
                        label=class_name,
                        linewidth=2
                    )
                ax.legend()
            else:
                ax.plot(
                    pdp_result['grid'],
                    pdp_result['values'],
                    linewidth=2,
                    color='blue'
                )
                ax.fill_between(
                    pdp_result['grid'],
                    pdp_result['values'],
                    alpha=0.3
                )
            
            ax.set_xlabel(f'Feature: {pdp_result["feature"]}')
            ax.set_ylabel('Average Prediction')
            ax.set_title(f'PDP: {pdp_result["feature"]}')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        if save_path:
            ensure_directory(save_path.parent)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Multiple PDP plots saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig