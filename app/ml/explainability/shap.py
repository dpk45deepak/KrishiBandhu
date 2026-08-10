"""
SHAP (SHapley Additive exPlanations) explainer.
"""

from typing import Any, Optional, Union, Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from ..common.models import BaseMLModel
from ..common.exceptions import EvaluationError
from ..common.utils import ensure_directory


class SHAPExplainer:
    """
    SHAP-based model explainer.
    Provides local and global explanations using SHAP values.
    """
    
    def __init__(
        self,
        model: BaseMLModel,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            background_data: Background data for KernelExplainer
            feature_names: Feature names
            **kwargs: Additional arguments for SHAP explainer
        """
        self.model = model
        self.feature_names = feature_names
        self.kwargs = kwargs
        
        self._explainer = None
        self._shap_values = None
        self._expected_value = None
        
        # Set up background data if not provided
        if background_data is None:
            # Use a small sample from training data if available
            self.background_data = None
        else:
            if isinstance(background_data, pd.DataFrame):
                background_data = background_data.values
            self.background_data = background_data[:100]  # Limit to 100 samples
        
        logger.info("SHAP explainer initialized")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'SHAPExplainer':
        """
        Fit the SHAP explainer.
        
        Args:
            X: Data to explain
            
        Returns:
            Self
        """
        try:
            import shap
            
            # Convert to numpy array
            if isinstance(X, pd.DataFrame):
                X = X.values
                if self.feature_names is None:
                    self.feature_names = X.columns.tolist()
            
            # Get the underlying model
            model_obj = self.model.get_model()
            
            # Choose explainer type based on model
            try:
                # Try TreeExplainer for tree-based models
                self._explainer = shap.TreeExplainer(
                    model_obj,
                    feature_names=self.feature_names,
                    **self.kwargs
                )
            except:
                try:
                    # Try KernelExplainer as fallback
                    if self.background_data is None:
                        self.background_data = X[:100]  # Use first 100 samples
                    
                    self._explainer = shap.KernelExplainer(
                        model_obj.predict,
                        self.background_data,
                        feature_names=self.feature_names,
                        **self.kwargs
                    )
                except:
                    # Try LinearExplainer for linear models
                    self._explainer = shap.LinearExplainer(
                        model_obj,
                        X,
                        feature_names=self.feature_names,
                        **self.kwargs
                    )
            
            # Calculate SHAP values
            self._shap_values = self._explainer.shap_values(X)
            self._expected_value = self._explainer.expected_value
            
            logger.info("SHAP explainer fitted successfully")
            return self
            
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        except Exception as e:
            raise EvaluationError(f"Failed to fit SHAP explainer: {str(e)}") from e
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Explain a single instance.
        
        Args:
            instance: Single instance to explain
            feature_names: Feature names
            
        Returns:
            Dictionary with SHAP values and explanation
        """
        if self._explainer is None:
            raise ValueError("SHAP explainer not fitted. Call fit() first.")
        
        try:
            # Convert to numpy array
            if isinstance(instance, pd.Series):
                if feature_names is None:
                    feature_names = instance.index.tolist()
                instance = instance.values
            
            # Ensure 2D
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)
            
            # Calculate SHAP values for this instance
            shap_values = self._explainer.shap_values(instance)
            
            # Format for single instance
            if isinstance(shap_values, list):
                # Multi-class
                shap_vals = shap_values[0][0] if shap_values else None
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    # Multi-class with shape (n_classes, n_samples, n_features)
                    shap_vals = shap_values[:, 0, :]
                else:
                    shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            
            # Calculate base value
            if isinstance(self._expected_value, list):
                base_value = self._expected_value[0]
            else:
                base_value = self._expected_value
            
            # Create explanation
            explanation = {
                'shap_values': shap_vals if isinstance(shap_vals, np.ndarray) else np.array(shap_vals),
                'base_value': base_value,
                'feature_names': feature_names or self.feature_names,
                'prediction': self.model.predict(instance)[0],
            }
            
            # Calculate feature contributions
            if explanation['feature_names']:
                contributions = [
                    {
                        'feature': name,
                        'shap_value': float(val),
                        'importance': float(abs(val))
                    }
                    for name, val in zip(explanation['feature_names'], explanation['shap_values'])
                ]
                contributions.sort(key=lambda x: x['importance'], reverse=True)
                explanation['contributions'] = contributions
            
            return explanation
            
        except Exception as e:
            raise EvaluationError(f"Failed to explain instance: {str(e)}") from e
    
    def plot_summary(
        self,
        max_display: int = 20,
        save_path: Optional[Path] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Plot SHAP summary plot.
        
        Args:
            max_display: Maximum number of features to display
            save_path: Path to save figure
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        if self._shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit() first.")
        
        try:
            import shap
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot summary
            if isinstance(self._shap_values, list):
                # Multi-class
                shap.summary_plot(
                    self._shap_values,
                    feature_names=self.feature_names,
                    max_display=max_display,
                    show=False,
                    **kwargs
                )
            else:
                shap.summary_plot(
                    self._shap_values,
                    feature_names=self.feature_names,
                    max_display=max_display,
                    show=False,
                    **kwargs
                )
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot SHAP summary: {str(e)}") from e
    
    def plot_importance(
        self,
        max_display: int = 20,
        save_path: Optional[Path] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Plot feature importance based on SHAP values.
        
        Args:
            max_display: Maximum number of features to display
            save_path: Path to save figure
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        if self._shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit() first.")
        
        try:
            import shap
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate mean absolute SHAP values
            if isinstance(self._shap_values, list):
                # Multi-class - average across classes
                shap_importance = np.mean([np.abs(v).mean(0) for v in self._shap_values], axis=0)
            else:
                shap_importance = np.abs(self._shap_values).mean(0)
            
            # Create DataFrame
            if self.feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(shap_importance))]
            else:
                feature_names = self.feature_names
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': shap_importance
            }).sort_values('importance', ascending=True)
            
            # Take top N
            if len(importance_df) > max_display:
                importance_df = importance_df.tail(max_display)
            
            # Plot
            sns.barplot(
                data=importance_df,
                x='importance',
                y='feature',
                palette='viridis_r',
                ax=ax
            )
            
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_ylabel('Feature')
            ax.set_title('SHAP Feature Importance')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                logger.info(f"SHAP importance plot saved to {save_path}")
            
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot SHAP importance: {str(e)}") from e
    
    def plot_waterfall(
        self,
        instance_idx: int = 0,
        save_path: Optional[Path] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Plot SHAP waterfall plot for a specific instance.
        
        Args:
            instance_idx: Index of instance to explain
            save_path: Path to save figure
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        if self._shap_values is None:
            raise ValueError("SHAP values not calculated. Call fit() first.")
        
        try:
            import shap
            
            # Get SHAP values for instance
            if isinstance(self._shap_values, list):
                # Multi-class - use first class
                shap_vals = self._shap_values[0][instance_idx]
            else:
                shap_vals = self._shap_values[instance_idx]
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals,
                    base_values=self._expected_value if not isinstance(self._expected_value, list) else self._expected_value[0],
                    data=None,
                    feature_names=self.feature_names
                ),
                show=False,
                **kwargs
            )
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                logger.info(f"SHAP waterfall plot saved to {save_path}")
            
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot SHAP waterfall: {str(e)}") from e
    
    def get_shap_values(self) -> np.ndarray:
        """Get SHAP values."""
        if self._shap_values is None:
            raise ValueError("SHAP values not calculated")
        return self._shap_values
    
    def get_expected_value(self) -> float:
        """Get expected/base value."""
        if self._expected_value is None:
            raise ValueError("Expected value not calculated")
        return self._expected_value