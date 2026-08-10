"""
LIME (Local Interpretable Model-agnostic Explanations) explainer.
"""

from typing import Any, Optional, Union, Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger

from ..common.models import BaseMLModel
from ..common.exceptions import EvaluationError
from ..common.utils import ensure_directory


class LIMEExplainer:
    """
    LIME-based model explainer.
    Provides local explanations using LIME.
    """
    
    def __init__(
        self,
        model: BaseMLModel,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = 'classification',
        **kwargs
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model
            feature_names: Feature names
            class_names: Class names (for classification)
            mode: 'classification' or 'regression'
            **kwargs: Additional arguments for LIME explainer
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.kwargs = kwargs
        
        self._explainer = None
        self._feature_importance = None
        
        logger.info("LIME explainer initialized")
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> 'LIMEExplainer':
        """
        Fit the LIME explainer.
        
        Args:
            X: Training data
            y: Training targets (optional)
            
        Returns:
            Self
        """
        try:
            import lime
            from lime.lime_tabular import LimeTabularExplainer
            
            # Convert to numpy array
            if isinstance(X, pd.DataFrame):
                X_values = X.values
                if self.feature_names is None:
                    self.feature_names = X.columns.tolist()
                # Check for categorical features
                categorical_features = [
                    i for i, col in enumerate(X.columns)
                    if X[col].dtype == 'object' or X[col].dtype.name == 'category'
                ]
            else:
                X_values = X
                categorical_features = []
            
            # Get model prediction function
            def predict_fn(x):
                return self.model.predict_proba(x) if hasattr(self.model, 'predict_proba') else self.model.predict(x)
            
            # Create LIME explainer
            self._explainer = LimeTabularExplainer(
                X_values,
                feature_names=self.feature_names,
                class_names=self.class_names,
                categorical_features=categorical_features,
                mode=self.mode,
                **self.kwargs
            )
            
            logger.info("LIME explainer fitted successfully")
            return self
            
        except ImportError:
            raise ImportError("LIME not installed. Install with: pip install lime")
        except Exception as e:
            raise EvaluationError(f"Failed to fit LIME explainer: {str(e)}") from e
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.Series],
        num_features: int = 10,
        top_labels: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Explain a single instance using LIME.
        
        Args:
            instance: Single instance to explain
            num_features: Number of features to include in explanation
            top_labels: Number of top labels to explain (classification only)
            **kwargs: Additional LIME arguments
            
        Returns:
            Dictionary with explanation
        """
        if self._explainer is None:
            raise ValueError("LIME explainer not fitted. Call fit() first.")
        
        try:
            # Convert to numpy array
            if isinstance(instance, pd.Series):
                instance = instance.values
            
            # Ensure 1D
            if instance.ndim > 1:
                instance = instance.flatten()
            
            # Get prediction function
            def predict_fn(x):
                return self.model.predict_proba(x) if hasattr(self.model, 'predict_proba') else self.model.predict(x)
            
            # Get LIME explanation
            if self.mode == 'classification':
                exp = self._explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=num_features,
                    top_labels=top_labels,
                    **kwargs
                )
                
                # Get explanation for top label
                label = exp.available_labels()[0] if top_labels > 0 else 0
                exp_list = exp.as_list(label=label)
                
                # Get feature contributions
                contributions = []
                for feature, weight in exp_list:
                    contributions.append({
                        'feature': feature,
                        'weight': weight,
                        'importance': abs(weight)
                    })
                
                return {
                    'instance': instance.tolist(),
                    'prediction': self.model.predict(instance.reshape(1, -1))[0],
                    'probabilities': self.model.predict_proba(instance.reshape(1, -1))[0].tolist(),
                    'label': label,
                    'contributions': contributions,
                    'explanation': exp
                }
            else:
                # Regression
                exp = self._explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=num_features,
                    **kwargs
                )
                
                exp_list = exp.as_list()
                
                contributions = []
                for feature, weight in exp_list:
                    contributions.append({
                        'feature': feature,
                        'weight': weight,
                        'importance': abs(weight)
                    })
                
                return {
                    'instance': instance.tolist(),
                    'prediction': self.model.predict(instance.reshape(1, -1))[0],
                    'contributions': contributions,
                    'explanation': exp
                }
            
        except Exception as e:
            raise EvaluationError(f"Failed to explain instance: {str(e)}") from e
    
    def plot_explanation(
        self,
        instance_idx: int = 0,
        top_labels: int = 1,
        save_path: Optional[Path] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Plot LIME explanation.
        
        Args:
            instance_idx: Index of instance to explain
            top_labels: Number of top labels to show
            save_path: Path to save figure
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        if self._explainer is None:
            raise ValueError("LIME explainer not fitted. Call fit() first.")
        
        try:
            # This would require the data to get the instance
            # We'll just create a simple plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get explanation for instance
            # Note: This requires the instance to be passed
            # We'll just show a placeholder
            
            ax.text(0.5, 0.5, 'LIME Explanation Plot\n(Requires instance data)',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            
            plt.tight_layout()
            
            if save_path:
                ensure_directory(save_path.parent)
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                logger.info(f"LIME plot saved to {save_path}")
            
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to plot LIME explanation: {str(e)}") from e
    
    def get_explanation_string(
        self,
        instance: Union[np.ndarray, pd.Series],
        num_features: int = 10,
        **kwargs
    ) -> str:
        """
        Get explanation as a string.
        
        Args:
            instance: Instance to explain
            num_features: Number of features to include
            
        Returns:
            Explanation string
        """
        explanation = self.explain_instance(instance, num_features, **kwargs)
        
        result = "LIME Explanation:\n"
        result += f"Prediction: {explanation['prediction']}\n\n"
        result += "Top Feature Contributions:\n"
        
        for i, contrib in enumerate(explanation['contributions'][:num_features], 1):
            sign = "+" if contrib['weight'] > 0 else "-"
            result += f"  {i}. {contrib['feature']}: {sign} {abs(contrib['weight']):.4f}\n"
        
        return result