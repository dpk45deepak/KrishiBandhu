"""
Feature importance methods for model interpretation.
"""

from typing import Any, Optional, Union, Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from loguru import logger

from ..common.models import BaseMLModel
from ..common.exceptions import EvaluationError
from ..common.utils import ensure_directory


class FeatureImportance:
    """
    Feature importance calculator.
    Provides built-in feature importance for tree-based models.
    """
    
    def __init__(self, model: BaseMLModel, feature_names: Optional[List[str]] = None):
        """
        Initialize feature importance calculator.
        
        Args:
            model: Trained model
            feature_names: Feature names
        """
        self.model = model
        self.feature_names = feature_names
        self._importance = None
    
    def calculate(self) -> Dict[str, float]:
        """
        Calculate built-in feature importance.
        
        Returns:
            Dictionary of feature importance scores
        """
        try:
            # Try to get feature importance from model
            if hasattr(self.model, 'get_feature_importance'):
                importance = self.model.get_feature_importance()
            elif hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_).flatten()
            else:
                raise ValueError("Model does not have built-in feature importance")
            
            # Get feature names
            if self.feature_names is None:
                if hasattr(self.model, '_feature_names'):
                    self.feature_names = self.model._feature_names
                else:
                    self.feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            # Create dictionary
            self._importance = dict(zip(self.feature_names, importance))
            
            # Sort by importance
            self._importance = dict(
                sorted(self._importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            logger.info(f"Feature importance calculated for {len(self._importance)} features")
            return self._importance
            
        except Exception as e:
            raise EvaluationError(f"Failed to calculate feature importance: {str(e)}") from e
    
    def plot(
        self,
        top_n: int = 20,
        save_path: Optional[Path] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save figure
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        if self._importance is None:
            self.calculate()
        
        # Get top N features
        top_features = dict(list(self._importance.items())[:top_n])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(top_features.keys()),
            'importance': list(top_features.values())
        })
        
        sns.barplot(
            data=importance_df,
            x='importance',
            y='feature',
            palette='viridis_r',
            ax=ax
        )
        
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'Feature Importance (Top {len(top_features)})')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            ensure_directory(save_path.parent)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig


class PermutationImportance:
    """
    Permutation importance calculator.
    Model-agnostic feature importance using permutation.
    """
    
    def __init__(
        self,
        model: BaseMLModel,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize permutation importance calculator.
        
        Args:
            model: Trained model
            X: Features
            y: Targets
            feature_names: Feature names
            n_repeats: Number of repetitions for permutation
            random_state: Random seed
            **kwargs: Additional arguments
        """
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.kwargs = kwargs
        
        self._importance = None
        self._importance_std = None
        
        logger.info("Permutation importance initialized")
    
    def calculate(
        self,
        scoring: Optional[str] = None,
        n_jobs: int = -1
    ) -> Dict[str, float]:
        """
        Calculate permutation importance.
        
        Args:
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary of feature importance scores
        """
        try:
            from sklearn.inspection import permutation_importance
            
            # Convert to numpy arrays
            if isinstance(self.X, pd.DataFrame):
                X = self.X.values
                if self.feature_names is None:
                    self.feature_names = self.X.columns.tolist()
            else:
                X = self.X
            
            if isinstance(self.y, pd.Series):
                y = self.y.values
            else:
                y = self.y
            
            # Get the underlying model
            model_obj = self.model.get_model() if hasattr(self.model, 'get_model') else self.model
            
            # Calculate permutation importance
            result = permutation_importance(
                model_obj,
                X,
                y,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                scoring=scoring,
                n_jobs=n_jobs,
                **self.kwargs
            )
            
            # Store results
            self._importance = result.importances_mean
            self._importance_std = result.importances_std
            
            # Create dictionary
            if self.feature_names is None:
                self.feature_names = [f'Feature_{i}' for i in range(len(self._importance))]
            
            importance_dict = dict(zip(self.feature_names, self._importance))
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            logger.info(f"Permutation importance calculated for {len(importance_dict)} features")
            return importance_dict
            
        except Exception as e:
            raise EvaluationError(f"Failed to calculate permutation importance: {str(e)}") from e
    
    def plot(
        self,
        top_n: int = 20,
        save_path: Optional[Path] = None,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Plot permutation importance.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save figure
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        if self._importance is None:
            self.calculate()
        
        # Get top N features
        importance_dict = dict(zip(self.feature_names, self._importance))
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        top_features = dict(list(sorted_importance.items())[:top_n])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create DataFrame with standard deviation
        importance_df = pd.DataFrame({
            'feature': list(top_features.keys()),
            'importance': list(top_features.values()),
            'std': [self._importance_std[i] for i in range(len(top_features))]
        })
        
        # Plot with error bars
        ax.barh(
            importance_df['feature'],
            importance_df['importance'],
            xerr=importance_df['std'],
            capsize=5,
            color=sns.color_palette('viridis_r', len(importance_df))
        )
        
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'Permutation Importance (Top {len(top_features)})')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            ensure_directory(save_path.parent)
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Permutation importance plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig