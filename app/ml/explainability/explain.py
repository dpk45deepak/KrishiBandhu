"""
Unified model explainer and report generator.
"""

from typing import Any, Optional, Union, Dict, List, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

from ..common.models import BaseMLModel
from ..common.exceptions import EvaluationError
from ..common.utils import ensure_directory, save_json, get_timestamp

from .shap import SHAPExplainer
from .lime import LIMEExplainer
from .feature_importance import FeatureImportance, PermutationImportance
from .pdp import PDPExplainer


class ModelExplainer:
    """
    Unified model explainer that combines multiple explanation methods.
    """
    
    def __init__(
        self,
        model: BaseMLModel,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize model explainer.
        
        Args:
            model: Trained model
            X_train: Training data
            y_train: Training targets
            feature_names: Feature names
            **kwargs: Additional arguments
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_importance = None
        self.permutation_importance = None
        self.pdp_explainer = None
        
        # Set up feature names
        if self.feature_names is None and isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        logger.info("Model explainer initialized")
    
    def fit_all(self, **kwargs) -> 'ModelExplainer':
        """
        Fit all explainers.
        
        Returns:
            Self
        """
        try:
            # Fit SHAP
            logger.info("Fitting SHAP explainer...")
            self.shap_explainer = SHAPExplainer(self.model, self.feature_names)
            self.shap_explainer.fit(self.X_train)
            
            # Fit LIME
            logger.info("Fitting LIME explainer...")
            mode = 'classification' if hasattr(self.model, 'get_n_classes') else 'regression'
            self.lime_explainer = LIMEExplainer(
                self.model,
                self.feature_names,
                mode=mode
            )
            self.lime_explainer.fit(self.X_train, self.y_train)
            
            # Feature importance
            logger.info("Calculating feature importance...")
            self.feature_importance = FeatureImportance(self.model, self.feature_names)
            self.feature_importance.calculate()
            
            # Permutation importance
            logger.info("Calculating permutation importance...")
            self.permutation_importance = PermutationImportance(
                self.model,
                self.X_train,
                self.y_train,
                self.feature_names
            )
            self.permutation_importance.calculate()
            
            # PDP
            logger.info("Initializing PDP explainer...")
            self.pdp_explainer = PDPExplainer(self.model, self.X_train, self.feature_names)
            
            logger.info("All explainers fitted successfully")
            return self
            
        except Exception as e:
            raise EvaluationError(f"Failed to fit explainers: {str(e)}") from e
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.Series],
        method: str = 'shap',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Explain a single instance.
        
        Args:
            instance: Instance to explain
            method: Explanation method ('shap', 'lime')
            **kwargs: Additional arguments
            
        Returns:
            Explanation dictionary
        """
        if method == 'shap':
            if self.shap_explainer is None:
                raise ValueError("SHAP explainer not fitted")
            return self.shap_explainer.explain_instance(instance, **kwargs)
        elif method == 'lime':
            if self.lime_explainer is None:
                raise ValueError("LIME explainer not fitted")
            return self.lime_explainer.explain_instance(instance, **kwargs)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
    
    def get_feature_importance(self, method: str = 'builtin') -> Dict[str, float]:
        """
        Get feature importance.
        
        Args:
            method: 'builtin' or 'permutation'
            
        Returns:
            Dictionary of feature importance
        """
        if method == 'builtin':
            if self.feature_importance is None:
                raise ValueError("Feature importance not calculated")
            return self.feature_importance._importance
        elif method == 'permutation':
            if self.permutation_importance is None:
                raise ValueError("Permutation importance not calculated")
            importance_dict = dict(
                zip(self.permutation_importance.feature_names,
                    self.permutation_importance._importance)
            )
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def generate_report(
        self,
        save_path: Path,
        include_plots: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation report.
        
        Args:
            save_path: Path to save report
            include_plots: Whether to include plots
            **kwargs: Additional arguments
            
        Returns:
            Report dictionary
        """
        try:
            report = {
                'model_name': self.model.name,
                'model_type': 'classification' if hasattr(self.model, 'get_n_classes') else 'regression',
                'timestamp': get_timestamp(),
                'feature_names': self.feature_names,
                'n_features': len(self.feature_names) if self.feature_names else 0,
            }
            
            # Feature importance
            report['feature_importance'] = {
                'builtin': self.get_feature_importance('builtin'),
                'permutation': self.get_feature_importance('permutation')
            }
            
            # SHAP summary
            if self.shap_explainer:
                report['shap'] = {
                    'expected_value': float(self.shap_explainer.get_expected_value()),
                    'global_importance': self.get_feature_importance('shap') if hasattr(self, 'get_shap_importance') else None
                }
            
            # Save report
            report_path = save_path / 'explanation_report.json'
            save_json(report, report_path)
            
            # Save plots if requested
            if include_plots:
                plots_dir = save_path / 'plots'
                ensure_directory(plots_dir)
                
                # Feature importance plot
                if self.feature_importance:
                    self.feature_importance.plot(
                        save_path=plots_dir / 'feature_importance.png',
                        show=False
                    )
                
                # Permutation importance plot
                if self.permutation_importance:
                    self.permutation_importance.plot(
                        save_path=plots_dir / 'permutation_importance.png',
                        show=False
                    )
                
                # SHAP summary plot
                if self.shap_explainer:
                    self.shap_explainer.plot_summary(
                        save_path=plots_dir / 'shap_summary.png',
                        show=False
                    )
                    self.shap_explainer.plot_importance(
                        save_path=plots_dir / 'shap_importance.png',
                        show=False
                    )
            
            logger.info(f"Explanation report saved to {save_path}")
            return report
            
        except Exception as e:
            raise EvaluationError(f"Failed to generate explanation report: {str(e)}") from e


class ExplanationReport:
    """
    Utility class for generating explanation reports.
    """
    
    @staticmethod
    def generate(
        model: BaseMLModel,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        save_path: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a complete explanation report.
        
        Args:
            model: Trained model
            X: Data to explain
            y: Targets (optional)
            save_path: Path to save report
            **kwargs: Additional arguments
            
        Returns:
            Report dictionary
        """
        explainer = ModelExplainer(model, X, y, **kwargs)
        explainer.fit_all()
        
        if save_path:
            return explainer.generate_report(save_path)
        else:
            # Return basic report without saving
            return {
                'model_name': model.name,
                'feature_importance': explainer.get_feature_importance('builtin'),
                'permutation_importance': explainer.get_feature_importance('permutation')
            }