"""
Base tuner class and factory for hyperparameter optimization.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union, Callable, Type
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from ..common.models import BaseMLModel
from ..common.exceptions import TuningError
from ..common.utils import ensure_directory, save_json, get_timestamp


@dataclass
class TuningResult:
    """
    Result of hyperparameter tuning.
    """
    best_params: Dict[str, Any]
    best_score: float
    best_model: Optional[BaseMLModel] = None
    all_scores: List[float] = field(default_factory=list)
    all_params: List[Dict[str, Any]] = field(default_factory=list)
    n_trials: int = 0
    tuning_time_seconds: float = 0.0
    study_name: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': self.n_trials,
            'tuning_time_seconds': self.tuning_time_seconds,
            'study_name': self.study_name,
            'additional_info': self.additional_info
        }


class BaseTuner(ABC):
    """
    Abstract base class for hyperparameter tuners.
    """
    
    def __init__(
        self,
        model_class: Type[BaseMLModel],
        param_space: Dict[str, Any],
        n_trials: int = 100,
        metric: str = 'accuracy',
        direction: str = 'maximize',
        random_state: int = 42,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize the tuner.
        
        Args:
            model_class: Model class to tune
            param_space: Parameter space to search
            n_trials: Number of trials
            metric: Metric to optimize
            direction: 'maximize' or 'minimize'
            random_state: Random seed
            verbose: Verbosity level
            **kwargs: Additional arguments
        """
        self.model_class = model_class
        self.param_space = param_space
        self.n_trials = n_trials
        self.metric = metric
        self.direction = direction
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs
        
        self._best_params = None
        self._best_score = None
        self._best_model = None
        self._all_scores = []
        self._all_params = []
        
        logger.info(f"Initialized {self.__class__.__name__} with {n_trials} trials")
    
    @abstractmethod
    def tune(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        cv: int = 5,
        **kwargs
    ) -> TuningResult:
        """
        Perform hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            cv: Number of cross-validation folds
            **kwargs: Additional arguments
            
        Returns:
            TuningResult object
        """
        pass
    
    def _evaluate_model(
        self,
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        cv: int = 5
    ) -> float:
        """
        Evaluate a model with given parameters.
        
        Args:
            params: Model parameters
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            cv: Number of CV folds
            
        Returns:
            Score
        """
        try:
            # Create model with parameters
            model = self.model_class(**params)
            
            if X_val is not None and y_val is not None:
                # Use validation set
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Calculate score
                if hasattr(model, 'get_n_classes'):
                    # Classification
                    from ..evaluation.classification_metrics import ClassificationMetrics
                    metrics = ClassificationMetrics.calculate(y_val, y_pred)
                    score = metrics.get(self.metric, 0)
                else:
                    # Regression
                    from ..evaluation.regression_metrics import RegressionMetrics
                    metrics = RegressionMetrics.calculate(y_val, y_pred)
                    score = metrics.get(self.metric, 0)
                
                # Adjust direction
                if self.direction == 'minimize':
                    score = -score
                
                return score
            else:
                # Use cross-validation
                from sklearn.model_selection import cross_val_score
                from sklearn.metrics import make_scorer
                
                # Determine scorer
                if hasattr(model, 'get_n_classes'):
                    # Classification
                    if self.metric in ['accuracy', 'precision', 'recall', 'f1']:
                        scorer = self.metric
                    else:
                        scorer = 'accuracy'
                else:
                    # Regression
                    if self.metric in ['r2', 'mae', 'mse', 'rmse']:
                        scorer = self.metric
                    else:
                        scorer = 'r2'
                
                # Get the underlying sklearn model
                sklearn_model = model.get_model() if hasattr(model, 'get_model') else model
                
                # Cross-validation scores
                scores = cross_val_score(
                    sklearn_model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scorer
                )
                
                score = scores.mean()
                
                # Adjust direction
                if self.direction == 'minimize':
                    score = -score
                
                return score
                
        except Exception as e:
            logger.warning(f"Failed to evaluate model with params {params}: {e}")
            return -np.inf if self.direction == 'maximize' else np.inf
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters found."""
        if self._best_params is None:
            raise TuningError("Tuning not performed yet")
        return self._best_params
    
    def get_best_score(self) -> float:
        """Get the best score found."""
        if self._best_score is None:
            raise TuningError("Tuning not performed yet")
        return self._best_score
    
    def get_best_model(self) -> BaseMLModel:
        """Get the best model found."""
        if self._best_model is None:
            raise TuningError("Tuning not performed yet")
        return self._best_model
    
    def get_results(self) -> Dict[str, Any]:
        """Get all tuning results."""
        return {
            'best_params': self._best_params,
            'best_score': self._best_score,
            'n_trials': len(self._all_scores),
            'all_scores': self._all_scores,
            'all_params': self._all_params
        }
    
    def save_results(self, path: Path) -> None:
        """
        Save tuning results to disk.
        
        Args:
            path: Path to save results
        """
        results = self.get_results()
        results['timestamp'] = get_timestamp()
        ensure_directory(path.parent)
        save_json(results, path)
        logger.info(f"Tuning results saved to {path}")


class TunerFactory:
    """
    Factory for creating tuners.
    """
    
    @staticmethod
    def create(
        tuner_type: str,
        model_class: Type[BaseMLModel],
        param_space: Dict[str, Any],
        **kwargs
    ) -> BaseTuner:
        """
        Create a tuner instance.
        
        Args:
            tuner_type: Type of tuner ('grid', 'random', 'optuna', 'bayesian')
            model_class: Model class to tune
            param_space: Parameter space
            **kwargs: Additional arguments
            
        Returns:
            Tuner instance
        """
        if tuner_type == 'grid':
            from .grid_search import GridSearchTuner
            return GridSearchTuner(model_class, param_space, **kwargs)
        elif tuner_type == 'random':
            from .random_search import RandomSearchTuner
            return RandomSearchTuner(model_class, param_space, **kwargs)
        elif tuner_type == 'optuna':
            from .optuna_search import OptunaTuner
            return OptunaTuner(model_class, param_space, **kwargs)
        elif tuner_type == 'bayesian':
            from .bayesian import BayesianTuner
            return BayesianTuner(model_class, param_space, **kwargs)
        else:
            raise ValueError(f"Unknown tuner type: {tuner_type}")