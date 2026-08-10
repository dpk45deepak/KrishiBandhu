"""
XGBoost Classifier wrapper.
"""

from typing import Any, Optional, Union, Dict
import numpy as np
import pandas as pd
from loguru import logger

from ..common.models import ClassificationModel
from ..common.exceptions import TrainingError, PredictionError


class XGBoostClassifier(ClassificationModel):
    """
    XGBoost Classifier wrapper with unified interface.
    """
    
    def __init__(
        self,
        name: str = "XGBoostClassifier",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        eval_metric: str = 'logloss',
        early_stopping_rounds: Optional[int] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self._additional_kwargs = kwargs
        
        self._model = None
        self._eval_result = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[tuple] = None,
        verbose: bool = False
    ) -> 'XGBoostClassifier':
        """Fit the XGBoost model."""
        try:
            import xgboost as xgb
            
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Store feature info
            self._n_features = X.shape[1]
            
            # Determine objective
            n_classes = len(np.unique(y))
            if n_classes == 2:
                objective = 'binary:logistic'
            else:
                objective = 'multi:softprob'
            
            # Create model
            self._model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                objective=objective,
                eval_metric=self.eval_metric,
                **self._additional_kwargs
            )
            
            # Fit with early stopping if eval_set provided
            if eval_set is not None:
                self._model.fit(
                    X, y,
                    eval_set=eval_set,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=verbose
                )
            else:
                self._model.fit(X, y)
            
            self._is_fitted = True
            
            # Store classes
            self._classes = self._model.classes_
            self._n_classes = len(self._classes)
            
            # Store eval result
            if hasattr(self._model, 'evals_result'):
                self._eval_result = self._model.evals_result()
            
            return self
            
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        except Exception as e:
            raise TrainingError(f"XGBoost training failed: {str(e)}") from e
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise PredictionError("Model not fitted")
        
        try:
            if isinstance(X, pd.DataFrame):
                X = X.values
            return self._model.predict(X)
        except Exception as e:
            raise PredictionError(f"Prediction failed: {str(e)}") from e
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise PredictionError("Model not fitted")
        
        try:
            if isinstance(X, pd.DataFrame):
                X = X.values
            return self._model.predict_proba(X)
        except Exception as e:
            raise PredictionError(f"Probability prediction failed: {str(e)}") from e
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'scale_pos_weight': self.scale_pos_weight,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'eval_metric': self.eval_metric,
            'early_stopping_rounds': self.early_stopping_rounds,
            **self._additional_kwargs
        }
    
    def get_model(self):
        """Get the underlying XGBoost model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model
    
    def get_feature_importance(self, importance_type: str = 'weight') -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        try:
            return self._model.feature_importances_
        except:
            return None
    
    def get_eval_result(self) -> Optional[Dict]:
        """Get evaluation results from training."""
        return self._eval_result