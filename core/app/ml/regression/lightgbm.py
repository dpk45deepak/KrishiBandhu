"""
LightGBM Regressor wrapper.
"""

from typing import Any, Optional, Union, Dict, List
import numpy as np
import pandas as pd
from loguru import logger

from ..common.models import RegressionModel
from ..common.exceptions import TrainingError, PredictionError


class LightGBMRegressor(RegressionModel):
    """
    LightGBM Regressor wrapper with unified interface.
    """
    
    def __init__(
        self,
        name: str = "LightGBMRegressor",
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._additional_kwargs = kwargs
        
        self._model = None
        self._eval_result = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[tuple]] = None,
        verbose: bool = False
    ) -> 'LightGBMRegressor':
        """Fit the LightGBM model."""
        try:
            import lightgbm as lgb
            
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Store feature info
            self._n_features = X.shape[1]
            
            # Create model
            self._model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                min_split_gain=self.min_split_gain,
                min_child_weight=self.min_child_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                objective='regression',
                **self._additional_kwargs
            )
            
            # Fit with early stopping if eval_set provided
            if eval_set is not None:
                # Convert eval_set to LightGBM format
                lgb_eval_set = []
                for X_eval, y_eval in eval_set:
                    if isinstance(X_eval, pd.DataFrame):
                        X_eval = X_eval.values
                    if isinstance(y_eval, pd.Series):
                        y_eval = y_eval.values
                    lgb_eval_set.append((X_eval, y_eval))
                
                self._model.fit(
                    X, y,
                    eval_set=lgb_eval_set,
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
            else:
                self._model.fit(X, y)
            
            self._is_fitted = True
            
            return self
            
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
        except Exception as e:
            raise TrainingError(f"LightGBM training failed: {str(e)}") from e
    
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
        """Not applicable for regression."""
        raise NotImplementedError("predict_proba is not supported for regression")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_split_gain': self.min_split_gain,
            'min_child_weight': self.min_child_weight,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            **self._additional_kwargs
        }
    
    def get_model(self):
        """Get the underlying LightGBM model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model
    
    def get_feature_importance(self, importance_type: str = 'split') -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        try:
            return self._model.feature_importances_
        except:
            return None