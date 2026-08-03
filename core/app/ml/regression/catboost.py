"""
CatBoost Regressor wrapper.
"""

from typing import Any, Optional, Union, Dict, List
import numpy as np
import pandas as pd
from loguru import logger

from ..common.models import RegressionModel
from ..common.exceptions import TrainingError, PredictionError


class CatBoostRegressor(RegressionModel):
    """
    CatBoost Regressor wrapper with unified interface.
    """
    
    def __init__(
        self,
        name: str = "CatBoostRegressor",
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3.0,
        border_count: int = 254,
        bagging_temperature: float = 1.0,
        random_strength: float = 1.0,
        rsm: float = 1.0,
        random_seed: int = 42,
        thread_count: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.bagging_temperature = bagging_temperature
        self.random_strength = random_strength
        self.rsm = rsm
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.verbose = verbose
        self._additional_kwargs = kwargs
        
        self._model = None
        self._eval_result = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[tuple]] = None,
        cat_features: Optional[List[int]] = None
    ) -> 'CatBoostRegressor':
        """Fit the CatBoost model."""
        try:
            from catboost import CatBoostRegressor as CatBoostModel
            
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X_values = X.values
                if cat_features is None:
                    # Auto-detect categorical features
                    cat_features = [i for i, col in enumerate(X.columns) 
                                  if X[col].dtype == 'object' or X[col].dtype.name == 'category']
            else:
                X_values = X
            
            if isinstance(y, pd.Series):
                y = y.values
            
            # Store feature info
            self._n_features = X_values.shape[1]
            
            # Create model
            self._model = CatBoostModel(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                border_count=self.border_count,
                bagging_temperature=self.bagging_temperature,
                random_strength=self.random_strength,
                rsm=self.rsm,
                random_seed=self.random_seed,
                thread_count=self.thread_count,
                verbose=self.verbose,
                **self._additional_kwargs
            )
            
            # Fit with early stopping if eval_set provided
            if eval_set is not None:
                # Convert eval_set to CatBoost format
                catboost_eval_set = []
                for X_eval, y_eval in eval_set:
                    if isinstance(X_eval, pd.DataFrame):
                        X_eval = X_eval.values
                    if isinstance(y_eval, pd.Series):
                        y_eval = y_eval.values
                    catboost_eval_set.append((X_eval, y_eval))
                
                self._model.fit(
                    X_values, y,
                    eval_set=catboost_eval_set,
                    cat_features=cat_features,
                    early_stopping_rounds=10
                )
            else:
                self._model.fit(
                    X_values, y,
                    cat_features=cat_features
                )
            
            self._is_fitted = True
            
            return self
            
        except ImportError:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
        except Exception as e:
            raise TrainingError(f"CatBoost training failed: {str(e)}") from e
    
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
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'l2_leaf_reg': self.l2_leaf_reg,
            'border_count': self.border_count,
            'bagging_temperature': self.bagging_temperature,
            'random_strength': self.random_strength,
            'rsm': self.rsm,
            'random_seed': self.random_seed,
            'thread_count': self.thread_count,
            'verbose': self.verbose,
            **self._additional_kwargs
        }
    
    def get_model(self):
        """Get the underlying CatBoost model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        try:
            return self._model.get_feature_importance()
        except:
            return None