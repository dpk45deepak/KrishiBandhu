"""
Random Forest Regressor wrapper.
"""

from typing import Any, Optional, Union, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForest

from ..common.models import RegressionModel
from ..common.exceptions import TrainingError, PredictionError


class RandomForestRegressor(RegressionModel):
    """
    Random Forest Regressor wrapper with unified interface.
    """
    
    def __init__(
        self,
        name: str = "RandomForestRegressor",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int] = 'sqrt',
        bootstrap: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._additional_kwargs = kwargs
        
        self._model = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'RandomForestRegressor':
        """Fit the Random Forest model."""
        try:
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Store feature info
            self._n_features = X.shape[1]
            
            # Create and fit model
            self._model = SklearnRandomForest(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self._additional_kwargs
            )
            
            self._model.fit(X, y)
            self._is_fitted = True
            
            return self
            
        except Exception as e:
            raise TrainingError(f"Random Forest training failed: {str(e)}") from e
    
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
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            **self._additional_kwargs
        }
    
    def get_model(self) -> SklearnRandomForest:
        """Get the underlying sklearn model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        if hasattr(self._model, 'feature_importances_'):
            return self._model.feature_importances_
        return None