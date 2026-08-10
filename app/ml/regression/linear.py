"""
Linear Regression wrapper.
"""

from typing import Any, Optional, Union, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinear
from sklearn.preprocessing import StandardScaler

from ..common.models import RegressionModel
from ..common.exceptions import TrainingError, PredictionError


class LinearRegression(RegressionModel):
    """
    Linear Regression wrapper with unified interface.
    """
    
    def __init__(
        self,
        name: str = "LinearRegression",
        fit_intercept: bool = True,
        normalize: bool = False,
        copy_X: bool = True,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self._additional_kwargs = kwargs
        
        self._model = None
        self._scaler = StandardScaler()
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'LinearRegression':
        """Fit the Linear Regression model."""
        try:
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Store feature info
            self._n_features = X.shape[1]
            
            # Scale features for better numerical stability
            if self.normalize:
                X = self._scaler.fit_transform(X)
            
            # Create and fit model
            self._model = SklearnLinear(
                fit_intercept=self.fit_intercept,
                normalize=False,  # Already handled
                copy_X=self.copy_X,
                n_jobs=self.n_jobs,
                **self._additional_kwargs
            )
            
            self._model.fit(X, y)
            self._is_fitted = True
            
            return self
            
        except Exception as e:
            raise TrainingError(f"Linear Regression training failed: {str(e)}") from e
    
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
            
            if self.normalize:
                X = self._scaler.transform(X)
            
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
            'fit_intercept': self.fit_intercept,
            'normalize': self.normalize,
            'copy_X': self.copy_X,
            'n_jobs': self.n_jobs,
            **self._additional_kwargs
        }
    
    def get_model(self) -> SklearnLinear:
        """Get the underlying sklearn model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model
    
    def get_coefficients(self) -> np.ndarray:
        """Get model coefficients."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model.coef_
    
    def get_intercept(self) -> float:
        """Get model intercept."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model.intercept_