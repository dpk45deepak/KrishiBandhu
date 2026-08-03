"""
Lasso Regressor wrapper.
"""

from typing import Any, Optional, Union, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.preprocessing import StandardScaler

from ..common.models import RegressionModel
from ..common.exceptions import TrainingError, PredictionError


class LassoRegressor(RegressionModel):
    """
    Lasso Regressor wrapper with unified interface.
    """
    
    def __init__(
        self,
        name: str = "Lasso",
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._additional_kwargs = kwargs
        
        self._model = None
        self._scaler = StandardScaler()
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'LassoRegressor':
        """Fit the Lasso model."""
        try:
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Store feature info
            self._n_features = X.shape[1]
            
            # Scale features
            X_scaled = self._scaler.fit_transform(X)
            
            # Create and fit model
            self._model = SklearnLasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                **self._additional_kwargs
            )
            
            self._model.fit(X_scaled, y)
            self._is_fitted = True
            
            return self
            
        except Exception as e:
            raise TrainingError(f"Lasso training failed: {str(e)}") from e
    
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
            X_scaled = self._scaler.transform(X)
            return self._model.predict(X_scaled)
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
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            **self._additional_kwargs
        }
    
    def get_model(self) -> SklearnLasso:
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