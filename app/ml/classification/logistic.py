"""
Logistic Regression Classifier wrapper.
"""

from typing import Any, Optional, Union, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogistic
from sklearn.preprocessing import StandardScaler

from ..common.models import ClassificationModel
from ..common.exceptions import TrainingError, PredictionError


class LogisticRegressionClassifier(ClassificationModel):
    """
    Logistic Regression Classifier wrapper with unified interface.
    """
    
    def __init__(
        self,
        name: str = "LogisticRegression",
        penalty: str = 'l2',
        C: float = 1.0,
        solver: str = 'lbfgs',
        max_iter: int = 1000,
        tol: float = 1e-4,
        class_weight: Optional[Union[str, Dict]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._additional_kwargs = kwargs
        
        self._model = None
        self._scaler = StandardScaler()
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'LogisticRegressionClassifier':
        """Fit the Logistic Regression model."""
        try:
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Store feature info
            self._n_features = X.shape[1]
            
            # Scale features for better convergence
            X_scaled = self._scaler.fit_transform(X)
            
            # Create and fit model
            self._model = SklearnLogistic(
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                tol=self.tol,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self._additional_kwargs
            )
            
            self._model.fit(X_scaled, y)
            self._is_fitted = True
            
            # Store classes
            self._classes = self._model.classes_
            self._n_classes = len(self._classes)
            
            return self
            
        except Exception as e:
            raise TrainingError(f"Logistic Regression training failed: {str(e)}") from e
    
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
        """Predict class probabilities."""
        if not self._is_fitted:
            raise PredictionError("Model not fitted")
        
        try:
            if isinstance(X, pd.DataFrame):
                X = X.values
            X_scaled = self._scaler.transform(X)
            return self._model.predict_proba(X_scaled)
        except Exception as e:
            raise PredictionError(f"Probability prediction failed: {str(e)}") from e
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'penalty': self.penalty,
            'C': self.C,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            **self._additional_kwargs
        }
    
    def get_model(self) -> SklearnLogistic:
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
        return self._model.intercept_[0]