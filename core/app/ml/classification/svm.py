"""
Support Vector Machine Classifier wrapper.
"""

from typing import Any, Optional, Union, Dict
import numpy as np
import pandas as pd
from sklearn.svm import SVC as SklearnSVC
from sklearn.preprocessing import StandardScaler

from ..common.models import ClassificationModel
from ..common.exceptions import TrainingError, PredictionError


class SVMClassifier(ClassificationModel):
    """
    Support Vector Machine Classifier wrapper with unified interface.
    """
    
    def __init__(
        self,
        name: str = "SVMClassifier",
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: Optional[Union[str, float]] = 'scale',
        degree: int = 3,
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = True,
        tol: float = 1e-3,
        max_iter: int = -1,
        class_weight: Optional[Union[str, Dict]] = None,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self._additional_kwargs = kwargs
        
        self._model = None
        self._scaler = StandardScaler()
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'SVMClassifier':
        """Fit the SVM model."""
        try:
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Store feature info
            self._n_features = X.shape[1]
            
            # Scale features for SVM
            X_scaled = self._scaler.fit_transform(X)
            
            # Create and fit model
            self._model = SklearnSVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                shrinking=self.shrinking,
                probability=self.probability,
                tol=self.tol,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                **self._additional_kwargs
            )
            
            self._model.fit(X_scaled, y)
            self._is_fitted = True
            
            # Store classes
            self._classes = self._model.classes_
            self._n_classes = len(self._classes)
            
            return self
            
        except Exception as e:
            raise TrainingError(f"SVM training failed: {str(e)}") from e
    
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
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'shrinking': self.shrinking,
            'probability': self.probability,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            **self._additional_kwargs
        }
    
    def get_model(self) -> SklearnSVC:
        """Get the underlying sklearn model."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model
    
    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model.support_vectors_
    
    def get_support_indices(self) -> np.ndarray:
        """Get indices of support vectors."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model.support_