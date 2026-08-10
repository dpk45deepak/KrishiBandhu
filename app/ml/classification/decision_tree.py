"""
Decision Tree Classifier wrapper.
"""

from typing import Any, Optional, Union, Dict
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from ..common.models import ClassificationModel
from ..common.exceptions import TrainingError, PredictionError


class DecisionTreeClassifier(ClassificationModel):
    """
    Decision Tree Classifier wrapper with unified interface.
    """
    
    def __init__(
        self,
        name: str = "DecisionTreeClassifier",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, int]] = None,
        criterion: str = 'gini',
        splitter: str = 'best',
        class_weight: Optional[Union[str, Dict]] = None,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.splitter = splitter
        self.class_weight = class_weight
        self.random_state = random_state
        self._additional_kwargs = kwargs
        
        self._model = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'DecisionTreeClassifier':
        """Fit the Decision Tree model."""
        try:
            # Convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values
            
            # Store feature info
            self._n_features = X.shape[1]
            
            # Create and fit model
            self._model = SklearnDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                criterion=self.criterion,
                splitter=self.splitter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                **self._additional_kwargs
            )
            
            self._model.fit(X, y)
            self._is_fitted = True
            
            # Store classes
            self._classes = self._model.classes_
            self._n_classes = len(self._classes)
            
            return self
            
        except Exception as e:
            raise TrainingError(f"Decision Tree training failed: {str(e)}") from e
    
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
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'criterion': self.criterion,
            'splitter': self.splitter,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            **self._additional_kwargs
        }
    
    def get_model(self) -> SklearnDecisionTree:
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
    
    def get_tree(self):
        """Get the underlying tree structure."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model.tree_