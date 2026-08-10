"""
Base model interfaces for the ML framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, TypeVar, Generic
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path

T = TypeVar('T')  # Type variable for model instance


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""
    
    model_id: str
    model_version: str
    model_name: str
    model_type: str  # 'classification' or 'regression'
    training_timestamp: datetime = Field(default_factory=datetime.now)
    training_dataset_version: Optional[str] = None
    feature_version: Optional[str] = None
    
    # Training info
    training_time_seconds: Optional[float] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    n_classes: Optional[int] = None
    
    # Performance metrics
    metrics: dict[str, float] = Field(default_factory=dict)
    
    # Hyperparameters
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    
    # Model info
    checksum: Optional[str] = None
    model_size_mb: Optional[float] = None
    
    # Additional metadata
    additional_info: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class BaseMLModel(ABC, Generic[T]):
    """
    Abstract base class for all ML models.
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self._model: Optional[T] = None
        self._is_fitted: bool = False
        self._metadata: Optional[ModelMetadata] = None
        self._hyperparameters = kwargs
        
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'BaseMLModel':
        """
        Fit the model to the training data.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities (for classification).
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        pass
    
    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """
        Get the model's hyperparameters.
        
        Returns:
            Dictionary of hyperparameters
        """
        pass
    
    @abstractmethod
    def get_model(self) -> T:
        """
        Get the underlying model object.
        
        Returns:
            The underlying model instance
        """
        pass
    
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        if self._metadata is None:
            raise ValueError("Model metadata not available")
        return self._metadata
    
    def set_metadata(self, metadata: ModelMetadata) -> None:
        """Set model metadata."""
        self._metadata = metadata
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, fitted={self._is_fitted})"


class ClassificationModel(BaseMLModel, ABC):
    """Base class for classification models."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._n_classes: Optional[int] = None
        self._classes: Optional[np.ndarray] = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'ClassificationModel':
        """Fit the model."""
        # Store classes information
        y_array = y.values if isinstance(y, pd.Series) else np.array(y)
        self._classes = np.unique(y_array)
        self._n_classes = len(self._classes)
        return super().fit(X, y)
    
    def get_classes(self) -> np.ndarray:
        """Get the unique classes."""
        if self._classes is None:
            raise ValueError("Model not fitted yet")
        return self._classes
    
    def get_n_classes(self) -> int:
        """Get number of classes."""
        if self._n_classes is None:
            raise ValueError("Model not fitted yet")
        return self._n_classes


class RegressionModel(BaseMLModel, ABC):
    """Base class for regression models."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'RegressionModel':
        """Fit the model."""
        return super().fit(X, y)