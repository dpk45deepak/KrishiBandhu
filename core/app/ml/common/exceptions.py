"""
Custom exceptions for the ML framework.
"""

from typing import Optional, Any


class AgriMLError(Exception):
    """Base exception for all ML framework errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ModelNotFoundError(AgriMLError):
    """Raised when a requested model is not found."""
    pass


class ModelLoadError(AgriMLError):
    """Raised when a model fails to load."""
    pass


class ModelSaveError(AgriMLError):
    """Raised when a model fails to save."""
    pass


class DatasetError(AgriMLError):
    """Raised when there are issues with the dataset."""
    pass


class PreprocessingError(AgriMLError):
    """Raised when preprocessing fails."""
    pass


class TrainingError(AgriMLError):
    """Raised when model training fails."""
    pass


class PredictionError(AgriMLError):
    """Raised when prediction fails."""
    pass


class EvaluationError(AgriMLError):
    """Raised when evaluation fails."""
    pass


class TuningError(AgriMLError):
    """Raised when hyperparameter tuning fails."""
    pass


class RegistryError(AgriMLError):
    """Raised when model registry operations fail."""
    pass


class ValidationError(AgriMLError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(AgriMLError):
    """Raised when there are configuration issues."""
    pass