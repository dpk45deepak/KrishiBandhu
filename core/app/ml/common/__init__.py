"""
Common module for the ML framework.
Provides reusable components for all ML models.
"""

from .exceptions import (
    AgriMLError,
    ModelNotFoundError,
    ModelLoadError,
    ModelSaveError,
    DatasetError,
    PreprocessingError,
    TrainingError,
    PredictionError,
    EvaluationError,
    TuningError,
    RegistryError,
    ValidationError,
    ConfigurationError
)

from .utils import (
    generate_checksum,
    generate_model_version,
    get_timestamp,
    ensure_directory,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    validate_dataframe,
    log_metrics,
    get_memory_usage,
    Timer
)

from .datasets import DatasetConfig, DatasetLoader
from .preprocessing import PreprocessingConfig, FeatureTransformer
from .models import (
    ModelMetadata,
    BaseMLModel,
    ClassificationModel,
    RegressionModel
)

__all__ = [
    # Exceptions
    'AgriMLError',
    'ModelNotFoundError',
    'ModelLoadError',
    'ModelSaveError',
    'DatasetError',
    'PreprocessingError',
    'TrainingError',
    'PredictionError',
    'EvaluationError',
    'TuningError',
    'RegistryError',
    'ValidationError',
    'ConfigurationError',
    
    # Utils
    'generate_checksum',
    'generate_model_version',
    'get_timestamp',
    'ensure_directory',
    'save_json',
    'load_json',
    'save_pickle',
    'load_pickle',
    'validate_dataframe',
    'log_metrics',
    'get_memory_usage',
    'Timer',
    
    # Datasets
    'DatasetConfig',
    'DatasetLoader',
    
    # Preprocessing
    'PreprocessingConfig',
    'FeatureTransformer',
    
    # Models
    'ModelMetadata',
    'BaseMLModel',
    'ClassificationModel',
    'RegressionModel',
]