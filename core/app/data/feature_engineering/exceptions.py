# app/data/feature_engineering/exceptions.py
class FeatureEngineeringError(Exception):
    """Base exception for feature engineering errors."""
    pass


class FeatureGenerationError(FeatureEngineeringError):
    """Raised when feature generation fails."""
    pass


class FeatureRegistryError(FeatureEngineeringError):
    """Raised when registry operations fail."""
    pass


class FeatureValidationError(FeatureEngineeringError):
    """Raised when feature validation fails."""
    pass


class FeatureSelectionError(FeatureEngineeringError):
    """Raised when feature selection fails."""
    pass


class EncodingError(FeatureEngineeringError):
    """Raised when encoding fails."""
    pass


class TransformationError(FeatureEngineeringError):
    """Raised when transformation fails."""
    pass


class FeatureNotFoundError(FeatureEngineeringError):
    """Raised when a feature is not found."""
    pass


class FeatureVersionError(FeatureEngineeringError):
    """Raised when version operations fail."""
    pass


class StoreBackendError(FeatureEngineeringError):
    """Raised when store backend operations fail."""
    pass