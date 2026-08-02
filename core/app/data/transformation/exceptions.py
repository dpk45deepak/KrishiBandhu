"""Custom exceptions for the standardization framework."""


class StandardizationError(Exception):
    """Base exception for standardization errors."""
    pass


class SchemaNotFoundError(StandardizationError):
    """Raised when a schema is not found."""
    pass


class SchemaValidationError(StandardizationError):
    """Raised when schema validation fails."""
    pass


class ColumnMappingError(StandardizationError):
    """Raised when column mapping fails."""
    pass


class UnitConversionError(StandardizationError):
    """Raised when unit conversion fails."""
    pass


class CategoryNormalizationError(StandardizationError):
    """Raised when category normalization fails."""
    pass


class RegistryError(StandardizationError):
    """Raised for registry operations errors."""
    pass


class MetadataGenerationError(StandardizationError):
    """Raised when metadata generation fails."""
    pass