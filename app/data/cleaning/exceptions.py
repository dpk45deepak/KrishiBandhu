"""Custom exceptions for the cleaning module."""

from typing import Optional, Any, Dict


class CleaningError(Exception):
    """Base exception for cleaning errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class StrategyError(CleaningError):
    """Exception raised for strategy-related errors."""
    pass


class TransformationError(CleaningError):
    """Exception raised for transformation errors."""
    pass


class PipelineError(CleaningError):
    """Exception raised for pipeline-related errors."""
    pass


class ConfigurationError(CleaningError):
    """Exception raised for configuration errors."""
    pass


class ValidationError(CleaningError):
    """Exception raised for validation errors during cleaning."""
    pass


class ReportGenerationError(CleaningError):
    """Exception raised for report generation errors."""
    pass


class ColumnNotFoundError(CleaningError):
    """Exception raised when a column is not found."""
    pass


class DataTypeError(CleaningError):
    """Exception raised for data type errors."""
    pass


class UnitConversionError(CleaningError):
    """Exception raised for unit conversion errors."""
    pass