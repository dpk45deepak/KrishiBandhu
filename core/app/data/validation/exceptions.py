"""Custom exceptions for the AgriMind AI validation framework.

All validation exceptions inherit from :class:`ValidationException` so that
callers can catch a single base type while still having access to rich,
actionable error messages.
"""

from __future__ import annotations

from typing import Any


class ValidationException(Exception):
    """Base exception for all validation framework errors.

    Args:
        message: Human-readable description of the error.
        details: Optional structured context (e.g. column name, row indices).
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the validation exception.

        Args:
            message: Human-readable description of the error.
            details: Optional structured context (e.g. column name, row indices).
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return a string representation including structured details."""
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


class SchemaNotFoundException(ValidationException):
    """Raised when a requested schema file cannot be located on disk."""


class InvalidSchemaException(ValidationException):
    """Raised when a schema is malformed, incomplete, or fails validation."""


class MissingColumnException(ValidationException):
    """Raised when a required column is absent from the dataset.

    The ``details`` dict contains a ``required_columns`` key listing every
    missing column.
    """


class DuplicateColumnException(ValidationException):
    """Raised when the dataset contains unexpected duplicate column names."""


class BusinessRuleViolation(ValidationException):
    """Raised when a business-level validation rule is violated.

    Used by :class:`~app.data.validation.rules.BusinessRule` implementations
    and by the :class:`~app.data.validation.validator.ValidationEngine` when
    strict mode is enabled and a business rule fails.
    """


class RuleRegistrationError(ValidationException):
    """Raised when a rule cannot be registered with the validation engine."""
