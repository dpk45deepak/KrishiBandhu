"""Tests for validation framework exceptions."""

import pytest

from app.data.validation.exceptions import (
    BusinessRuleViolation,
    DuplicateColumnException,
    InvalidSchemaException,
    MissingColumnException,
    RuleRegistrationError,
    SchemaNotFoundException,
    ValidationException,
)


class TestValidationException:
    """Test the base ValidationException."""

    def test_message_only(self) -> None:
        """Test exception with only a message."""
        exc = ValidationException("boom")
        assert str(exc) == "boom"
        assert exc.message == "boom"
        assert exc.details == {}

    def test_message_with_details(self) -> None:
        """Test exception with structured details."""
        exc = ValidationException("boom", details={"col": "x"})
        assert exc.details == {"col": "x"}
        assert "details={'col': 'x'}" in str(exc)

    def test_is_exception(self) -> None:
        """Test it is a real Exception subclass."""
        assert issubclass(ValidationException, Exception)
        with pytest.raises(Exception):
            raise ValidationException("boom")


class TestSpecializedExceptions:
    """Test all specialized exception types."""

    def test_schema_not_found(self) -> None:
        """Test SchemaNotFoundException is raised and captured."""
        with pytest.raises(SchemaNotFoundException) as exc_info:
            raise SchemaNotFoundException("schema missing")
        assert isinstance(exc_info.value, ValidationException)

    def test_invalid_schema(self) -> None:
        """Test InvalidSchemaException."""
        with pytest.raises(InvalidSchemaException) as exc_info:
            raise InvalidSchemaException("bad schema")
        assert isinstance(exc_info.value, ValidationException)

    def test_missing_column(self) -> None:
        """Test MissingColumnException with details."""
        with pytest.raises(MissingColumnException) as exc_info:
            raise MissingColumnException("Column 'temp' missing", details={"column": "temp"})
        assert exc_info.value.details["column"] == "temp"

    def test_duplicate_column(self) -> None:
        """Test DuplicateColumnException."""
        with pytest.raises(DuplicateColumnException):
            raise DuplicateColumnException("dup columns")

    def test_business_rule_violation(self) -> None:
        """Test BusinessRuleViolation."""
        with pytest.raises(BusinessRuleViolation) as exc_info:
            raise BusinessRuleViolation("Strict mode failed", details={"validation_score": 0.4})
        assert exc_info.value.details["validation_score"] == 0.4

    def test_rule_registration_error(self) -> None:
        """Test RuleRegistrationError."""
        with pytest.raises(RuleRegistrationError):
            raise RuleRegistrationError("cannot register")

    def test_all_are_validation_exceptions(self) -> None:
        """Test every specialized exception inherits from ValidationException."""
        for exc_type in [
            SchemaNotFoundException,
            InvalidSchemaException,
            MissingColumnException,
            DuplicateColumnException,
            BusinessRuleViolation,
            RuleRegistrationError,
        ]:
            assert issubclass(exc_type, ValidationException)
