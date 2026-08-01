"""Tests for validation framework Pydantic models."""

from datetime import datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.data.validation.models import (
    ColumnStatistic,
    RuleStatistic,
    ValidationError,
    ValidationReport,
    ValidationResult,
    ValidationRule,
    ValidationSeverity,
    ValidationSummary,
)


class TestValidationSeverity:
    """Test the ValidationSeverity enum."""

    def test_weights(self) -> None:
        """Test severity weights."""
        assert ValidationSeverity.CRITICAL.weight == 1.0
        assert ValidationSeverity.ERROR.weight == 0.7
        assert ValidationSeverity.WARNING.weight == 0.3
        assert ValidationSeverity.INFO.weight == 0.0

    def test_string_values(self) -> None:
        """Test enum string values."""
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.CRITICAL.value == "critical"


class TestValidationRule:
    """Test the ValidationRule model."""

    def test_defaults(self) -> None:
        """Test defaults are applied."""
        rule = ValidationRule(name="test", type="range")
        assert rule.enabled is True
        assert rule.severity == ValidationSeverity.ERROR
        assert rule.column is None
        assert rule.parameters == {}

    def test_custom_cfg(self) -> None:
        """Test custom configuration is preserved."""
        rule = ValidationRule(
            name="temp_range",
            type="range",
            column="temperature",
            parameters={"min": -20, "max": 60},
            severity=ValidationSeverity.CRITICAL,
        )
        assert rule.parameters == {"min": -20, "max": 60}
        assert rule.severity == ValidationSeverity.CRITICAL

    def test_extra_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(PydanticValidationError):
            ValidationRule(name="x", type="y", unexpected="field")

    def test_empty_name_rejected(self) -> None:
        """Test that an empty name is rejected."""
        with pytest.raises(PydanticValidationError):
            ValidationRule(name="", type="range")


class TestValidationError:
    """Test the ValidationError model."""

    def test_row_count_defaults(self) -> None:
        """Test row_count is derived from row_indices when absent."""
        error = ValidationError(
            rule_name="range",
            rule_type="range",
            column="temp",
            message="boom",
            row_indices=[1, 2, 3],
        )
        assert error.row_count == 3

    def test_explicit_row_count(self) -> None:
        """Test explicit row_count is preserved."""
        error = ValidationError(
            rule_name="range",
            rule_type="range",
            message="boom",
            row_indices=[1, 2],
            row_count=2,
        )
        assert error.row_count == 2

    def test_serialization(self) -> None:
        """Test JSON serialization round-trip."""
        error = ValidationError(
            rule_name="range",
            rule_type="range",
            column="temp",
            severity=ValidationSeverity.WARNING,
            message="boom",
            row_indices=[0],
        )
        data = error.model_dump(mode="json")
        assert data["severity"] == "warning"
        assert data["row_indices"] == [0]
        restored = ValidationError.model_validate(data)
        assert restored.message == "boom"


class TestValidationResult:
    """Test the ValidationResult model."""

    def test_passing_result(self) -> None:
        """Test a passing result."""
        result = ValidationResult(rule_name="range", rule_type="range", column="temp", passed=True)
        assert result.passed is True
        assert result.errors == []
        assert result.failed_count == 0

    def test_failing_result_derives_failed_count(self) -> None:
        """Test failed_count is derived from errors."""
        error = ValidationError(
            rule_name="range",
            rule_type="range",
            column="temp",
            message="out of range",
            row_indices=[0, 1],
        )
        result = ValidationResult(
            rule_name="range",
            rule_type="range",
            column="temp",
            passed=False,
            errors=[error],
        )
        assert result.failed_count == 2

    def test_compact_serialization(self) -> None:
        """Test serialization preserves structure."""
        result = ValidationResult(rule_name="range", rule_type="range", passed=True, total_rows=10)
        data = result.model_dump()
        assert data["total_rows"] == 10
        assert data["passed"] is True


class TestStatisticsModels:
    """Test ColumnStatistic and RuleStatistic."""

    def test_column_statistic_defaults(self) -> None:
        """Test defaults of ColumnStatistic."""
        stat = ColumnStatistic(column="temp")
        assert stat.rules_checked == 0
        assert stat.rules_failed == 0
        assert stat.errors == 0

    def test_rule_statistic_defaults(self) -> None:
        """Test defaults of RuleStatistic."""
        stat = RuleStatistic(rule_name="r", rule_type="range")
        assert stat.ran is False
        assert stat.passed is False


class TestValidationSummary:
    """Test the ValidationSummary model."""

    def test_derived_fields(self) -> None:
        """Test derived fields are computed post-init."""
        summary = ValidationSummary(
            dataset_name="ds",
            total_rows=100,
            rows_passed=90,
            rules_checked=10,
            rules_passed=8,
            total_errors=5,
        )
        assert summary.rows_failed == 10
        assert summary.rules_failed == 2
        assert summary.passed is False

    def test_passing_summary(self) -> None:
        """Test a fully passing summary."""
        summary = ValidationSummary(
            dataset_name="ds",
            total_rows=50,
            rows_passed=50,
            rules_checked=5,
            rules_passed=5,
            total_errors=0,
        )
        assert summary.passed is True
        assert summary.validation_score == 1.0

    def test_timestamp_parse_from_string(self) -> None:
        """Test timestamp can be parsed from ISO string."""
        summary = ValidationSummary(dataset_name="ds", timestamp="2024-01-01T12:00:00")
        assert isinstance(summary.timestamp, datetime)
        assert summary.timestamp.year == 2024


class TestValidationReport:
    """Test the ValidationReport model."""

    def test_empty_report(self) -> None:
        """Test report with minimal summary."""
        summary = ValidationSummary(dataset_name="ds")
        report = ValidationReport(summary=summary)
        assert report.results == []
        assert report.failure_reasons == []

    def test_full_report_serialization(self) -> None:
        """Test full report JSON serialization."""
        summary = ValidationSummary(dataset_name="ds", total_rows=10)
        error = ValidationError(
            rule_name="range", rule_type="range", column="t", message="x", row_indices=[1]
        )
        result = ValidationResult(
            rule_name="range",
            rule_type="range",
            column="t",
            passed=False,
            errors=[error],
        )
        report = ValidationReport(
            summary=summary,
            results=[result],
            failure_reasons=["[error] x"],
            metadata={"schema": "crop"},
        )
        data = report.model_dump(mode="json")
        assert data["metadata"]["schema"] == "crop"
        assert data["failure_reasons"] == ["[error] x"]
        assert data["results"][0]["rule_name"] == "range"
