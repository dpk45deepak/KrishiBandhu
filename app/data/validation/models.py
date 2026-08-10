"""Pydantic models for the AgriMind AI validation framework.

These models are fully serializable (JSON, dict) and used throughout the
validation pipeline: rules emit :class:`ValidationResult`, the engine
aggregates into :class:`ValidationSummary`, and reports are built from
:class:`ValidationReport`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ValidationSeverity(StrEnum):
    """Severity level assigned to a validation error.

    Attributes:
        CRITICAL: Data cannot proceed; pipeline must halt.
        ERROR: A rule was violated; requires attention.
        WARNING: Potential quality concern; non-blocking.
        INFO: Informational observation.
    """

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @property
    def weight(self) -> float:
        """Return the numeric weight used when computing the validation score."""
        return {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.ERROR: 0.7,
            ValidationSeverity.WARNING: 0.3,
            ValidationSeverity.INFO: 0.0,
        }[self]


class ValidationRule(BaseModel):
    """Configuration for a single validation rule.

    Attributes:
        name: Unique rule name.
        type: Rule type (e.g. 'required_column', 'range').
        column: Target column name; optional for cross-column rules.
        parameters: Rule-specific parameters (min, max, allowed_values, etc.).
        severity: Severity assigned when the rule fails.
        enabled: Whether the rule is active.
        message: Optional custom failure message template.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Unique rule name")
    type: str = Field(..., min_length=1, description="Rule type identifier")
    column: str | None = Field(default=None, description="Target column name")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Rule-specific parameters")
    severity: ValidationSeverity = Field(
        default=ValidationSeverity.ERROR, description="Failure severity"
    )
    enabled: bool = Field(default=True, description="Whether the rule is active")
    message: str | None = Field(default=None, description="Custom failure message template")

    @field_validator("parameters")
    @classmethod
    def validate_parameters_not_none(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Ensure parameters is never None after model construction."""
        return value or {}


class ValidationError(BaseModel):
    """A single validation failure.

    Attributes:
        rule_name: Name of the rule that failed.
        rule_type: Type of the rule that failed.
        column: Column the error relates to (if any).
        severity: Severity of the failure.
        message: Human-readable failure description.
        row_indices: Optional list of affected row indices.
        row_count: Number of affected rows (0 if not row-specific).
        details: Optional structured payload for machine consumption.
    """

    model_config = ConfigDict(extra="forbid")

    rule_name: str
    rule_type: str
    column: str | None = None
    severity: ValidationSeverity = ValidationSeverity.ERROR
    message: str
    row_indices: list[int] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_row_count(self) -> ValidationError:
        """Derive row_count from row_indices when not explicitly set."""
        if self.row_count == 0 and self.row_indices:
            self.row_count = len(self.row_indices)
        return self


class ValidationResult(BaseModel):
    """Structured output produced by a single validation rule.

    Attributes:
        rule_name: Name of the rule that produced this result.
        rule_type: Type of the rule.
        column: Column the rule operated on (if any).
        passed: Whether the rule passed.
        severity: Severity of the rule when it fails.
        errors: List of individual validation errors.
        total_rows: Number of rows examined.
        failed_count: Number of rows that failed.
        details: Additional structured information.
    """

    model_config = ConfigDict(extra="forbid")

    rule_name: str
    rule_type: str
    column: str | None = None
    passed: bool
    severity: ValidationSeverity = ValidationSeverity.ERROR
    errors: list[ValidationError] = Field(default_factory=list)
    total_rows: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_failed_count(self) -> ValidationResult:
        """Derive failed_count from the union of failed row indices."""
        if self.failed_count == 0 and self.errors:
            failed_rows = {idx for error in self.errors for idx in error.row_indices}
            if failed_rows:
                self.failed_count = len(failed_rows)
        return self


class ColumnStatistic(BaseModel):
    """Per-column validation statistics for reports.

    Attributes:
        column: Column name.
        rules_checked: Number of rules that checked this column.
        rules_failed: Number of rules that failed for this column.
        errors: Count of individual validation errors.
        failed_rows: Number of rows failing at least one column rule.
    """

    model_config = ConfigDict(extra="forbid")

    column: str
    rules_checked: int = 0
    rules_failed: int = 0
    errors: int = 0
    failed_rows: int = 0


class RuleStatistic(BaseModel):
    """Statistics for a single rule across a validation run.

    Attributes:
        rule_name: Name of the rule.
        rule_type: Type of the rule.
        ran: Whether the rule executed.
        passed: Whether the rule passed.
        errors: Number of errors emitted.
        failed_rows: Number of rows that failed this rule.
    """

    model_config = ConfigDict(extra="forbid")

    rule_name: str
    rule_type: str
    ran: bool = False
    passed: bool = False
    errors: int = 0
    failed_rows: int = 0


class ValidationSummary(BaseModel):
    """Aggregated summary of a validation run.

    Attributes:
        dataset_name: Name of the validated dataset.
        total_rows: Number of rows validated.
        total_columns: Number of columns validated.
        rows_passed: Number of rows with no failures.
        rows_failed: Number of rows with at least one failure.
        rules_checked: Number of rule executions.
        rules_passed: Number of passing rule executions.
        rules_failed: Number of failing rule executions.
        total_errors: Total individual error count.
        severity_counts: Error counts broken down by severity.
        validation_score: Overall score in [0, 1].
        passed: Whether the dataset passed validation.
        strict_mode: Whether strict mode (fail-fast) was enabled.
        timestamp: When the validation ran.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    total_rows: int = 0
    total_columns: int = 0
    rows_passed: int = 0
    rows_failed: int = 0
    rules_checked: int = 0
    rules_passed: int = 0
    rules_failed: int = 0
    total_errors: int = 0
    severity_counts: dict[str, int] = Field(default_factory=dict)
    validation_score: float = Field(default=1.0, ge=0.0, le=1.0)
    passed: bool = True
    strict_mode: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context: Any) -> None:
        """Recompute derived fields after construction."""
        self.rules_failed = self.rules_checked - self.rules_passed
        self.rows_failed = self.total_rows - self.rows_passed
        self.passed = self.rules_failed == 0 and self.total_errors == 0

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value: Any) -> Any:
        """Accept ISO-8601 strings for timestamps."""
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value


class ValidationReport(BaseModel):
    """Complete serializable validation report.

    Attributes:
        summary: The aggregated validation summary.
        results: Per-rule validation results.
        column_statistics: Per-column statistics.
        rule_statistics: Per-rule statistics.
        failure_reasons: Human-readable list of failure descriptions.
        metadata: Arbitrary run metadata (schema path, engine, duration, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    summary: ValidationSummary
    results: list[ValidationResult] = Field(default_factory=list)
    column_statistics: list[ColumnStatistic] = Field(default_factory=list)
    rule_statistics: list[RuleStatistic] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
