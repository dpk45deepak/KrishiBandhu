"""Validation rules for the AgriMind AI validation framework.

Every rule inherits from :class:`BaseRule` and implements a single
responsibility. Rules are stateless with respect to data; all configuration
is injected via the constructor, making them reusable and independently
testable. The :class:`RuleRegistry` allows users to register custom rules
(including agricultural business rules) without modifying the engine.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any, ClassVar

import pandas as pd
from loguru import logger

from app.data.validation.exceptions import (
    InvalidSchemaException,
    MissingColumnException,
    RuleRegistrationError,
)
from app.data.validation.models import (
    ValidationError,
    ValidationResult,
    ValidationSeverity,
)
from app.data.validation.schema import ColumnDefinition, ValidationSchema

RuleFactory = Callable[..., "BaseRule"]
BusinessFunction = Callable[..., ValidationResult]


def _load_function(dotted_path: str) -> Callable[..., Any]:
    """Import a callable from a dotted path string.

    Args:
        dotted_path: Fully qualified import path (e.g. 'pkg.mod.func').

    Returns:
        The imported callable.

    Raises:
        ValueError: If the path is not dotted or the function is missing.
    """
    module_name, _, func_name = dotted_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Function path must be dotted: {dotted_path}")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _series_for(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a column series, raising a meaningful error if absent.

    Args:
        df: DataFrame to extract from.
        column: Column name.

    Returns:
        The requested column series.

    Raises:
        MissingColumnException: If the column is not present.
    """
    if column not in df.columns:
        raise MissingColumnException(
            f"Column '{column}' not found in dataset",
            details={"column": column, "available_columns": sorted(df.columns.tolist())},
        )
    return df[column]


def _resolve_column(rule: BaseRule, parameters: dict[str, Any]) -> str:
    """Resolve the target column for a rule from params or the rule itself.

    Args:
        rule: The rule being executed.
        parameters: Rule parameters that may contain a 'column' key.

    Returns:
        The resolved column name.

    Raises:
        MissingColumnException: If no column can be resolved.
    """
    column = parameters.get("column") or rule.column
    if column is None:
        raise MissingColumnException(
            f"Rule '{rule.name}' requires a 'column' parameter",
            details={"rule": rule.name, "rule_type": rule.rule_type},
        )
    return column


class BaseRule(ABC):
    """Abstract base class for all validation rules.

    Subclasses implement :meth:`validate` and provide a ``rule_type``.
    """

    rule_type: ClassVar[str] = "base"

    def __init__(
        self,
        name: str,
        column: str | None = None,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize a rule.

        Args:
            name: Unique rule name.
            column: Target column (None for dataset-level rules).
            parameters: Rule-specific parameters.
            severity: Severity when the rule fails.
            message: Optional custom failure message.
        """
        self.name = name
        self.column = column
        self.parameters = parameters or {}
        self.severity = self._coerce_severity(severity)
        self.message = message
        logger.debug(f"Instantiated {self.__class__.__name__} '{name}'")

    @staticmethod
    def _coerce_severity(severity: ValidationSeverity | str) -> ValidationSeverity:
        """Coerce a severity value into a ValidationSeverity enum member."""
        if isinstance(severity, ValidationSeverity):
            return severity
        return ValidationSeverity(severity)

    def make_error(
        self,
        message: str,
        row_indices: list[int] | None = None,
        details: dict[str, Any] | None = None,
    ) -> ValidationError:
        """Create a structured validation error for this rule.

        Args:
            message: Human-readable failure message.
            row_indices: Affected row indices.
            details: Optional structured payload.

        Returns:
            A ValidationError instance.
        """
        return ValidationError(
            rule_name=self.name,
            rule_type=self.rule_type,
            column=self.column,
            severity=self.severity,
            message=message,
            row_indices=list(row_indices or []),
            details=details or {},
        )

    def pass_result(
        self, df: pd.DataFrame, details: dict[str, Any] | None = None
    ) -> ValidationResult:
        """Build a passing ValidationResult.

        Args:
            df: DataFrame validated.
            details: Optional structured details.

        Returns:
            A passing ValidationResult.
        """
        return ValidationResult(
            rule_name=self.name,
            rule_type=self.rule_type,
            column=self.column,
            passed=True,
            severity=self.severity,
            total_rows=len(df),
            failed_count=0,
            details=details or {},
        )

    def fail_result(
        self,
        df: pd.DataFrame,
        errors: list[ValidationError],
        details: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Build a failing ValidationResult from a list of errors.

        Args:
            df: DataFrame validated.
            errors: Validation errors produced by the rule.
            details: Optional structured details.

        Returns:
            A failing ValidationResult.
        """
        failed_rows = {idx for error in errors for idx in error.row_indices}
        return ValidationResult(
            rule_name=self.name,
            rule_type=self.rule_type,
            column=self.column,
            passed=False,
            severity=self.severity,
            errors=errors,
            total_rows=len(df),
            failed_count=len(failed_rows),
            details=details or {},
        )

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Execute the rule against a DataFrame.

        Args:
            df: Pandas DataFrame to validate.

        Returns:
            A structured ValidationResult.
        """


class RequiredColumnRule(BaseRule):
    """Ensure that all required columns are present in the dataset."""

    rule_type = "required_column"

    def __init__(
        self,
        name: str = "required_column",
        column: str | None = None,
        required_columns: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize the rule with the set of required columns."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        self.required_columns = required_columns or ([column] if column else [])

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Check that all required columns exist in the DataFrame."""
        missing = [col for col in self.required_columns if col not in df.columns]
        if not missing:
            return self.pass_result(df, {"required_columns": self.required_columns})
        error = self.make_error(
            f"Required column(s) missing: {missing}",
            details={"missing_columns": missing, "required_columns": self.required_columns},
        )
        return self.fail_result(df, [error], {"missing_columns": missing})


class DataTypeRule(BaseRule):
    """Validate that column values match the expected data type."""

    rule_type = "dtype"

    def __init__(
        self,
        name: str = "dtype",
        column: str | None = None,
        dtype: str = "object",
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize the rule with an expected dtype."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        self.dtype = (parameters or {}).get("dtype", dtype)

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Check every non-null value against the expected dtype."""
        series = _series_for(df, self.column or "")
        invalid = self._invalid_mask(series)
        indices = series.index[invalid].tolist()
        if not indices:
            return self.pass_result(df, {"dtype": self.dtype})
        error = self.make_error(
            f"Column '{self.column}' contains values incompatible with dtype '{self.dtype}' "
            f"({len(indices)} row(s))",
            row_indices=indices,
            details={"dtype": self.dtype, "invalid_count": len(indices)},
        )
        return self.fail_result(df, [error])

    def _invalid_mask(self, series: pd.Series) -> pd.Series:
        """Compute a boolean mask of rows failing the dtype check."""
        dtype = self.dtype
        notna = series.notna()
        if dtype in ("int", "integer"):
            numeric = pd.to_numeric(series, errors="coerce")
            return notna & (numeric.isna() | (numeric % 1 != 0))
        if dtype in ("float", "number"):
            numeric = pd.to_numeric(series, errors="coerce")
            return notna & numeric.isna()
        if dtype == "bool":
            truthy = {True, False, "true", "false", "True", "False"}
            return notna & ~series.isin(truthy)
        if dtype in ("datetime", "date"):
            parsed = pd.to_datetime(series, errors="coerce")
            return notna & parsed.isna()
        if dtype == "str":
            return notna & series.map(lambda value: not isinstance(value, str))
        return pd.Series(False, index=series.index)


class NullValueRule(BaseRule):
    """Validate that null values are within allowed proportions."""

    rule_type = "null_value"

    def __init__(
        self,
        name: str = "null_value",
        column: str | None = None,
        nullable: bool = True,
        max_null_ratio: float = 1.0,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize with nullability and a maximum acceptable null ratio."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        self.nullable = nullable
        self.max_null_ratio = max_null_ratio

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Count nulls and fail if not nullable or ratio exceeds the limit."""
        series = _series_for(df, self.column or "")
        null_mask = series.isna()
        null_count = int(null_mask.sum())
        null_ratio = null_count / len(series) if len(series) else 0.0
        if null_count == 0:
            return self.pass_result(df, {"null_count": 0, "null_ratio": 0.0})

        if self.nullable and null_ratio <= self.max_null_ratio:
            return self.pass_result(
                df,
                {
                    "null_count": null_count,
                    "null_ratio": null_ratio,
                    "max_null_ratio": self.max_null_ratio,
                },
            )

        error = self.make_error(
            f"Column '{self.column}' has {null_count} null value(s) "
            f"({null_ratio:.2%}); nullable={self.nullable}, "
            f"max_null_ratio={self.max_null_ratio}",
            row_indices=series.index[null_mask].tolist(),
            details={"null_count": null_count, "null_ratio": null_ratio},
        )
        return self.fail_result(
            df,
            [error],
            {
                "null_count": null_count,
                "null_ratio": null_ratio,
                "max_null_ratio": self.max_null_ratio,
            },
        )


class DuplicateRule(BaseRule):
    """Detect duplicate rows in the dataset."""

    rule_type = "duplicate"

    def __init__(
        self,
        name: str = "duplicate",
        column: str | None = None,
        subset: list[str] | None = None,
        max_duplicates: int = 0,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.WARNING,
        message: str | None = None,
    ) -> None:
        """Initialize with an optional subset and a duplicate tolerance."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        self.subset = subset
        self.max_duplicates = max_duplicates

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Flag rows that duplicate another row (within the subset)."""
        duplicate_mask = df.duplicated(subset=self.subset, keep="first")
        indices = df.index[duplicate_mask].tolist()
        if len(indices) <= self.max_duplicates:
            return self.pass_result(
                df, {"duplicate_count": len(indices), "max_duplicates": self.max_duplicates}
            )
        error = self.make_error(
            f"Found {len(indices)} duplicate row(s) (max allowed: {self.max_duplicates})",
            row_indices=indices,
            details={"duplicate_count": len(indices), "max_duplicates": self.max_duplicates},
        )
        return self.fail_result(df, [error])


class UniqueRule(BaseRule):
    """Validate that a column contains only unique values."""

    rule_type = "unique"

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Flag rows whose value duplicates an earlier value in the column."""
        series = _series_for(df, self.column or "")
        duplicate_mask = series.duplicated(keep="first")
        indices = series.index[duplicate_mask].tolist()
        if not indices:
            return self.pass_result(df, {"duplicate_count": 0})
        error = self.make_error(
            f"Column '{self.column}' contains {len(indices)} duplicate value(s)",
            row_indices=indices,
            details={"duplicate_count": len(indices)},
        )
        return self.fail_result(df, [error])


class RangeRule(BaseRule):
    """Validate that numeric values fall within [min_value, max_value]."""

    rule_type = "range"

    def __init__(
        self,
        name: str = "range",
        column: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize with inclusive numeric bounds."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        params = parameters or {}
        self.min_value = params.get("min", min_value)
        self.max_value = params.get("max", max_value)

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Flag rows outside the configured numeric bounds."""
        series = _series_for(df, self.column or "")
        numeric = pd.to_numeric(series, errors="coerce")
        invalid_mask = pd.Series(False, index=series.index)
        if self.min_value is not None:
            invalid_mask |= numeric < self.min_value
        if self.max_value is not None:
            invalid_mask |= numeric > self.max_value
        invalid_mask &= series.notna()

        indices = series.index[invalid_mask].tolist()
        if not indices:
            return self.pass_result(df, {"min": self.min_value, "max": self.max_value})
        bounds = self._format_bounds()
        error = self.make_error(
            f"Column '{self.column}' has {len(indices)} value(s) outside range {bounds}",
            row_indices=indices,
            details={"min": self.min_value, "max": self.max_value, "invalid_count": len(indices)},
        )
        return self.fail_result(df, [error])

    def _format_bounds(self) -> str:
        """Format the configured bounds for error messages."""
        lower = "-inf" if self.min_value is None else self.min_value
        upper = "inf" if self.max_value is None else self.max_value
        return f"[{lower}, {upper}]"


class RegexRule(BaseRule):
    """Validate that string values match a regular expression pattern."""

    rule_type = "regex"

    def __init__(
        self,
        name: str = "regex",
        column: str | None = None,
        pattern: str | None = None,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize with a regex pattern."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        self.pattern = (parameters or {}).get("pattern", pattern)

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Flag non-null values that do not match the pattern."""
        if not self.pattern:
            raise RuleRegistrationError(f"Rule '{self.name}' requires a 'pattern' parameter")
        series = _series_for(df, self.column or "")
        as_string = series.astype(str)
        valid_mask = as_string.str.match(self.pattern, na=False)
        invalid_mask = series.notna() & ~valid_mask
        indices = series.index[invalid_mask].tolist()
        if not indices:
            return self.pass_result(df, {"pattern": self.pattern})
        error = self.make_error(
            f"Column '{self.column}' has {len(indices)} value(s) not matching pattern '{self.pattern}'",
            row_indices=indices,
            details={"pattern": self.pattern, "invalid_count": len(indices)},
        )
        return self.fail_result(df, [error])


class AllowedValuesRule(BaseRule):
    """Validate that values belong to a fixed allowlist."""

    rule_type = "allowed_values"

    def __init__(
        self,
        name: str = "allowed_values",
        column: str | None = None,
        allowed_values: list[Any] | None = None,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize with the list of allowed values."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        self.allowed_values = list((parameters or {}).get("allowed_values", allowed_values or []))

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Flag values not present in the allowlist."""
        series = _series_for(df, self.column or "")
        invalid_mask = series.notna() & ~series.isin(self.allowed_values)
        indices = series.index[invalid_mask].tolist()
        if not indices:
            return self.pass_result(df, {"allowed_values": self.allowed_values})
        error = self.make_error(
            f"Column '{self.column}' has {len(indices)} value(s) not in the allowed set",
            row_indices=indices,
            details={
                "allowed_values": self.allowed_values,
                "invalid_count": len(indices),
                "unexpected_values": sorted(set(series.loc[indices].astype(str))),
            },
        )
        return self.fail_result(df, [error])


class MinLengthRule(BaseRule):
    """Validate minimum string length."""

    rule_type = "min_length"

    def __init__(
        self,
        name: str = "min_length",
        column: str | None = None,
        min_length: int = 0,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize with a minimum string length."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        self.min_length = int((parameters or {}).get("min_length", min_length))

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Flag non-null strings shorter than the minimum."""
        series = _series_for(df, self.column or "")
        lengths = series.astype(str).str.len()
        invalid_mask = series.notna() & (lengths < self.min_length)
        indices = series.index[invalid_mask].tolist()
        if not indices:
            return self.pass_result(df, {"min_length": self.min_length})
        error = self.make_error(
            f"Column '{self.column}' has {len(indices)} value(s) shorter than {self.min_length} chars",
            row_indices=indices,
            details={"min_length": self.min_length, "invalid_count": len(indices)},
        )
        return self.fail_result(df, [error])


class MaxLengthRule(BaseRule):
    """Validate maximum string length."""

    rule_type = "max_length"

    def __init__(
        self,
        name: str = "max_length",
        column: str | None = None,
        max_length: int = 0,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize with a maximum string length."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        self.max_length = int((parameters or {}).get("max_length", max_length))

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Flag non-null strings longer than the maximum."""
        series = _series_for(df, self.column or "")
        lengths = series.astype(str).str.len()
        invalid_mask = series.notna() & (lengths > self.max_length)
        indices = series.index[invalid_mask].tolist()
        if not indices:
            return self.pass_result(df, {"max_length": self.max_length})
        error = self.make_error(
            f"Column '{self.column}' has {len(indices)} value(s) longer than {self.max_length} chars",
            row_indices=indices,
            details={"max_length": self.max_length, "invalid_count": len(indices)},
        )
        return self.fail_result(df, [error])


class ConstantColumnRule(BaseRule):
    """Validate that a column is constant (single value) or equals a fixed value."""

    rule_type = "constant_column"

    def __init__(
        self,
        name: str = "constant_column",
        column: str | None = None,
        constant_value: Any | None = None,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.WARNING,
        message: str | None = None,
    ) -> None:
        """Initialize with an optional expected constant value."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        self.constant_value = (parameters or {}).get("constant_value", constant_value)

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Flag rows that diverge from the expected constant value."""
        series = _series_for(df, self.column or "")
        if self.constant_value is not None:
            invalid_mask = series.notna() & (series != self.constant_value)
            message = f"Column '{self.column}' must equal constant '{self.constant_value}'"
        else:
            non_null = series.dropna()
            if non_null.empty:
                return self.pass_result(df, {"constant": True, "mode": "detect"})
            invalid_mask = series.notna() & (series != non_null.iloc[0])
            message = f"Column '{self.column}' is not constant"

        indices = series.index[invalid_mask].tolist()
        if not indices:
            return self.pass_result(df, {"constant": True, "constant_value": self.constant_value})
        error = self.make_error(
            f"{message} ({len(indices)} violating row(s))",
            row_indices=indices,
            details={"constant_value": self.constant_value, "invalid_count": len(indices)},
        )
        return self.fail_result(df, [error])


class BusinessRule(BaseRule):
    """Wrap an arbitrary callable as a reusable validation rule.

    Business rules receive ``(df, parameters, rule)`` and return a
    :class:`ValidationResult`. New agricultural rules can be registered
    without touching the engine by adding them to the :class:`RuleRegistry`.
    """

    rule_type = "business"

    def __init__(
        self,
        name: str,
        function: BusinessFunction | None = None,
        function_path: str | None = None,
        column: str | None = None,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize with either a callable or an importable path."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        if function is None and function_path:
            function = _load_function(function_path)
        if function is None:
            raise RuleRegistrationError(
                f"Business rule '{name}' requires a callable or function_path"
            )
        self.function = function

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Delegate validation to the wrapped callable."""
        return self.function(df, self.parameters, self)


class CrossColumnRule(BaseRule):
    """Validate relationships between two columns.

    Supports either a comparison operator (eq/ne/gt/ge/lt/le) or a custom
    callable with signature ``(df, parameters, rule) -> ValidationResult``.
    """

    rule_type = "cross_column"

    _OPERATORS: ClassVar[dict[str, Callable[[pd.Series, pd.Series], pd.Series]]] = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
    }

    def __init__(
        self,
        name: str = "cross_column",
        column: str | None = None,
        column_a: str | None = None,
        column_b: str | None = None,
        comparison: str | None = None,
        function: BusinessFunction | None = None,
        parameters: dict[str, Any] | None = None,
        severity: ValidationSeverity | str = ValidationSeverity.ERROR,
        message: str | None = None,
    ) -> None:
        """Initialize with columns and a comparison or custom function."""
        super().__init__(
            name=name,
            column=column,
            parameters=parameters,
            severity=severity,
            message=message,
        )
        params = parameters or {}
        self.column_a = params.get("column_a", column_a)
        self.column_b = params.get("column_b", column_b)
        self.comparison = params.get("comparison", comparison)
        self.function = function

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Run the cross-column comparison or custom function."""
        if self.function is not None:
            return self.function(df, self.parameters, self)
        if self.column_a is None or self.column_b is None:
            raise RuleRegistrationError(
                f"Cross-column rule '{self.name}' requires 'column_a' and 'column_b'"
            )
        if self.comparison not in self._OPERATORS:
            raise RuleRegistrationError(
                f"Rule '{self.name}' has unsupported comparison '{self.comparison}'"
            )

        col_a = _series_for(df, self.column_a)
        col_b = _series_for(df, self.column_b)
        both_present = col_a.notna() & col_b.notna()
        valid = self._OPERATORS[self.comparison](col_a, col_b)
        invalid_mask = both_present & ~valid
        indices = df.index[invalid_mask].tolist()
        if not indices:
            return self.pass_result(
                df,
                {
                    "column_a": self.column_a,
                    "column_b": self.column_b,
                    "comparison": self.comparison,
                },
            )
        error = self.make_error(
            f"Cross-column check failed: '{self.column_a}' {self.comparison} "
            f"'{self.column_b}' violated in {len(indices)} row(s)",
            row_indices=indices,
            details={
                "column_a": self.column_a,
                "column_b": self.column_b,
                "comparison": self.comparison,
            },
        )
        return self.fail_result(df, [error])


class RuleRegistry:
    """Registry mapping rule type names to rule factories.

    Users extend the framework by calling :meth:`register` with a new type
    name and a callable (a rule class or factory) without modifying the
    :class:`ValidationEngine`.
    """

    def __init__(self) -> None:
        """Initialize the registry with an empty factory map."""
        self._factories: dict[str, RuleFactory] = {}

    def register(self, rule_type: str, factory: RuleFactory) -> None:
        """Register a rule factory under a type name.

        Args:
            rule_type: Unique rule type identifier.
            factory: Callable returning a BaseRule.
        """
        if not callable(factory):
            raise RuleRegistrationError(f"Factory for '{rule_type}' is not callable")
        if rule_type in self._factories:
            logger.warning(f"Overwriting existing rule factory for '{rule_type}'")
        self._factories[rule_type] = factory
        logger.debug(f"Registered rule factory '{rule_type}'")

    def unregister(self, rule_type: str) -> None:
        """Remove a rule factory from the registry.

        Args:
            rule_type: Rule type to remove.
        """
        self._factories.pop(rule_type, None)
        logger.debug(f"Unregistered rule factory '{rule_type}'")

    def has(self, rule_type: str) -> bool:
        """Return whether a rule type is registered."""
        return rule_type in self._factories

    def select(self, rule_types: list[str]) -> list[str]:
        """Return the subset of requested rule types that are registered."""
        return [rt for rt in rule_types if self.has(rt)]

    def create(self, rule_type: str, **kwargs: Any) -> BaseRule:
        """Instantiate a rule from the registry.

        Args:
            rule_type: Registered rule type.
            **kwargs: Constructor arguments for the rule.

        Returns:
            A configured BaseRule instance.

        Raises:
            RuleRegistrationError: If the type is not registered.
        """
        factory = self._factories.get(rule_type)
        if factory is None:
            raise RuleRegistrationError(
                f"No rule factory registered for type '{rule_type}'. "
                f"Available types: {sorted(self._factories)}"
            )
        return factory(**kwargs)

    def rule_types(self) -> list[str]:
        """Return all registered rule type names."""
        return sorted(self._factories)


# ---------------------------------------------------------------------------
# Agricultural business rules
# ---------------------------------------------------------------------------


def _numeric_bounds_check(
    df: pd.DataFrame,
    rule: BusinessRule,
    lower: float,
    upper: float,
    label: str,
) -> ValidationResult:
    """Shared implementation for numeric range business rules."""
    series = _series_for(df, rule.column or "")
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_mask = series.notna() & ((numeric < lower) | (numeric > upper))
    indices = series.index[invalid_mask].tolist()
    if not indices:
        return rule.pass_result(df, {"column": rule.column, "min": lower, "max": upper})
    error = rule.make_error(
        f"{rule.name}: {label} must be between {lower} and {upper} "
        f"({len(indices)} row(s) violated)",
        row_indices=indices,
        details={"min": lower, "max": upper, "invalid_count": len(indices)},
    )
    return rule.fail_result(df, [error])


def _non_negative_check(df: pd.DataFrame, rule: BusinessRule, label: str) -> ValidationResult:
    """Shared implementation for non-negative business rules."""
    series = _series_for(df, rule.column or "")
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_mask = series.notna() & (numeric < 0)
    indices = series.index[invalid_mask].tolist()
    if not indices:
        return rule.pass_result(df, {"column": rule.column, "min": 0})
    error = rule.make_error(
        f"{rule.name}: {label} must be >= 0 ({len(indices)} row(s) violated)",
        row_indices=indices,
        details={"min": 0, "invalid_count": len(indices)},
    )
    return rule.fail_result(df, [error])


def temperature_range(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: temperature must be between -20 and 60 degrees Celsius."""
    rule.column = _resolve_column(rule, parameters)
    return _numeric_bounds_check(df, rule, -20.0, 60.0, "Temperature (°C)")


def humidity_range(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: humidity must be between 0 and 100 percent."""
    rule.column = _resolve_column(rule, parameters)
    return _numeric_bounds_check(df, rule, 0.0, 100.0, "Humidity (%)")


def rainfall_non_negative(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: rainfall must be non-negative (mm)."""
    rule.column = _resolve_column(rule, parameters)
    return _non_negative_check(df, rule, "Rainfall (mm)")


def nitrogen_range(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: nitrogen must be between 0 and 300 kg/ha."""
    rule.column = _resolve_column(rule, parameters)
    return _numeric_bounds_check(df, rule, 0.0, 300.0, "Nitrogen (kg/ha)")


def phosphorus_range(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: phosphorus must be between 0 and 150 kg/ha."""
    rule.column = _resolve_column(rule, parameters)
    return _numeric_bounds_check(df, rule, 0.0, 150.0, "Phosphorus (kg/ha)")


def potassium_range(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: potassium must be between 0 and 300 kg/ha."""
    rule.column = _resolve_column(rule, parameters)
    return _numeric_bounds_check(df, rule, 0.0, 300.0, "Potassium (kg/ha)")


def yield_non_negative(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: yield must be non-negative."""
    rule.column = _resolve_column(rule, parameters)
    return _non_negative_check(df, rule, "Yield")


def crop_name_not_empty(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: crop name cannot be empty or blank."""
    rule.column = _resolve_column(rule, parameters)
    series = _series_for(df, rule.column)
    stripped = series.fillna("").astype(str).str.strip()
    invalid_mask = stripped == ""
    indices = series.index[invalid_mask].tolist()
    if not indices:
        return rule.pass_result(df, {"column": rule.column})
    error = rule.make_error(
        f"{rule.name}: Crop name cannot be empty ({len(indices)} row(s) violated)",
        row_indices=indices,
        details={"invalid_count": len(indices)},
    )
    return rule.fail_result(df, [error])


def state_not_numeric(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: state must not be a numeric value."""
    rule.column = _resolve_column(rule, parameters)
    series = _series_for(df, rule.column)
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_mask = numeric.notna()
    indices = series.index[invalid_mask].tolist()
    if not indices:
        return rule.pass_result(df, {"column": rule.column})
    error = rule.make_error(
        f"{rule.name}: State cannot be numeric ({len(indices)} row(s) violated)",
        row_indices=indices,
        details={"invalid_count": len(indices)},
    )
    return rule.fail_result(df, [error])


def district_not_null(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: district must not be null."""
    rule.column = _resolve_column(rule, parameters)
    series = _series_for(df, rule.column)
    null_mask = series.isna()
    indices = series.index[null_mask].tolist()
    if not indices:
        return rule.pass_result(df, {"column": rule.column})
    error = rule.make_error(
        f"{rule.name}: District cannot be null ({len(indices)} row(s) violated)",
        row_indices=indices,
        details={"invalid_count": len(indices)},
    )
    return rule.fail_result(df, [error])


def date_not_in_future(
    df: pd.DataFrame, parameters: dict[str, Any], rule: BaseRule
) -> ValidationResult:
    """Business rule: date must not be in the future."""
    rule.column = _resolve_column(rule, parameters)
    series = _series_for(df, rule.column)
    parsed = pd.to_datetime(series, errors="coerce")
    today = pd.Timestamp(datetime.now().date())
    invalid_mask = parsed.notna() & (parsed > today)
    indices = series.index[invalid_mask].tolist()
    if not indices:
        return rule.pass_result(df, {"column": rule.column})
    error = rule.make_error(
        f"{rule.name}: Date cannot be in the future ({len(indices)} row(s) violated)",
        row_indices=indices,
        details={"invalid_count": len(indices)},
    )
    return rule.fail_result(df, [error])


def _business_factory(
    rule_type: str, function: BusinessFunction, default_column: str
) -> RuleFactory:
    """Create a registry factory that wraps a business function in a BusinessRule."""

    def factory(**kwargs: Any) -> BusinessRule:
        return BusinessRule(
            name=kwargs.get("name", function.__name__),
            column=kwargs.get("column", default_column),
            function=function,
            parameters=kwargs.get("parameters", {}),
            severity=kwargs.get("severity", ValidationSeverity.ERROR),
            message=kwargs.get("message"),
        )

    return factory


def register_agri_business_rules(registry: RuleRegistry) -> None:
    """Register all built-in agricultural business rules on a registry.

    Args:
        registry: The registry to populate.
    """
    rule_factories = [
        ("business.temperature", temperature_range, "temperature"),
        ("business.humidity", humidity_range, "humidity"),
        ("business.rainfall", rainfall_non_negative, "rainfall"),
        ("business.nitrogen", nitrogen_range, "nitrogen"),
        ("business.phosphorus", phosphorus_range, "phosphorus"),
        ("business.potassium", potassium_range, "potassium"),
        ("business.yield", yield_non_negative, "yield"),
        ("business.crop_name", crop_name_not_empty, "crop_name"),
        ("business.state", state_not_numeric, "state"),
        ("business.district", district_not_null, "district"),
        ("business.date", date_not_in_future, "date"),
    ]
    for rule_type, function, column in rule_factories:
        registry.register(rule_type, _business_factory(rule_type, function, column))


def _build_default_registry() -> RuleRegistry:
    """Construct the default registry with all built-in rule types."""
    registry = RuleRegistry()
    registry.register("required_column", RequiredColumnRule)
    registry.register("dtype", DataTypeRule)
    registry.register("data_type", DataTypeRule)
    registry.register("null_value", NullValueRule)
    registry.register("null", NullValueRule)
    registry.register("duplicate", DuplicateRule)
    registry.register("duplicate_rows", DuplicateRule)
    registry.register("unique", UniqueRule)
    registry.register("range", RangeRule)
    registry.register("regex", RegexRule)
    registry.register("allowed_values", AllowedValuesRule)
    registry.register("min_length", MinLengthRule)
    registry.register("max_length", MaxLengthRule)
    registry.register("constant_column", ConstantColumnRule)
    registry.register("business", BusinessRule)
    registry.register("cross_column", CrossColumnRule)
    register_agri_business_rules(registry)
    logger.debug(f"Default registry built with {len(registry.rule_types())} rule types")
    return registry


DEFAULT_REGISTRY = _build_default_registry()


# ---------------------------------------------------------------------------
# Schema -> rule construction
# ---------------------------------------------------------------------------


def _rules_for_column(col_def: ColumnDefinition, registry: RuleRegistry) -> list[BaseRule]:
    """Build the rule set for a single schema column definition."""
    rules: list[BaseRule] = []

    if col_def.required:
        rules.append(
            registry.create(
                "required_column",
                name=f"required_{col_def.name}",
                column=col_def.name,
                required_columns=[col_def.name],
                severity=ValidationSeverity.ERROR,
            )
        )

    rules.append(
        registry.create(
            "dtype",
            name=f"dtype_{col_def.name}",
            column=col_def.name,
            dtype=col_def.dtype,
            severity=ValidationSeverity.ERROR,
        )
    )

    # Always add a null rule; the engine config supplies the default ratio.
    rules.append(
        registry.create(
            "null_value",
            name=f"null_{col_def.name}",
            column=col_def.name,
            nullable=col_def.nullable,
            severity=ValidationSeverity.ERROR,
        )
    )

    if col_def.unique:
        rules.append(
            registry.create(
                "unique",
                name=f"unique_{col_def.name}",
                column=col_def.name,
                severity=ValidationSeverity.ERROR,
            )
        )

    if col_def.allowed_values is not None:
        rules.append(
            registry.create(
                "allowed_values",
                name=f"allowed_{col_def.name}",
                column=col_def.name,
                allowed_values=col_def.allowed_values,
                severity=ValidationSeverity.ERROR,
            )
        )

    if col_def.regex:
        rules.append(
            registry.create(
                "regex",
                name=f"regex_{col_def.name}",
                column=col_def.name,
                pattern=col_def.regex,
                severity=ValidationSeverity.ERROR,
            )
        )

    if col_def.min_value is not None or col_def.max_value is not None:
        rules.append(
            registry.create(
                "range",
                name=f"range_{col_def.name}",
                column=col_def.name,
                parameters={"min": col_def.min_value, "max": col_def.max_value},
                severity=ValidationSeverity.ERROR,
            )
        )

    if col_def.min_length is not None:
        rules.append(
            registry.create(
                "min_length",
                name=f"min_length_{col_def.name}",
                column=col_def.name,
                min_length=col_def.min_length,
                severity=ValidationSeverity.ERROR,
            )
        )

    if col_def.max_length is not None:
        rules.append(
            registry.create(
                "max_length",
                name=f"max_length_{col_def.name}",
                column=col_def.name,
                max_length=col_def.max_length,
                severity=ValidationSeverity.ERROR,
            )
        )

    if "constant_value" in col_def.model_fields_set:
        rules.append(
            registry.create(
                "constant_column",
                name=f"constant_{col_def.name}",
                column=col_def.name,
                constant_value=col_def.constant_value,
                severity=ValidationSeverity.WARNING,
            )
        )

    return rules


def build_rules_from_schema(
    schema: ValidationSchema,
    registry: RuleRegistry | None = None,
) -> list[BaseRule]:
    """Instantiate every rule implied by a validation schema.

    Args:
        schema: The parsed validation schema.
        registry: Registry used to create rules (defaults to built-ins).

    Returns:
        A list of configured BaseRule instances.

    Raises:
        InvalidSchemaException: If a business rule function cannot be imported.
    """
    registry = registry or DEFAULT_REGISTRY
    rules: list[BaseRule] = []

    for _col_name, col_def in schema.columns.items():
        rules.extend(_rules_for_column(col_def, registry))

    for bdef in schema.business_rules:
        try:
            function = _load_function(bdef.function)
        except Exception as exc:
            raise InvalidSchemaException(
                f"Cannot import business rule function '{bdef.function}': {exc}",
                details={"rule": bdef.name, "function": bdef.function},
            ) from exc
        rules.append(
            BusinessRule(
                name=bdef.name,
                function=function,
                column=bdef.parameters.get("column"),
                parameters=bdef.parameters,
                severity=bdef.severity,
            )
        )

    logger.info(
        f"Built {len(rules)} validation rules from schema '{schema.name}' "
        f"({len(schema.columns)} columns, {len(schema.business_rules)} business rules)"
    )
    return rules
