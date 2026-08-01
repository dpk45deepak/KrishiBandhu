"""Validation engine for the AgriMind AI validation framework.

The :class:`ValidationEngine` orchestrates the entire validation pipeline:
schema loading, rule construction, execution, aggregation, and reporting.
It supports single dataframes, multiple dataframes, and directory-level
validation with fail-fast and strict modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from loguru import logger
from pydantic import BaseModel, Field

from app.data.ingestion.loader import DataLoader
from app.data.validation.exceptions import (
    BusinessRuleViolation,
    InvalidSchemaException,
    ValidationException,
)
from app.data.validation.models import (
    ColumnStatistic,
    RuleStatistic,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    ValidationSummary,
)
from app.data.validation.rules import (
    DEFAULT_REGISTRY,
    BaseRule,
    RuleRegistry,
    build_rules_from_schema,
)
from app.data.validation.schema import SchemaLoader, ValidationSchema
from app.utils.dataset_scanner import DatasetScanner


class ValidationEngineConfig(BaseModel):
    """Runtime configuration for the validation engine.

    Attributes:
        strict_mode: Raise on the first ERROR-level failure.
        fail_fast: Stop rule execution after the first failing rule.
        max_missing_percentage: Default max null ratio for null_value rules.
        duplicate_threshold: Default max duplicate rows for duplicate rules.
        report_generation: Whether reports are generated after validation.
    """

    strict_mode: bool = False
    fail_fast: bool = False
    max_missing_percentage: float = Field(default=0.1, ge=0.0, le=1.0)
    duplicate_threshold: int = Field(default=0, ge=0)
    report_generation: bool = True


class ValidationEngine:
    """Coordinates schema loading, rule execution, and result aggregation.

    The engine is the single entry point for validating datasets. It accepts
    pandas DataFrames or file paths, runs every configured rule, and returns
    a :class:`ValidationReport` with rich statistics.
    """

    def __init__(
        self,
        schema: str | Path | dict[str, Any] | ValidationSchema | None = None,
        registry: RuleRegistry | None = None,
        config: ValidationEngineConfig | None = None,
        schema_dir: str | Path | None = None,
    ) -> None:
        """Initialize the validation engine.

        Args:
            schema: Schema dict, YAML path, or parsed ValidationSchema.
            registry: Rule registry (defaults to built-in rules).
            config: Runtime engine configuration.
            schema_dir: Optional directory to search for schema files.
        """
        self._loader = SchemaLoader(schema_dir=schema_dir)
        self._data_loader = DataLoader(engine="pandas")
        self.registry = registry or DEFAULT_REGISTRY
        self.config = config or ValidationEngineConfig()
        self.schema = self._resolve_schema(schema)
        self.validate = self._bound_validate()
        logger.debug(
            f"ValidationEngine initialized | schema={self.schema.name if self.schema else None} "
            f"| strict={self.config.strict_mode} | fail_fast={self.config.fail_fast}"
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _resolve_schema(
        self, schema: str | Path | dict[str, Any] | ValidationSchema | None
    ) -> ValidationSchema | None:
        """Resolve a schema from any supported input form."""
        if schema is None:
            logger.debug("No schema provided; engine will accept explicit rule lists")
            return None
        if isinstance(schema, ValidationSchema):
            return schema
        return self._loader.load(schema)

    def _bound_validate(self) -> Any:
        """Return a validate callable that supports rule-list overrides."""

        def validate(
            df: pd.DataFrame,
            dataset_name: str | None = None,
            rules: list[BaseRule] | None = None,
        ) -> ValidationReport:
            return self.validate_dataframe(df, dataset_name, rules)

        return validate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_rule(self, rule_type: str, factory: Any) -> None:
        """Register a custom rule factory with the engine.

        Args:
            rule_type: Unique rule type identifier.
            factory: Callable returning a BaseRule.
        """
        self.registry.register(rule_type, factory)

    def validate_dataframe(
        self,
        df: pd.DataFrame | pl.DataFrame,
        dataset_name: str | None = None,
        rules: list[BaseRule] | None = None,
    ) -> ValidationReport:
        """Validate a single DataFrame (pandas or polars).

        Args:
            df: DataFrame to validate.
            dataset_name: Display name for the dataset.
            rules: Explicit rule list; defaults to rules built from the schema.

        Returns:
            A complete ValidationReport.

        Raises:
            InvalidSchemaException: If no schema or rules are configured.
            ValidationException: If fail-fast mode encounters a failure.
        """
        name = dataset_name or "unknown_dataset"
        df = self._to_pandas(df)
        df = self._apply_aliases(df)
        effective_rules = rules if rules is not None else self._build_rules()

        if not effective_rules:
            raise InvalidSchemaException(
                f"No validation rules available for dataset '{name}'",
                details={"dataset": name},
            )

        logger.info(f"Validating '{name}' with {len(effective_rules)} rules")
        collected: list[ValidationResult] = []

        for rule in effective_rules:
            result = self._execute_rule(rule, df)
            if result is not None:
                collected.append(result)
                if self._should_halt(result):
                    self._raise_fail_fast(name, collected)

        report = self._build_report(name, df, collected)
        self._log_summary(report)
        if self.config.strict_mode and not report.summary.passed:
            raise BusinessRuleViolation(
                f"Dataset '{name}' failed strict validation",
                details={"validation_score": report.summary.validation_score},
            )
        return report

    def load_and_validate(
        self,
        file_path: str | Path,
        dataset_name: str | None = None,
    ) -> ValidationReport:
        """Load a dataset from disk and validate it.

        Args:
            file_path: Path to a supported dataset file.
            dataset_name: Optional display name.

        Returns:
            A complete ValidationReport.
        """
        file_path = Path(file_path)
        name = dataset_name or file_path.stem
        logger.info(f"Loading dataset '{name}' from {file_path}")
        raw = self._data_loader.load(file_path)
        df = self._to_pandas(raw)
        report = self.validate_dataframe(df, dataset_name=name)
        report.metadata["file_path"] = str(file_path)
        return report

    def validate_multiple(
        self,
        dataframes: dict[str, pd.DataFrame],
    ) -> dict[str, ValidationReport]:
        """Validate multiple named DataFrames.

        Args:
            dataframes: Mapping of dataset name to DataFrame.

        Returns:
            Mapping of dataset name to ValidationReport.
        """
        reports: dict[str, ValidationReport] = {}
        for name, df in dataframes.items():
            logger.debug(f"Validating dataset '{name}' from mapping")
            reports[name] = self.validate_dataframe(df, dataset_name=name)
        return reports

    def validate_directory(
        self,
        directory: str | Path,
        pattern: str = "*",
        recursive: bool = True,
    ) -> dict[str, ValidationReport]:
        """Validate every supported dataset in a directory.

        Args:
            directory: Directory to scan for datasets.
            pattern: Glob pattern to filter files.
            recursive: Whether to scan subdirectories.

        Returns:
            Mapping of dataset name to ValidationReport.
        """
        scanner = DatasetScanner(data_dir=str(directory))
        results = scanner.scan(recursive=recursive)
        reports: dict[str, ValidationReport] = {}
        for ds in results:
            if pattern != "*" and not Path(ds.file_path).name.endswith(pattern[1:]):
                continue
            try:
                reports[Path(ds.file_path).stem] = self.load_and_validate(ds.file_path)
            except ValidationException as exc:
                logger.error(f"Skipping '{ds.filename}': {exc}")
        return reports

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _build_rules(self) -> list[BaseRule]:
        """Build rule instances from the engine schema."""
        if self.schema is None:
            return []
        rules = build_rules_from_schema(self.schema, registry=self.registry)
        return self._apply_config_defaults(rules)

    def _apply_config_defaults(self, rules: list[BaseRule]) -> list[BaseRule]:
        """Overlay engine-config defaults on rules that support them."""
        from app.data.validation.rules import DuplicateRule, NullValueRule

        for rule in rules:
            if isinstance(rule, NullValueRule):
                if rule.max_null_ratio >= 1.0 and rule.nullable:
                    rule.max_null_ratio = self.config.max_missing_percentage
            if isinstance(rule, DuplicateRule) and rule.max_duplicates == 0:
                rule.max_duplicates = self.config.duplicate_threshold
        return rules

    def _apply_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns that match schema aliases to canonical names."""
        if self.schema is None:
            return df
        alias_map = self.schema.all_aliases
        rename_map = {col: alias_map[col] for col in df.columns if col in alias_map}
        if rename_map:
            logger.info(f"Applied column aliases: {rename_map}")
            df = df.rename(columns=rename_map)
        return df

    def _execute_rule(self, rule: BaseRule, df: pd.DataFrame) -> ValidationResult | None:
        """Execute a rule, skipping gracefully when its column is missing."""
        try:
            return rule.validate(df)
        except Exception as exc:
            from app.data.validation.exceptions import MissingColumnException

            if isinstance(exc, MissingColumnException):
                logger.warning(f"Skipping rule '{rule.name}': {exc}")
                return None
            logger.error(f"Rule '{rule.name}' raised unexpected error: {exc}")
            raise ValidationException(
                f"Rule '{rule.name}' failed to execute: {exc}",
                details={"rule": rule.name, "rule_type": rule.rule_type},
            ) from exc

    def _should_halt(self, result: ValidationResult) -> bool:
        """Determine whether fail-fast stops execution."""
        return self.config.fail_fast and not result.passed

    def _raise_fail_fast(self, dataset_name: str, collected: list[ValidationResult]) -> None:
        """Raise an exception when fail-fast is triggered."""
        report = self._build_report(dataset_name, pd.DataFrame(), collected)
        self._log_summary(report)
        raise BusinessRuleViolation(
            f"Fail-fast: dataset '{dataset_name}' failed validation",
            details={"failures": [r.rule_name for r in collected if not r.passed]},
        )

    def _build_report(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        results: list[ValidationResult],
    ) -> ValidationReport:
        """Aggregate validation results into a complete report."""
        total_rows = len(df)
        failed_row_indices = {
            idx
            for result in results
            if not result.passed
            for error in result.errors
            for idx in error.row_indices
        }
        total_errors = sum(len(result.errors) for result in results)
        rules_passed = sum(1 for result in results if result.passed)
        severity_counts = self._severity_breakdown(results)
        score = self._compute_score(results, total_rows)

        summary = ValidationSummary(
            dataset_name=dataset_name,
            total_rows=total_rows,
            total_columns=len(df.columns),
            rows_passed=total_rows - len(failed_row_indices),
            rows_failed=len(failed_row_indices),
            rules_checked=len(results),
            rules_passed=rules_passed,
            total_errors=total_errors,
            severity_counts=severity_counts,
            validation_score=score,
            strict_mode=self.config.strict_mode,
        )

        return ValidationReport(
            summary=summary,
            results=results,
            column_statistics=self._column_statistics(results),
            rule_statistics=self._rule_statistics(results),
            failure_reasons=self._failure_reasons(results),
            metadata={"schema": self.schema.name if self.schema else None},
        )

    def _severity_breakdown(self, results: list[ValidationResult]) -> dict[str, int]:
        """Count errors by severity across all results."""
        counts: dict[str, int] = {sev.value: 0 for sev in ValidationSeverity}
        for result in results:
            for error in result.errors:
                counts[error.severity.value] = counts.get(error.severity.value, 0) + 1
        return counts

    def _compute_score(self, results: list[ValidationResult], total_rows: int) -> float:
        """Compute the overall validation score in [0, 1]."""
        failing = [result for result in results if not result.passed]
        if not failing:
            return 1.0

        total_penalty = 0.0
        for result in failing:
            penalty = result.severity.weight
            if total_rows > 0:
                penalty *= min(1.0, result.failed_count / total_rows)
            total_penalty += penalty
        return max(0.0, 1.0 - min(1.0, total_penalty))

    def _column_statistics(self, results: list[ValidationResult]) -> list[ColumnStatistic]:
        """Compute per-column statistics from validation results."""
        stats_map: dict[str, ColumnStatistic] = {}
        for result in results:
            if result.column is None:
                continue
            stat = stats_map.setdefault(result.column, ColumnStatistic(column=result.column))
            stat.rules_checked += 1
            failed_rows = {idx for error in result.errors for idx in error.row_indices}
            stat.failed_rows = max(stat.failed_rows, len(failed_rows))
            if not result.passed:
                stat.rules_failed += 1
                stat.errors += len(result.errors)
        return list(stats_map.values())

    def _rule_statistics(self, results: list[ValidationResult]) -> list[RuleStatistic]:
        """Compute per-rule statistics from validation results."""
        stats: list[RuleStatistic] = []
        for result in results:
            failed_rows = {idx for error in result.errors for idx in error.row_indices}
            stats.append(
                RuleStatistic(
                    rule_name=result.rule_name,
                    rule_type=result.rule_type,
                    ran=True,
                    passed=result.passed,
                    errors=len(result.errors),
                    failed_rows=len(failed_rows),
                )
            )
        return stats

    def _failure_reasons(self, results: list[ValidationResult]) -> list[str]:
        """Collect human-readable failure reasons."""
        reasons: list[str] = []
        for result in results:
            if not result.passed:
                for error in result.errors:
                    reasons.append(f"[{error.severity.value}] {error.message}")
        return reasons

    def _log_summary(self, report: ValidationReport) -> None:
        """Log a concise summary of the validation run."""
        s = report.summary
        logger.info(
            f"Validation complete for '{s.dataset_name}': score={s.validation_score:.3f} | "
            f"rows={s.total_rows} passed={s.rows_passed} failed={s.rows_failed} | "
            f"rules={s.rules_checked} passed={s.rules_passed} failed={s.rules_failed} | "
            f"errors={s.total_errors}"
        )

    @staticmethod
    def _to_pandas(frame: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
        """Convert a polars DataFrame to pandas if needed."""
        if isinstance(frame, pl.DataFrame):
            logger.debug("Converting polars DataFrame to pandas")
            return frame.to_pandas()
        return frame
