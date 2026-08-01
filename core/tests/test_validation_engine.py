"""Tests for the ValidationEngine."""

from pathlib import Path

import pandas as pd
import pytest

from app.data.validation.exceptions import (
    BusinessRuleViolation,
    InvalidSchemaException,
    SchemaNotFoundException,
)
from app.data.validation.rules import (
    BaseRule,
    RangeRule,
    RuleRegistry,
)
from app.data.validation.schema import ValidationSchema, build_schema_from_columns
from app.data.validation.validator import ValidationEngine, ValidationEngineConfig


class TestValidationEngineInit:
    """Test engine initialization."""

    def test_init_with_schema_dict(self, schema_dict: dict) -> None:
        """Test engine initializes from a schema dict."""
        engine = ValidationEngine(schema=schema_dict)
        assert engine.schema is not None
        assert engine.schema.name == "inline_schema"

    def test_init_with_schema_object(self, crop_schema: ValidationSchema) -> None:
        """Test engine initializes from a ValidationSchema object."""
        engine = ValidationEngine(schema=crop_schema)
        assert engine.schema == crop_schema

    def test_init_with_yaml_path(self, tmp_path: Path) -> None:
        """Test engine initializes from a YAML schema file."""
        schema_path = tmp_path / "schema.yaml"
        schema_path.write_text("name: file_schema\ncolumns:\n  temp:\n    dtype: float\n")
        engine = ValidationEngine(schema=str(schema_path))
        assert engine.schema.name == "file_schema"

    def test_init_without_schema(self) -> None:
        """Test engine with no schema for explicit rule lists."""
        engine = ValidationEngine()
        assert engine.schema is None

    def test_init_with_missing_schema_raises(self) -> None:
        """Test missing schema file raises SchemaNotFoundException."""
        with pytest.raises(SchemaNotFoundException):
            ValidationEngine(schema="missing_schema_xyz.yaml")

    def test_register_rule(self) -> None:
        """Test custom rule registration."""
        engine = ValidationEngine()
        engine.register_rule("my_rule", RangeRule)
        assert engine.registry.has("my_rule") is True


class TestValidateDataframe:
    """Test validating a single DataFrame."""

    def test_passing_dataset(self, sample_df: pd.DataFrame, crop_schema: ValidationSchema) -> None:
        """Test a valid dataset passes."""
        engine = ValidationEngine(schema=crop_schema)
        report = engine.validate_dataframe(sample_df, dataset_name="crops")
        assert report.summary.passed is True
        assert report.summary.validation_score == 1.0
        assert report.summary.rows_failed == 0

    def test_failing_dataset(self, invalid_df: pd.DataFrame, crop_schema: ValidationSchema) -> None:
        """Test an invalid dataset fails."""
        engine = ValidationEngine(schema=crop_schema)
        report = engine.validate_dataframe(invalid_df, dataset_name="bad_crops")
        assert report.summary.passed is False
        assert report.summary.rows_failed > 0
        assert report.summary.total_errors > 0
        assert report.summary.validation_score < 1.0

    def test_dataset_name_defaults(
        self, sample_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test dataset name defaults to unknown_dataset."""
        engine = ValidationEngine(schema=crop_schema)
        report = engine.validate_dataframe(sample_df)
        assert report.summary.dataset_name == "unknown_dataset"

    def test_explicit_rules_override_schema(
        self, sample_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test explicit rule lists override the schema rule set."""
        engine = ValidationEngine(schema=crop_schema)
        rule = RangeRule(name="temp_range", column="temperature", min_value=-20, max_value=60)
        report = engine.validate_dataframe(sample_df, dataset_name="crops", rules=[rule])
        assert report.summary.rules_checked == 1
        assert report.summary.passed is True

    def test_no_schema_no_rules_raises(self) -> None:
        """Test engine without schema and no explicit rules raises."""
        engine = ValidationEngine()
        with pytest.raises(InvalidSchemaException):
            engine.validate_dataframe(pd.DataFrame({"a": [1]}))

    def test_polars_input_converted(
        self, sample_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test a polars DataFrame is accepted and converted."""
        import polars as pl

        pl_df = pl.from_pandas(sample_df)
        engine = ValidationEngine(schema=crop_schema)
        report = engine.validate_dataframe(pl_df, dataset_name="polars_df")
        assert report.summary.total_rows == len(sample_df)

    def test_missing_required_column_fails(self, crop_schema: ValidationSchema) -> None:
        """Test a dataset missing a required column fails."""
        df = pd.DataFrame({"temperature": [25.0]})
        engine = ValidationEngine(schema=crop_schema)
        report = engine.validate_dataframe(df, dataset_name="missing_col")
        assert report.summary.passed is False
        assert report.summary.total_errors > 0

    def test_validation_score_reflects_severity(self) -> None:
        """Test the score reflects severity-weighted penalties."""
        schema = build_schema_from_columns({"x": {"dtype": "float", "min_value": 0}})
        engine = ValidationEngine(schema=schema)
        df = pd.DataFrame({"x": [-1, -2, -3]})
        report = engine.validate_dataframe(df, dataset_name="all_fail")
        # One ERROR rule (weight 0.7) with all rows failing => score 0.3
        assert report.summary.validation_score == pytest.approx(0.3)
        assert report.summary.passed is False


class TestStrictMode:
    """Test strict mode behaviour."""

    def test_strict_mode_raises(
        self, invalid_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test strict mode raises BusinessRuleViolation on failure."""
        engine = ValidationEngine(
            schema=crop_schema,
            config=ValidationEngineConfig(strict_mode=True),
        )
        with pytest.raises(BusinessRuleViolation):
            engine.validate_dataframe(invalid_df, dataset_name="strict_bad")

    def test_strict_mode_passes(
        self, sample_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test strict mode does not raise for valid data."""
        engine = ValidationEngine(
            schema=crop_schema,
            config=ValidationEngineConfig(strict_mode=True),
        )
        report = engine.validate_dataframe(sample_df, dataset_name="strict_good")
        assert report.summary.passed is True


class TestFailFast:
    """Test fail-fast mode."""

    def test_fail_fast_raises(
        self, invalid_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test fail-fast raises BusinessRuleViolation."""
        engine = ValidationEngine(
            schema=crop_schema,
            config=ValidationEngineConfig(fail_fast=True),
        )
        with pytest.raises(BusinessRuleViolation):
            engine.validate_dataframe(invalid_df, dataset_name="fast_fail")

    def test_fail_fast_stops_early(self) -> None:
        """Test fail-fast stops after the first failing rule."""
        counter = {"executions": 0}

        class CountingRule(BaseRule):
            rule_type = "counting"

            def __init__(self, name: str, column: str | None = None) -> None:
                super().__init__(name=name, column=column)

            def validate(self, df: pd.DataFrame) -> object:
                counter["executions"] += 1
                from app.data.validation.models import ValidationResult

                return ValidationResult(
                    rule_name=self.name,
                    rule_type=self.rule_type,
                    column=self.column,
                    passed=False,
                    errors=[self.make_error("always fails", row_indices=list(df.index))],
                    total_rows=len(df),
                    failed_count=len(df),
                )

        registry = RuleRegistry()
        registry.register("counting", CountingRule)
        engine = ValidationEngine(
            schema=build_schema_from_columns({"a": {"dtype": "float"}}),
            registry=registry,
            config=ValidationEngineConfig(fail_fast=True),
        )
        rules = [CountingRule("r1"), CountingRule("r2"), CountingRule("r3")]
        with pytest.raises(BusinessRuleViolation):
            engine.validate_dataframe(pd.DataFrame({"a": [1.0]}), dataset_name="ff", rules=rules)
        assert counter["executions"] == 1


class TestConfigIntegration:
    """Test engine honours engine config defaults."""

    def test_max_missing_percentage_applied(self) -> None:
        """Test engine default max null ratio applies to nullable columns."""
        schema = build_schema_from_columns({"x": {"dtype": "float"}})
        engine = ValidationEngine(
            schema=schema,
            config=ValidationEngineConfig(max_missing_percentage=0.25),
        )
        # 3 of 4 null = 75% > 25%
        df = pd.DataFrame({"x": [1.0, None, None, None]})
        report = engine.validate_dataframe(df, dataset_name="missing")
        assert report.summary.passed is False

    def test_duplicate_threshold_applied(self) -> None:
        """Test engine duplicate threshold is applied."""
        schema = build_schema_from_columns({"a": {"dtype": "int"}, "b": {"dtype": "int"}})
        engine = ValidationEngine(
            schema=schema,
            config=ValidationEngineConfig(duplicate_threshold=2),
        )
        df = pd.DataFrame({"a": [1, 1, 1, 2], "b": [5, 5, 5, 6]})
        report = engine.validate_dataframe(df, dataset_name="dups")
        assert report.summary.passed is True


class TestAliases:
    """Test schema alias handling."""

    def test_alias_columns_renamed(self) -> None:
        """Test alias columns are renamed to canonical names before validation."""
        schema = build_schema_from_columns(
            {
                "temperature": {"dtype": "float", "aliases": ["temp"], "min_value": -20},
            }
        )
        engine = ValidationEngine(schema=schema)
        df = pd.DataFrame({"temp": [25.0, 30.0]})
        report = engine.validate_dataframe(df, dataset_name="alias")
        assert report.summary.passed is True


class TestValidateMultipleAndDirectory:
    """Test multi-dataframe and directory validation."""

    def test_validate_multiple(
        self, sample_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test validating multiple named dataframes."""
        engine = ValidationEngine(schema=crop_schema)
        reports = engine.validate_multiple({"crop_a": sample_df, "crop_b": sample_df.copy()})
        assert set(reports) == {"crop_a", "crop_b"}
        assert reports["crop_a"].summary.passed is True

    def test_load_and_validate(
        self, tmp_path: Path, sample_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test loading a CSV file and validating it."""
        csv_path = tmp_path / "crops.csv"
        sample_df.to_csv(csv_path, index=False)
        engine = ValidationEngine(schema=crop_schema)
        report = engine.load_and_validate(csv_path)
        assert report.summary.dataset_name == "crops"
        assert report.metadata["file_path"] == str(csv_path)
        assert report.summary.passed is True

    def test_validate_directory(
        self, tmp_path: Path, sample_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test validating every dataset in a directory."""
        (tmp_path / "crop1.csv").write_text("")
        sample_df.to_csv(tmp_path / "crop1.csv", index=False)
        sample_df.to_csv(tmp_path / "crop2.csv", index=False)
        engine = ValidationEngine(schema=crop_schema)
        reports = engine.validate_directory(tmp_path)
        assert set(reports) == {"crop1", "crop2"}
        assert reports["crop1"].summary.passed is True


class TestReportAggregation:
    """Test report aggregation details."""

    def test_column_statistics(
        self, invalid_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test column statistics are populated."""
        engine = ValidationEngine(schema=crop_schema)
        report = engine.validate_dataframe(invalid_df, dataset_name="stats")
        stats = report.column_statistics
        assert any(s.column == "temperature" for s in stats)
        temp_stat = [s for s in stats if s.column == "temperature"][0]
        assert temp_stat.rules_checked >= 1

    def test_rule_statistics(self, sample_df: pd.DataFrame, crop_schema: ValidationSchema) -> None:
        """Test rule statistics are populated."""
        engine = ValidationEngine(schema=crop_schema)
        report = engine.validate_dataframe(sample_df, dataset_name="stats")
        assert len(report.rule_statistics) == report.summary.rules_checked
        assert all(s.ran for s in report.rule_statistics)
        assert report.summary.rules_passed == report.summary.rules_checked

    def test_severity_breakdown(
        self, invalid_df: pd.DataFrame, crop_schema: ValidationSchema
    ) -> None:
        """Test severity counts are populated."""
        engine = ValidationEngine(schema=crop_schema)
        report = engine.validate_dataframe(invalid_df, dataset_name="sev")
        assert sum(report.summary.severity_counts.values()) == report.summary.total_errors

    def test_failure_reasons(self, invalid_df: pd.DataFrame, crop_schema: ValidationSchema) -> None:
        """Test failure reasons list is populated."""
        engine = ValidationEngine(schema=crop_schema)
        report = engine.validate_dataframe(invalid_df, dataset_name="reasons")
        assert len(report.failure_reasons) > 0
        assert all(reason.startswith("[") for reason in report.failure_reasons)
