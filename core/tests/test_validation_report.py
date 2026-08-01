"""Tests for the ValidationReportGenerator."""

import json
from pathlib import Path

import pandas as pd

from app.data.validation.models import (
    RuleStatistic,
    ValidationError,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    ValidationSummary,
)
from app.data.validation.report import ValidationReportGenerator


def _make_report(dataset_name: str = "crops") -> ValidationReport:
    """Build a ValidationReport fixture with one failing rule."""
    summary = ValidationSummary(
        dataset_name=dataset_name,
        total_rows=4,
        rows_passed=2,
        rows_failed=2,
        rules_checked=3,
        rules_passed=2,
        total_errors=2,
        severity_counts={
            "critical": 0,
            "error": 2,
            "warning": 0,
            "info": 0,
        },
        validation_score=0.6,
        passed=False,
    )
    error = ValidationError(
        rule_name="range_temperature",
        rule_type="range",
        column="temperature",
        message="Column 'temperature' has 2 value(s) outside range [-20, 60]",
        row_indices=[1, 2],
        row_count=2,
        details={"min": -20, "max": 60},
    )
    result = ValidationResult(
        rule_name="range_temperature",
        rule_type="range",
        column="temperature",
        passed=False,
        severity=ValidationSeverity.ERROR,
        errors=[error],
        total_rows=4,
        failed_count=2,
    )
    passing = ValidationResult(
        rule_name="dtype_humidity",
        rule_type="dtype",
        column="humidity",
        passed=True,
        total_rows=4,
        failed_count=0,
    )
    return ValidationReport(
        summary=summary,
        results=[result, passing],
        rule_statistics=[
            RuleStatistic(
                rule_name="range_temperature",
                rule_type="range",
                ran=True,
                passed=False,
                errors=1,
                failed_rows=2,
            ),
            RuleStatistic(
                rule_name="dtype_humidity",
                rule_type="dtype",
                ran=True,
                passed=True,
                errors=0,
                failed_rows=0,
            ),
        ],
        failure_reasons=["[error] Column 'temperature' has 2 value(s) outside range [-20, 60]"],
        metadata={"schema": "crop_schema"},
    )


class TestValidationReportGenerator:
    """Test the ValidationReportGenerator."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test output directory is created on init."""
        output = tmp_path / "reports" / "validation"
        ValidationReportGenerator(output_dir=output)
        assert output.exists()

    def test_generate_html(self, tmp_path: Path) -> None:
        """Test HTML report generation."""
        generator = ValidationReportGenerator(output_dir=tmp_path)
        path = generator.generate_html(_make_report())
        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "Validation Report" in content
        assert "crops" in content
        assert "range_temperature" in content

    def test_generate_json(self, tmp_path: Path) -> None:
        """Test JSON report generation."""
        generator = ValidationReportGenerator(output_dir=tmp_path)
        path = generator.generate_json(_make_report())
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["summary"]["dataset_name"] == "crops"
        assert data["metadata"]["schema"] == "crop_schema"
        assert data["results"][0]["rule_name"] == "range_temperature"

    def test_generate_markdown(self, tmp_path: Path) -> None:
        """Test Markdown report generation."""
        generator = ValidationReportGenerator(output_dir=tmp_path)
        path = generator.generate_markdown(_make_report())
        content = Path(path).read_text(encoding="utf-8")
        assert "Validation Report" in content
        assert "range_temperature" in content
        assert "Severity Counts" in content

    def test_generate_failed_rows_csv_with_data(self, tmp_path: Path) -> None:
        """Test CSV generation with failed rows."""
        report = _make_report()
        failed_df = pd.DataFrame({"temperature": [70.0, -30.0]})
        generator = ValidationReportGenerator(output_dir=tmp_path)
        path = generator.generate_failed_rows_csv(report, failed_df)
        assert Path(path).exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 2
        assert "temperature" in loaded.columns

    def test_generate_failed_rows_csv_without_data(self, tmp_path: Path) -> None:
        """Test CSV generation skips when no failed rows."""
        report = _make_report()
        generator = ValidationReportGenerator(output_dir=tmp_path)
        path = generator.generate_failed_rows_csv(report, None)
        assert path == ""

    def test_generate_all(self, tmp_path: Path) -> None:
        """Test generating all report formats."""
        report = _make_report()
        failed_df = pd.DataFrame({"temperature": [70.0]})
        generator = ValidationReportGenerator(output_dir=tmp_path)
        paths = generator.generate_all(report, failed_df)
        assert "html" in paths and "json" in paths
        assert "markdown" in paths and "csv" in paths
        for fmt, path in paths.items():
            assert Path(path).exists(), f"{fmt} file missing"

    def test_build_failed_rows(self) -> None:
        """Test extracting failed rows from the original dataframe."""
        report = _make_report()
        df = pd.DataFrame({"temperature": [25.0, 70.0, -30.0, 22.0]})
        report.results[0].errors[0].row_indices = [1, 2]
        generator = ValidationReportGenerator(output_dir="reports")
        failed = generator.build_failed_rows(df, report)
        assert len(failed) == 2
        assert list(failed["temperature"]) == [70.0, -30.0]

    def test_build_failed_rows_empty(self) -> None:
        """Test build_failed_rows returns empty for passing report."""
        report = _make_report()
        report.results[0].passed = True
        report.results[0].errors = []
        generator = ValidationReportGenerator(output_dir="reports")
        failed = generator.build_failed_rows(pd.DataFrame({"a": [1]}), report)
        assert failed.empty

    def test_render_html_contains_charts(self, tmp_path: Path) -> None:
        """Test HTML includes Plotly chart divs."""
        generator = ValidationReportGenerator(output_dir=tmp_path)
        path = generator.generate_html(_make_report())
        content = Path(path).read_text(encoding="utf-8")
        assert "plotly" in content.lower()
