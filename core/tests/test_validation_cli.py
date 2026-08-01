"""Tests for the validate CLI command."""

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from main import app


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def crop_csv(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    """Write a valid crop dataset to a temp CSV."""
    csv_path = tmp_path / "crops.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


class TestValidateCommand:
    """Test the validate CLI command."""

    def test_validate_single_file(self, cli_runner: CliRunner, crop_csv: Path) -> None:
        """Test validating a single file succeeds."""
        result = cli_runner.invoke(
            app,
            ["validate", str(crop_csv), "--schema", "crop_schema"],
        )
        assert result.exit_code == 0
        assert "Validation complete" in result.output
        assert "crops" in result.output

    def test_validate_single_file_with_strict(self, cli_runner: CliRunner, crop_csv: Path) -> None:
        """Test strict mode flag."""
        result = cli_runner.invoke(
            app,
            ["validate", str(crop_csv), "--schema", "crop_schema", "--strict"],
        )
        # Valid data passes even in strict mode
        assert result.exit_code == 0

    def test_validate_nonexistent_file(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test validating a missing file."""
        missing = tmp_path / "nonexistent.csv"
        result = cli_runner.invoke(app, ["validate", str(missing)])
        assert result.exit_code == 1
        assert "File not found" in result.output

    def test_validate_generates_reports(
        self, cli_runner: CliRunner, crop_csv: Path, tmp_path: Path
    ) -> None:
        """Test validate generates report artifacts."""
        output = tmp_path / "validation_reports"
        result = cli_runner.invoke(
            app,
            ["validate", str(crop_csv), "--schema", "crop_schema", "--output", str(output)],
        )
        assert result.exit_code == 0
        html = output / "validation_report.html"
        json_report = output / "validation_report.json"
        md = output / "validation_report.md"
        assert html.exists()
        assert json_report.exists()
        assert md.exists()
