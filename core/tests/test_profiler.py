"""Tests for the DataProfiler module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.data.profiling.profiler import (
    DataProfiler,
    ColumnProfile,
    NumericSummary,
    CategoricalSummary,
    ProfilingResult,
)
from app.constants.constants import OUTLIER_IQR_MULTIPLIER


def create_test_csv(file_path: Path) -> Path:
    """Create a test CSV file with known properties."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n = 200

    data = {
        "numeric_normal": np.random.normal(50, 10, n),
        "numeric_uniform": np.random.uniform(0, 100, n),
        "integer_col": np.random.randint(0, 100, n),
        "categorical_col": np.random.choice(["A", "B", "C", "D"], n),
        "binary_col": np.random.choice([0, 1], n),
        "constant_col": np.ones(n),
        "missing_col": np.random.choice([1.0, np.nan], n, p=[0.7, 0.3]),
        "id_col": range(n),
        "text_col": [f"text_{i}" for i in range(n)],
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path


class TestDataProfiler:
    """Test the DataProfiler class."""

    def test_profile_csv(self, tmp_path: Path) -> None:
        """Test basic profiling of a CSV file."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler(engine="pandas")
        result = profiler.profile(csv_path)

        assert isinstance(result, ProfilingResult)
        assert result.filename == "test.csv"
        assert result.row_count == 200
        assert result.column_count == 9
        assert result.shape == (200, 9)

    def test_profile_returns_columns(self, tmp_path: Path) -> None:
        """Test that profiling returns column profiles."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        assert len(result.columns) == 9
        column_names = {c.name for c in result.columns}
        assert "numeric_normal" in column_names
        assert "categorical_col" in column_names
        assert "constant_col" in column_names
        assert "missing_col" in column_names

    def test_numeric_column_detection(self, tmp_path: Path) -> None:
        """Test that numeric columns are correctly identified."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        assert "numeric_normal" in result.numeric_columns
        assert "numeric_uniform" in result.numeric_columns
        assert "integer_col" in result.numeric_columns

    def test_categorical_column_detection(self, tmp_path: Path) -> None:
        """Test that categorical columns are correctly identified."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        assert "categorical_col" in result.categorical_columns
        assert "text_col" in result.categorical_columns

    def test_constant_column_detection(self, tmp_path: Path) -> None:
        """Test that constant columns are detected."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        assert "constant_col" in result.constant_columns

    def test_missing_values_detection(self, tmp_path: Path) -> None:
        """Test that missing values are counted."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        # missing_col has ~30% missing
        assert result.total_missing > 0
        # Find the missing_col profile
        missing_profile = next(c for c in result.columns if c.name == "missing_col")
        assert missing_profile.missing_count > 0
        assert missing_profile.missing_ratio > 0.2
        assert missing_profile.missing_ratio < 0.4

    def test_numeric_summary(self, tmp_path: Path) -> None:
        """Test that numeric summaries are computed."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        normal_profile = next(c for c in result.columns if c.name == "numeric_normal")
        assert normal_profile.numeric_summary is not None
        assert normal_profile.numeric_summary.mean is not None
        assert 40 < normal_profile.numeric_summary.mean < 60
        assert normal_profile.numeric_summary.std is not None
        assert normal_profile.numeric_summary.min is not None
        assert normal_profile.numeric_summary.max is not None
        assert normal_profile.numeric_summary.q1 is not None
        assert normal_profile.numeric_summary.q3 is not None

    def test_categorical_summary(self, tmp_path: Path) -> None:
        """Test that categorical summaries are computed."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        cat_profile = next(c for c in result.columns if c.name == "categorical_col")
        assert cat_profile.categorical_summary is not None
        assert cat_profile.categorical_summary.unique_count == 4
        assert len(cat_profile.categorical_summary.top_values) > 0

    def test_quality_score_range(self, tmp_path: Path) -> None:
        """Test that quality score is between 0 and 1."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        assert 0.0 <= result.quality_score <= 1.0

    def test_quality_score_perfect_data(self, tmp_path: Path) -> None:
        """Test that perfect data gets a high quality score."""
        file_path = tmp_path / "perfect.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "a": np.random.normal(50, 10, 1000),
            "b": np.random.normal(30, 5, 1000),
            "c": ["A", "B"] * 500,
        })
        df.to_csv(file_path, index=False)

        profiler = DataProfiler()
        result = profiler.profile(file_path)
        assert result.quality_score > 0.7

    def test_target_candidates(self, tmp_path: Path) -> None:
        """Test that target candidates are identified."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        # binary_col should be a target candidate
        binary_profile = next(c for c in result.columns if c.name == "binary_col")
        assert binary_profile.is_target_candidate

    def test_suggested_ml_task_regression(self, tmp_path: Path) -> None:
        """Test ML task suggestion for regression data."""
        file_path = tmp_path / "regression.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.random.seed(42)
        # 30 rows with many repeated target values -> <90% unique -> becomes candidate
        df = pd.DataFrame({
            "feature": np.random.normal(50, 10, 30),
            "target": np.repeat(np.arange(10), 3)[:30],  # only 10 uniques out of 30 rows
        })
        df.to_csv(file_path, index=False)

        profiler = DataProfiler()
        result = profiler.profile(file_path)
        assert result.suggested_ml_task == "regression"

    def test_suggested_ml_task_classification(self, tmp_path: Path) -> None:
        """Test ML task suggestion for classification data."""
        file_path = tmp_path / "classification.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "feature": np.random.normal(50, 10, 200),
            "target": np.random.choice(["yes", "no"], 200),
        })
        df.to_csv(file_path, index=False)

        profiler = DataProfiler()
        result = profiler.profile(file_path)
        # The 'target' column is a low-cardinality categorical -> classification
        assert result.suggested_ml_task == "classification"

    def test_correlation_matrix(self, tmp_path: Path) -> None:
        """Test that correlation matrix is computed."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        assert len(result.correlation_matrix) > 0
        # Check that all numeric columns are in the correlation matrix
        for col in result.numeric_columns:
            assert col in result.correlation_matrix

    def test_high_cardinality_detection(self, tmp_path: Path) -> None:
        """Test that high cardinality columns are detected."""
        csv_path = create_test_csv(tmp_path / "test.csv")
        profiler = DataProfiler()
        result = profiler.profile(csv_path)

        # id_col has 200 unique values -> high cardinality
        assert "id_col" in result.high_cardinality_columns

    def test_profiling_result_dataclass(self) -> None:
        """Test ProfilingResult dataclass creation."""
        result = ProfilingResult(
            filename="test.csv",
            file_path="/path/to/test.csv",
            shape=(100, 5),
            column_count=5,
            row_count=100,
            total_missing=10,
            total_missing_ratio=0.02,
            duplicate_rows=0,
            duplicate_ratio=0.0,
            memory_usage_mb=0.5,
        )
        assert result.filename == "test.csv"
        assert result.row_count == 100
        assert result.suggested_ml_task == "unknown"
