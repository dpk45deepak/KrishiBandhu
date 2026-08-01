"""Tests for the DatasetScanner module."""

import csv
import os
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app.utils.dataset_scanner import DatasetScanner, DatasetInfo
from app.constants.constants import SUPPORTED_FILE_FORMATS


def create_sample_csv(file_path: Path, rows: int = 100, cols: int = 5) -> Path:
    """Create a sample CSV file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [f"col_{i}" for i in range(cols)]
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(rows):
            writer.writerow([f"val_{i}_{j}" for j in range(cols)])
    return file_path


def create_sample_parquet(file_path: Path, rows: int = 100, cols: int = 5) -> Path:
    """Create a sample Parquet file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = {f"col_{i}": list(range(rows)) for i in range(cols)}
    table = pa.table(data)
    pq.write_table(table, file_path)
    return file_path


def create_sample_excel(file_path: Path, rows: int = 50, cols: int = 4) -> Path:
    """Create a sample Excel file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = {f"col_{i}": list(range(rows)) for i in range(cols)}
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False, engine="openpyxl")
    return file_path


class TestDatasetScanner:
    """Test the DatasetScanner class."""

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """Test scanning an empty directory returns no datasets."""
        scanner = DatasetScanner(data_dir=tmp_path)
        results = scanner.scan()
        assert len(results) == 0

    def test_scan_directory_with_csv(self, tmp_path: Path) -> None:
        """Test that CSV files are detected."""
        csv_path = create_sample_csv(tmp_path / "test.csv")
        scanner = DatasetScanner(data_dir=tmp_path)
        results = scanner.scan()
        assert len(results) == 1
        assert results[0].filename == "test.csv"
        assert results[0].extension == ".csv"
        assert results[0].rows == 100
        assert results[0].columns == 5

    def test_scan_directory_with_parquet(self, tmp_path: Path) -> None:
        """Test that Parquet files are detected."""
        pq_path = create_sample_parquet(tmp_path / "test.parquet")
        scanner = DatasetScanner(data_dir=tmp_path)
        results = scanner.scan()
        assert len(results) == 1
        assert results[0].filename == "test.parquet"
        assert results[0].extension == ".parquet"
        assert results[0].rows == 100
        assert results[0].columns == 5

    def test_scan_directory_with_excel(self, tmp_path: Path) -> None:
        """Test that Excel files are detected."""
        xlsx_path = create_sample_excel(tmp_path / "test.xlsx")
        scanner = DatasetScanner(data_dir=tmp_path)
        results = scanner.scan()
        assert len(results) == 1
        assert results[0].filename == "test.xlsx"
        assert results[0].extension == ".xlsx"
        assert results[0].rows == 50
        assert results[0].columns == 4

    def test_scan_directory_ignores_unsupported(self, tmp_path: Path) -> None:
        """Test that unsupported file types are ignored."""
        (tmp_path / "notes.txt").write_text("some text")
        (tmp_path / "data.json").write_text('{"key": "value"}')
        scanner = DatasetScanner(data_dir=tmp_path)
        results = scanner.scan()
        assert len(results) == 0

    def test_scan_recursive_finds_nested_files(self, tmp_path: Path) -> None:
        """Test recursive scanning finds files in subdirectories."""
        nested_dir = tmp_path / "subdir" / "nested"
        create_sample_csv(nested_dir / "deep.csv")
        scanner = DatasetScanner(data_dir=tmp_path)
        results = scanner.scan(recursive=True)
        assert len(results) == 1
        assert results[0].filename == "deep.csv"

    def test_scan_non_recursive_ignores_subdirs(self, tmp_path: Path) -> None:
        """Test non-recursive scanning ignores subdirectories."""
        nested_dir = tmp_path / "subdir"
        create_sample_csv(nested_dir / "deep.csv")
        scanner = DatasetScanner(data_dir=tmp_path)
        results = scanner.scan(recursive=False)
        assert len(results) == 0

    def test_scan_multiple_files(self, tmp_path: Path) -> None:
        """Test scanning with multiple supported files."""
        create_sample_csv(tmp_path / "a.csv")
        create_sample_csv(tmp_path / "b.csv")
        create_sample_parquet(tmp_path / "c.parquet")
        scanner = DatasetScanner(data_dir=tmp_path)
        results = scanner.scan()
        assert len(results) == 3

    def test_get_summary_dataframe(self, tmp_path: Path) -> None:
        """Test the summary dataframe output."""
        create_sample_csv(tmp_path / "test.csv", rows=100, cols=5)
        scanner = DatasetScanner(data_dir=tmp_path)
        scanner.scan()
        df = scanner.get_summary_dataframe()
        assert len(df) == 1
        assert df["filename"][0] == "test.csv"
        assert df["rows"][0] == 100
        assert df["columns"][0] == 5

    def test_count_property(self, tmp_path: Path) -> None:
        """Test the count property."""
        create_sample_csv(tmp_path / "test.csv")
        scanner = DatasetScanner(data_dir=tmp_path)
        scanner.scan()
        assert scanner.count == 1

    def test_datasets_property(self, tmp_path: Path) -> None:
        """Test the datasets property returns list of DatasetInfo."""
        create_sample_csv(tmp_path / "test.csv")
        scanner = DatasetScanner(data_dir=tmp_path)
        scanner.scan()
        datasets = scanner.datasets
        assert len(datasets) == 1
        assert isinstance(datasets[0], DatasetInfo)

    def test_file_size_is_positive(self, tmp_path: Path) -> None:
        """Test that file_size_bytes is positive for real files."""
        create_sample_csv(tmp_path / "test.csv", rows=1000, cols=10)
        scanner = DatasetScanner(data_dir=tmp_path)
        results = scanner.scan()
        assert results[0].file_size_bytes > 0
        assert results[0].file_size_mb > 0
