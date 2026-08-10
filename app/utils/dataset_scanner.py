"""Dataset scanning and discovery engine for AgriMind AI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import pandas as pd
import polars as pl
from loguru import logger
from rich.console import Console
from rich.table import Table

from app.constants.constants import FOLDER_RAW, SUPPORTED_FILE_FORMATS
from app.utils.decorators import timer
from app.utils.file_utils import get_file_extension, get_file_size_bytes
from app.utils.path_utils import resolve_path


@dataclass
class DatasetInfo:
    """Metadata information for a single dataset."""

    filename: str
    file_path: Path
    extension: str
    rows: int | None = None
    columns: int | None = None
    file_size_bytes: int = 0
    file_size_mb: float = 0.0
    memory_usage_mb: float = 0.0
    column_names: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    has_header: bool = True
    detected_separator: str = ","


class DatasetScanner:
    """Scans directories for supported datasets and collects metadata.

    Automatically discovers CSV, XLS, XLSX, and Parquet files in the
    configured raw data directory and extracts dataset metadata without
    loading the full dataset into memory where possible.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Initialize the scanner.

        Args:
            data_dir: Directory to scan. Defaults to data/raw.
        """
        self.data_dir = resolve_path(data_dir or FOLDER_RAW)
        self._datasets: list[DatasetInfo] = []
        self._console = Console()
        logger.info(f"DatasetScanner initialized for directory: {self.data_dir}")

    @timer
    def scan(self, recursive: bool = True) -> list[DatasetInfo]:
        """Scan the data directory and collect dataset metadata.

        Args:
            recursive: Whether to scan subdirectories recursively.

        Returns:
            List of DatasetInfo objects for each discovered file.
        """
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")
            return []

        self._datasets = []
        pattern = "**/*" if recursive else "*"

        for file_path in self.data_dir.glob(pattern):
            if not file_path.is_file():
                continue

            ext = get_file_extension(file_path)
            if ext not in SUPPORTED_FILE_FORMATS:
                continue

            try:
                info = self._inspect_file(file_path, ext)
                self._datasets.append(info)
                logger.debug(f"Scanned: {info.filename} | {info.rows} rows x {info.columns} cols")
            except Exception as e:
                logger.warning(f"Could not scan {file_path.name}: {e}")

        logger.info(
            f"Scan complete: found {len(self._datasets)} datasets in {self.data_dir}"
        )
        return self._datasets

    def _inspect_file(self, file_path: Path, ext: str) -> DatasetInfo:
        """Inspect a single file and extract metadata.

        Attempts to use fast header-only reads for row/column estimation.
        Falls back to full reads for memory estimation.

        Args:
            file_path: Path to the file.
            ext: File extension (lowercase, with dot).

        Returns:
            DatasetInfo for the file.
        """
        info = DatasetInfo(
            filename=file_path.name,
            file_path=file_path,
            extension=ext,
            file_size_bytes=get_file_size_bytes(file_path),
            file_size_mb=get_file_size_bytes(file_path) / (1024 * 1024),
        )

        if ext == ".csv":
            self._inspect_csv(info)
        elif ext in (".xls", ".xlsx"):
            self._inspect_excel(info)
        elif ext == ".parquet":
            self._inspect_parquet(info)

        return info

    def _inspect_csv(self, info: DatasetInfo) -> None:
        """Inspect a CSV file using efficient partial reads."""
        try:
            # Use polars for fast scanning
            scan = pl.scan_csv(
                str(info.file_path),
                infer_schema_length=100,
                n_rows=None,
            )
            # Get column count from schema
            schema = scan.collect_schema()
            info.columns = len(schema)
            info.column_names = list(schema.names())
            info.dtypes = {name: str(dtype) for name, dtype in schema.items()}

            # Estimate row count by reading first 10K rows
            try:
                small_sample = scan.head(10000).collect()
                info.rows = len(small_sample)
                if info.rows == 10000:
                    # Try faster row count via wc-like approach
                    info.rows = self._estimate_row_count_fast(info.file_path)
            except Exception:
                # Fallback: read all with polars
                df = pl.read_csv(str(info.file_path), infer_schema_length=100)
                info.rows = len(df)
                info.columns = len(df.columns)
                info.column_names = df.columns
                info.dtypes = {c: str(df[c].dtype) for c in df.columns}

            # Memory estimate
            try:
                small = pl.read_csv(str(info.file_path), n_rows=1000)
                per_row = small.estimated_size("mb") / len(small) if len(small) > 0 else 0
                info.memory_usage_mb = round(per_row * (info.rows or 0), 4)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"CSV inspection failed for {info.filename}: {e}")

    def _inspect_excel(self, info: DatasetInfo) -> None:
        """Inspect an Excel file using pandas."""
        try:
            df = pd.read_excel(info.file_path, nrows=10000, engine="openpyxl")
            info.rows = len(df)
            info.columns = len(df.columns)
            info.column_names = list(df.columns)
            info.dtypes = {c: str(df[c].dtype) for c in df.columns}
            info.memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 4)

            # If we read 10K rows, estimate real row count
            if info.rows == 10000:
                df_full = pd.read_excel(info.file_path, engine="openpyxl")
                info.rows = len(df_full)
                info.memory_usage_mb = round(
                    df_full.memory_usage(deep=True).sum() / (1024 * 1024), 4
                )
        except Exception as e:
            logger.warning(f"Excel inspection failed for {info.filename}: {e}")

    def _inspect_parquet(self, info: DatasetInfo) -> None:
        """Inspect a Parquet file using PyArrow metadata."""
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(info.file_path)
            info.rows = pf.metadata.num_rows
            info.columns = len(pf.schema_arrow.names)
            info.column_names = pf.schema_arrow.names
            info.dtypes = {
                name: str(pf.schema_arrow.field(name).type)
                for name in pf.schema_arrow.names
            }

            # Memory estimate from file size * compression ratio estimate (~3x)
            info.memory_usage_mb = round(
                (info.file_size_bytes * 3) / (1024 * 1024), 4
            )
        except Exception as e:
            logger.warning(f"Parquet inspection failed for {info.filename}: {e}")

    @staticmethod
    def _estimate_row_count_fast(file_path: Path) -> int | None:
        """Quick row count for CSV by counting newlines.

        Args:
            file_path: Path to CSV file.

        Returns:
            Approximate row count, or None if it fails.
        """
        try:
            # Read first 100KB to estimate rows
            with open(file_path, "rb") as f:
                sample = f.read(1024 * 100)
            first_chunk_rows = sample.count(b"\n") - 1  # subtract header
            if first_chunk_rows <= 0:
                return None

            file_size = os.path.getsize(file_path)
            if file_size <= len(sample):
                return first_chunk_rows

            # Estimate: (total_size / sample_size) * rows_in_sample
            estimated = int((file_size / len(sample)) * first_chunk_rows)
            return max(estimated, 0)
        except Exception:
            return None

    def get_summary_dataframe(self) -> pl.DataFrame:
        """Return dataset metadata as a Polars DataFrame.

        Returns:
            DataFrame with columns: filename, extension, rows, columns,
            file_size_mb, memory_usage_mb.
        """
        if not self._datasets:
            return pl.DataFrame(
                {
                    "filename": [],
                    "extension": [],
                    "rows": [],
                    "columns": [],
                    "file_size_mb": [],
                    "memory_usage_mb": [],
                }
            )

        return pl.DataFrame(
            {
                "filename": [d.filename for d in self._datasets],
                "extension": [d.extension for d in self._datasets],
                "rows": [d.rows or 0 for d in self._datasets],
                "columns": [d.columns or 0 for d in self._datasets],
                "file_size_mb": [round(d.file_size_mb, 4) for d in self._datasets],
                "memory_usage_mb": [round(d.memory_usage_mb, 4) for d in self._datasets],
            }
        )

    def print_summary_table(self) -> None:
        """Print a formatted summary table of all scanned datasets using Rich."""
        if not self._datasets:
            self._console.print("[yellow]No datasets found.[/yellow]")
            return

        table = Table(title=f"Dataset Scan Results — {self.data_dir}")
        table.add_column("Filename", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Rows", justify="right")
        table.add_column("Cols", justify="right")
        table.add_column("File Size", justify="right")
        table.add_column("Est. Memory", justify="right")

        total_rows = 0
        total_memory = 0.0
        for d in self._datasets:
            rows_str = f"{d.rows:,}" if d.rows else "?"
            table.add_row(
                d.filename,
                d.extension.upper(),
                rows_str,
                str(d.columns or "?"),
                f"{d.file_size_mb:.2f} MB",
                f"{d.memory_usage_mb:.2f} MB",
            )
            total_rows += d.rows or 0
            total_memory += d.memory_usage_mb

        self._console.print(table)
        self._console.print(
            f"\n[bold]Summary:[/bold] {len(self._datasets)} datasets, "
            f"{total_rows:,} total rows, "
            f"{total_memory:.2f} MB estimated memory"
        )

    @property
    def datasets(self) -> list[DatasetInfo]:
        """Return the list of scanned datasets."""
        return self._datasets

    @property
    def count(self) -> int:
        """Return the number of scanned datasets."""
        return len(self._datasets)
