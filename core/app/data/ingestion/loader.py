"""Data loading engine for AgriMind AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import polars as pl
from loguru import logger

from app.constants.constants import SUPPORTED_FILE_FORMATS
from app.utils.decorators import timer
from app.utils.file_utils import get_file_extension


class DataLoader:
    """Loads datasets from supported file formats into pandas or polars DataFrames.

    Supports CSV, XLS, XLSX, and Parquet formats.
    """

    def __init__(self, engine: Literal["pandas", "polars"] = "polars") -> None:
        """Initialize the DataLoader.

        Args:
            engine: Default DataFrame engine ('pandas' or 'polars').
        """
        self.engine = engine
        logger.debug(f"DataLoader initialized with engine: {engine}")

    @timer
    def load(
        self,
        file_path: str | Path,
        engine: Literal["pandas", "polars"] | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame | pd.DataFrame:
        """Load a dataset from a file.

        Args:
            file_path: Path to the data file.
            engine: Override the default engine for this load.
            **kwargs: Additional arguments passed to the reader function.

        Returns:
            DataFrame in the specified engine format.

        Raises:
            ValueError: If the file format is unsupported.
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = get_file_extension(file_path)
        if ext not in SUPPORTED_FILE_FORMATS:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {SUPPORTED_FILE_FORMATS}"
            )

        engine = engine or self.engine
        logger.info(f"Loading {file_path} using {engine} engine")
        result = self._load_with_engine(file_path, ext, engine, **kwargs)
        logger.info(
            f"Loaded {file_path.name}: {len(result)} rows, {len(result.columns)} columns"
        )
        return result

    def _load_with_engine(
        self,
        file_path: Path,
        ext: str,
        engine: Literal["pandas", "polars"],
        **kwargs: Any,
    ) -> pl.DataFrame | pd.DataFrame:
        """Route file loading to the appropriate engine and format handler.

        Args:
            file_path: Path to the file.
            ext: File extension.
            engine: Target DataFrame engine.
            **kwargs: Additional reader arguments.

        Returns:
            Loaded DataFrame.
        """
        loaders = {
            "pandas": self._load_pandas,
            "polars": self._load_polars,
        }
        loader = loaders[engine]
        return loader(file_path, ext, **kwargs)

    def _load_pandas(self, file_path: Path, ext: str, **kwargs: Any) -> pd.DataFrame:
        """Load a file into a pandas DataFrame.

        Args:
            file_path: Path to the file.
            ext: File extension.
            **kwargs: Additional pandas read_* arguments.

        Returns:
            Pandas DataFrame.
        """
        readers = {
            ".csv": pd.read_csv,
            ".xls": pd.read_excel,
            ".xlsx": pd.read_excel,
            ".parquet": pd.read_parquet,
        }
        reader = readers.get(ext)
        if reader is None:
            raise ValueError(f"No pandas reader for format: {ext}")

        engine_kwargs: dict[str, Any] = {}
        if ext in (".xls", ".xlsx"):
            engine_kwargs["engine"] = "openpyxl" if ext == ".xlsx" else "xlrd"

        logger.debug(f"Reading {file_path.name} with pandas")
        return reader(file_path, **{**engine_kwargs, **kwargs})

    def _load_polars(self, file_path: Path, ext: str, **kwargs: Any) -> pl.DataFrame:
        """Load a file into a polars DataFrame.

        Args:
            file_path: Path to the file.
            ext: File extension.
            **kwargs: Additional polars read_* arguments.

        Returns:
            Polars DataFrame.
        """
        readers = {
            ".csv": pl.read_csv,
            ".xls": self._read_excel_to_polars,
            ".xlsx": self._read_excel_to_polars,
            ".parquet": pl.read_parquet,
        }
        reader = readers.get(ext)
        if reader is None:
            raise ValueError(f"No polars reader for format: {ext}")

        logger.debug(f"Reading {file_path.name} with polars")
        return reader(file_path, **kwargs)

    @staticmethod
    def _read_excel_to_polars(file_path: Path, **kwargs: Any) -> pl.DataFrame:
        """Read an Excel file into a polars DataFrame via pandas conversion.

        Args:
            file_path: Path to the Excel file.
            **kwargs: Additional pandas read_excel arguments.

        Returns:
            Polars DataFrame.
        """
        ext = get_file_extension(file_path)
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        pandas_df = pd.read_excel(file_path, engine=engine, **kwargs)
        return pl.from_pandas(pandas_df)
