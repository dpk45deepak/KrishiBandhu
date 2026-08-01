"""Data profiling engine for AgriMind AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from scipy.stats import skew, kurtosis

from app.constants.constants import (
    CORRELATION_THRESHOLD,
    MAX_UNIQUE_FOR_HIGH_CARDINALITY,
    OUTLIER_IQR_MULTIPLIER,
)
from app.data.ingestion.loader import DataLoader
from app.utils.decorators import timer


@dataclass
class NumericSummary:
    """Summary statistics for a numeric column."""

    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    outlier_count: int = 0
    outlier_ratio: float = 0.0
    zero_count: int = 0
    negative_count: int = 0


@dataclass
class CategoricalSummary:
    """Summary statistics for a categorical column."""

    unique_count: int = 0
    top_values: list[tuple[str, int]] = field(default_factory=list)
    cardinality_ratio: float = 0.0
    is_high_cardinality: bool = False


@dataclass
class ColumnProfile:
    """Complete profile for a single column."""

    name: str
    dtype: str
    missing_count: int = 0
    missing_ratio: float = 0.0
    unique_count: int = 0
    unique_ratio: float = 0.0
    is_constant: bool = False
    is_numeric: bool = False
    is_categorical: bool = False
    is_target_candidate: bool = False
    numeric_summary: NumericSummary | None = None
    categorical_summary: CategoricalSummary | None = None


@dataclass
class ProfilingResult:
    """Complete profiling result for a dataset."""

    filename: str
    file_path: str
    shape: tuple[int, int]
    column_count: int
    row_count: int
    total_missing: int
    total_missing_ratio: float
    duplicate_rows: int
    duplicate_ratio: float
    memory_usage_mb: float
    columns: list[ColumnProfile] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    high_cardinality_columns: list[str] = field(default_factory=list)
    target_candidates: list[str] = field(default_factory=list)
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    high_correlations: list[tuple[str, str, float]] = field(default_factory=list)
    suggested_ml_task: str = "unknown"
    quality_score: float = 0.0


class DataProfiler:
    """Profiles datasets and computes comprehensive statistical summaries.

    Computes shape, missing values, duplicates, unique counts, data types,
    numerical summaries (mean, median, std, skewness, kurtosis, outliers),
    categorical summaries, correlation matrix, constant column detection,
    high cardinality detection, target column suggestions, and ML task inference.
    """

    def __init__(self, engine: Literal["pandas", "polars"] = "pandas") -> None:
        """Initialize the DataProfiler.

        Args:
            engine: The DataFrame engine to use for profiling.
        """
        self.engine = engine
        self._loader = DataLoader(engine=engine)
        logger.debug(f"DataProfiler initialized with engine: {engine}")

    @timer
    def profile(self, file_path: str | Path) -> ProfilingResult:
        """Profile a dataset from a file.

        Args:
            file_path: Path to the data file.

        Returns:
            A ProfilingResult with complete dataset statistics.
        """
        file_path = Path(file_path)
        logger.info(f"Profiling {file_path.name}")

        # Load the data
        df = self._loader.load(file_path, engine=self.engine)

        # Convert to pandas for profiling if polars
        if isinstance(df, pl.DataFrame):
            pdf = df.to_pandas()
        else:
            pdf = df

        # Build profile
        result = self._build_profiling_result(pdf, file_path)
        result.quality_score = self._compute_quality_score(result)

        logger.info(
            f"Profiled {file_path.name}: {result.shape[0]} rows, "
            f"{result.shape[1]} cols, quality={result.quality_score:.2f}"
        )
        return result

    def _build_profiling_result(self, df: pd.DataFrame, file_path: Path) -> ProfilingResult:
        """Build a ProfilingResult from a pandas DataFrame.

        Args:
            df: Pandas DataFrame to profile.
            file_path: Path to the source file.

        Returns:
            A populated ProfilingResult.
        """
        n_rows, n_cols = df.shape

        result = ProfilingResult(
            filename=file_path.name,
            file_path=str(file_path),
            shape=(n_rows, n_cols),
            column_count=n_cols,
            row_count=n_rows,
            total_missing=int(df.isnull().sum().sum()),
            total_missing_ratio=float(df.isnull().sum().sum() / (n_rows * n_cols))
            if n_rows * n_cols > 0
            else 0.0,
            duplicate_rows=int(df.duplicated().sum()),
            duplicate_ratio=float(df.duplicated().sum() / n_rows) if n_rows > 0 else 0.0,
            memory_usage_mb=float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        )

        # Profile each column
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        constant_cols: list[str] = []
        high_cardinality_cols: list[str] = []
        target_candidates: list[str] = []

        for col in df.columns:
            profile = self._profile_column(df, col)
            result.columns.append(profile)

            if profile.is_numeric:
                numeric_cols.append(col)
            if profile.is_categorical:
                categorical_cols.append(col)
            if profile.is_constant:
                constant_cols.append(col)
            if profile.categorical_summary and profile.categorical_summary.is_high_cardinality:
                high_cardinality_cols.append(col)
            if profile.is_target_candidate:
                target_candidates.append(col)

        result.numeric_columns = numeric_cols
        result.categorical_columns = categorical_cols
        result.constant_columns = constant_cols
        result.high_cardinality_columns = high_cardinality_cols
        result.target_candidates = target_candidates

        # Correlation matrix (numeric columns only)
        result.correlation_matrix = self._compute_correlations(df, numeric_cols)
        result.high_correlations = self._find_high_correlations(
            result.correlation_matrix, numeric_cols
        )

        # ML task suggestion
        result.suggested_ml_task = self._suggest_ml_task(
            result, df, target_candidates, numeric_cols, categorical_cols
        )

        return result

    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        """Profile a single column.

        Args:
            df: The DataFrame.
            col: Column name.

        Returns:
            A ColumnProfile for the column.
        """
        series = df[col]
        n_rows = len(df)
        missing = int(series.isnull().sum())
        missing_ratio = missing / n_rows if n_rows > 0 else 0.0
        unique = int(series.nunique())
        unique_ratio = unique / n_rows if n_rows > 0 else 0.0
        is_constant = unique <= 1

        dtype = str(series.dtype)
        is_numeric = self._is_numeric_dtype(series)
        is_categorical = self._is_categorical_dtype(series) or (
            not is_numeric and not is_constant and unique <= MAX_UNIQUE_FOR_HIGH_CARDINALITY
        )

        profile = ColumnProfile(
            name=col,
            dtype=dtype,
            missing_count=missing,
            missing_ratio=round(missing_ratio, 4),
            unique_count=unique,
            unique_ratio=round(unique_ratio, 4),
            is_constant=is_constant,
            is_numeric=is_numeric,
            is_categorical=is_categorical,
        )

        # Numeric summary
        if is_numeric:
            profile.numeric_summary = self._compute_numeric_summary(series)
            profile.is_target_candidate = self._is_target_candidate(
                series, profile.numeric_summary, True
            )

        # Categorical summary
        if is_categorical:
            profile.categorical_summary = self._compute_categorical_summary(series)
            if profile.categorical_summary:
                profile.is_target_candidate = self._is_target_candidate_categorical(
                    series, profile.categorical_summary
                )
        else:
            # For non-categorical columns, still compute high cardinality
            if unique > MAX_UNIQUE_FOR_HIGH_CARDINALITY and not is_constant:
                profile.categorical_summary = CategoricalSummary(
                    unique_count=unique,
                    cardinality_ratio=unique_ratio,
                    is_high_cardinality=True,
                )

        return profile

    def _compute_numeric_summary(self, series: pd.Series) -> NumericSummary:
        """Compute numeric summary statistics for a series.

        Args:
            series: Numeric pandas Series.

        Returns:
            NumericSummary with computed statistics.
        """
        clean = series.dropna()

        if len(clean) == 0:
            return NumericSummary()

        summary = NumericSummary(
            mean=float(clean.mean()),
            median=float(clean.median()),
            std=float(clean.std()) if len(clean) > 1 else 0.0,
            min=float(clean.min()),
            max=float(clean.max()),
            q1=float(clean.quantile(0.25)),
            q3=float(clean.quantile(0.75)),
        )
        summary.iqr = summary.q3 - summary.q1 if summary.q3 and summary.q1 else None

        # Skewness and kurtosis
        if len(clean) > 2:
            try:
                summary.skewness = float(skew(clean, bias=False))
                summary.kurtosis = float(kurtosis(clean, bias=False))
            except Exception:
                summary.skewness = 0.0
                summary.kurtosis = 0.0

        # Outliers using IQR
        if summary.iqr is not None and summary.iqr > 0:
            lower = summary.q1 - OUTLIER_IQR_MULTIPLIER * summary.iqr
            upper = summary.q3 + OUTLIER_IQR_MULTIPLIER * summary.iqr
            outlier_count = int(((clean < lower) | (clean > upper)).sum())
            summary.outlier_count = outlier_count
            summary.outlier_ratio = round(outlier_count / len(clean), 4) if len(clean) > 0 else 0.0

        # Zero and negative counts
        if pd.api.types.is_integer_dtype(clean) or pd.api.types.is_float_dtype(clean):
            summary.zero_count = int((clean == 0).sum())
            summary.negative_count = int((clean < 0).sum())

        return summary

    def _compute_categorical_summary(self, series: pd.Series) -> CategoricalSummary:
        """Compute categorical summary statistics.

        Args:
            series: Categorical pandas Series.

        Returns:
            CategoricalSummary with computed statistics.
        """
        clean = series.dropna()
        n_rows = len(clean)

        if n_rows == 0:
            return CategoricalSummary()

        unique_count = int(clean.nunique())
        cardinality_ratio = unique_count / n_rows if n_rows > 0 else 0.0

        value_counts = clean.value_counts()
        top_values = value_counts.head(10)
        top_list: list[tuple[str, int]] = [
            (str(idx), int(count))
            for idx, count in top_values.items()
        ]

        return CategoricalSummary(
            unique_count=unique_count,
            top_values=top_list,
            cardinality_ratio=round(cardinality_ratio, 4),
            is_high_cardinality=unique_count > MAX_UNIQUE_FOR_HIGH_CARDINALITY,
        )

    def _compute_correlations(
        self, df: pd.DataFrame, numeric_cols: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise Pearson correlations for numeric columns.

        Args:
            df: The DataFrame.
            numeric_cols: List of numeric column names.

        Returns:
            Nested dict: {col1: {col2: correlation}}.
        """
        if len(numeric_cols) < 2:
            return {}

        corr_matrix = df[numeric_cols].corr(numeric_only=True)
        result: dict[str, dict[str, float]] = {}
        for col in corr_matrix.columns:
            result[col] = {}
            for other in corr_matrix.columns:
                val = corr_matrix.loc[col, other]
                if not pd.isna(val):
                    result[col][other] = round(float(val), 4)
        return result

    def _find_high_correlations(
        self, corr_matrix: dict[str, dict[str, float]], numeric_cols: list[str]
    ) -> list[tuple[str, str, float]]:
        """Find highly correlated column pairs.

        Args:
            corr_matrix: Correlation matrix.
            numeric_cols: List of numeric column names.

        Returns:
            List of (col1, col2, |correlation|) tuples with |corr| > threshold.
        """
        high: list[tuple[str, str, float]] = []
        seen: set[tuple[str, str]] = set()
        for col in numeric_cols:
            for other in numeric_cols:
                if col >= other or col not in corr_matrix or other not in corr_matrix[col]:
                    continue
                val = abs(corr_matrix[col][other])
                if val > CORRELATION_THRESHOLD:
                    high.append((col, other, round(val, 4)))
                    seen.add((col, other))
        return sorted(high, key=lambda x: -x[2])

    def _is_numeric_dtype(self, series: pd.Series) -> bool:
        """Check if a series has numeric dtype (int or float).

        Args:
            series: The column series.

        Returns:
            True if numeric.
        """
        return pd.api.types.is_numeric_dtype(series)

    def _is_categorical_dtype(self, series: pd.Series) -> bool:
        """Check if a series has categorical or string dtype.

        Args:
            series: The column series.

        Returns:
            True if categorical, object, string, or StringDtype.
        """
        return (
            pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        )

    def _is_target_candidate(
        self, series: pd.Series, summary: NumericSummary, is_numeric: bool
    ) -> bool:
        """Determine if a numeric column is a good target candidate.

        Criteria: low missing ratio, reasonable unique count,
        not constant, not an ID column.

        Args:
            series: The column series.
            summary: NumericSummary for the column.
            is_numeric: Whether the column is numeric.

        Returns:
            True if the column is a target candidate.
        """
        if not is_numeric:
            return False
        if summary is None:
            return False
        # Not a candidate if constant
        if series.nunique() <= 1:
            return False
        # Not a candidate if very high cardinality (likely an ID)
        if series.nunique() > len(series) * 0.9:
            return False
        # Low missing is preferred
        missing_ratio = series.isnull().sum() / len(series) if len(series) > 0 else 1.0
        if missing_ratio > 0.5:
            return False
        return True

    def _is_target_candidate_categorical(
        self, series: pd.Series, summary: CategoricalSummary
    ) -> bool:
        """Determine if a categorical column is a good target candidate.

        Criteria: binary or low cardinality (classification target).

        Args:
            series: The column series.
            summary: CategoricalSummary for the column.

        Returns:
            True if the column might be a classification target.
        """
        if summary is None:
            return False
        # Good for classification: 2-10 classes, low missing
        if not (2 <= summary.unique_count <= 20):
            return False
        missing_ratio = series.isnull().sum() / len(series) if len(series) > 0 else 1.0
        return missing_ratio < 0.5

    def _suggest_ml_task(
        self,
        result: ProfilingResult,
        df: pd.DataFrame,
        target_candidates: list[str],
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> str:
        """Suggest the most appropriate ML task based on data characteristics.

        Args:
            result: Current ProfilingResult.
            df: The DataFrame.
            target_candidates: List of target candidate column names.
            numeric_cols: List of numeric column names.
            categorical_cols: List of categorical column names.

        Returns:
            Suggested ML task: 'classification', 'regression', 'clustering', or 'unknown'.
        """
        if not target_candidates:
            # No obvious target — could be clustering
            if len(numeric_cols) >= 3:
                return "clustering"
            return "unknown"

        # Check the first target candidate
        target = target_candidates[0]
        if target in numeric_cols:
            # Check if it's a binary (0/1) classification target or regression
            unique_vals = df[target].nunique()
            if unique_vals <= 10:
                return "classification"
            return "regression"
        elif target in categorical_cols:
            return "classification"

        return "unknown"

    def _compute_quality_score(self, result: ProfilingResult) -> float:
        """Compute an overall data quality score.

        Uses weighted components:
        - Missing values (25%)
        - Duplicates (20%)
        - Datatype consistency (15%)
        - Constant columns (15%)
        - Outliers (25%)

        Args:
            result: The profiling result.

        Returns:
            Quality score between 0 and 1.
        """
        from app.constants.constants import (
            WEIGHT_CONSTANT_COLUMNS,
            WEIGHT_DTYPE_CONSISTENCY,
            WEIGHT_DUPLICATES,
            WEIGHT_MISSING,
            WEIGHT_OUTLIERS,
        )

        # Missing score (inverse of missing ratio)
        missing_score = 1.0 - result.total_missing_ratio

        # Duplicate score (inverse of duplicate ratio)
        duplicate_score = 1.0 - result.duplicate_ratio

        # Datatype consistency: prefer typed dtypes over object
        if result.column_count > 0:
            typed = sum(
                1
                for c in result.columns
                if c.is_numeric or "categorical" in str(c.dtype)
            )
            dtype_score = typed / result.column_count
        else:
            dtype_score = 1.0

        # Constant columns score
        if result.column_count > 0:
            constant_score = 1.0 - (len(result.constant_columns) / result.column_count)
        else:
            constant_score = 1.0

        # Outlier score (average across numeric columns)
        outliers_scores: list[float] = []
        for col_profile in result.columns:
            if col_profile.numeric_summary and col_profile.numeric_summary.outlier_ratio > 0:
                outliers_scores.append(1.0 - col_profile.numeric_summary.outlier_ratio)
            elif col_profile.numeric_summary:
                outliers_scores.append(1.0)

        outlier_score = (
            float(np.mean(outliers_scores)) if outliers_scores else 1.0
        )

        score = (
            WEIGHT_MISSING * missing_score
            + WEIGHT_DUPLICATES * duplicate_score
            + WEIGHT_DTYPE_CONSISTENCY * dtype_score
            + WEIGHT_CONSTANT_COLUMNS * constant_score
            + WEIGHT_OUTLIERS * outlier_score
        )

        return round(score, 4)
