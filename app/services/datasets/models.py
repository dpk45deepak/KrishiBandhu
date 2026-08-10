# app/services/datasets/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class DatasetStatus(str, Enum):
    UPLOADING = "uploading"
    SCANNING = "scanning"
    PROFILING = "profiling"
    VALIDATING = "validating"
    CLEANING = "cleaning"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class DatasetFormat(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    EXCEL = "excel"
    GEOJSON = "geojson"
    TIFF = "tiff"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Statistical profile of a single column - from existing Profiler module."""
    name: str
    dtype: str
    count: int
    null_count: int
    null_percentage: float
    unique_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[Any] = None
    max: Optional[Any] = None
    quantiles: Dict[str, float] = field(default_factory=dict)
    top_values: List[Dict[str, Any]] = field(default_factory=list)
    histogram: Optional[Dict[str, Any]] = None


@dataclass
class DatasetProfile:
    """Full dataset profile from existing Profiler."""
    row_count: int
    column_count: int
    total_size_bytes: int
    columns: List[ColumnProfile]
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    missing_patterns: Optional[Dict[str, Any]] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ValidationRule:
    """Validation rule definition."""
    column: str
    rule_type: str  # "not_null", "range", "in_set", "regex", "unique", "custom"
    params: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # "error", "warning"


@dataclass
class ValidationIssue:
    """Single validation issue."""
    rule: ValidationRule
    row_indices: List[int]
    affected_count: int
    message: str


@dataclass
class ValidationReport:
    """Complete validation report from existing Validator."""
    is_valid: bool
    total_rows: int
    rules_count: int
    issues: List[ValidationIssue]
    error_count: int
    warning_count: int
    summary: str
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    handle_nulls: Optional[Dict[str, str]] = None  # column -> strategy
    drop_duplicates: bool = False
    outlier_method: Optional[str] = None  # "iqr", "zscore", "isolation_forest"
    outlier_columns: Optional[List[str]] = None
    text_cleaning: Optional[List[str]] = None  # columns to clean text
    custom_transforms: Optional[Dict[str, str]] = None  # column -> expression


@dataclass
class StandardizationConfig:
    """Configuration for data standardization."""
    date_columns: Optional[List[str]] = None
    numeric_scaling: Optional[str] = None  # "standard", "minmax", "robust"
    categorical_encoding: Optional[str] = None  # "onehot", "label", "target"
    text_normalization: bool = False
    coordinate_system: Optional[str] = None  # For geospatial data


@dataclass
class DatasetCreate:
    """Request to create/register a dataset."""
    name: str
    description: str = ""
    format: DatasetFormat = DatasetFormat.CSV
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetVersion:
    """Dataset version information."""
    version_id: UUID
    dataset_id: UUID
    version_number: int
    file_path: str
    row_count: int
    column_count: int
    size_bytes: int
    checksum: str
    parent_version: Optional[UUID] = None
    changelog: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DatasetResponse:
    """Full dataset information returned by the API."""
    id: UUID
    name: str
    description: str
    format: DatasetFormat
    status: DatasetStatus
    tags: List[str]
    metadata: Dict[str, Any]
    current_version: Optional[DatasetVersion] = None
    profile: Optional[DatasetProfile] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None