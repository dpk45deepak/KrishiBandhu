"""Application-wide constants for AgriMind AI."""

from pathlib import Path
from typing import Final

# Random seed for reproducibility
RANDOM_SEED: Final[int] = 42

# Supported dataset types
SUPPORTED_DATASET_TYPES: Final[set[str]] = {"tabular", "timeseries", "text", "image"}

# Supported file formats
SUPPORTED_FILE_FORMATS: Final[list[str]] = [".csv", ".xls", ".xlsx", ".parquet"]

# Supported file extensions mapped to reader types
FILE_FORMAT_READERS: Final[dict[str, str]] = {
    ".csv": "csv",
    ".xls": "excel",
    ".xlsx": "excel",
    ".parquet": "parquet",
}

# Folder names
FOLDER_RAW: Final[str] = "data/raw"
FOLDER_INTERIM: Final[str] = "data/interim"
FOLDER_PROCESSED: Final[str] = "data/processed"
FOLDER_FEATURE_STORE: Final[str] = "data/feature_store"
FOLDER_MODELS: Final[str] = "models"
FOLDER_REPORTS_PROFILING: Final[str] = "reports/profiling"
FOLDER_REPORTS_VALIDATION: Final[str] = "reports/validation"
FOLDER_REPORTS_EDA: Final[str] = "reports/eda"
FOLDER_LOGS: Final[str] = "logs"
FOLDER_SCRIPTS: Final[str] = "scripts"
FOLDER_DOCS: Final[str] = "docs"

# Default colors for plotting
COLORS: Final[dict[str, str]] = {
    "primary": "#2E7D32",
    "secondary": "#4CAF50",
    "accent": "#81C784",
    "warning": "#FF9800",
    "danger": "#F44336",
    "info": "#2196F3",
    "background": "#F5F5F5",
    "text": "#212121",
    "missing": "#FF5722",
    "duplicate": "#9C27B0",
    "outlier": "#E91E63",
}

# Report names
REPORT_PROFILE_HTML: Final[str] = "profile_report.html"
REPORT_PROFILE_JSON: Final[str] = "profile_report.json"
REPORT_PROFILE_MARKDOWN: Final[str] = "profile_summary.md"
REPORT_VALIDATION_HTML: Final[str] = "validation_report.html"
REPORT_EDA_HTML: Final[str] = "eda_report.html"

# Quality score weights
WEIGHT_MISSING: Final[float] = 0.25
WEIGHT_DUPLICATES: Final[float] = 0.20
WEIGHT_DTYPE_CONSISTENCY: Final[float] = 0.15
WEIGHT_CONSTANT_COLUMNS: Final[float] = 0.15
WEIGHT_OUTLIERS: Final[float] = 0.25

# Profiling thresholds
MAX_UNIQUE_FOR_HIGH_CARDINALITY: Final[int] = 50
CORRELATION_THRESHOLD: Final[float] = 0.7
OUTLIER_IQR_MULTIPLIER: Final[float] = 1.5

# Memory thresholds (in bytes)
MEMORY_WARNING_MB: Final[int] = 500
MEMORY_CRITICAL_MB: Final[int] = 1000

# Project root detection
PROJECT_ROOT_MARKERS: Final[list[str]] = ["pyproject.toml", "configs", "main.py"]


def get_project_root() -> Path:
    """Detect the project root directory by looking for marker files."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if any((current / marker).exists() for marker in PROJECT_ROOT_MARKERS):
            return current
        current = current.parent
    return Path.cwd()
