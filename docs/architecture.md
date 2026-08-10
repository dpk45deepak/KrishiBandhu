# AgriMind AI Architecture

## Overview

AgriMind AI is an enterprise-grade agricultural intelligence platform built on a modular, layered architecture. The foundation layer provides data discovery, profiling, validation, and reporting capabilities that power downstream ML pipelines and API services.

## Architecture Layers

### 1. Configuration Layer (`app/config/`)

Uses Pydantic v2 for schema-validated configuration loaded from `configs/config.yaml`. The `Settings` model auto-creates required directories on load. Falls back to sensible defaults if no config file exists.

**Key classes:**
- `Settings` — Root model with nested `ProjectConfig`, `PathsConfig`, `LoggingConfig`, `ReportingConfig`

### 2. Logging Layer (`app/logger/`)

Loguru-based logging with dual output: colored console (via stderr) and rotating file compression (ZIP). Log files are organized by date in `logs/`.

**Key functions:**
- `setup_logger()` — Configures handlers with rotation, retention, and formatting
- `get_logger()` — Returns child-logged logger instance

### 3. Utilities Layer (`app/utils/`)

Reusable components shared across the application.

| Module | Key Components |
|--------|---------------|
| `file_utils.py` | `get_file_extension()`, `is_supported_format()`, `scan_directory()`, `count_files()` |
| `path_utils.py` | `resolve_path()`, `ensure_dir()`, `list_dirs()`, `get_relative_path()` |
| `memory.py` | `get_dataframe_memory_usage()`, `format_bytes()`, `get_object_size()` |
| `decorators.py` | `@timer`, `@exception_handler`, `@log_entry_exit` |
| `dataset_scanner.py` | `DatasetScanner` class |

### 4. Data Layer (`app/data/`)

#### 4.1 Ingestion (`app/data/ingestion/`)

`DataLoader` supports CSV, XLS, XLSX, and Parquet formats via both pandas and polars engines. Automatically selects the appropriate reader based on file extension.

#### 4.2 Profiling (`app/data/profiling/`)

Three components work together:

1. **DataProfiler** — Computes all statistical summaries: shape, missing values, duplicates, unique counts, dtypes, numerical stats (mean, median, std, skewness, kurtosis, IQR, outliers), categorical stats, correlation matrix, constant column detection, high cardinality detection, target suggestion, ML task inference.

2. **ReportGenerator** — Converts `ProfilingResult` into three formats:
   - HTML with embedded Plotly interactive charts
   - JSON with full structured data
   - Markdown human-readable summary

3. **DataQualityScore** (built into profiler) — Weighted formula:
   - Missing (25%)
   - Duplicates (20%)
   - Datatype consistency (15%)
   - Constant columns (15%)
   - Outliers (25%)

### 5. CLI Layer (`main.py`)

Typer-based CLI with three commands:

- `scan` — Discover datasets in `data/raw/` and display Rich table
- `profile` — Profile one or all datasets, generate reports
- `report` — List available profiling reports

### 6. Future Layers

| Layer | Future Sprint |
|-------|--------------|
| `app/data/validation/` | Schema validation with Great Expectations |
| `app/data/cleaning/` | Data cleaning pipeline |
| `app/data/transformation/` | Feature transformation |
| `app/data/feature_engineering/` | Domain-specific feature engineering |
| `app/ml/` | Classification, regression, explainability, tuning, evaluation |
| `app/services/` | FastAPI REST backend |
| Frontend | React dashboard |

## Data Flow

```
data/raw/*.{csv,xls,xlsx,parquet}
        │
        ▼
  DatasetScanner ──► Rich summary table
        │
        ▼
  DataLoader ──► pandas/polars DataFrame
        │
        ▼
  DataProfiler ──► ProfilingResult
        │
        ▼
  ReportGenerator ──► HTML + JSON + Markdown
```

## Design Principles

1. **SOLID** — Single responsibility per class (Scanner, Loader, Profiler, Generator)
2. **Type hints everywhere** — No `Any` unless unavoidable
3. **Logging** — Every significant action is logged via Loguru
4. **Error handling** — Graceful degradation with informative messages
5. **No placeholders** — Production code only
6. **SOLID file structure** — One class/concern per file (< 200 lines where possible)
