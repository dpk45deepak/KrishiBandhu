# Data Guide

## Data Layers

AgriMind AI includes a modular data layer in `app/data`.

### Ingestion

- Implemented in `app/data/ingestion/loader.py`.
- `DataLoader` supports reading CSV, XLS, XLSX, and Parquet files.
- The engine can be `pandas` or `polars`.
- Excel files are converted to polars via pandas when needed.

### Profiling

- Implemented in `app/data/profiling/profiler.py`.
- `DataProfiler` calculates dataset statistics, missing values, duplicates, data types, numeric summaries, categorical summaries, and correlations.
- Profiling results are represented by `ProfilingResult`, `ColumnProfile`, `NumericSummary`, and `CategoricalSummary`.
- The profiler infers a suggested ML task and computes a quality score.

### Reporting

- Implemented in `app/data/profiling/report_generator.py`.
- `ReportGenerator` emits HTML, JSON, and Markdown reports.
- HTML reports include Plotly charts for missing values, type distribution, correlations, and outliers.

### Validation

- Implemented in `app/data/validation/validator.py`.
- `ValidationEngine` supports schema loading, rule execution, strict mode, fail-fast, and directory-level validation.
- CLI validation loads records using `DataLoader` and generates reports with `ValidationReportGenerator`.

### Utilities

- `app/utils/dataset_scanner.py` discovers supported files and reports dataset metadata.
- `app/utils/file_utils.py` contains helpers for file extension detection and supported format checks.
- `app/utils/path_utils.py` manages filesystem paths.

## Supported Dataset Formats

- `.csv`
- `.xls`
- `.xlsx`
- `.parquet`

## Data Paths

Default paths are defined in `app/config/config.py` and created automatically when configuration loads:

- `data/raw`
- `data/interim`
- `data/processed`
- `data/feature_store`
- `reports/profiling`
- `reports/validation`
- `reports/eda`

## Workflow

1. Place datasets in `data/raw/`.
2. Use `python main.py scan` to discover files.
3. Use `python main.py profile` to generate dataset profiles.
4. Use `python main.py validate` to validate datasets against a schema.
5. Use `python main.py clean` to run cleaning operations.

## Configuration

The data layer is configured by `configs/config.yaml` and additional YAML files for cleaning or pipeline behavior.
