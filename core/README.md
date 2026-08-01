# AgriMind AI — Agricultural Intelligence Platform

Enterprise-grade foundation for agricultural machine learning and data engineering.

## Architecture

```
AgriMind/
├── app/                    # Core application package
│   ├── config/             # Pydantic v2 configuration from YAML
│   ├── constants/          # Application-wide constants
│   ├── logger/             # Loguru setup (colored console + rotating file)
│   ├── utils/              # File/path utils, DatasetScanner, decorators
│   ├── data/
│   │   ├── ingestion/      # DataLoader — CSV, XLS, XLSX, Parquet
│   │   ├── profiling/      # DataProfiler, ReportGenerator, QualityScore
│   │   ├── validation/     # Schema validation (future)
│   │   ├── cleaning/       # Data cleaning pipeline (future)
│   │   ├── transformation/ # Feature transformation (future)
│   │   └── feature_engineering/ # Feature engineering (future)
│   ├── ml/                 # ML pipelines (future sprints)
│   └── services/           # API services (future)
├── configs/                # YAML configuration files
├── data/                   # Data storage
│   ├── raw/                # Source datasets
│   ├── interim/            # Intermediate processed data
│   ├── processed/          # Final processed data
│   └── feature_store/      # Feature store artifacts
├── models/                 # Trained model artifacts
├── reports/                # Generated reports
│   ├── profiling/          # Profile reports (HTML, JSON, Markdown)
│   ├── validation/         # Validation reports
│   └── eda/                # Exploratory analysis reports
├── tests/                  # pytest test suite
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── logs/                   # Rotating log files
└── main.py                 # CLI entry point (Typer)
```

## Setup

### Prerequisites

- Python 3.12+
- uv (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd AgriMind

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Quick Start

```bash
# Place your datasets in data/raw/

# Scan for datasets
python main.py scan

# Profile all datasets and generate reports
python main.py profile

# Profile a specific file
python main.py profile data/raw/my_dataset.csv

# View available reports
python main.py report

# Enable verbose logging
python main.py --verbose scan
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `scan` | Scan data directory and display dataset summary |
| `profile` | Profile datasets and generate HTML/JSON/Markdown reports |
| `report` | Show overview of available profiling reports |

## Reports

Profiling generates three report formats:

- **HTML** — Interactive report with Plotly visualizations (correlation heatmaps, missing value charts, outlier analysis)
- **JSON** — Full structured profiling data for programmatic consumption
- **Markdown** — Human-readable summary suitable for documentation

### Profiling Features

- Shape, column names, data types
- Missing value detection and ratio
- Duplicate row detection
- Unique value counts
- Numerical summary (mean, median, std, min, max, Q1, Q3, IQR)
- Skewness and kurtosis
- Outlier detection (IQR method)
- Correlation matrix (Pearson)
- High correlation pair detection
- Constant column detection
- High cardinality column detection
- Target column candidate suggestion
- ML task inference (classification, regression, clustering)
- **Data Quality Score** (weighted: missing 25%, duplicates 20%, dtype consistency 15%, constants 15%, outliers 25%)

## Configuration

Edit `configs/config.yaml` to customize:

- Paths for data, reports, logs
- Logging level, rotation, retention
- Report format and visualization settings
- Random seed for reproducibility
- Supported file extensions

## Testing

```bash
pytest tests/ -v
```

## Tech Stack

- **Python 3.12** — Modern Python with pattern matching and type hints
- **Pandas** — Data manipulation (profiling engine)
- **Polars** — High-performance data loading
- **NumPy** — Numerical computing
- **PyArrow** — Parquet file support
- **Pydantic v2** — Configuration validation
- **Typer** — CLI framework
- **Rich** — Beautiful terminal output
- **Loguru** — Logging (colored console + rotating files)
- **YAML** — Configuration
- **Plotly** — Interactive visualizations
- **Scikit-Learn** — Statistical utilities
- **Pytest** — Testing framework
- **Ruff** — Linting
- **Black** — Code formatting

## Project Status

**Current Sprint: Foundation**

- ✅ Project structure and configuration
- ✅ Logging system
- ✅ Dataset discovery engine
- ✅ Data profiler
- ✅ Report generation (HTML, JSON, Markdown)
- ✅ Data quality scoring
- ✅ CLI interface
- ✅ Tests (Config, Scanner, Profiler)
- ❌ ML models (future sprint)
- ❌ API services (future sprint)
- ❌ Frontend (future sprint)

## License

Proprietary — Agricultural Intelligence Platform
