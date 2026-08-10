# CLI Guide

## Overview

The AgriMind AI CLI is implemented with Typer in `main.py` and the `app/cli` package. It provides dataset scanning, profiling, report discovery, validation, and cleaning workflows.

## Entry Point

- `python main.py` or `agrimind` if installed via `uv sync`.

## Commands

### scan

Discover datasets in a directory.

Usage:

```bash
python main.py scan
python main.py scan data/raw --recursive
```

Behavior:

- Loads configuration from `configs/config.yaml` via `app.config.config.load_config()`.
- Uses `app.utils.dataset_scanner.DatasetScanner` to scan the `data/raw/` directory by default.
- Prints a Rich table summary with dataset counts, row totals, and memory estimates.

### profile

Profile one dataset or all datasets in `data/raw/`.

Usage:

```bash
python main.py profile
python main.py profile data/raw/my_dataset.csv
python main.py profile data/raw/my_dataset.csv --output reports/profiling
```

Behavior:

- Uses `app.data.profiling.profiler.DataProfiler` to compute dataset statistics.
- Generates HTML, JSON, and Markdown reports using `app.data.profiling.report_generator.ReportGenerator`.
- Outputs profile summaries, quality scores, and report file paths.

### report

List available profiling report files.

Usage:

```bash
python main.py report
python main.py report reports/profiling
```

Behavior:

- Searches the report directory for HTML, JSON, and Markdown output files.
- Prints a summary of discovered reports.

### validate

Validate datasets against a schema.

Usage:

```bash
python main.py validate
python main.py validate data/raw/my_dataset.csv --schema configs/crop_schema.yaml
```

Behavior:

- Loads validation config from `configs/config.yaml` and optional CLI overrides.
- Uses `app.data.validation.validator.ValidationEngine` with `ValidationEngineConfig`.
- Builds validation reports with `app.data.validation.report.ValidationReportGenerator`.
- Displays validation score, passed/failed counts, and report file locations.

### clean

Clean a dataset or directory of datasets.

Usage:

```bash
python main.py clean data/raw/my_dataset.csv --strategy configs/cleaning.yaml --save-interim --report reports/cleaning
```

Behavior:

- Loads cleaning strategy from YAML into `app.data.cleaning.CleaningConfig`.
- Executes cleaning with `app.data.cleaning.CleaningPipeline`.
- Supports directory cleaning, parallel workers, and optional report generation.

## Configuration

Key configuration is loaded from `configs/config.yaml` and `configs/cleaning.yaml`.

- `config.yaml` controls paths, logging, reporting, validation, and supported file formats.
- `cleaning.yaml` controls cleaning strategies and pipeline behavior.

## Extensibility

The CLI is intentionally lightweight and delegates work to service classes. New commands can be added by implementing a register function in `app/cli/` and wiring it into `main.py`.
