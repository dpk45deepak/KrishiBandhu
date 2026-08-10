# Testing Guide

## Overview

Tests live under `tests/` and use `pytest` for unit and integration coverage.

## Running Tests

```bash
pytest tests/ -v
```

## Test Configuration

- `pyproject.toml` configures `pytest` to discover `tests/test_*.py` and `test_*` functions.
- The repository uses `pytest-cov` for coverage reporting.

## Test Strategy

- Test configuration loading and defaults.
- Test dataset scanning and file discovery.
- Test profiling and report generation.
- Test validation engine behavior and rule handling.
- Test CLI command behavior where possible.

## Existing Tests

Key test modules include:

- `tests/test_config.py`
- `tests/test_scanner.py`
- `tests/test_profiler.py`
- `tests/test_validation_engine.py`
- `tests/test_validation_report.py`

## Writing New Tests

- Use the `tmp_path` fixture for filesystem isolation.
- Keep tests deterministic by using sample datasets under `tests/` or generated data.
- Avoid depending on external resources.

## Test Quality

- Validate both success and failure paths.
- Use expressive assertions and descriptive test names.
- Keep test functions focused and small.
