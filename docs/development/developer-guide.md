# Developer Guide

## Repository Structure

- `app/` — Main application package.
- `configs/` — YAML configuration files used by the application.
- `data/` — Data inputs and generated data artifacts.
- `docs/` — Project documentation.
- `frontend/` — Web UI assets.
- `models/` — Trained model artifacts.
- `reports/` — Generated analysis reports.
- `tests/` — Pytest test suite.
- `main.py` — CLI entry point for the Typer application.

## Setup

1. Install Python 3.12+.
2. Install dependencies with `uv sync` or `pip install -e .`.
3. Ensure the project configuration exists in `configs/config.yaml`.

## Running the CLI

- `python main.py scan`
- `python main.py profile`
- `python main.py report`
- `python main.py validate`
- `python main.py clean`

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- `ruff` is configured in `pyproject.toml`.
- `black` formatting uses a 100-character line length.

## Adding Documentation

- Use `docs/README.md` as the documentation portal.
- Add new feature docs under the matching docs subfolder.
- Keep API docs in `docs/api/api-reference.md`.

## Notes for Contributors

- Prefer explicit typing and data classes.
- Keep CLI commands thin; delegate business logic to service modules.
- Use `app/config/config.py` for centralized configuration loading.
- Keep `app/data` modules focused on data operations and `app/services` focused on API workflows.
