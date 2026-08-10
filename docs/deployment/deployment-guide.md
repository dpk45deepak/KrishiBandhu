# Deployment Guide

## Current Deployment State

AgriMind AI is packaged as a Python project with `pyproject.toml`. The current repository contains:

- `main.py` — CLI entry point.
- `app/services/api/app.py` — FastAPI application factory.
- `pyproject.toml` — dependency and build settings.

## Running Locally

1. Install dependencies:

```bash
uv sync
```

or

```bash
pip install -e .
```

2. Run the CLI:

```bash
python main.py scan
```

3. Start the FastAPI app if a server entrypoint is added.

## Service Deployment Notes

The FastAPI factory is available at `app/services/api/app.py`. A production deployment would typically use a server like Uvicorn or Hypercorn.

The codebase does not currently include a Dockerfile or Kubernetes manifest.

## Configuration

Deployment relies on configuration from `configs/config.yaml` and runtime environment variables supported by the application.

## Recommended Production Path

- Install dependencies in a virtual environment.
- Configure logging and paths in `configs/config.yaml`.
- Run the API using a production ASGI server.
- Use the CLI for local data operations and maintenance.
