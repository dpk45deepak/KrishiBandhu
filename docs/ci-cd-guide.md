# CI/CD Guide

## Overview

This project does not currently include a fully configured CI/CD pipeline, but it is structured to support automated quality gates and deployment.

## Recommended CI/CD Steps

1. Install dependencies in a clean environment.
2. Run `ruff` linting.
3. Run `black --check` for formatting.
4. Run `pytest tests/ -v`.
5. Optionally run `pytest --cov=app` for coverage.

## Suggested Workflow

- Use GitHub Actions or a similar platform.
- Include separate jobs for linting, testing, and packaging.
- Fail the pipeline on lint or test failures.

## Packaging

The project is configured for Hatchling in `pyproject.toml`.

- The package is named `agrimind-ai`.
- The CLI entry point is `agrimind = main:app`.

## Future CI/CD Enhancements

- Add a Docker build step.
- Add code coverage reporting.
- Add API contract validation for FastAPI endpoints.
- Add release publishing to PyPI or an internal package index.
