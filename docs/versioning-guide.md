# Versioning Guide

## Overview

AgriMind AI uses semantic versioning for the Python package.

## Package Version

- Version is declared in `pyproject.toml` under `[project]` as `0.1.0`.

## Source Control

- Use Git for branch-based development.
- Keep feature work on topic branches.
- Merge to main only after review and passing tests.

## Release Process

1. Bump `version` in `pyproject.toml`.
2. Update documentation if needed.
3. Run tests and linting.
4. Tag the release in Git with `vX.Y.Z`.

## Changelog

- The repository does not currently include a changelog file.
- Maintain a manual changelog in release notes or `docs/` for major versions.

## Dataset and Model Versioning

- Dataset versioning is supported conceptually by the service models under `app/services/datasets/models.py`.
- Feature store and model registry objects include version metadata.

## Notes

The current implementation uses in-memory structures for runtime objects. Persistent versioning support can be added with a database or artifact store in a later sprint.
