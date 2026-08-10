# AgriMind AI Documentation

Welcome to the AgriMind AI documentation portal. This documentation reflects the current codebase and implementation available in the `core` repository.

## Documentation Overview

- [Architecture](../architecture.md)
- [CLI Guide](./cli/usage.md)
- [API Reference](./api/api-reference.md)
- [Data Guide](./data/data-guide.md)
- [Pipeline Guide](./pipeline/pipeline-guide.md)
- [Developer Guide](./development/developer-guide.md)
- [ML Guide](./ml/ml-guide.md)
- [Services Guide](./services/services-guide.md)
- [Feature Store Guide](./feature_store/feature-store-guide.md)
- [Deployment Guide](./deployment/deployment-guide.md)
- [Configuration Guide](./configuration-guide.md)
- [Coding Standards](./coding-standards.md)
- [Testing Guide](./testing-guide.md)
- [CI/CD Guide](./ci-cd-guide.md)
- [Versioning Guide](./versioning-guide.md)
- [Examples](./examples/quick-start.md)

## Project Status

AgriMind AI is an agricultural intelligence platform built around:

- A Python CLI powered by Typer for dataset scanning, profiling, reporting, validation, and cleaning.
- A modular data processing stack in `app/data/` for ingestion, profiling, validation, transformation, feature engineering, and pipeline orchestration.
- A service layer in `app/services/` intended to support REST APIs, feature store, monitoring, ML workflows, inference, and WebSocket integration.
- A configuration system based on `Pydantic v2` and YAML.

## How to Use This Documentation

- Start with the [CLI Guide](./cli/usage.md) for command-line workflows.
- Read the [API Reference](./api/api-reference.md) for available HTTP endpoints and service contracts.
- Consult the [Data Guide](./data/data-guide.md) for dataset ingestion, profiling, and validation behavior.
- Review the [Pipeline Guide](./pipeline/pipeline-guide.md) for the orchestrator and stage model.
- Use the [Developer Guide](./development/developer-guide.md) for contributing, testing, and project structure.

## Notes

This documentation is generated from the live codebase. It does not invent missing features and is intended to remain synchronized with implemented modules.
