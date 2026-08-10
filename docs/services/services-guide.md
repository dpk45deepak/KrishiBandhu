# Services Guide

## Overview

`app/services` contains the platform service layer and FastAPI integration points for AgriMind AI.

## API Factory

The FastAPI app factory is defined in `app/services/api/app.py`.

- Creates the application with `FastAPI(...)`.
- Configures CORS using `app.config.settings`.
- Includes routers for health, auth, datasets, pipeline, ML, feature store, inference, reports, monitoring, and WebSocket.
- Uses an async lifespan manager to start reusable services.

## Authentication and Permissions

Auth dependencies are implemented in `app/services/api/dependencies.py`.

- `get_current_user` supports JWT and API key authentication.
- `require_permission(permission)` enforces RBAC permissions from `app/services/auth/models.py`.
- The auth router is in `app/services/auth/router.py`.

## Service Layer Patterns

Each domain exposes:

- A router (`router.py`) for HTTP routing and request validation.
- A service class (`service.py`) for business logic.
- Models (`models.py`) for data contracts and DTOs.

### Key domains

- **Auth** — user registration, token issuance, API key management.
- **Datasets** — dataset creation, upload, profiling, validation, cleaning, standardization, versioning.
- **Pipeline** — pipeline definitions, runs, asynchronous execution, cancellation.
- **ML** — model lifecycle, training, evaluation, tuning, predictions, explainability, deployment.
- **Feature Store** — feature group registration, online feature serving, statistics, lineage.
- **Inference** — endpoint creation, prediction, batch jobs.
- **Reports** — report generation and export.
- **Monitoring** — metrics, alerts, system observations.
- **Health** — liveness/readiness and component checks.
- **WebSocket** — real-time client connections and channel subscriptions.

## Logging

All service modules use `app.logger.get_logger(...)` for structured logging. Service events and errors are logged consistently.

## Current Implementation State

The service layer is designed as an integrated backend stack, with API-first router definitions and in-memory service state.

- Persistence, database integration, and production deployment are scaffolded but not fully implemented in all modules.
- Future work can add database backends, background task queues, and expanded auth storage.
