# API Reference

The AgriMind AI API is structured around FastAPI routers in `app/services`. The application factory is defined in `app/services/api/app.py` and exposes routes under `/api`.

## API Base Path

All routes are mounted under `/api`.

## Authentication

### POST /api/auth/register

Register a new user.

Request body:

- `username`: string
- `email`: string
- `password`: string
- `full_name`: string
- `role`: one of `admin`, `data_scientist`, `analyst`, `viewer`, `service`

Response:

- `message`, `username`, `id`

### POST /api/auth/token

OAuth2 password grant for JWT access.

Form fields:

- `username`
- `password`

Response:

- `access_token`, `token_type`, `expires_in`, optional `refresh_token`

### POST /api/auth/refresh

Refresh an access token.

Request body:

- `refresh_token`

Response:

- `access_token`, `token_type`, `expires_in`, `refresh_token`

### GET /api/auth/me

Get current authenticated user details.

## Datasets

### POST /api/datasets

Create a dataset entry.

Requires `dataset:write` permission.

Request body:

- `name`, `description`, `format`, `tags`, `metadata`

### POST /api/datasets/{dataset_id}/upload

Upload a dataset file and scan it.

Requires `dataset:write` permission.

### GET /api/datasets

List datasets.

Query params:

- `status`
- `tags`

Requires `dataset:read` permission.

### GET /api/datasets/{dataset_id}

Get dataset details.

Requires `dataset:read` permission.

### POST /api/datasets/{dataset_id}/profile

Generate a dataset profile.

Requires `dataset:read` permission.

### POST /api/datasets/{dataset_id}/validate

Validate a dataset against rules.

Requires `dataset:read` permission.

### POST /api/datasets/{dataset_id}/clean

Clean a dataset and create a new version.

Requires `dataset:write` permission.

### POST /api/datasets/{dataset_id}/standardize

Standardize a dataset and create a new version.

Requires `dataset:write` permission.

### GET /api/datasets/{dataset_id}/versions

List dataset versions.

Requires `dataset:read` permission.

### GET /api/datasets/{dataset_id}/download

Download dataset content.

Requires `dataset:read` permission.

### DELETE /api/datasets/{dataset_id}

Delete a dataset.

Requires `dataset:delete` permission.

## Pipeline

### POST /api/pipeline

Create a pipeline definition.

Requires `pipeline:write` permission.

### GET /api/pipeline

List pipelines.

Requires `pipeline:read` permission.

### GET /api/pipeline/{pipeline_id}

Get pipeline details.

Requires `pipeline:read` permission.

### POST /api/pipeline/{pipeline_id}/run

Execute a pipeline synchronously.

Requires `pipeline:execute` permission.

### POST /api/pipeline/{pipeline_id}/run-async

Execute a pipeline asynchronously.

Requires `pipeline:execute` permission.

### GET /api/pipeline/{pipeline_id}/runs

List pipeline runs.

Requires `pipeline:read` permission.

### GET /api/pipeline/{pipeline_id}/runs/{run_id}

Get a specific run.

Requires `pipeline:read` permission.

### POST /api/pipeline/{pipeline_id}/runs/{run_id}/cancel

Cancel a running pipeline.

Requires `pipeline:execute` permission.

### PATCH /api/pipeline/{pipeline_id}/status

Update pipeline status.

Requires `pipeline:write` permission.

### DELETE /api/pipeline/{pipeline_id}

Delete a pipeline.

Requires `pipeline:write` permission.

## Machine Learning

### POST /api/ml/models

Register a new model.

Requires `model:train` permission.

### GET /api/ml/models

List models.

Requires `model:read` permission.

### GET /api/ml/models/{model_id}

Get model details.

Requires `model:read` permission.

### DELETE /api/ml/models/{model_id}

Delete a model.

Requires `model:deploy` permission.

### POST /api/ml/models/{model_id}/train

Train a model.

Requires `model:train` permission.

### GET /api/ml/training-jobs

List training jobs.

Requires `model:read` permission.

### POST /api/ml/models/{model_id}/evaluate

Evaluate a trained model.

Requires `model:read` permission.

### POST /api/ml/models/{model_id}/tune

Tune model hyperparameters.

Requires `model:train` permission.

### POST /api/ml/models/{model_id}/predict

Make predictions with a model.

Requires `predict:execute` permission.

### POST /api/ml/predict?model_id={model_id}

Batch predict using query parameter model_id.

Requires `predict:execute` permission.

### GET /api/ml/models/{model_id}/explain

Get explainability report.

Requires `model:read` permission.

### POST /api/ml/models/{model_id}/deploy

Deploy a model.

Requires `model:deploy` permission.

## Feature Store

### POST /api/features/groups

Create a feature group.

Requires `feature:write` permission.

### GET /api/features/groups

List feature groups.

Requires `feature:read` permission.

### GET /api/features/groups/{group_name}

Get group details.

Requires `feature:read` permission.

### GET /api/features/groups/{group_name}/features?entity_ids=...

Fetch online feature vectors.

Requires `feature:read` permission.

### DELETE /api/features/groups/{group_name}

Delete a feature group.

Requires `feature:write` permission.

## Inference

### POST /api/inference/endpoints

Create an inference endpoint.

Requires `model:deploy` permission.

### GET /api/inference/endpoints

List endpoints.

Requires `model:read` permission.

### GET /api/inference/endpoints/{name}

Get endpoint details.

Requires `model:read` permission.

### POST /api/inference/endpoints/{name}/predict

Predict through the endpoint.

Requires `predict:execute` permission.

### POST /api/inference/endpoints/{name}/batch

Run batch inference.

Requires `predict:execute` permission.

### GET /api/inference/jobs

List inference jobs.

Requires `predict:read` permission.

### GET /api/inference/jobs/{job_id}

Get job status.

Requires `predict:read` permission.

### PATCH /api/inference/endpoints/{name}/config

Update endpoint configuration.

Requires `model:deploy` permission.

### DELETE /api/inference/endpoints/{name}

Delete an endpoint.

Requires `model:deploy` permission.

## Health

### GET /api/health

Get full system health.

### GET /api/health/ready

Readiness probe.

### GET /api/health/live

Liveness probe.

### GET /api/health/{component}

Get component health.

## Monitoring

### GET /api/monitoring/metrics/{metric_type}

Get metric summaries.

Requires `admin` permission.

### GET /api/monitoring/metrics/{metric_type}/history

Query metric history.

Requires `admin` permission.

### POST /api/monitoring/alerts/rules

Create an alert rule.

Requires `admin` permission.

### GET /api/monitoring/alerts/rules

List alert rules.

Requires `admin` permission.

### GET /api/monitoring/alerts

List triggered alerts.

Requires `admin` permission.

### POST /api/monitoring/alerts/{alert_id}/acknowledge

Acknowledge an alert.

Requires `admin` permission.

### GET /api/monitoring/system

Get current system metrics.

Requires `admin` permission.

## WebSocket

### GET /api/ws/stats

Get WebSocket service statistics.

### WebSocket /api/ws/connect

Connect to the WebSocket endpoint with optional query params:

- `channels`
- `token`

The socket supports subscribe/unsubscribe messages and Pong responses.
