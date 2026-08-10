# Feature Store Guide

## Overview

The feature store layer is implemented in `app/services/feature_store` and is intended to manage feature registration, storage, and online serving.

## Key Components

- `app/services/feature_store/router.py` — HTTP API routes for feature store operations.
- `app/services/feature_store/service.py` — business logic for feature ingestion, retrieval, statistics, and lineage.
- `app/services/feature_store/models.py` — data models for features, groups, statistics, and vectors.

## Feature Group Management

### Create a Feature Group

`POST /api/features/groups`

Required inputs:

- `name`
- `description`
- `features`
- Optional `dataset_id`
- Optional `entity_key`

Requires `feature:write` permission.

### List Feature Groups

`GET /api/features/groups`

Optional filter by `dataset_id`.

Requires `feature:read` permission.

### Get Feature Group Details

`GET /api/features/groups/{group_name}`

Retrieves feature metadata, statistics, lineage, and timestamps.

Requires `feature:read` permission.

### Delete a Feature Group

`DELETE /api/features/groups/{group_name}`

Requires `feature:write` permission.

## Online Feature Serving

`GET /api/features/groups/{group_name}/features?entity_ids=...`

Returns feature values for specific entity IDs from the online store.

- If feature values are missing, the service can optionally compute on demand.
- The online store is an in-memory dictionary in the current implementation.

## Ingestion and Statistics

`FeatureStoreService` supports ingesting feature values from a DataFrame and computing statistics.

- Uses existing data standardization and profiling mechanisms.
- Stores feature statistics and lineage information.
- Optionally loads feature vectors into an online store.

## Notes

The current feature store is a proof-of-concept in-memory service. It defines the shape of a feature management layer and can be extended to persistent storage or real-time stores in future sprints.
