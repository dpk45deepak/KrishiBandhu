# Machine Learning Guide

## Overview

The ML capabilities in AgriMind AI are split between the CLI package in `app/ml` and the service layer in `app/services/ml`.

## ML CLI

The CLI is implemented in `app/ml/cli.py` and exposes commands for:

- `train` — train a classification or regression model
- `predict` — make predictions from a saved model or registry model
- `evaluate` — evaluate a trained model on test data
- `compare` — compare multiple registered models
- `tune` — run hyperparameter tuning

### Design

- `DatasetConfig` and `DatasetLoader` in `app/ml/common/datasets.py` manage dataset loading and preprocessing.
- `ModelTrainer` in `app/ml/common/trainer.py` centralizes training logic.
- `ModelPredictor` in `app/ml/common/predictor.py` supports batch predictions and probability output.
- `ModelPersistence` in `app/ml/common/persistence.py` persists model artifacts.
- `ModelRegistry` in `app/ml/common/registry.py` stores registry metadata under `models/registry`.
- Evaluation and reporting are handled by `app/ml/evaluation/evaluator.py` and `app/ml/evaluation/reports.py`.
- Hyperparameter tuning uses `app/ml/tuning/tuner.py` and concrete tuner implementations in `app/ml/tuning/`.

## ML Services

The API service layer for ML is in `app/services/ml`.

### Key capabilities

- Model registration and discovery
- Training and tuning jobs
- Evaluation reports
- Prediction endpoints
- Explainability reports
- Model deployment

### Router `app/services/ml/router.py`

This router exposes endpoints for:

- `POST /api/ml/models`
- `GET /api/ml/models`
- `GET /api/ml/models/{model_id}`
- `DELETE /api/ml/models/{model_id}`
- `POST /api/ml/models/{model_id}/train`
- `GET /api/ml/training-jobs`
- `POST /api/ml/models/{model_id}/evaluate`
- `POST /api/ml/models/{model_id}/tune`
- `POST /api/ml/models/{model_id}/predict`
- `POST /api/ml/predict`
- `GET /api/ml/models/{model_id}/explain`
- `POST /api/ml/models/{model_id}/deploy`

### Service `app/services/ml/service.py`

The service layer manages model lifecycle and training flows, including:

- registering model metadata
- validating pipeline parameters
- orchestrating training and asynchronous job execution
- returning evaluation and tuning results
- prediction requests
- deployment metadata

## Notes

The ML codebase is designed for extensibility. Concrete model implementations are organized by task type under `app/ml/classification/` and `app/ml/regression/`.

The service API is permission-based and integrates with the broader auth system.
