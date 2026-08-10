# Pipeline Guide

## Overview

The AgriMind AI pipeline system is centered in `app/data/pipeline` and exposes CLI controls through `app/data/pipeline/pipeline.py`.

## Core Components

- `PipelineOrchestrator` (`app/data/pipeline/orchestrator.py`)
- `StageRegistry` (`app/data/pipeline/registry.py`)
- `PipelineScheduler` (`app/data/pipeline/scheduler.py`)
- `StageExecutor` (`app/data/pipeline/executor.py`)
- `StateManager` (`app/data/pipeline/state.py`)
- `PipelineMonitor` (`app/data/pipeline/monitor.py`)
- `HookManager` (`app/data/pipeline/hooks.py`)

## Pipeline Configuration

Pipeline definitions are loaded from YAML or JSON via `PipelineOrchestrator._load_config()`.

The configuration model is defined in `app/data/pipeline/models.py` and includes:

- `stages`: list of `StageConfig`
- `schedule`: optional cron expression
- `notifications`
- `concurrency_limit`
- `max_retries`
- `timeout_seconds`
- `tags`

Each `StageConfig` includes:

- `stage_type`
- `params`
- `depends_on`
- `on_failure`
- `retry_count`
- `timeout_seconds`
- `condition`

## Execution Modes

The orchestrator supports sequential and parallel execution using `PipelineScheduler` and `StageExecutor`.

- Parallel execution is coordinated when multiple stages are runnable at the same dependency level.
- Checkpointing can persist pipeline state between stage runs.

## Pipeline CLI

The pipeline CLI is implemented in `app/data/pipeline/pipeline.py`.

### run

Run the pipeline with optional overrides:

```bash
python main.py pipeline run
python main.py pipeline run data/raw/my_dataset.csv
python main.py pipeline run --config configs/pipeline.yaml
python main.py pipeline run --dry-run
```

### status

List available checkpoints:

```bash
python main.py pipeline status
```

### clean

Delete checkpoint and pipeline reports:

```bash
python main.py pipeline clean
```

### list_stages

List supported pipeline stages.

## Stage Types

Defined in `app/data/pipeline/models.py`:

- `scan`
- `profile`
- `validate`
- `clean`
- `standardize`
- `feature_engineer`
- `feature_store`
- `train`
- `evaluate`
- `tune`
- `explain`
- `register`
- `predict`
- `report`
- `custom`

## Runtime Behavior

The orchestrator:

1. Loads configuration.
2. Initializes context, state, hooks, monitor, scheduler, and executor.
3. Validates stage registration and dependencies.
4. Executes stages in order or in parallel.
5. Saves checkpoints when enabled.
6. Generates reports and emits pipeline events.

## Current Implementation Notes

- The pipeline CLI loads a config file and creates `PipelineOrchestrator`.
- Stage registration is expected to be plugged into the orchestrator via `register_stages()` in future implementation.
- The orchestrator currently includes detailed execution sequencing and error handling.
