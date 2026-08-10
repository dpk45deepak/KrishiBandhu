# Configuration Guide

## Overview

AgriMind AI uses YAML-based configuration with Pydantic v2 validation.

## Primary Configuration File

- `configs/config.yaml` is the main application configuration file.
- When missing, the application falls back to defaults declared in `app/config/config.py`.

## Settings Structure

The root settings model is `app.config.config.Settings`.

Nested configuration groups:

- `project` — project metadata
- `paths` — filesystem directories for data, reports, logs, and docs
- `logging` — logger level, rotation, retention, and format
- `reporting` — profiling report output settings
- `validation` — validation engine runtime options
- `random_seed` — reproducibility default
- `supported_extensions` — supported input file formats

## Path Defaults

Defaults are created automatically during config load:

- `data/raw`
- `data/interim`
- `data/processed`
- `data/feature_store`
- `models`
- `reports/profiling`
- `reports/validation`
- `reports/eda`
- `logs`
- `scripts`
- `docs`

## Logging

The logging config includes:

- `level` — debug or info
- `rotation` — when to rotate log files
- `retention` — how long to keep logs
- `format` — console log format
- `file_format` — file log format
- `colored` — colorized console output

## Validation Settings

Validation config controls:

- `strict_mode`
- `fail_fast`
- `max_missing_percentage`
- `duplicate_threshold`
- `report_generation`
- `default_schema`

## Extending Configuration

To add a new config section:

1. Extend `app.config.config.Settings` with a new Pydantic model.
2. Update the YAML schema in `configs/config.yaml`.
3. Use `load_config()` to access values across the application.

## Runtime Behavior

- `load_config()` reads YAML and validates it.
- Missing files fall back to default settings.
- The config loader also creates missing directories automatically.
