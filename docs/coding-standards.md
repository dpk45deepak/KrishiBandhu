# Coding Standards

## Overview

AgriMind AI follows Python best practices and consistent formatting rules.

## Formatting

- `black` with line length 100.
- `ruff` linting is configured in `pyproject.toml`.
- Ignore `E501` line length errors in `__init__.py` files.

## Type Safety

- Prefer explicit type annotations over `Any`.
- Use `typing` constructs such as `Optional`, `Literal`, `Protocol`, and `dataclass`.
- Keep runtime `Any` usage limited to adapter or loader code only.

## Project Patterns

- Keep CLI commands thin; business logic belongs in service or data modules.
- Each module should have a single responsibility.
- Use data classes and Pydantic models for structured data and configuration.
- Log meaningful actions and errors with `Loguru`.

## File Structure

- `app/config` — config models and loader
- `app/core` — runtime, plugin, scheduler, registry
- `app/data` — ingestion, profiling, validation, transformation, pipeline
- `app/services` — API service layer and domain-specific business logic
- `app/utils` — shared helpers and decorators

## Documentation

- Public classes and functions should include Google-style docstrings.
- Keep docstrings accurate and aligned with implementation.
- Document new modules in `docs/`.

## Testing

- Write tests in `tests/` using `pytest`.
- Follow the naming conventions: `test_*.py` and `test_*` functions.
- Test both success and error conditions.

## Review Checklist

- Does the code follow the existing package structure?
- Are all new public APIs documented?
- Are errors handled gracefully?
- Is logging present for important operations?
- Are config values loaded from `app/config/config.py` when applicable?
