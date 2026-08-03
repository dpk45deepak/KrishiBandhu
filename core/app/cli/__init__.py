# app/cli/__init__.py
"""
AgriMind CLI - Unified command-line interface.

Every command calls services through the API client layer.
No direct business logic - all operations go through services.
"""
from app.cli.main import app

__all__ = ["app"]