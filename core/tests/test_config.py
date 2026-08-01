"""Tests for the configuration module."""

from pathlib import Path

import pytest
import yaml

from app.config.config import Settings, load_config
from app.constants.constants import get_project_root


class TestSettings:
    """Test the Settings model."""

    def test_default_settings(self) -> None:
        """Test that Settings can be created with defaults."""
        settings = Settings()
        assert settings.project.name == "AgriMind AI"
        assert settings.project.version == "0.1.0"
        assert settings.project.environment == "development"
        assert settings.random_seed == 42
        assert ".csv" in settings.supported_extensions

    def test_paths_exist_after_validation(self) -> None:
        """Test that ensure_paths_exist creates directories."""
        settings = Settings()
        root = get_project_root()

        # Check that the logging path was created via the validator
        logs_dir = root / settings.paths.logs
        assert logs_dir.exists()

        # Check reports path
        reports_dir = root / settings.paths.reports_profiling
        assert reports_dir.exists()


class TestLoadConfig:
    """Test loading config from YAML."""

    def test_load_from_existing_file(self) -> None:
        """Test loading from the project's config.yaml."""
        config = load_config()
        assert config.project.name == "AgriMind AI"
        assert isinstance(config.random_seed, int)

    def test_load_from_custom_path(self, tmp_path: Path) -> None:
        """Test loading from a custom config path."""
        custom_config = {
            "project": {"name": "Test Project", "environment": "development"},
            "paths": {
                "data_raw": str(tmp_path / "data" / "raw"),
                "logs": str(tmp_path / "logs"),
                "reports_profiling": str(tmp_path / "reports" / "profiling"),
            },
            "random_seed": 123,
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(custom_config, f)

        config = load_config(config_file)
        assert config.project.name == "Test Project"
        assert config.random_seed == 123

    def test_load_nonexistent_file_fallback(self) -> None:
        """Test that missing config file falls back to defaults."""
        config = load_config("/nonexistent/path/config.yaml")
        assert config.project.name == "AgriMind AI"
        assert config.project.version == "0.1.0"

    def test_settings_round_trip(self, tmp_path: Path) -> None:
        """Test that Settings can serialize back to dict."""
        settings = Settings()
        data = settings.model_dump()
        assert data["project"]["name"] == "AgriMind AI"
        assert data["random_seed"] == 42
        assert "supported_extensions" in data
