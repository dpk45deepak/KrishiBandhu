"""Configuration management for AgriMind AI using Pydantic v2 + YAML."""

from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from app.constants.constants import get_project_root


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str = "AgriMind AI"
    version: str = "0.1.0"
    description: str = "Agricultural Intelligence Platform"
    environment: Literal["development", "staging", "production"] = "development"


class PathsConfig(BaseModel):
    """Filesystem path configuration."""

    data_raw: str = "data/raw"
    data_interim: str = "data/interim"
    data_processed: str = "data/processed"
    data_feature_store: str = "data/feature_store"
    models: str = "models"
    reports_profiling: str = "reports/profiling"
    reports_validation: str = "reports/validation"
    reports_eda: str = "reports/eda"
    logs: str = "logs"
    scripts: str = "scripts"
    docs: str = "docs"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "DEBUG"
    rotation: str = "10 MB"
    retention: str = "30 days"
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> | "
        "<level>{message}</level>"
    )
    file_format: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name} | {message}"
    colored: bool = True


class ReportingConfig(BaseModel):
    """Reporting configuration for profiling and validation."""

    profiling_format: Literal["html", "json", "all"] = "html"
    include_plots: bool = True
    correlation_threshold: float = 0.7
    max_unique_values_for_high_cardinality: int = 50
    outlier_iqr_multiplier: float = 1.5


class Settings(BaseModel):
    """Root configuration model for AgriMind AI."""

    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    logging: LoggingConfig = LoggingConfig()
    reporting: ReportingConfig = ReportingConfig()
    random_seed: int = 42
    supported_extensions: list[str] = Field(
        default_factory=lambda: [".csv", ".xls", ".xlsx", ".parquet"]
    )

    @model_validator(mode="after")
    def ensure_paths_exist(self) -> "Settings":
        """Create all configured directories if they don't exist."""
        root = get_project_root()
        path_attrs = [
            self.paths.data_raw,
            self.paths.data_interim,
            self.paths.data_processed,
            self.paths.data_feature_store,
            self.paths.models,
            self.paths.reports_profiling,
            self.paths.reports_validation,
            self.paths.reports_eda,
            self.paths.logs,
            self.paths.scripts,
            self.paths.docs,
        ]
        for path_str in path_attrs:
            full_path = root / path_str
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {full_path}")
        return self


def load_config(config_path: str | Path | None = None) -> Settings:
    """Load configuration from a YAML file with fallback to defaults.

    Args:
        config_path: Path to config.yaml. If None, looks in configs/ directory.

    Returns:
        A validated Settings instance.
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        settings = Settings()
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        settings = Settings(**raw_config)

    # Ensure all paths exist
    settings.ensure_paths_exist()
    return settings
