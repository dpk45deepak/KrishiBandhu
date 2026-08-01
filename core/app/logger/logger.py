"""Logging configuration for AgriMind AI using Loguru."""

import sys
from pathlib import Path

from loguru import logger

from app.constants.constants import FOLDER_LOGS, get_project_root


def setup_logger(
    level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "30 days",
    colored: bool = True,
) -> None:
    """Configure Loguru with console and rotating file handlers.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        rotation: Max file size before rotation (e.g., '10 MB', '1 GB').
        retention: How long to keep old log files (e.g., '30 days').
        colored: Whether to use colored console output.
    """
    # Remove default handler
    logger.remove()

    # Console handler with optional coloring
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    if colored:
        logger.add(sys.stderr, format=console_format, level=level, colorize=True)
    else:
        plain_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        )
        logger.add(sys.stderr, format=plain_format, level=level, colorize=False)

    # File handler with rotation
    logs_dir = get_project_root() / FOLDER_LOGS
    logs_dir.mkdir(parents=True, exist_ok=True)

    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} | {message}"
    )
    logger.add(
        sink=str(logs_dir / "agrimind_{time:YYYY-MM-DD}.log"),
        format=file_format,
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logger.debug(
        "Logger initialized | level={level} | rotation={rotation} | "
        "retention={retention} | logs_dir={logs_dir}",
        level=level,
        rotation=rotation,
        retention=retention,
        logs_dir=logs_dir,
    )


def get_logger(name: str | None = None):
    """Get a logger instance, optionally child-logged under a name.

    Args:
        name: Sub-logger name, typically __name__.

    Returns:
        A Loguru logger instance.
    """
    return logger.bind(name=name or __name__)
