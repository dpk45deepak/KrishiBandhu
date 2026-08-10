"""File utility functions for AgriMind AI."""

import os
from pathlib import Path
from typing import Generator

from loguru import logger

from app.constants.constants import SUPPORTED_FILE_FORMATS


def get_file_extension(file_path: str | Path) -> str:
    """Extract and normalize the file extension.

    Args:
        file_path: Path to the file.

    Returns:
        Lowercase file extension including the dot.
    """
    return Path(file_path).suffix.lower()


def is_supported_format(file_path: str | Path) -> bool:
    """Check if a file has a supported extension.

    Args:
        file_path: Path to the file.

    Returns:
        True if the extension is supported.
    """
    ext = get_file_extension(file_path)
    supported = ext in SUPPORTED_FILE_FORMATS
    if not supported:
        logger.debug(f"Unsupported file format: {ext} for {file_path}")
    return supported


def get_file_size_mb(file_path: str | Path) -> float:
    """Get file size in megabytes.

    Args:
        file_path: Path to the file.

    Returns:
        File size in MB.
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def get_file_size_bytes(file_path: str | Path) -> int:
    """Get file size in bytes.

    Args:
        file_path: Path to the file.

    Returns:
        File size in bytes.
    """
    return os.path.getsize(file_path)


def scan_directory(
    directory: str | Path,
    recursive: bool = True,
) -> Generator[Path, None, None]:
    """Scan a directory for supported data files.

    Args:
        directory: Path to the directory to scan.
        recursive: Whether to scan subdirectories recursively.

    Yields:
        Path objects for each supported data file found.
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return

    pattern = "**/*" if recursive else "*"
    for file_path in directory.glob(pattern):
        if file_path.is_file() and is_supported_format(file_path):
            logger.debug(f"Found supported file: {file_path}")
            yield file_path


def count_files(directory: str | Path, recursive: bool = True) -> int:
    """Count supported data files in a directory.

    Args:
        directory: Path to the directory.
        recursive: Whether to count recursively.

    Returns:
        Number of supported files found.
    """
    return sum(1 for _ in scan_directory(directory, recursive=recursive))
