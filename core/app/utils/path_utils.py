"""Path utility functions for AgriMind AI."""

from pathlib import Path
from typing import List

from loguru import logger

from app.constants.constants import get_project_root


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if it is not absolute.

    Args:
        path: A path string or Path object.

    Returns:
        An absolute Path.
    """
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (get_project_root() / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The resolved Path to the directory.
    """
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory: {resolved}")
    return resolved


def list_dirs(directory: str | Path) -> List[Path]:
    """List all subdirectories in a directory.

    Args:
        directory: Path to list subdirectories of.

    Returns:
        List of Path objects for each subdirectory.
    """
    directory = resolve_path(directory)
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    return [p for p in directory.iterdir() if p.is_dir()]


def get_relative_path(path: str | Path) -> Path:
    """Get the relative path from the project root.

    Args:
        path: Absolute or relative path.

    Returns:
        Path relative to project root.
    """
    resolved = resolve_path(path)
    root = get_project_root()
    try:
        return resolved.relative_to(root)
    except ValueError:
        return resolved
