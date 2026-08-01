"""Memory usage utilities for AgriMind AI."""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger


def get_dataframe_memory_usage(df: Any) -> dict[str, float]:
    """Estimate memory usage of a DataFrame.

    Works with both pandas and polars DataFrames.

    Args:
        df: A pandas or polars DataFrame.

    Returns:
        Dict with 'total_mb' and 'per_column' (list of {column, mb}).
    """
    try:
        # Polars path
        if hasattr(df, "estimated_size"):
            total_bytes = df.estimated_size("mb")
            total_mb = float(total_bytes)
            per_column: list[dict[str, Any]] = []
            for col in df.columns:
                col_bytes = df[col].estimated_size("mb")
                per_column.append({"column": col, "mb": float(col_bytes)})
        else:
            # Pandas path
            total_bytes = df.memory_usage(deep=True).sum()
            total_mb = total_bytes / (1024 * 1024)
            per_column = []
            for col in df.columns:
                col_bytes = df[col].memory_usage(deep=True)
                per_column.append({"column": col, "mb": col_bytes / (1024 * 1024)})
    except Exception as e:
        logger.warning(f"Could not estimate memory usage: {e}")
        return {"total_mb": 0.0, "per_column": []}

    result = {"total_mb": round(total_mb, 4), "per_column": per_column}
    logger.debug(f"DataFrame memory usage: {total_mb:.2f} MB")
    return result


def format_bytes(bytes_val: int) -> str:
    """Format bytes into a human-readable string.

    Args:
        bytes_val: Size in bytes.

    Returns:
        Human-readable size string (e.g., '2.5 MB', '1.2 GB').
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def get_object_size(obj: Any, seen: set | None = None) -> int:
    """Recursively estimate the size of a Python object.

    Args:
        obj: The object to measure.
        seen: Set of object ids already counted (for recursion guard).

    Returns:
        Approximate size in bytes.
    """
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(get_object_size(k, seen) + get_object_size(v, seen) for k, v in obj.items())
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_object_size(i, seen) for i in obj)
    return size
