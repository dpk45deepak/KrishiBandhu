"""Decorator utilities for AgriMind AI."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """Decorator that logs the execution time of a function.

    Args:
        func: The function to time.

    Returns:
        Wrapped function with timing.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"{func.__name__} completed in {elapsed:.4f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} failed after {elapsed:.4f}s: {e}")
            raise

    return wrapper  # type: ignore[return-value]


def exception_handler(
    default_return: Any = None,
    log_level: str = "ERROR",
    reraise: bool = False,
) -> Callable[[F], F]:
    """Decorator that catches exceptions and logs them.

    Args:
        default_return: Value to return on exception (if not reraise).
        log_level: Log level for the error message.
        reraise: Whether to re-raise the exception after logging.

    Returns:
        Decorated function with exception handling.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(log_level, f"Exception in {func.__name__}: {e}")
                if reraise:
                    raise
                return default_return

        return wrapper  # type: ignore[return-value]

    return decorator


def log_entry_exit(func: F) -> F:
    """Decorator that logs function entry and exit.

    Args:
        func: The function to log.

    Returns:
        Wrapped function with entry/exit logging.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__}")
            return result
        except Exception as e:
            logger.debug(f"Exiting {func.__name__} with error: {e}")
            raise

    return wrapper  # type: ignore[return-value]
