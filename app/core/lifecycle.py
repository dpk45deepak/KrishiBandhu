"""Lifecycle hooks for runtime startup and shutdown."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class LifecycleManager:
    """Manages startup and shutdown callbacks for runtime services."""

    def __init__(self) -> None:
        self._startup_hooks: list[Callable[[], Any]] = []
        self._shutdown_hooks: list[Callable[[], Any]] = []

    def add_startup_hook(self, hook: Callable[[], Any]) -> None:
        self._startup_hooks.append(hook)

    def add_shutdown_hook(self, hook: Callable[[], Any]) -> None:
        self._shutdown_hooks.append(hook)

    def startup(self) -> None:
        for hook in self._startup_hooks:
            hook()

    def shutdown(self) -> None:
        for hook in reversed(self._shutdown_hooks):
            hook()
