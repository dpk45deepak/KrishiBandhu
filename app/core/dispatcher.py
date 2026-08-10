"""Task dispatcher for background work and runtime actions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Dispatcher:
    """Simple dispatcher that routes actions to registered handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(self, action: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        self._handlers[action] = handler

    def dispatch(self, action: str, payload: dict[str, Any] | None = None) -> Any:
        if action not in self._handlers:
            raise KeyError(f"No handler registered for action '{action}'")
        return self._handlers[action](payload or {})
