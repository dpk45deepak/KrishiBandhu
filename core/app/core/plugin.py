"""Plugin abstractions for extending the runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PlatformPlugin(ABC):
    """Base class for runtime plugins."""

    name: str = "plugin"

    @abstractmethod
    def register(self, container: Any, event_bus: Any, dispatcher: Any, scheduler: Any) -> None:
        """Register plugin components with the runtime."""

    def health_check(self) -> dict[str, Any]:
        return {"name": self.name, "status": "ok"}
