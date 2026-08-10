"""Example plugin that demonstrates runtime integration hooks."""

from __future__ import annotations

from typing import Any

from app.core.event_bus import PlatformEvent
from app.core.plugin import PlatformPlugin


class ExamplePlugin(PlatformPlugin):
    """Registers a simple event handler and a dispatcher callback."""

    def __init__(self) -> None:
        self.name = "example-plugin"

    def register(self, container: Any, event_bus: Any, dispatcher: Any, scheduler: Any) -> None:
        def handle_event(event: PlatformEvent) -> None:
            if event.name == "dataset.scanned":
                container.resolve("registry").register("last_dataset", event.payload or {})

        event_bus.subscribe("dataset.scanned", handle_event)
        dispatcher.register("dataset.scan", lambda payload: payload)
        scheduler.add_job("example-job", lambda: None, interval_seconds=60)
