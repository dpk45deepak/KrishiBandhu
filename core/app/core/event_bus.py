"""Simple event bus for platform-wide integration events."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PlatformEvent:
    """Base event emitted by runtime components."""

    name: str
    payload: dict[str, Any] | None = None


class EventBus:
    """Lightweight pub/sub event bus with topic-based handlers."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[PlatformEvent], None]]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: Callable[[PlatformEvent], None]) -> None:
        self._subscribers[event_name].append(handler)

    def publish(self, event: PlatformEvent) -> None:
        for handler in self._subscribers.get(event.name, []):
            handler(event)
