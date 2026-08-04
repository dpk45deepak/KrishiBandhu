"""Dependency abstractions for the AgriMind runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class SupportsDependencyInjection(Protocol):
    """Protocol for objects that can be resolved from the container."""


@dataclass(slots=True)
class DependencyDefinition:
    """Registration definition for a dependency."""

    name: str
    factory: Callable[[], Any]
    singleton: bool = True
    instance: Any | None = field(default=None, repr=False)

    def resolve(self) -> Any:
        if self.singleton and self.instance is not None:
            return self.instance
        value = self.factory()
        if self.singleton:
            self.instance = value
        return value
