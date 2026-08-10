"""Registry primitives for runtime components and plugins."""

from __future__ import annotations

from typing import Any


class ComponentRegistry:
    """Simple in-process registry for runtime components."""

    def __init__(self) -> None:
        self._components: dict[str, Any] = {}

    def register(self, name: str, component: Any) -> None:
        self._components[name] = component

    def get(self, name: str) -> Any:
        if name not in self._components:
            raise KeyError(f"Component '{name}' is not registered")
        return self._components[name]

    def has(self, name: str) -> bool:
        return name in self._components

    def all(self) -> dict[str, Any]:
        return dict(self._components)
