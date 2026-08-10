"""Dependency injection container for AgriMind runtime components."""

from __future__ import annotations

from typing import Any, Callable

from app.core.dependency import DependencyDefinition
from app.core.registry import ComponentRegistry


class Container:
    """Minimal IoC container with lazy registration and singleton behavior."""

    def __init__(self) -> None:
        self._definitions: dict[str, DependencyDefinition] = {}
        self.registry = ComponentRegistry()

    def register(self, name: str, factory: Callable[[], Any], *, singleton: bool = True) -> None:
        self._definitions[name] = DependencyDefinition(name=name, factory=factory, singleton=singleton)

    def register_instance(self, name: str, instance: Any) -> None:
        self.registry.register(name, instance)

    def resolve(self, name: str) -> Any:
        if self.registry.has(name):
            return self.registry.get(name)
        if name not in self._definitions:
            raise KeyError(f"Dependency '{name}' is not registered")
        value = self._definitions[name].resolve()
        self.registry.register(name, value)
        return value

    def build(self) -> None:
        for name in list(self._definitions):
            self.resolve(name)
