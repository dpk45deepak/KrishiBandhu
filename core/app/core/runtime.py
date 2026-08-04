"""Top-level runtime orchestrator for AgriMind AI."""

from __future__ import annotations

from app.config.config import load_config
from app.core.container import Container
from app.core.dispatcher import Dispatcher
from app.core.event_bus import EventBus, PlatformEvent
from app.core.lifecycle import LifecycleManager
from app.core.plugin import PlatformPlugin
from app.core.registry import ComponentRegistry
from app.core.scheduler import Scheduler
from app.logger.logger import get_logger, setup_logger

from app.core.example_plugin import ExamplePlugin
from app.core.platform_plugin import PlatformServicesPlugin


class AgriMindRuntime:
    """Coordinates the platform runtime and its subsystems."""

    def __init__(self, *, config: object | None = None) -> None:
        self.config = config or load_config()
        self.logger = get_logger("app.core.runtime")
        setup_logger(level="INFO", colored=True)
        self.container = Container()
        self.registry = ComponentRegistry()
        self.event_bus = EventBus()
        self.dispatcher = Dispatcher()
        self.scheduler = Scheduler()
        self.lifecycle = LifecycleManager()
        self.plugins: list[PlatformPlugin] = [ExamplePlugin(), PlatformServicesPlugin()]

    def register_plugin(self, plugin: PlatformPlugin) -> None:
        self.plugins.append(plugin)

    def start(self) -> None:
        self.logger.info("Starting AgriMind runtime")
        self.container.register_instance("config", self.config)
        self.container.register_instance("event_bus", self.event_bus)
        self.container.register_instance("dispatcher", self.dispatcher)
        self.container.register_instance("scheduler", self.scheduler)
        self.container.register_instance("registry", self.registry)
        self.container.register_instance("lifecycle", self.lifecycle)
        self.container.build()

        for plugin in self.plugins:
            plugin.register(self.container, self.event_bus, self.dispatcher, self.scheduler)

        self.lifecycle.startup()
        self.event_bus.publish(PlatformEvent(name="runtime.started", payload={"status": "ok"}))

    def stop(self) -> None:
        self.logger.info("Stopping AgriMind runtime")
        self.lifecycle.shutdown()
        self.event_bus.publish(PlatformEvent(name="runtime.stopped", payload={"status": "ok"}))

    def health_check(self) -> dict[str, object]:
        return {
            "status": "ok",
            "components": [
                "container",
                "registry",
                "event_bus",
                "dispatcher",
                "scheduler",
                "lifecycle",
            ],
            "plugins": [plugin.name for plugin in self.plugins],
        }
