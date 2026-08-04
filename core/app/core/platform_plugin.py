"""Concrete runtime plugin that registers existing platform services."""

from __future__ import annotations

from typing import Any

from app.core.plugin import PlatformPlugin
from app.logger.logger import get_logger
from app.services.ml.service import MLService
from app.services.pipeline.service import PipelineService
from app.services.reports.service import ReportService


class PlatformServicesPlugin(PlatformPlugin):
    """Registers the main application services with the runtime container."""

    def __init__(self) -> None:
        self.name = "platform-services"
        self.logger = get_logger(__name__)

    def register(self, container: Any, event_bus: Any, dispatcher: Any, scheduler: Any) -> None:
        services = {
            "ml_service": MLService,
            "pipeline_service": PipelineService,
            "report_service": ReportService,
        }

        for name, service_cls in services.items():
            container.register(name, service_cls)
            self.logger.info("Registered service", name=name)

        dispatcher.register("training.start", lambda payload: payload)
        dispatcher.register("pipeline.run", lambda payload: payload)
        dispatcher.register("report.generate", lambda payload: payload)

        scheduler.add_job("pipeline-monitor", lambda: None, interval_seconds=60)
        scheduler.add_job("report-scheduler", lambda: None, interval_seconds=300)
