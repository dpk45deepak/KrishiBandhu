# app/services/health/service.py
import time
from typing import Dict

from app.config import settings
from app.logger import get_logger
from app.services.health.models import ComponentHealth, HealthStatus, SystemHealth
from app.utils.decorators import timed

logger = get_logger(__name__)


class HealthService:
    """Health monitoring service consuming existing config and logger modules."""
    
    def __init__(self):
        self._start_time = time.time()
        self._component_checks = {
            "data_platform": self._check_data_platform,
            "ml_platform": self._check_ml_platform,
            "config": self._check_config,
        }
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time
    
    @timed
    async def _check_data_platform(self) -> ComponentHealth:
        """Check data platform health using existing data module."""
        try:
            from app.data import DataPlatform
            platform = DataPlatform()
            await platform.ping()
            return ComponentHealth(
                name="data_platform",
                status=HealthStatus.HEALTHY,
                metadata={"version": platform.version}
            )
        except Exception as e:
            logger.error(f"Data platform health check failed: {e}")
            return ComponentHealth(
                name="data_platform",
                status=HealthStatus.UNHEALTHY,
                error=str(e)
            )
    
    @timed
    async def _check_ml_platform(self) -> ComponentHealth:
        """Check ML platform health using existing ml module."""
        try:
            from app.ml import MLPlatform
            platform = MLPlatform()
            await platform.ping()
            return ComponentHealth(
                name="ml_platform",
                status=HealthStatus.HEALTHY,
                metadata={"version": platform.version}
            )
        except Exception as e:
            logger.error(f"ML platform health check failed: {e}")
            return ComponentHealth(
                name="ml_platform",
                status=HealthStatus.UNHEALTHY,
                error=str(e)
            )
    
    @timed
    async def _check_config(self) -> ComponentHealth:
        """Verify configuration is valid."""
        try:
            from app.config import settings
            _ = settings.APP_NAME  # Validate config loads
            return ComponentHealth(
                name="config",
                status=HealthStatus.HEALTHY,
                metadata={
                    "app_name": settings.APP_NAME,
                    "environment": settings.ENVIRONMENT,
                }
            )
        except Exception as e:
            logger.error(f"Config health check failed: {e}")
            return ComponentHealth(
                name="config",
                status=HealthStatus.UNHEALTHY,
                error=str(e)
            )
    
    async def check_all(self) -> SystemHealth:
        """Run all component health checks."""
        components: Dict[str, ComponentHealth] = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_fn in self._component_checks.items():
            start = time.monotonic()
            result = await check_fn()
            result.latency_ms = (time.monotonic() - start) * 1000
            components[name] = result
            
            if result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return SystemHealth(
            status=overall_status,
            version=settings.VERSION,
            uptime_seconds=self.uptime_seconds,
            components=components,
        )
    
    async def check_component(self, component_name: str) -> ComponentHealth:
        """Check a specific component."""
        if component_name not in self._component_checks:
            return ComponentHealth(
                name=component_name,
                status=HealthStatus.UNHEALTHY,
                error=f"Unknown component: {component_name}"
            )
        return await self._component_checks[component_name]()