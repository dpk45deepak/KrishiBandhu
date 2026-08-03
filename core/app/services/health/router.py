# app/services/health/router.py
from fastapi import APIRouter, Depends

from app.services.health.models import ComponentHealth, SystemHealth
from app.services.health.service import HealthService
from app.services.api.dependencies import get_health_service

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=SystemHealth)
async def get_health(
    service: HealthService = Depends(get_health_service),
) -> SystemHealth:
    """Get full system health status."""
    return await service.check_all()


@router.get("/ready")
async def readiness_check(
    service: HealthService = Depends(get_health_service),
):
    """Kubernetes readiness probe - checks all critical components."""
    health = await service.check_all()
    if health.status == "unhealthy":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "components": health.components}
        )
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe - minimal check."""
    return {"status": "alive"}


@router.get("/{component}", response_model=ComponentHealth)
async def get_component_health(
    component: str,
    service: HealthService = Depends(get_health_service),
) -> ComponentHealth:
    """Get health status for a specific component."""
    return await service.check_component(component)