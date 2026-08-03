# app/services/monitoring/router.py
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.logger import get_logger
from app.services.api.dependencies import get_current_user, require_permission
from app.services.auth.models import Permission
from app.services.monitoring.models import (
    AlertRule,
    AlertSeverity,
    MetricType,
    SystemMetrics,
)
from app.services.monitoring.service import MonitoringService

logger = get_logger(__name__)
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

_monitoring = MonitoringService()


def get_monitoring() -> MonitoringService:
    return _monitoring


@router.get("/metrics/{metric_type}", response_model=Dict[str, Any])
async def get_metric_summary(
    metric_type: MetricType,
    window_seconds: int = Query(3600, description="Time window in seconds"),
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
    service: MonitoringService = Depends(get_monitoring),
):
    """Get summary statistics for a metric."""
    return await service.get_metric_summary(metric_type, window_seconds)


@router.get("/metrics/{metric_type}/history", response_model=List[dict])
async def query_metrics(
    metric_type: MetricType,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
    service: MonitoringService = Depends(get_monitoring),
):
    """Query metric history."""
    from datetime import datetime
    start = datetime.fromisoformat(start_time) if start_time else None
    end = datetime.fromisoformat(end_time) if end_time else None
    
    records = await service.query_metrics(metric_type, start, end)
    return [
        {
            "id": str(r.id),
            "value": r.value,
            "tags": r.tags,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in records
    ]


@router.post("/alerts/rules", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_alert_rule(
    rule: AlertRule,
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
    service: MonitoringService = Depends(get_monitoring),
):
    """Create an alert rule."""
    created = await service.create_alert_rule(rule)
    return {
        "id": str(created.id),
        "name": created.name,
        "metric_type": created.metric_type.value,
        "condition": created.condition,
        "severity": created.severity.value,
    }


@router.get("/alerts/rules", response_model=List[dict])
async def list_alert_rules(
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
    service: MonitoringService = Depends(get_monitoring),
):
    """List alert rules."""
    rules = await service.get_alert_rules()
    return [
        {
            "id": str(r.id),
            "name": r.name,
            "metric_type": r.metric_type.value,
            "condition": r.condition,
            "severity": r.severity.value,
            "enabled": r.enabled,
        }
        for r in rules
    ]


@router.get("/alerts", response_model=List[dict])
async def list_alerts(
    acknowledged: Optional[bool] = Query(None),
    severity: Optional[AlertSeverity] = Query(None),
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
    service: MonitoringService = Depends(get_monitoring),
):
    """List triggered alerts."""
    alerts = await service.get_alerts(acknowledged=acknowledged, severity=severity)
    return [
        {
            "id": str(a.id),
            "rule_name": a.rule_name,
            "severity": a.severity.value,
            "metric_type": a.metric_type.value,
            "current_value": a.current_value,
            "message": a.message,
            "acknowledged": a.acknowledged,
            "triggered_at": a.triggered_at.isoformat(),
        }
        for a in alerts
    ]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
    service: MonitoringService = Depends(get_monitoring),
):
    """Acknowledge an alert."""
    acknowledged = await service.acknowledge_alert(alert_id, current_user["username"])
    if not acknowledged:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert acknowledged"}


@router.get("/system", response_model=SystemMetrics)
async def get_system_metrics(
    current_user: dict = Depends(require_permission(Permission.ADMIN.value)),
    service: MonitoringService = Depends(get_monitoring),
):
    """Get current system metrics."""
    return await service.get_system_metrics()