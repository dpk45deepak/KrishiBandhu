# app/services/health/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SystemHealth:
    status: HealthStatus
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))