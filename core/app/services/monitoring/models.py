# app/services/monitoring/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class MetricType(str, Enum):
    # API metrics
    API_REQUEST_COUNT = "api_request_count"
    API_LATENCY = "api_latency"
    API_ERROR_COUNT = "api_error_count"
    
    # Pipeline metrics
    PIPELINE_RUN_COUNT = "pipeline_run_count"
    PIPELINE_DURATION = "pipeline_duration"
    PIPELINE_SUCCESS_RATE = "pipeline_success_rate"
    
    # ML metrics
    MODEL_TRAINING_TIME = "model_training_time"
    MODEL_ACCURACY = "model_accuracy"
    MODEL_PREDICTION_COUNT = "model_prediction_count"
    MODEL_DRIFT = "model_drift"
    
    # Data metrics
    DATA_INGEST_RATE = "data_ingest_rate"
    DATA_QUALITY_SCORE = "data_quality_score"
    FEATURE_DRIFT = "feature_drift"
    
    # System metrics
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    
    # Custom
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricRecord:
    """A single metric data point."""
    id: UUID
    metric_type: MetricType
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AlertRule:
    """Rule for triggering alerts."""
    id: UUID
    name: str
    metric_type: MetricType
    condition: str  # e.g., "> 0.9", "< 100", "== 0"
    severity: AlertSeverity = AlertSeverity.WARNING
    window_seconds: int = 300  # Evaluation window
    cooldown_seconds: int = 600  # Minimum time between alerts
    enabled: bool = True
    notifications: List[str] = field(default_factory=list)  # Email/webhook URLs
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """A triggered alert."""
    id: UUID
    rule_id: UUID
    rule_name: str
    severity: AlertSeverity
    metric_type: MetricType
    current_value: float
    threshold_condition: str
    message: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None


@dataclass
class DashboardConfig:
    """Configuration for a monitoring dashboard."""
    id: UUID
    name: str
    panels: List[Dict[str, Any]]
    refresh_interval_seconds: int = 30
    layout: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """Snapshot of system metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    api_requests_per_minute: float
    avg_latency_ms: float
    active_pipelines: int
    models_deployed: int
    feature_groups: int
    total_predictions_today: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))