# app/services/websocket/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import WebSocket


class WSMessageType(str, Enum):
    # Pipeline events
    PIPELINE_RUN_STARTED = "pipeline.run.started"
    PIPELINE_STAGE_PROGRESS = "pipeline.stage.progress"
    PIPELINE_STAGE_COMPLETED = "pipeline.stage.completed"
    PIPELINE_STAGE_FAILED = "pipeline.stage.failed"
    PIPELINE_RUN_COMPLETED = "pipeline.run.completed"
    PIPELINE_LOG = "pipeline.log"
    
    # Training events
    TRAINING_STARTED = "training.started"
    TRAINING_PROGRESS = "training.progress"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"
    
    # Data events
    DATASET_UPLOADED = "dataset.uploaded"
    DATASET_PROFILE_COMPLETED = "dataset.profile.completed"
    DATASET_VALIDATION_COMPLETED = "dataset.validation.completed"
    FEATURE_INGEST_COMPLETED = "feature.ingest.completed"
    
    # System events
    SYSTEM_METRICS = "system.metrics"
    ALERT_TRIGGERED = "alert.triggered"
    HEALTH_CHECK = "health.check"
    
    # User notifications
    NOTIFICATION = "notification"
    ERROR = "error"


class WSChannel(str, Enum):
    """Pre-defined WebSocket channels."""
    SYSTEM = "system"
    PIPELINES = "pipelines"
    TRAINING = "training"
    DATASETS = "datasets"
    ALERTS = "alerts"
    METRICS = "metrics"


@dataclass
class WSMessage:
    """WebSocket message format."""
    type: WSMessageType
    channel: WSChannel
    data: Dict[str, Any]
    message_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sender: Optional[str] = None


@dataclass
class WSConnection:
    """Represents an active WebSocket connection."""
    id: UUID
    websocket: WebSocket
    channels: List[WSChannel]
    user_id: Optional[str] = None
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)