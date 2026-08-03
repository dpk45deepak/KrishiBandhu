# app/services/pipeline/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class PipelineStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class StageType(str, Enum):
    SCAN = "scan"
    PROFILE = "profile"
    VALIDATE = "validate"
    CLEAN = "clean"
    STANDARDIZE = "standardize"
    FEATURE_ENGINEER = "feature_engineer"
    FEATURE_STORE = "feature_store"
    TRAIN = "train"
    EVALUATE = "evaluate"
    TUNE = "tune"
    EXPLAIN = "explain"
    REGISTER = "register"
    PREDICT = "predict"
    REPORT = "report"
    CUSTOM = "custom"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class StageConfig:
    """Configuration for a single pipeline stage."""
    stage_type: StageType
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # stage names this depends on
    on_failure: str = "stop"  # "stop", "skip", "continue"
    retry_count: int = 0
    timeout_seconds: int = 3600
    condition: Optional[str] = None  # Python expression to evaluate


@dataclass
class PipelineStage:
    """Runtime state of a pipeline stage during execution."""
    id: UUID
    name: str
    stage_type: StageType
    config: StageConfig
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path/URI
    retry_attempt: int = 0


@dataclass
class RunLog:
    """Log entry for a pipeline run."""
    timestamp: datetime
    level: str
    stage_name: Optional[str]
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRun:
    """A single execution of a pipeline."""
    id: UUID
    pipeline_id: UUID
    pipeline_name: str
    status: StageStatus = StageStatus.PENDING
    stages: List[PipelineStage] = field(default_factory=list)
    logs: List[RunLog] = field(default_factory=list)
    triggered_by: Optional[str] = None
    trigger_type: str = "manual"  # "manual", "scheduled", "webhook", "api"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    input_params: Dict[str, Any] = field(default_factory=dict)
    output_artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    stages: List[StageConfig]
    schedule: Optional[str] = None  # Cron expression
    notifications: Dict[str, Any] = field(default_factory=dict)
    concurrency_limit: int = 1
    max_retries: int = 0
    timeout_seconds: int = 7200
    tags: List[str] = field(default_factory=list)


@dataclass
class PipelineCreate:
    """Request to create a pipeline."""
    name: str
    description: str = ""
    config: PipelineConfig
    dataset_id: Optional[str] = None


@dataclass
class PipelineResponse:
    """Pipeline information returned by API."""
    id: UUID
    name: str
    description: str
    status: PipelineStatus
    config: PipelineConfig
    dataset_id: Optional[str] = None
    latest_run: Optional[PipelineRun] = None
    run_count: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None