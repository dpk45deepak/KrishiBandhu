"""Pipeline domain models with Pydantic validation."""

from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, ConfigDict, validator
from pathlib import Path


class StageStatus(str, Enum):
    """Execution status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class StageType(str, Enum):
    """Types of pipeline stages."""
    SCAN = "scan"
    PROFILE = "profile"
    VALIDATE = "validate"
    CLEAN = "clean"
    STANDARDIZE = "standardize"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_STORE = "feature_store"
    EDA = "eda"
    SAVE = "save"


class ExecutionMode(str, Enum):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


class StageConfig(BaseModel):
    """Configuration for a single pipeline stage."""
    model_config = ConfigDict(extra="forbid")
    
    name: StageType
    enabled: bool = True
    retry_count: int = Field(default=3, ge=0, le=10)
    retry_delay: int = Field(default=5, ge=1, le=300)
    timeout: int = Field(default=3600, ge=60, le=86400)
    depends_on: List[StageType] = Field(default_factory=list)
    parallel_group: Optional[str] = None
    condition: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(default="agrimind_pipeline")
    version: str = Field(default="1.0.0")
    stages: List[StageConfig]
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_parallel_workers: int = Field(default=4, ge=1, le=16)
    enable_checkpointing: bool = True
    enable_metrics: bool = True
    enable_hooks: bool = True
    artifact_dir: Path = Field(default=Path("reports/pipeline"))
    log_level: str = Field(default="INFO")
    
    @validator("artifact_dir")
    def validate_artifact_dir(cls, v: Path) -> Path:
        """Ensure artifact directory is valid."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class StageMetadata(BaseModel):
    """Metadata for a stage execution."""
    stage_type: StageType
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    retry_count: int = 0
    error: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    output_path: Optional[Path] = None


class PipelineState(BaseModel):
    """Complete pipeline execution state."""
    pipeline_id: str
    config: PipelineConfig
    stages: Dict[StageType, StageMetadata]
    current_stage: Optional[StageType] = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: StageStatus = StageStatus.PENDING
    checkpoint: Optional[Path] = None
    execution_graph: Dict[StageType, Set[StageType]] = Field(default_factory=dict)
    shared_context: Dict[str, Any] = Field(default_factory=dict)


class PipelineEvent(BaseModel):
    """Event emitted during pipeline execution."""
    event_type: str
    pipeline_id: str
    stage_type: Optional[StageType] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    """Result of a stage execution."""
    success: bool
    stage_type: StageType
    metadata: StageMetadata
    output: Optional[Any] = None
    error: Optional[Exception] = None