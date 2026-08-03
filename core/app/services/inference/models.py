# app/services/inference/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class InferenceStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


class EndpointStatus(str, Enum):
    ACTIVE = "active"
    DRAINING = "draining"
    INACTIVE = "inactive"
    FAILED = "failed"


@dataclass
class InferenceConfig:
    """Configuration for inference serving."""
    batch_size: int = 32
    timeout_ms: int = 5000
    max_retries: int = 2
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    explain_predictions: bool = False
    return_probabilities: bool = True
    threshold: Optional[float] = None
    preprocess: Optional[Dict[str, Any]] = None
    postprocess: Optional[Dict[str, Any]] = None


@dataclass
class ModelEndpoint:
    """A deployed model endpoint."""
    id: UUID
    name: str
    model_id: UUID
    model_version: int
    endpoint_path: str
    status: EndpointStatus = EndpointStatus.ACTIVE
    config: InferenceConfig = field(default_factory=InferenceConfig)
    metrics: Dict[str, Any] = field(default_factory=dict)
    deployed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_request_at: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class InferenceRequest:
    """Single or batch inference request."""
    instances: List[Dict[str, Any]]
    config_override: Optional[InferenceConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Inference response."""
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    explanations: Optional[List[Dict[str, Any]]] = None
    status: InferenceStatus = InferenceStatus.SUCCESS
    model_id: UUID = field(default_factory=uuid4)
    model_version: int = 1
    latency_ms: float = 0.0
    errors: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchInferenceRequest:
    """Large batch inference job."""
    dataset_path: str
    output_path: Optional[str] = None
    config_override: Optional[InferenceConfig] = None
    chunk_size: int = 10000
    notify_on_completion: Optional[List[str]] = None  # Email/webhook URLs


@dataclass
class InferenceJob:
    """Track a batch inference job."""
    id: UUID
    endpoint_id: UUID
    status: str = "queued"  # queued, running, completed, failed
    total_instances: int = 0
    processed_instances: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    output_path: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None