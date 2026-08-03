# app/services/feature_store/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class FeatureType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    DATETIME = "datetime"
    GEOSPATIAL = "geospatial"
    EMBEDDING = "embedding"


class FeatureSource(str, Enum):
    RAW = "raw"
    ENGINEERED = "engineered"
    AGGREGATED = "aggregated"
    EXTERNAL = "external"
    DERIVED = "derived"


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""
    name: str
    dtype: FeatureType
    source: FeatureSource
    description: str = ""
    transformation: Optional[str] = None  # SQL/Python expression
    source_columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class FeatureGroup:
    """A collection of related features."""
    id: UUID
    name: str
    description: str
    features: List[FeatureDefinition]
    dataset_id: Optional[str] = None
    entity_key: str = "id"  # Primary key for joining
    timestamp_column: Optional[str] = None
    version: int = 1
    status: str = "active"  # active, deprecated, archived
    statistics: Optional["FeatureStats"] = None
    lineage: Optional["FeatureLineage"] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None


@dataclass
class FeatureVector:
    """A materialized set of feature values."""
    entity_id: str
    feature_values: Dict[str, Any]
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureStats:
    """Statistical summary of a feature group."""
    row_count: int
    feature_stats: Dict[str, Dict[str, Any]]  # feature_name -> {mean, std, min, max, ...}
    missing_counts: Dict[str, int]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    drift_metrics: Optional[Dict[str, Any]] = None
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FeatureLineage:
    """Track where features came from."""
    source_dataset_id: Optional[str] = None
    pipeline_run_id: Optional[str] = None
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    git_commit: Optional[str] = None


@dataclass
class FeatureStoreConfig:
    """Configuration for feature store operations."""
    online_store_enabled: bool = True
    offline_store_enabled: bool = True
    cache_ttl_seconds: int = 3600
    compute_on_demand: bool = False
    version_policy: str = "manual"  # manual, auto, semver