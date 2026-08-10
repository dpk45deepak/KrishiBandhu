"""
Data models for versioning framework using Pydantic.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Set
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator
from pathlib import Path


class VersionStatus(str, Enum):
    """Status of a versioned entity."""
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"


class EntityType(str, Enum):
    """Types of versionable entities."""
    DATASET = "dataset"
    ARTIFACT = "artifact"
    FEATURE = "feature"
    SCHEMA = "schema"
    MODEL = "model"
    PIPELINE = "pipeline"


class SemanticVersion(BaseModel):
    """Semantic version representation."""
    major: int = Field(ge=0)
    minor: int = Field(ge=0)
    patch: int = Field(ge=0)
    pre_release: Optional[str] = None
    build_metadata: Optional[str] = None

    def __str__(self) -> str:
        base = f"v{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            base = f"{base}-{self.pre_release}"
        if self.build_metadata:
            base = f"{base}+{self.build_metadata}"
        return base

    def increment_major(self) -> "SemanticVersion":
        return SemanticVersion(
            major=self.major + 1,
            minor=0,
            patch=0,
            pre_release=None,
            build_metadata=None
        )

    def increment_minor(self) -> "SemanticVersion":
        return SemanticVersion(
            major=self.major,
            minor=self.minor + 1,
            patch=0,
            pre_release=None,
            build_metadata=None
        )

    def increment_patch(self) -> "SemanticVersion":
        return SemanticVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch + 1,
            pre_release=None,
            build_metadata=None
        )

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse version string like v1.2.3 or 1.2.3-alpha+001."""
        clean = version_str.lstrip('v')
        pre_release = None
        build_metadata = None

        if '+' in clean:
            clean, build_metadata = clean.split('+')
        if '-' in clean:
            clean, pre_release = clean.split('-')

        parts = clean.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")

        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
            pre_release=pre_release,
            build_metadata=build_metadata
        )


class ChecksumInfo(BaseModel):
    """Checksum information for a versioned entity."""
    sha256: str
    md5: str
    file_size: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BaseMetadata(BaseModel):
    """Base metadata for all versioned entities."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    version: SemanticVersion
    status: VersionStatus = VersionStatus.DRAFT
    entity_type: EntityType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    modified_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    description: Optional[str] = None
    tags: Set[str] = Field(default_factory=set)

    class Config:
        use_enum_values = True
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class DatasetMetadata(BaseMetadata):
    """Metadata specific to datasets."""
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    schema_version: SemanticVersion
    source: Optional[str] = None
    license: Optional[str] = None
    pipeline_version: Optional[SemanticVersion] = None
    processing_history: List["ProcessingStep"] = Field(default_factory=list)
    checksum: Optional[ChecksumInfo] = None
    file_path: Optional[Path] = None
    format: str = "parquet"
    compression: Optional[str] = None
    partitioning_columns: List[str] = Field(default_factory=list)
    column_names: List[str] = Field(default_factory=list)
    column_types: Dict[str, str] = Field(default_factory=dict)
    null_counts: Dict[str, int] = Field(default_factory=dict)
    unique_counts: Dict[str, int] = Field(default_factory=dict)
    descriptive_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class ArtifactMetadata(BaseMetadata):
    """Metadata specific to artifacts (models, reports, etc.)."""
    artifact_type: str
    file_path: Optional[Path] = None
    checksum: Optional[ChecksumInfo] = None
    dependencies: List[UUID] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    training_dataset_version: Optional[SemanticVersion] = None
    training_dataset_id: Optional[UUID] = None
    framework: Optional[str] = None
    framework_version: Optional[str] = None


class FeatureMetadata(BaseMetadata):
    """Metadata specific to features."""
    feature_type: str
    data_type: str
    nullable: bool = True
    description: Optional[str] = None
    derived_from: List[str] = Field(default_factory=list)
    transformation_logic: Optional[str] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)
    cardinality: Optional[int] = None
    distribution: Optional[Dict[str, float]] = None
    missing_rate: float = Field(ge=0.0, le=1.0, default=0.0)


class SchemaMetadata(BaseMetadata):
    """Metadata specific to schemas."""
    schema_definition: Dict[str, Any] = Field(default_factory=dict)
    version_evolution: List[Dict[str, Any]] = Field(default_factory=list)
    compatibility: str = "backward"
    validation_rules: Dict[str, Any] = Field(default_factory=dict)


class ProcessingStep(BaseModel):
    """A single step in the processing pipeline."""
    step_id: UUID = Field(default_factory=uuid4)
    step_name: str
    step_type: str
    input_version: SemanticVersion
    output_version: SemanticVersion
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: float
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output_checksum: Optional[ChecksumInfo] = None
    status: str = "success"
    logs: Optional[str] = None


class LineageNode(BaseModel):
    """A node in the lineage graph."""
    entity_id: UUID
    entity_name: str
    entity_type: EntityType
    version: SemanticVersion
    checksum: Optional[ChecksumInfo] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LineageEdge(BaseModel):
    """An edge in the lineage graph representing a transformation."""
    source_id: UUID
    target_id: UUID
    transformation_type: str
    transformation_description: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class VersionCompareResult(BaseModel):
    """Result of comparing two versions."""
    version_a: SemanticVersion
    version_b: SemanticVersion
    metadata_diff: Dict[str, Any] = Field(default_factory=dict)
    schema_diff: Dict[str, Any] = Field(default_factory=dict)
    data_diff: Dict[str, Any] = Field(default_factory=dict)
    checksum_match: bool = False
    row_count_diff: Optional[int] = None
    column_count_diff: Optional[int] = None
    missing_columns_a: List[str] = Field(default_factory=list)
    missing_columns_b: List[str] = Field(default_factory=list)
    changed_columns: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    summary: str = ""