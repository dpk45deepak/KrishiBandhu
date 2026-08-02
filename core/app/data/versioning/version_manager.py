"""
Version Manager - Central orchestrator for all versioning operations.
"""

from typing import Optional, List, Dict, Any, Union, Set
from pathlib import Path
from uuid import UUID
from datetime import datetime
import pandas as pd
from loguru import logger

from .models import (
    SemanticVersion,
    DatasetMetadata,
    ArtifactMetadata,
    FeatureMetadata,
    SchemaMetadata,
    VersionStatus,
    EntityType,
    ChecksumInfo,
    ProcessingStep,
    VersionCompareResult
)
from .dataset_registry import DatasetRegistry
from .artifact_registry import ArtifactRegistry
from .feature_registry import FeatureRegistry
from .schema_registry import SchemaRegistry
from .lineage import LineageTracker
from .checksum import ChecksumGenerator
from .storage import StorageManager
from .report import ReportGenerator
from .exceptions import (
    VersionNotFoundError,
    DatasetNotFoundError,
    ArtifactNotFoundError,
    VersionConflictError,
    ChecksumMismatchError,
    RollbackError,
    StorageError
)


class VersionManager:
    """
    Central orchestrator for managing all versioning operations.

    Provides a unified interface for:
    - Dataset versioning
    - Artifact versioning
    - Feature versioning
    - Schema versioning
    - Lineage tracking
    - Rollback operations
    - Version comparison
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.versioning_path = self.base_path / 'versioning_data'

        # Initialize components
        self.storage = StorageManager(self.base_path)
        self.checksum_generator = ChecksumGenerator()

        self.dataset_registry = DatasetRegistry(self.versioning_path / 'registries')
        self.artifact_registry = ArtifactRegistry(self.versioning_path / 'registries')
        self.feature_registry = FeatureRegistry(self.versioning_path / 'registries')
        self.schema_registry = SchemaRegistry(self.versioning_path / 'registries')

        self.lineage_tracker = LineageTracker(self.versioning_path / 'lineage')
        self.report_generator = ReportGenerator(
            self.dataset_registry,
            self.artifact_registry,
            self.feature_registry,
            self.schema_registry,
            self.lineage_tracker,
            self.base_path / 'reports' / 'versioning'
        )

        logger.info("VersionManager initialized successfully")

    # ========== Dataset Versioning ==========

    def create_dataset_version(
        self,
        df: pd.DataFrame,
        name: str,
        version_type: str = 'patch',
        source: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        license: Optional[str] = None,
        format: str = 'parquet',
        compression: Optional[str] = None
    ) -> DatasetMetadata:
        """
        Create a new version of a dataset.

        Args:
            df: DataFrame to version
            name: Name of the dataset
            version_type: Type of version (major, minor, patch)
            source: Source of the dataset
            description: Description of the dataset
            tags: Tags for the dataset
            license: License information
            format: File format
            compression: Compression algorithm

        Returns:
            Dataset metadata
        """
        # Determine version
        if name in self.dataset_registry._cache:
            existing_versions = self.dataset_registry.list_versions(name)
            if existing_versions:
                latest = max(existing_versions.keys())

                if version_type == 'major':
                    version = latest.increment_major()
                elif version_type == 'minor':
                    version = latest.increment_minor()
                else:  # patch
                    version = latest.increment_patch()
            else:
                version = SemanticVersion(major=1, minor=0, patch=0)
        else:
            version = SemanticVersion(major=1, minor=0, patch=0)

        # Store dataset
        file_path = self.storage.store_dataset(
            df,
            name,
            version,
            format=format,
            compression=compression
        )

        # Generate checksum
        checksum = self.checksum_generator.generate_file_checksum(file_path)

        # Create metadata
        metadata = DatasetMetadata(
            name=name,
            version=version,
            entity_type=EntityType.DATASET,
            status=VersionStatus.DRAFT,
            rows=len(df),
            columns=len(df.columns),
            schema_version=version,
            source=source,
            license=license,
            description=description,
            tags=tags or set(),
            checksum=checksum,
            file_path=file_path,
            format=format,
            compression=compression,
            column_names=df.columns.tolist(),
            column_types={col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            null_counts=df.isnull().sum().to_dict(),
            unique_counts=df.nunique().to_dict()
        )

        # Register metadata
        metadata = self.dataset_registry.register_dataset(metadata)

        # Add lineage node
        self.lineage_tracker.add_node(
            metadata.id,
            name,
            EntityType.DATASET,
            version,
            checksum.dict() if checksum else None
        )

        logger.info(f"Created dataset version {name} v{version}")
        return metadata

    def get_dataset(
        self,
        name: str,
        version: Optional[Union[str, SemanticVersion]] = None
    ) -> DatasetMetadata:
        """Get a dataset by name and optional version."""
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return self.dataset_registry.get_dataset(name, version)

    def load_dataset(
        self,
        name: str,
        version: Optional[Union[str, SemanticVersion]] = None
    ) -> pd.DataFrame:
        """Load a dataset from storage."""
        metadata = self.get_dataset(name, version)

        if not metadata.file_path or not metadata.file_path.exists():
            raise StorageError(f"Dataset file not found: {metadata.file_path}")

        if metadata.checksum:
            self.checksum_generator.verify_checksum(
                metadata.file_path,
                metadata.checksum
            )

        return self.storage.load_dataset(
            name,
            metadata.version,
            format=metadata.format
        )

    def list_datasets(self) -> List[str]:
        """List all dataset names."""
        return list(self.dataset_registry._cache.keys())

    def list_dataset_versions(self, name: str) -> List[str]:
        """List all versions of a dataset."""
        versions = self.dataset_registry.list_versions(name)
        return [str(v) for v in versions.keys()]

    # ========== Artifact Versioning ==========

    def create_artifact_version(
        self,
        data: Union[bytes, Dict[str, Any]],
        name: str,
        artifact_type: str,
        version_type: str = 'patch',
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        framework: Optional[str] = None,
        framework_version: Optional[str] = None,
        training_dataset_id: Optional[UUID] = None,
        training_dataset_version: Optional[SemanticVersion] = None,
        file_extension: str = 'json'
    ) -> ArtifactMetadata:
        """Create a new version of an artifact."""
        # Determine version
        if name in self.artifact_registry._cache:
            existing_versions = self.artifact_registry.list_versions(name)
            if existing_versions:
                latest = max(existing_versions.keys())
                if version_type == 'major':
                    version = latest.increment_major()
                elif version_type == 'minor':
                    version = latest.increment_minor()
                else:
                    version = latest.increment_patch()
            else:
                version = SemanticVersion(major=1, minor=0, patch=0)
        else:
            version = SemanticVersion(major=1, minor=0, patch=0)

        # Store artifact
        file_path = self.storage.store_artifact(
            data,
            name,
            version,
            artifact_type,
            file_extension=file_extension
        )

        # Generate checksum
        checksum = self.checksum_generator.generate_file_checksum(file_path)

        # Create metadata
        metadata = ArtifactMetadata(
            name=name,
            version=version,
            entity_type=EntityType.ARTIFACT,
            status=VersionStatus.DRAFT,
            artifact_type=artifact_type,
            description=description,
            tags=tags or set(),
            checksum=checksum,
            file_path=file_path,
            dependencies=[],
            parameters=parameters or {},
            metrics=metrics or {},
            training_dataset_id=training_dataset_id,
            training_dataset_version=training_dataset_version,
            framework=framework,
            framework_version=framework_version
        )

        # Register metadata
        metadata = self.artifact_registry.register_artifact(metadata)

        # Add lineage node
        self.lineage_tracker.add_node(
            metadata.id,
            name,
            EntityType.ARTIFACT,
            version,
            checksum.dict() if checksum else None
        )

        # Add lineage edge from training dataset if specified
        if training_dataset_id:
            self.lineage_tracker.add_edge(
                training_dataset_id,
                metadata.id,
                "trained_on",
                f"Trained on dataset version {training_dataset_version}",
                {"dataset_id": str(training_dataset_id)}
            )

        logger.info(f"Created artifact version {name} v{version}")
        return metadata

    def get_artifact(
        self,
        name: str,
        version: Optional[Union[str, SemanticVersion]] = None
    ) -> ArtifactMetadata:
        """Get an artifact by name and optional version."""
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return self.artifact_registry.get_artifact(name, version)

    def load_artifact(
        self,
        name: str,
        version: Optional[Union[str, SemanticVersion]] = None
    ) -> Union[Dict[str, Any], bytes]:
        """Load an artifact from storage."""
        metadata = self.get_artifact(name, version)

        if not metadata.file_path or not metadata.file_path.exists():
            raise StorageError(f"Artifact file not found: {metadata.file_path}")

        if metadata.checksum:
            self.checksum_generator.verify_checksum(
                metadata.file_path,
                metadata.checksum
            )

        ext = metadata.file_path.suffix.lstrip('.')
        return self.storage.load_artifact(
            name,
            metadata.version,
            metadata.artifact_type,
            file_extension=ext
        )

    # ========== Feature Versioning ==========

    def create_feature_version(
        self,
        name: str,
        feature_type: str,
        data_type: str,
        version_type: str = 'patch',
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        derived_from: Optional[List[str]] = None,
        transformation_logic: Optional[str] = None,
        nullable: bool = True,
        statistics: Optional[Dict[str, Any]] = None,
        cardinality: Optional[int] = None,
        missing_rate: float = 0.0
    ) -> FeatureMetadata:
        """Create a new version of a feature."""
        # Determine version
        if name in self.feature_registry._cache:
            existing_versions = self.feature_registry.list_versions(name)
            if existing_versions:
                latest = max(existing_versions.keys())
                if version_type == 'major':
                    version = latest.increment_major()
                elif version_type == 'minor':
                    version = latest.increment_minor()
                else:
                    version = latest.increment_patch()
            else:
                version = SemanticVersion(major=1, minor=0, patch=0)
        else:
            version = SemanticVersion(major=1, minor=0, patch=0)

        # Create metadata
        metadata = FeatureMetadata(
            name=name,
            version=version,
            entity_type=EntityType.FEATURE,
            status=VersionStatus.DRAFT,
            feature_type=feature_type,
            data_type=data_type,
            description=description,
            tags=tags or set(),
            derived_from=derived_from or [],
            transformation_logic=transformation_logic,
            nullable=nullable,
            statistics=statistics or {},
            cardinality=cardinality,
            missing_rate=missing_rate
        )

        # Register metadata
        metadata = self.feature_registry.register_feature(metadata)

        # Add lineage node
        self.lineage_tracker.add_node(
            metadata.id,
            name,
            EntityType.FEATURE,
            version
        )

        logger.info(f"Created feature version {name} v{version}")
        return metadata

    def get_feature(
        self,
        name: str,
        version: Optional[Union[str, SemanticVersion]] = None
    ) -> FeatureMetadata:
        """Get a feature by name and optional version."""
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return self.feature_registry.get_feature(name, version)

    # ========== Schema Versioning ==========

    def create_schema_version(
        self,
        name: str,
        schema_definition: Dict[str, Any],
        version_type: str = 'patch',
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        compatibility: str = 'backward',
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> SchemaMetadata:
        """Create a new version of a schema."""
        # Determine version
        if name in self.schema_registry._cache:
            existing_versions = self.schema_registry.list_versions(name)
            if existing_versions:
                latest = max(existing_versions.keys())
                if version_type == 'major':
                    version = latest.increment_major()
                elif version_type == 'minor':
                    version = latest.increment_minor()
                else:
                    version = latest.increment_patch()
            else:
                version = SemanticVersion(major=1, minor=0, patch=0)
        else:
            version = SemanticVersion(major=1, minor=0, patch=0)

        # Create metadata
        metadata = SchemaMetadata(
            name=name,
            version=version,
            entity_type=EntityType.SCHEMA,
            status=VersionStatus.DRAFT,
            schema_definition=schema_definition,
            description=description,
            tags=tags or set(),
            compatibility=compatibility,
            validation_rules=validation_rules or {},
            version_evolution=[]
        )

        # Register metadata
        metadata = self.schema_registry.register_schema(metadata)

        # Add lineage node
        self.lineage_tracker.add_node(
            metadata.id,
            name,
            EntityType.SCHEMA,
            version
        )

        logger.info(f"Created schema version {name} v{version}")
        return metadata

    def get_schema(
        self,
        name: str,
        version: Optional[Union[str, SemanticVersion]] = None
    ) -> SchemaMetadata:
        """Get a schema by name and optional version."""
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        return self.schema_registry.get_schema(name, version)

    # ========== Version Comparison ==========

    def compare_versions(
        self,
        entity_type: str,
        entity_name: str,
        version_a: Union[str, SemanticVersion],
        version_b: Union[str, SemanticVersion]
    ) -> Dict[str, Any]:
        """Compare two versions of an entity."""
        if isinstance(version_a, str):
            version_a = SemanticVersion.parse(version_a)
        if isinstance(version_b, str):
            version_b = SemanticVersion.parse(version_b)

        if entity_type == 'dataset':
            return self.dataset_registry.compare_versions(
                entity_name,
                version_a,
                version_b
            )
        elif entity_type == 'artifact':
            return self.artifact_registry.compare_artifacts(
                entity_name,
                version_a,
                version_b
            )
        elif entity_type == 'feature':
            return self.feature_registry.compare_features(
                entity_name,
                version_a,
                version_b
            )
        elif entity_type == 'schema':
            return self.schema_registry.get_schema_diff(
                entity_name,
                version_a,
                version_b
            )
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")

    # ========== Rollback ==========

    def rollback_dataset(
        self,
        name: str,
        target_version: Union[str, SemanticVersion],
        reason: Optional[str] = None
    ) -> DatasetMetadata:
        """Rollback a dataset to a previous version."""
        if isinstance(target_version, str):
            target_version = SemanticVersion.parse(target_version)

        # Get target version metadata
        target_metadata = self.dataset_registry.get_dataset(name, target_version)

        # Get current version (latest)
        current_metadata = self.dataset_registry.get_dataset(name)

        # Check if target version is older
        if target_version >= current_metadata.version:
            raise RollbackError(
                f"Target version {target_version} is not older than current version {current_metadata.version}"
            )

        # Create a new version with the target data
        df = self.load_dataset(name, target_version)

        # Create rollback version
        rollback_metadata = self.create_dataset_version(
            df,
            name,
            version_type='minor',
            source=f"Rollback from v{current_metadata.version} to v{target_version}",
            description=f"Rolled back to version {target_version}. Reason: {reason or 'Not specified'}"
        )

        # Add rollback to processing history
        step = ProcessingStep(
            step_name="rollback",
            step_type="rollback",
            input_version=current_metadata.version,
            output_version=rollback_metadata.version,
            duration_seconds=0,
            parameters={
                "target_version": str(target_version),
                "reason": reason,
                "previous_version": str(current_metadata.version)
            }
        )
        self.dataset_registry.add_processing_step(
            name,
            rollback_metadata.version,
            step
        )

        logger.info(f"Rolled back dataset {name} to version {target_version}")
        return rollback_metadata

    def rollback_artifact(
        self,
        name: str,
        target_version: Union[str, SemanticVersion],
        reason: Optional[str] = None
    ) -> ArtifactMetadata:
        """Rollback an artifact to a previous version."""
        if isinstance(target_version, str):
            target_version = SemanticVersion.parse(target_version)

        # Get target version metadata
        target_metadata = self.artifact_registry.get_artifact(name, target_version)

        # Get current version (latest)
        current_metadata = self.artifact_registry.get_artifact(name)

        # Check if target version is older
        if target_version >= current_metadata.version:
            raise RollbackError(
                f"Target version {target_version} is not older than current version {current_metadata.version}"
            )

        # Load the target artifact
        artifact_data = self.load_artifact(name, target_version)

        # Determine file extension
        ext = target_metadata.file_path.suffix.lstrip('.') if target_metadata.file_path else 'json'

        # Create rollback version
        rollback_metadata = self.create_artifact_version(
            artifact_data,
            name,
            artifact_type=target_metadata.artifact_type,
            version_type='minor',
            description=f"Rolled back to version {target_version}. Reason: {reason or 'Not specified'}",
            tags=target_metadata.tags,
            parameters=target_metadata.parameters,
            metrics=target_metadata.metrics,
            framework=target_metadata.framework,
            framework_version=target_metadata.framework_version,
            file_extension=ext
        )

        logger.info(f"Rolled back artifact {name} to version {target_version}")
        return rollback_metadata

    # ========== Lineage ==========

    def get_lineage(
        self,
        entity_id: Union[str, UUID],
        depth: Optional[int] = None,
        include_upstream: bool = True,
        include_downstream: bool = True
    ) -> Dict[str, Any]:
        """Get lineage for a specific entity."""
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)

        return self.lineage_tracker.get_lineage(
            entity_id,
            depth=depth,
            include_upstream=include_upstream,
            include_downstream=include_downstream
        )

    def get_upstream_sources(self, entity_id: Union[str, UUID]) -> List[Dict[str, Any]]:
        """Get all upstream sources for an entity."""
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)

        return self.lineage_tracker.get_upstream_sources(entity_id)

    def get_downstream_consumers(self, entity_id: Union[str, UUID]) -> List[Dict[str, Any]]:
        """Get all downstream consumers for an entity."""
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)

        return self.lineage_tracker.get_downstream_consumers(entity_id)

    def detect_impact(
        self,
        entity_id: Union[str, UUID],
        change_type: str = 'MODIFIED'
    ) -> Dict[str, Any]:
        """Detect the impact of a change to an entity."""
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)

        return self.lineage_tracker.detect_impact(entity_id, change_type)

    # ========== Reports ==========

    def generate_version_report(
        self,
        entity_name: str,
        entity_type: str = 'dataset',
        output_format: str = 'html'
    ) -> str:
        """Generate a version report."""
        return self.report_generator.generate_version_report(
            entity_name,
            entity_type,
            output_format
        )

    def generate_lineage_report(
        self,
        entity_id: Union[str, UUID],
        depth: int = 3,
        output_format: str = 'html'
    ) -> str:
        """Generate a lineage report."""
        if isinstance(entity_id, str):
            entity_id = UUID(entity_id)

        return self.report_generator.generate_lineage_graph_report(
            str(entity_id),
            depth,
            output_format
        )

    def generate_artifact_report(
        self,
        artifact_name: str,
        version: Optional[Union[str, SemanticVersion]] = None,
        output_format: str = 'html'
    ) -> str:
        """Generate an artifact report."""
        if isinstance(version, str):
            version = SemanticVersion.parse(version)

        return self.report_generator.generate_artifact_report(
            artifact_name,
            version,
            output_format
        )

    def generate_registry_summary_report(self) -> str:
        """Generate a registry summary report."""
        return self.report_generator.generate_registry_summary_report()

    def save_report(
        self,
        report_content: str,
        report_name: str,
        format: str = 'html'
    ) -> Path:
        """Save a report to file."""
        return self.report_generator.save_report(
            report_content,
            report_name,
            format
        )

    # ========== Utility Methods ==========

    def get_version_history(
        self,
        entity_type: str,
        entity_name: str
    ) -> List[Dict[str, Any]]:
        """Get the version history for an entity."""
        if entity_type == 'dataset':
            versions = self.dataset_registry.list_versions(entity_name)
            return [
                {
                    "version": str(v),
                    "status": meta.status.value,
                    "created_at": meta.created_at.isoformat(),
                    "rows": meta.rows,
                    "columns": meta.columns
                }
                for v, meta in versions.items()
            ]
        elif entity_type == 'artifact':
            versions = self.artifact_registry.list_versions(entity_name)
            return [
                {
                    "version": str(v),
                    "status": meta.status.value,
                    "created_at": meta.created_at.isoformat(),
                    "artifact_type": meta.artifact_type
                }
                for v, meta in versions.items()
            ]
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")

    def validate_dataset_checksum(
        self,
        name: str,
        version: Optional[Union[str, SemanticVersion]] = None
    ) -> bool:
        """Validate the checksum of a dataset."""
        metadata = self.get_dataset(name, version)

        if not metadata.checksum or not metadata.file_path:
            return False

        try:
            self.checksum_generator.verify_checksum(
                metadata.file_path,
                metadata.checksum
            )
            return True
        except ChecksumMismatchError:
            return False

    def detect_duplicate_datasets(
        self,
        name1: str,
        name2: str,
        version1: Optional[Union[str, SemanticVersion]] = None,
        version2: Optional[Union[str, SemanticVersion]] = None
    ) -> bool:
        """Detect if two datasets are duplicates."""
        df1 = self.load_dataset(name1, version1)
        df2 = self.load_dataset(name2, version2)

        return self.checksum_generator.detect_duplicate_dataset(df1, df2)