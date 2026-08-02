"""
Artifact registry for tracking all versioned artifacts (models, reports, pipelines, etc.).
"""

from typing import Optional, List, Dict, Any, Set
from uuid import UUID
from datetime import datetime
import json
from pathlib import Path
from loguru import logger

from .models import (
    ArtifactMetadata,
    SemanticVersion,
    VersionStatus,
    ChecksumInfo,
    EntityType
)
from .exceptions import (
    ArtifactNotFoundError,
    VersionNotFoundError,
    DuplicateEntityError,
    RegistryError
)


class ArtifactRegistry:
    """
    Registry for managing versioned artifacts.

    Tracks:
    - Models (ML models, statistical models)
    - Reports (EDA reports, validation reports)
    - Pipelines (processing pipelines)
    - Visualizations
    - Any other generated artifacts
    """

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[SemanticVersion, ArtifactMetadata]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.registry_path / 'artifact_registry.json'

        if not registry_file.exists():
            return

        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)

            for artifact_name, versions in data.items():
                for version_str, metadata_dict in versions.items():
                    version = SemanticVersion.parse(version_str)
                    if artifact_name not in self._cache:
                        self._cache[artifact_name] = {}

                    self._cache[artifact_name][version] = ArtifactMetadata(
                        **metadata_dict
                    )

            logger.info(f"Loaded artifact registry with {len(self._cache)} artifacts")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            raise RegistryError(f"Failed to load artifact registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.registry_path / 'artifact_registry.json'

        data = {}
        for artifact_name, versions in self._cache.items():
            data[artifact_name] = {}
            for version, metadata in versions.items():
                data[artifact_name][str(version)] = metadata.dict()

        try:
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved artifact registry: {registry_file}")

        except Exception as e:
            raise RegistryError(f"Failed to save artifact registry: {e}")

    def register_artifact(
        self,
        metadata: ArtifactMetadata,
        checksum_info: Optional[ChecksumInfo] = None
    ) -> ArtifactMetadata:
        """
        Register a new artifact version.

        Args:
            metadata: Artifact metadata
            checksum_info: Optional checksum information

        Returns:
            Registered metadata
        """
        artifact_name = metadata.name
        version = metadata.version

        # Check if version already exists
        if artifact_name in self._cache and version in self._cache[artifact_name]:
            raise DuplicateEntityError(
                f"Artifact {artifact_name} version {version} already exists"
            )

        # Add checksum if provided
        if checksum_info:
            metadata.checksum = checksum_info

        # Update timestamps
        metadata.modified_at = datetime.utcnow()

        # Initialize artifact entry if not exists
        if artifact_name not in self._cache:
            self._cache[artifact_name] = {}

        # Store in cache
        self._cache[artifact_name][version] = metadata

        # Save to disk
        self._save_registry()

        logger.info(f"Registered artifact {artifact_name} version {version}")
        return metadata

    def get_artifact(
        self,
        artifact_name: str,
        version: Optional[SemanticVersion] = None,
        artifact_type: Optional[str] = None,
        status: Optional[VersionStatus] = None
    ) -> ArtifactMetadata:
        """
        Get an artifact by name and optional version.

        Args:
            artifact_name: Name of the artifact
            version: Specific version (if None, get latest)
            artifact_type: Filter by artifact type
            status: Filter by status

        Returns:
            Artifact metadata

        Raises:
            ArtifactNotFoundError: If artifact not found
        """
        if artifact_name not in self._cache:
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_name}")

        versions = self._cache[artifact_name]

        # Filter by artifact type if specified
        if artifact_type:
            versions = {
                v: meta for v, meta in versions.items()
                if meta.artifact_type == artifact_type
            }
            if not versions:
                raise ArtifactNotFoundError(
                    f"No {artifact_type} artifacts found for {artifact_name}"
                )

        if version:
            if version not in versions:
                raise VersionNotFoundError(
                    f"Version {version} not found for artifact {artifact_name}"
                )
            return versions[version]

        # Filter by status if specified
        if status:
            filtered = {
                v: meta for v, meta in versions.items()
                if meta.status == status
            }
            if not filtered:
                raise ArtifactNotFoundError(
                    f"No {status} versions found for artifact {artifact_name}"
                )
            versions = filtered

        # Get latest version (highest semantic version)
        if not versions:
            raise ArtifactNotFoundError(f"No versions found for artifact {artifact_name}")

        latest_version = max(versions.keys())
        return versions[latest_version]

    def list_versions(
        self,
        artifact_name: str,
        artifact_type: Optional[str] = None,
        include_status: Optional[Set[VersionStatus]] = None
    ) -> Dict[SemanticVersion, ArtifactMetadata]:
        """
        List all versions of an artifact.

        Args:
            artifact_name: Name of the artifact
            artifact_type: Filter by artifact type
            include_status: Filter by status

        Returns:
            Dictionary mapping versions to metadata
        """
        if artifact_name not in self._cache:
            return {}

        versions = self._cache[artifact_name]

        if artifact_type:
            versions = {
                v: meta for v, meta in versions.items()
                if meta.artifact_type == artifact_type
            }

        if include_status:
            versions = {
                v: meta for v, meta in versions.items()
                if meta.status in include_status
            }

        return dict(sorted(versions.items(), key=lambda x: str(x[0])))

    def update_artifact_status(
        self,
        artifact_name: str,
        version: SemanticVersion,
        new_status: VersionStatus,
        reason: Optional[str] = None
    ) -> ArtifactMetadata:
        """
        Update the status of an artifact version.

        Args:
            artifact_name: Name of the artifact
            version: Version to update
            new_status: New status
            reason: Reason for status change

        Returns:
            Updated metadata
        """
        if artifact_name not in self._cache:
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_name}")

        if version not in self._cache[artifact_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for artifact {artifact_name}"
            )

        metadata = self._cache[artifact_name][version]
        metadata.status = new_status
        metadata.modified_at = datetime.utcnow()

        self._save_registry()
        logger.info(f"Updated artifact {artifact_name} version {version} status to {new_status}")

        return metadata

    def add_metrics(
        self,
        artifact_name: str,
        version: SemanticVersion,
        metrics: Dict[str, float]
    ) -> ArtifactMetadata:
        """
        Add or update metrics for an artifact.

        Args:
            artifact_name: Name of the artifact
            version: Version to update
            metrics: Metrics to add

        Returns:
            Updated metadata
        """
        if artifact_name not in self._cache:
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_name}")

        if version not in self._cache[artifact_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for artifact {artifact_name}"
            )

        metadata = self._cache[artifact_name][version]
        metadata.metrics.update(metrics)
        metadata.modified_at = datetime.utcnow()

        self._save_registry()
        logger.info(f"Added metrics to {artifact_name} version {version}")

        return metadata

    def get_artifacts_by_training_dataset(
        self,
        dataset_id: UUID,
        dataset_version: Optional[SemanticVersion] = None
    ) -> List[ArtifactMetadata]:
        """
        Find all artifacts trained on a specific dataset.

        Args:
            dataset_id: Dataset ID
            dataset_version: Dataset version

        Returns:
            List of artifact metadata
        """
        results = []

        for artifact_name, versions in self._cache.items():
            for version, metadata in versions.items():
                if metadata.training_dataset_id == dataset_id:
                    if dataset_version is None or metadata.training_dataset_version == dataset_version:
                        results.append(metadata)

        return results

    def get_artifact_lineage(
        self,
        artifact_name: str,
        version: SemanticVersion
    ) -> Dict[str, Any]:
        """
        Get lineage information for an artifact.

        Args:
            artifact_name: Name of the artifact
            version: Version to get lineage for

        Returns:
            Lineage information
        """
        if artifact_name not in self._cache:
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_name}")

        if version not in self._cache[artifact_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for artifact {artifact_name}"
            )

        metadata = self._cache[artifact_name][version]

        lineage_info = {
            "artifact_name": metadata.name,
            "version": str(metadata.version),
            "artifact_type": metadata.artifact_type,
            "training_dataset_id": str(metadata.training_dataset_id) if metadata.training_dataset_id else None,
            "training_dataset_version": str(metadata.training_dataset_version) if metadata.training_dataset_version else None,
            "dependencies": [str(dep) for dep in metadata.dependencies],
            "parameters": metadata.parameters,
            "metrics": metadata.metrics,
            "framework": metadata.framework,
            "framework_version": metadata.framework_version
        }

        return lineage_info

    def search_artifacts(
        self,
        query: Optional[str] = None,
        artifact_type: Optional[str] = None,
        status: Optional[VersionStatus] = None,
        tags: Optional[Set[str]] = None,
        min_metric_value: Optional[Dict[str, float]] = None
    ) -> List[ArtifactMetadata]:
        """
        Search for artifacts based on criteria.

        Args:
            query: Text search in name and description
            artifact_type: Filter by artifact type
            status: Filter by status
            tags: Filter by tags
            min_metric_value: Minimum metric values {metric_name: min_value}

        Returns:
            List of matching artifact metadata
        """
        results = []

        for artifact_name, versions in self._cache.items():
            # Get latest version for search
            if not versions:
                continue

            latest = max(versions.keys())
            metadata = versions[latest]

            # Apply filters
            if query:
                query_lower = query.lower()
                if not (
                    query_lower in metadata.name.lower() or
                    (metadata.description and query_lower in metadata.description.lower())
                ):
                    continue

            if artifact_type and metadata.artifact_type != artifact_type:
                continue

            if status and metadata.status != status:
                continue

            if tags and not tags.intersection(metadata.tags):
                continue

            if min_metric_value:
                matches = True
                for metric_name, min_value in min_metric_value.items():
                    if metric_name not in metadata.metrics or metadata.metrics[metric_name] < min_value:
                        matches = False
                        break
                if not matches:
                    continue

            results.append(metadata)

        return results

    def compare_artifacts(
        self,
        artifact_name: str,
        version_a: SemanticVersion,
        version_b: SemanticVersion
    ) -> Dict[str, Any]:
        """
        Compare two versions of an artifact.

        Args:
            artifact_name: Name of the artifact
            version_a: First version
            version_b: Second version

        Returns:
            Comparison results
        """
        if artifact_name not in self._cache:
            raise ArtifactNotFoundError(f"Artifact not found: {artifact_name}")

        if version_a not in self._cache[artifact_name]:
            raise VersionNotFoundError(
                f"Version {version_a} not found for artifact {artifact_name}"
            )

        if version_b not in self._cache[artifact_name]:
            raise VersionNotFoundError(
                f"Version {version_b} not found for artifact {artifact_name}"
            )

        meta_a = self._cache[artifact_name][version_a]
        meta_b = self._cache[artifact_name][version_b]

        # Compare metrics
        metric_diff = {}
        all_metrics = set(meta_a.metrics.keys()) | set(meta_b.metrics.keys())

        for metric in all_metrics:
            val_a = meta_a.metrics.get(metric)
            val_b = meta_b.metrics.get(metric)
            if val_a != val_b:
                metric_diff[metric] = {
                    "version_a": val_a,
                    "version_b": val_b,
                    "difference": val_b - val_a if val_a is not None and val_b is not None else None
                }

        comparison = {
            "version_a": str(version_a),
            "version_b": str(version_b),
            "artifact_type": {
                "a": meta_a.artifact_type,
                "b": meta_b.artifact_type
            },
            "framework": {
                "a": meta_a.framework,
                "b": meta_b.framework
            },
            "metrics_diffs": metric_diff,
            "parameter_diffs": {
                k: {"a": meta_a.parameters.get(k), "b": meta_b.parameters.get(k)}
                for k in set(meta_a.parameters.keys()) | set(meta_b.parameters.keys())
                if meta_a.parameters.get(k) != meta_b.parameters.get(k)
            },
            "training_dataset": {
                "a": str(meta_a.training_dataset_id) if meta_a.training_dataset_id else None,
                "b": str(meta_b.training_dataset_id) if meta_b.training_dataset_id else None
            },
            "checksum_match": (
                meta_a.checksum and meta_b.checksum and
                meta_a.checksum.sha256 == meta_b.checksum.sha256
            )
        }

        return comparison