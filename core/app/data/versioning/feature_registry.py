"""
Feature registry for tracking all versioned features.
"""

from typing import Optional, List, Dict, Any, Set
from uuid import UUID
from datetime import datetime
import json
from pathlib import Path
from loguru import logger

from .models import (
    FeatureMetadata,
    SemanticVersion,
    VersionStatus,
    EntityType
)
from .exceptions import (
    VersionNotFoundError,
    DuplicateEntityError,
    RegistryError
)


class FeatureRegistry:
    """
    Registry for managing versioned features.

    Tracks:
    - Feature definitions
    - Feature types and data types
    - Feature statistics
    - Feature lineage (derived from which columns)
    """

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[SemanticVersion, FeatureMetadata]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.registry_path / 'feature_registry.json'

        if not registry_file.exists():
            return

        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)

            for feature_name, versions in data.items():
                for version_str, metadata_dict in versions.items():
                    version = SemanticVersion.parse(version_str)
                    if feature_name not in self._cache:
                        self._cache[feature_name] = {}

                    self._cache[feature_name][version] = FeatureMetadata(
                        **metadata_dict
                    )

            logger.info(f"Loaded feature registry with {len(self._cache)} features")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            raise RegistryError(f"Failed to load feature registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.registry_path / 'feature_registry.json'

        data = {}
        for feature_name, versions in self._cache.items():
            data[feature_name] = {}
            for version, metadata in versions.items():
                data[feature_name][str(version)] = metadata.dict()

        try:
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved feature registry: {registry_file}")

        except Exception as e:
            raise RegistryError(f"Failed to save feature registry: {e}")

    def register_feature(
        self,
        metadata: FeatureMetadata
    ) -> FeatureMetadata:
        """
        Register a new feature version.

        Args:
            metadata: Feature metadata

        Returns:
            Registered metadata
        """
        feature_name = metadata.name
        version = metadata.version

        # Check if version already exists
        if feature_name in self._cache and version in self._cache[feature_name]:
            raise DuplicateEntityError(
                f"Feature {feature_name} version {version} already exists"
            )

        # Update timestamps
        metadata.modified_at = datetime.utcnow()

        # Initialize feature entry if not exists
        if feature_name not in self._cache:
            self._cache[feature_name] = {}

        # Store in cache
        self._cache[feature_name][version] = metadata

        # Save to disk
        self._save_registry()

        logger.info(f"Registered feature {feature_name} version {version}")
        return metadata

    def get_feature(
        self,
        feature_name: str,
        version: Optional[SemanticVersion] = None,
        status: Optional[VersionStatus] = None
    ) -> FeatureMetadata:
        """
        Get a feature by name and optional version.

        Args:
            feature_name: Name of the feature
            version: Specific version (if None, get latest)
            status: Filter by status

        Returns:
            Feature metadata

        Raises:
            VersionNotFoundError: If feature not found
        """
        if feature_name not in self._cache:
            raise VersionNotFoundError(f"Feature not found: {feature_name}")

        versions = self._cache[feature_name]

        if version:
            if version not in versions:
                raise VersionNotFoundError(
                    f"Version {version} not found for feature {feature_name}"
                )
            return versions[version]

        # Filter by status if specified
        if status:
            filtered = {
                v: meta for v, meta in versions.items()
                if meta.status == status
            }
            if not filtered:
                raise VersionNotFoundError(
                    f"No {status} versions found for feature {feature_name}"
                )
            versions = filtered

        # Get latest version (highest semantic version)
        if not versions:
            raise VersionNotFoundError(f"No versions found for feature {feature_name}")

        latest_version = max(versions.keys())
        return versions[latest_version]

    def list_versions(
        self,
        feature_name: str,
        include_status: Optional[Set[VersionStatus]] = None
    ) -> Dict[SemanticVersion, FeatureMetadata]:
        """
        List all versions of a feature.

        Args:
            feature_name: Name of the feature
            include_status: Filter by status

        Returns:
            Dictionary mapping versions to metadata
        """
        if feature_name not in self._cache:
            return {}

        versions = self._cache[feature_name]

        if include_status:
            versions = {
                v: meta for v, meta in versions.items()
                if meta.status in include_status
            }

        return dict(sorted(versions.items(), key=lambda x: str(x[0])))

    def update_feature_status(
        self,
        feature_name: str,
        version: SemanticVersion,
        new_status: VersionStatus,
        reason: Optional[str] = None
    ) -> FeatureMetadata:
        """
        Update the status of a feature version.

        Args:
            feature_name: Name of the feature
            version: Version to update
            new_status: New status
            reason: Reason for status change

        Returns:
            Updated metadata
        """
        if feature_name not in self._cache:
            raise VersionNotFoundError(f"Feature not found: {feature_name}")

        if version not in self._cache[feature_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for feature {feature_name}"
            )

        metadata = self._cache[feature_name][version]
        metadata.status = new_status
        metadata.modified_at = datetime.utcnow()

        self._save_registry()
        logger.info(f"Updated feature {feature_name} version {version} status to {new_status}")

        return metadata

    def update_statistics(
        self,
        feature_name: str,
        version: SemanticVersion,
        statistics: Dict[str, Any]
    ) -> FeatureMetadata:
        """
        Update statistics for a feature.

        Args:
            feature_name: Name of the feature
            version: Version to update
            statistics: Statistics to add

        Returns:
            Updated metadata
        """
        if feature_name not in self._cache:
            raise VersionNotFoundError(f"Feature not found: {feature_name}")

        if version not in self._cache[feature_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for feature {feature_name}"
            )

        metadata = self._cache[feature_name][version]
        metadata.statistics.update(statistics)
        metadata.modified_at = datetime.utcnow()

        self._save_registry()
        logger.info(f"Updated statistics for {feature_name} version {version}")

        return metadata

    def get_features_by_type(
        self,
        feature_type: str,
        status: Optional[VersionStatus] = None
    ) -> List[FeatureMetadata]:
        """
        Get all features of a specific type.

        Args:
            feature_type: Type of feature (e.g., 'numerical', 'categorical')
            status: Filter by status

        Returns:
            List of feature metadata
        """
        results = []

        for feature_name, versions in self._cache.items():
            if not versions:
                continue

            # Get latest version
            latest = max(versions.keys())
            metadata = versions[latest]

            if metadata.feature_type == feature_type:
                if status is None or metadata.status == status:
                    results.append(metadata)

        return results

    def get_feature_lineage(
        self,
        feature_name: str,
        version: SemanticVersion
    ) -> Dict[str, Any]:
        """
        Get lineage information for a feature.

        Args:
            feature_name: Name of the feature
            version: Version to get lineage for

        Returns:
            Lineage information
        """
        if feature_name not in self._cache:
            raise VersionNotFoundError(f"Feature not found: {feature_name}")

        if version not in self._cache[feature_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for feature {feature_name}"
            )

        metadata = self._cache[feature_name][version]

        lineage_info = {
            "feature_name": metadata.name,
            "version": str(metadata.version),
            "feature_type": metadata.feature_type,
            "data_type": metadata.data_type,
            "derived_from": metadata.derived_from,
            "transformation_logic": metadata.transformation_logic,
            "statistics": metadata.statistics,
            "cardinality": metadata.cardinality,
            "missing_rate": metadata.missing_rate
        }

        return lineage_info

    def search_features(
        self,
        query: Optional[str] = None,
        feature_type: Optional[str] = None,
        data_type: Optional[str] = None,
        status: Optional[VersionStatus] = None,
        tags: Optional[Set[str]] = None,
        max_missing_rate: Optional[float] = None
    ) -> List[FeatureMetadata]:
        """
        Search for features based on criteria.

        Args:
            query: Text search in name and description
            feature_type: Filter by feature type
            data_type: Filter by data type
            status: Filter by status
            tags: Filter by tags
            max_missing_rate: Maximum missing rate

        Returns:
            List of matching feature metadata
        """
        results = []

        for feature_name, versions in self._cache.items():
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

            if feature_type and metadata.feature_type != feature_type:
                continue

            if data_type and metadata.data_type != data_type:
                continue

            if status and metadata.status != status:
                continue

            if tags and not tags.intersection(metadata.tags):
                continue

            if max_missing_rate is not None and metadata.missing_rate > max_missing_rate:
                continue

            results.append(metadata)

        return results

    def get_feature_dependencies(
        self,
        feature_name: str,
        version: SemanticVersion
    ) -> List[str]:
        """
        Get all dependencies for a feature.

        Args:
            feature_name: Name of the feature
            version: Version to get dependencies for

        Returns:
            List of dependency names
        """
        if feature_name not in self._cache:
            raise VersionNotFoundError(f"Feature not found: {feature_name}")

        if version not in self._cache[feature_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for feature {feature_name}"
            )

        metadata = self._cache[feature_name][version]
        return metadata.derived_from

    def compare_features(
        self,
        feature_name: str,
        version_a: SemanticVersion,
        version_b: SemanticVersion
    ) -> Dict[str, Any]:
        """
        Compare two versions of a feature.

        Args:
            feature_name: Name of the feature
            version_a: First version
            version_b: Second version

        Returns:
            Comparison results
        """
        if feature_name not in self._cache:
            raise VersionNotFoundError(f"Feature not found: {feature_name}")

        if version_a not in self._cache[feature_name]:
            raise VersionNotFoundError(
                f"Version {version_a} not found for feature {feature_name}"
            )

        if version_b not in self._cache[feature_name]:
            raise VersionNotFoundError(
                f"Version {version_b} not found for feature {feature_name}"
            )

        meta_a = self._cache[feature_name][version_a]
        meta_b = self._cache[feature_name][version_b]

        comparison = {
            "version_a": str(version_a),
            "version_b": str(version_b),
            "feature_type": {
                "a": meta_a.feature_type,
                "b": meta_b.feature_type
            },
            "data_type": {
                "a": meta_a.data_type,
                "b": meta_b.data_type
            },
            "nullable": {
                "a": meta_a.nullable,
                "b": meta_b.nullable
            },
            "cardinality": {
                "a": meta_a.cardinality,
                "b": meta_b.cardinality
            },
            "missing_rate": {
                "a": meta_a.missing_rate,
                "b": meta_b.missing_rate,
                "difference": meta_b.missing_rate - meta_a.missing_rate
            },
            "derived_from": {
                "a": meta_a.derived_from,
                "b": meta_b.derived_from
            },
            "statistics_diff": {
                k: {"a": meta_a.statistics.get(k), "b": meta_b.statistics.get(k)}
                for k in set(meta_a.statistics.keys()) | set(meta_b.statistics.keys())
                if meta_a.statistics.get(k) != meta_b.statistics.get(k)
            },
            "transformation_logic": {
                "a": meta_a.transformation_logic,
                "b": meta_b.transformation_logic
            }
        }

        return comparison