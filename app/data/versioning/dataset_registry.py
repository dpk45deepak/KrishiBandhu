"""
Dataset registry for tracking all versioned datasets.
"""

from typing import Optional, List, Dict, Any, Set
from uuid import UUID
from datetime import datetime
import json
from pathlib import Path
from loguru import logger

from .models import (
    DatasetMetadata,
    SemanticVersion,
    VersionStatus,
    ChecksumInfo,
    ProcessingStep
)
from .exceptions import (
    DatasetNotFoundError,
    VersionNotFoundError,
    DuplicateEntityError,
    RegistryError
)


class DatasetRegistry:
    """
    Registry for managing versioned datasets.

    Tracks:
    - All datasets and their versions
    - Dataset metadata
    - Processing history
    - Version status
    """

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[SemanticVersion, DatasetMetadata]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.registry_path / 'dataset_registry.json'

        if not registry_file.exists():
            return

        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)

            for dataset_name, versions in data.items():
                for version_str, metadata_dict in versions.items():
                    version = SemanticVersion.parse(version_str)
                    if dataset_name not in self._cache:
                        self._cache[dataset_name] = {}

                    self._cache[dataset_name][version] = DatasetMetadata(
                        **metadata_dict
                    )

            logger.info(f"Loaded dataset registry with {len(self._cache)} datasets")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            raise RegistryError(f"Failed to load dataset registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.registry_path / 'dataset_registry.json'

        # Convert to serializable format
        data = {}
        for dataset_name, versions in self._cache.items():
            data[dataset_name] = {}
            for version, metadata in versions.items():
                data[dataset_name][str(version)] = metadata.dict()

        # Write with pretty formatting
        try:
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved dataset registry: {registry_file}")

        except Exception as e:
            raise RegistryError(f"Failed to save dataset registry: {e}")

    def register_dataset(
        self,
        metadata: DatasetMetadata,
        checksum_info: Optional[ChecksumInfo] = None
    ) -> DatasetMetadata:
        """
        Register a new dataset version.

        Args:
            metadata: Dataset metadata
            checksum_info: Optional checksum information

        Returns:
            Registered metadata
        """
        dataset_name = metadata.name
        version = metadata.version

        # Check if version already exists
        if dataset_name in self._cache and version in self._cache[dataset_name]:
            raise DuplicateEntityError(
                f"Dataset {dataset_name} version {version} already exists"
            )

        # Add checksum if provided
        if checksum_info:
            metadata.checksum = checksum_info

        # Update timestamps
        metadata.modified_at = datetime.utcnow()

        # Initialize dataset entry if not exists
        if dataset_name not in self._cache:
            self._cache[dataset_name] = {}

        # Store in cache
        self._cache[dataset_name][version] = metadata

        # Save to disk
        self._save_registry()

        logger.info(f"Registered dataset {dataset_name} version {version}")
        return metadata

    def get_dataset(
        self,
        dataset_name: str,
        version: Optional[SemanticVersion] = None,
        status: Optional[VersionStatus] = None
    ) -> DatasetMetadata:
        """
        Get a dataset by name and optional version.

        Args:
            dataset_name: Name of the dataset
            version: Specific version (if None, get latest)
            status: Filter by status

        Returns:
            Dataset metadata

        Raises:
            DatasetNotFoundError: If dataset not found
        """
        if dataset_name not in self._cache:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_name}")

        versions = self._cache[dataset_name]

        if version:
            if version not in versions:
                raise VersionNotFoundError(
                    f"Version {version} not found for dataset {dataset_name}"
                )
            return versions[version]

        # Get latest version
        if not versions:
            raise DatasetNotFoundError(f"No versions found for dataset {dataset_name}")

        # Filter by status if specified
        if status:
            filtered = {
                v: meta for v, meta in versions.items()
                if meta.status == status
            }
            if not filtered:
                raise DatasetNotFoundError(
                    f"No {status} versions found for dataset {dataset_name}"
                )
            versions = filtered

        # Get latest version (highest semantic version)
        latest_version = max(versions.keys())
        return versions[latest_version]

    def list_versions(
        self,
        dataset_name: str,
        include_status: Optional[Set[VersionStatus]] = None
    ) -> Dict[SemanticVersion, DatasetMetadata]:
        """
        List all versions of a dataset.

        Args:
            dataset_name: Name of the dataset
            include_status: Filter by status

        Returns:
            Dictionary mapping versions to metadata
        """
        if dataset_name not in self._cache:
            return {}

        versions = self._cache[dataset_name]

        if include_status:
            versions = {
                v: meta for v, meta in versions.items()
                if meta.status in include_status
            }

        return dict(sorted(versions.items(), key=lambda x: str(x[0])))

    def update_dataset_status(
        self,
        dataset_name: str,
        version: SemanticVersion,
        new_status: VersionStatus,
        reason: Optional[str] = None
    ) -> DatasetMetadata:
        """
        Update the status of a dataset version.

        Args:
            dataset_name: Name of the dataset
            version: Version to update
            new_status: New status
            reason: Reason for status change

        Returns:
            Updated metadata
        """
        if dataset_name not in self._cache:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_name}")

        if version not in self._cache[dataset_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for dataset {dataset_name}"
            )

        metadata = self._cache[dataset_name][version]
        metadata.status = new_status
        metadata.modified_at = datetime.utcnow()

        # Add to processing history if reason provided
        if reason:
            metadata.processing_history.append(
                ProcessingStep(
                    step_name=f"status_change_{new_status.value}",
                    step_type="status_update",
                    input_version=version,
                    output_version=version,
                    duration_seconds=0,
                    parameters={"reason": reason}
                )
            )

        self._save_registry()
        logger.info(f"Updated dataset {dataset_name} version {version} status to {new_status}")

        return metadata

    def add_processing_step(
        self,
        dataset_name: str,
        version: SemanticVersion,
        step: ProcessingStep
    ) -> DatasetMetadata:
        """
        Add a processing step to a dataset's history.

        Args:
            dataset_name: Name of the dataset
            version: Version to update
            step: Processing step to add

        Returns:
            Updated metadata
        """
        if dataset_name not in self._cache:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_name}")

        if version not in self._cache[dataset_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for dataset {dataset_name}"
            )

        metadata = self._cache[dataset_name][version]
        metadata.processing_history.append(step)
        metadata.modified_at = datetime.utcnow()

        self._save_registry()
        logger.info(f"Added processing step to {dataset_name} version {version}")

        return metadata

    def search_datasets(
        self,
        query: Optional[str] = None,
        status: Optional[VersionStatus] = None,
        tags: Optional[Set[str]] = None,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None
    ) -> List[DatasetMetadata]:
        """
        Search for datasets based on criteria.

        Args:
            query: Text search in name and description
            status: Filter by status
            tags: Filter by tags
            min_rows: Minimum number of rows
            max_rows: Maximum number of rows

        Returns:
            List of matching dataset metadata
        """
        results = []

        for dataset_name, versions in self._cache.items():
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

            if status and metadata.status != status:
                continue

            if tags and not tags.intersection(metadata.tags):
                continue

            if min_rows is not None and metadata.rows < min_rows:
                continue

            if max_rows is not None and metadata.rows > max_rows:
                continue

            results.append(metadata)

        return results

    def get_lineage(self, dataset_name: str, version: SemanticVersion) -> Dict[str, Any]:
        """
        Get lineage information for a specific dataset version.

        Args:
            dataset_name: Name of the dataset
            version: Version to get lineage for

        Returns:
            Lineage information
        """
        if dataset_name not in self._cache:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_name}")

        if version not in self._cache[dataset_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for dataset {dataset_name}"
            )

        metadata = self._cache[dataset_name][version]

        lineage_info = {
            "dataset_name": metadata.name,
            "version": str(metadata.version),
            "source": metadata.source,
            "pipeline_version": str(metadata.pipeline_version) if metadata.pipeline_version else None,
            "processing_steps": [
                {
                    "step_name": step.step_name,
                    "step_type": step.step_type,
                    "input_version": str(step.input_version),
                    "output_version": str(step.output_version),
                    "timestamp": step.timestamp.isoformat() if step.timestamp else None,
                    "parameters": step.parameters
                }
                for step in metadata.processing_history
            ],
            "schema_version": str(metadata.schema_version) if metadata.schema_version else None
        }

        return lineage_info

    def compare_versions(
        self,
        dataset_name: str,
        version_a: SemanticVersion,
        version_b: SemanticVersion
    ) -> Dict[str, Any]:
        """
        Compare two versions of a dataset.

        Args:
            dataset_name: Name of the dataset
            version_a: First version
            version_b: Second version

        Returns:
            Comparison results
        """
        if dataset_name not in self._cache:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_name}")

        if version_a not in self._cache[dataset_name]:
            raise VersionNotFoundError(
                f"Version {version_a} not found for dataset {dataset_name}"
            )

        if version_b not in self._cache[dataset_name]:
            raise VersionNotFoundError(
                f"Version {version_b} not found for dataset {dataset_name}"
            )

        meta_a = self._cache[dataset_name][version_a]
        meta_b = self._cache[dataset_name][version_b]

        comparison = {
            "version_a": str(version_a),
            "version_b": str(version_b),
            "rows": {
                "a": meta_a.rows,
                "b": meta_b.rows,
                "difference": meta_b.rows - meta_a.rows
            },
            "columns": {
                "a": meta_a.columns,
                "b": meta_b.columns,
                "difference": meta_b.columns - meta_a.columns
            },
            "schema_version": {
                "a": str(meta_a.schema_version) if meta_a.schema_version else None,
                "b": str(meta_b.schema_version) if meta_b.schema_version else None
            },
            "checksum_match": (
                meta_a.checksum and meta_b.checksum and
                meta_a.checksum.sha256 == meta_b.checksum.sha256
            ),
            "metadata_diffs": {
                k: {"a": getattr(meta_a, k), "b": getattr(meta_b, k)}
                for k in meta_a.dict().keys()
                if getattr(meta_a, k) != getattr(meta_b, k)
                and k not in ["checksum", "created_at", "modified_at"]
            }
        }

        return comparison