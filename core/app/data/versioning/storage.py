"""
Storage management for versioned entities.
"""

import shutil
from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd
import json
import yaml
from datetime import datetime
from loguru import logger

from .models import SemanticVersion, EntityType
from .exceptions import StorageError, UnsupportedFormatError
from .checksum import ChecksumGenerator


class StorageManager:
    """Manages physical storage of versioned entities."""

    def __init__(
        self,
        base_path: Union[str, Path],
        checksum_generator: Optional[ChecksumGenerator] = None
    ):
        self.base_path = Path(base_path)
        self.checksum_generator = checksum_generator or ChecksumGenerator()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directory structure."""
        directories = [
            self.base_path / 'raw',
            self.base_path / 'interim',
            self.base_path / 'processed',
            self.base_path / 'feature_store',
            self.base_path / 'versions' / 'raw',
            self.base_path / 'versions' / 'interim',
            self.base_path / 'versions' / 'processed',
            self.base_path / 'versions' / 'features',
            self.base_path / 'metadata',
            self.base_path / 'artifacts',
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    def get_version_path(
        self,
        entity_name: str,
        version: SemanticVersion,
        entity_type: EntityType,
        subdir: Optional[str] = None
    ) -> Path:
        """
        Get the storage path for a specific version.

        Args:
            entity_name: Name of the entity
            version: Semantic version
            entity_type: Type of entity
            subdir: Optional subdirectory

        Returns:
            Path to the versioned entity
        """
        # Map entity type to directory
        type_mapping = {
            EntityType.DATASET: 'versions/datasets',
            EntityType.ARTIFACT: 'artifacts',
            EntityType.FEATURE: 'versions/features',
            EntityType.SCHEMA: 'metadata/schemas',
            EntityType.MODEL: 'artifacts/models',
            EntityType.PIPELINE: 'artifacts/pipelines',
        }

        base_dir = self.base_path / type_mapping.get(entity_type, 'versions')
        version_dir = base_dir / entity_name / str(version)

        if subdir:
            version_dir = version_dir / subdir

        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir

    def store_dataset(
        self,
        df: pd.DataFrame,
        entity_name: str,
        version: SemanticVersion,
        format: str = 'parquet',
        compression: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Store a dataset in the versioned storage.

        Args:
            df: DataFrame to store
            entity_name: Name of the dataset
            version: Semantic version
            format: File format (parquet, csv, json)
            compression: Compression algorithm
            metadata: Additional metadata

        Returns:
            Path to the stored file

        Raises:
            UnsupportedFormatError: If format is not supported
        """
        version_path = self.get_version_path(
            entity_name,
            version,
            EntityType.DATASET
        )

        filename = f"{entity_name}_{version}.{format}"
        file_path = version_path / filename

        try:
            if format == 'parquet':
                df.to_parquet(
                    file_path,
                    compression=compression or 'snappy',
                    index=False
                )
            elif format == 'csv':
                df.to_csv(
                    file_path,
                    index=False,
                    compression=compression or None
                )
            elif format == 'json':
                df.to_json(
                    file_path,
                    orient='records',
                    compression=compression or None
                )
            else:
                raise UnsupportedFormatError(f"Unsupported format: {format}")

            # Generate checksum
            checksum = self.checksum_generator.generate_file_checksum(file_path)

            # Store metadata
            if metadata:
                self._store_metadata(
                    entity_name,
                    version,
                    EntityType.DATASET,
                    {**metadata, 'checksum': checksum.dict()}
                )

            logger.info(f"Stored dataset {entity_name} version {version} at {file_path}")
            return file_path

        except Exception as e:
            raise StorageError(f"Failed to store dataset: {e}")

    def load_dataset(
        self,
        entity_name: str,
        version: SemanticVersion,
        format: str = 'parquet'
    ) -> pd.DataFrame:
        """
        Load a dataset from versioned storage.

        Args:
            entity_name: Name of the dataset
            version: Semantic version
            format: File format

        Returns:
            Loaded DataFrame

        Raises:
            StorageError: If loading fails
        """
        version_path = self.get_version_path(
            entity_name,
            version,
            EntityType.DATASET
        )

        filename = f"{entity_name}_{version}.{format}"
        file_path = version_path / filename

        if not file_path.exists():
            raise StorageError(f"Dataset not found: {file_path}")

        try:
            if format == 'parquet':
                df = pd.read_parquet(file_path)
            elif format == 'csv':
                df = pd.read_csv(file_path)
            elif format == 'json':
                df = pd.read_json(file_path)
            else:
                raise UnsupportedFormatError(f"Unsupported format: {format}")

            logger.info(f"Loaded dataset {entity_name} version {version} from {file_path}")
            return df

        except Exception as e:
            raise StorageError(f"Failed to load dataset: {e}")

    def store_artifact(
        self,
        data: Union[bytes, Dict[str, Any]],
        entity_name: str,
        version: SemanticVersion,
        artifact_type: str,
        file_extension: str = 'json',
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Store an artifact in the versioned storage.

        Args:
            data: Data to store (bytes or dict)
            entity_name: Name of the artifact
            version: Semantic version
            artifact_type: Type of artifact
            file_extension: File extension for the artifact
            metadata: Additional metadata

        Returns:
            Path to the stored file
        """
        version_path = self.get_version_path(
            entity_name,
            version,
            EntityType.ARTIFACT,
            subdir=artifact_type
        )

        filename = f"{entity_name}_{version}.{file_extension}"
        file_path = version_path / filename

        try:
            if isinstance(data, dict):
                if file_extension == 'json':
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                elif file_extension == 'yaml' or file_extension == 'yml':
                    with open(file_path, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False)
                else:
                    raise UnsupportedFormatError(f"Unsupported format: {file_extension}")
            elif isinstance(data, bytes):
                with open(file_path, 'wb') as f:
                    f.write(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

            # Generate checksum
            checksum = self.checksum_generator.generate_file_checksum(file_path)

            # Store metadata
            if metadata:
                self._store_metadata(
                    entity_name,
                    version,
                    EntityType.ARTIFACT,
                    {
                        **metadata,
                        'artifact_type': artifact_type,
                        'checksum': checksum.dict()
                    }
                )

            logger.info(f"Stored artifact {entity_name} version {version} at {file_path}")
            return file_path

        except Exception as e:
            raise StorageError(f"Failed to store artifact: {e}")

    def load_artifact(
        self,
        entity_name: str,
        version: SemanticVersion,
        artifact_type: str,
        file_extension: str = 'json'
    ) -> Union[Dict[str, Any], bytes]:
        """
        Load an artifact from versioned storage.

        Args:
            entity_name: Name of the artifact
            version: Semantic version
            artifact_type: Type of artifact
            file_extension: File extension

        Returns:
            Loaded artifact data
        """
        version_path = self.get_version_path(
            entity_name,
            version,
            EntityType.ARTIFACT,
            subdir=artifact_type
        )

        filename = f"{entity_name}_{version}.{file_extension}"
        file_path = version_path / filename

        if not file_path.exists():
            raise StorageError(f"Artifact not found: {file_path}")

        try:
            if file_extension == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif file_extension in ['yaml', 'yml']:
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                with open(file_path, 'rb') as f:
                    return f.read()

        except Exception as e:
            raise StorageError(f"Failed to load artifact: {e}")

    def _store_metadata(
        self,
        entity_name: str,
        version: SemanticVersion,
        entity_type: EntityType,
        metadata: Dict[str, Any]
    ) -> Path:
        """Store metadata for a versioned entity."""
        metadata_path = self.base_path / 'metadata' / entity_type.value

        metadata_path.mkdir(parents=True, exist_ok=True)

        filename = f"{entity_name}_{version}_metadata.json"
        file_path = metadata_path / filename

        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.debug(f"Stored metadata for {entity_name} version {version}")
        return file_path

    def load_metadata(
        self,
        entity_name: str,
        version: SemanticVersion,
        entity_type: EntityType
    ) -> Dict[str, Any]:
        """Load metadata for a versioned entity."""
        metadata_path = self.base_path / 'metadata' / entity_type.value
        filename = f"{entity_name}_{version}_metadata.json"
        file_path = metadata_path / filename

        if not file_path.exists():
            return {}

        with open(file_path, 'r') as f:
            return json.load(f)

    def delete_version(
        self,
        entity_name: str,
        version: SemanticVersion,
        entity_type: EntityType,
        force: bool = False
    ) -> None:
        """
        Delete a version from storage.

        Args:
            entity_name: Name of the entity
            version: Semantic version
            entity_type: Type of entity
            force: Force deletion even if not archived
        """
        version_path = self.get_version_path(
            entity_name,
            version,
            entity_type
        )

        if not force:
            # Move to trash instead of deleting
            trash_path = self.base_path / '.trash' / entity_type.value
            trash_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            target_path = trash_path / f"{entity_name}_{version}_{timestamp}"

            shutil.move(str(version_path), str(target_path))
            logger.info(f"Moved version {version} to trash: {target_path}")
        else:
            shutil.rmtree(version_path, ignore_errors=True)
            logger.info(f"Force deleted version {version}: {version_path}")

        # Delete metadata
        metadata_path = self.base_path / 'metadata' / entity_type.value
        metadata_file = metadata_path / f"{entity_name}_{version}_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()

    def list_versions(
        self,
        entity_name: str,
        entity_type: EntityType
    ) -> Dict[SemanticVersion, Dict[str, Any]]:
        """
        List all versions of an entity.

        Returns:
            Dictionary mapping versions to their metadata
        """
        base_dir = self.base_path / entity_type.value / entity_name

        if not base_dir.exists():
            return {}

        versions = {}
        for version_dir in base_dir.iterdir():
            if version_dir.is_dir():
                try:
                    version = SemanticVersion.parse(version_dir.name)
                    metadata = self.load_metadata(entity_name, version, entity_type)
                    versions[version] = metadata
                except ValueError:
                    # Skip invalid version directories
                    continue

        return dict(sorted(versions.items(), key=lambda x: str(x[0])))

    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        total_size = 0
        file_count = 0

        for file_path in self.base_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'total_size_gb': total_size / (1024 * 1024 * 1024),
            'file_count': file_count,
            'base_path': str(self.base_path)
        }