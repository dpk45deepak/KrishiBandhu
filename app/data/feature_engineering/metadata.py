# app/data/feature_engineering/metadata.py
import json
import yaml
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import asdict
import pandas as pd
import numpy as np
from loguru import logger
from pydantic import ValidationError

from app.data.feature_engineering.models import (
    FeatureMetadata, FeatureDefinition, FeatureSet, FeatureType
)
from app.data.feature_engineering.exceptions import (
    FeatureRegistryError, FeatureValidationError
)


class FeatureMetadataManager:
    """
    Enterprise feature metadata manager.
    
    Handles creation, validation, serialization, and versioning of feature metadata.
    Implements comprehensive metadata tracking for reproducibility.
    """
    
    def __init__(self, metadata_path: Path):
        """
        Initialize metadata manager.
        
        Args:
            metadata_path: Base path for metadata storage
        """
        self.metadata_path = Path(metadata_path)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different metadata types
        self.feature_dir = self.metadata_path / 'features'
        self.version_dir = self.metadata_path / 'versions'
        self.snapshot_dir = self.metadata_path / 'snapshots'
        self.validation_dir = self.metadata_path / 'validation'
        
        for dir_path in [self.feature_dir, self.version_dir, 
                        self.snapshot_dir, self.validation_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self._metadata_cache: Dict[str, Dict[str, FeatureMetadata]] = {}
        self._version_history: Dict[str, List[str]] = {}
        self._load_metadata_cache()
    
    def _load_metadata_cache(self) -> None:
        """Load all metadata into cache for fast access."""
        try:
            # Load feature metadata
            feature_files = list(self.feature_dir.glob('*.json'))
            for file_path in feature_files:
                feature_name = file_path.stem
                self._metadata_cache[feature_name] = {}
                
                # Load all versions for this feature
                version_files = list(self.version_dir.glob(f'{feature_name}_*.json'))
                for vf in version_files:
                    try:
                        with open(vf, 'r') as f:
                            data = json.load(f)
                            metadata = FeatureMetadata(**data)
                            version = metadata.version
                            self._metadata_cache[feature_name][version] = metadata
                            
                            # Track version history
                            if feature_name not in self._version_history:
                                self._version_history[feature_name] = []
                            self._version_history[feature_name].append(version)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata from {vf}: {e}")
            
            # Sort version histories
            for feature_name in self._version_history:
                self._version_history[feature_name].sort()
            
            logger.info(f"Loaded metadata cache with {len(self._metadata_cache)} features")
            
        except Exception as e:
            logger.error(f"Failed to load metadata cache: {e}")
            self._metadata_cache = {}
    
    def create_metadata(
        self,
        feature_name: str,
        description: str,
        formula: str,
        data_type: str,
        feature_type: FeatureType,
        owner: str,
        version: str,
        source_columns: List[str],
        dependencies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        validation_rules: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> FeatureMetadata:
        """
        Create new feature metadata.
        
        Args:
            feature_name: Name of the feature
            description: Feature description
            formula: Formula or transformation used
            data_type: Data type of the feature
            feature_type: Type of feature
            owner: Feature owner
            version: Version string
            source_columns: Source columns used
            dependencies: Feature dependencies
            tags: Feature tags
            validation_rules: Validation rules
            parameters: Additional parameters
            
        Returns:
            FeatureMetadata instance
        """
        metadata = FeatureMetadata(
            feature_name=feature_name,
            description=description,
            formula=formula,
            data_type=data_type,
            feature_type=feature_type,
            owner=owner,
            version=version,
            source_columns=source_columns,
            dependencies=dependencies or [],
            tags=tags or [],
            validation_rules=validation_rules or {},
            created_date=datetime.now(),
            last_modified=datetime.now(),
            is_active=True
        )
        
        # Generate checksum
        metadata.checksum = self._generate_checksum(metadata)
        
        # Add transformation history
        metadata.transformation_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'created',
            'version': version,
            'parameters': parameters or {}
        })
        
        return metadata
    
    def _generate_checksum(self, metadata: FeatureMetadata) -> str:
        """
        Generate unique checksum for metadata.
        
        Args:
            metadata: Feature metadata
            
        Returns:
            Checksum string
        """
        # Create deterministic string from metadata
        data = (
            f"{metadata.feature_name}"
            f"{metadata.version}"
            f"{metadata.formula}"
            f"{'-'.join(sorted(metadata.source_columns))}"
            f"{'-'.join(sorted(metadata.dependencies))}"
            f"{metadata.feature_type}"
            f"{metadata.data_type}"
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def save_metadata(self, metadata: FeatureMetadata) -> None:
        """
        Save feature metadata to disk.
        
        Args:
            metadata: Feature metadata to save
        """
        try:
            # Validate metadata
            self.validate_metadata(metadata)
            
            # Save to feature directory
            feature_file = self.feature_dir / f"{metadata.feature_name}.json"
            feature_data = metadata.model_dump()
            
            # Update feature file with latest metadata
            with open(feature_file, 'w') as f:
                json.dump(feature_data, f, indent=2, default=str)
            
            # Save version-specific metadata
            version_file = self.version_dir / f"{metadata.feature_name}_{metadata.version}.json"
            with open(version_file, 'w') as f:
                json.dump(feature_data, f, indent=2, default=str)
            
            # Update cache
            if metadata.feature_name not in self._metadata_cache:
                self._metadata_cache[metadata.feature_name] = {}
            self._metadata_cache[metadata.feature_name][metadata.version] = metadata
            
            # Update version history
            if metadata.feature_name not in self._version_history:
                self._version_history[metadata.feature_name] = []
            if metadata.version not in self._version_history[metadata.feature_name]:
                self._version_history[metadata.feature_name].append(metadata.version)
                self._version_history[metadata.feature_name].sort()
            
            logger.info(f"Saved metadata for {metadata.feature_name} version {metadata.version}")
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to save metadata: {e}")
    
    def load_metadata(
        self,
        feature_name: str,
        version: Optional[str] = None
    ) -> FeatureMetadata:
        """
        Load feature metadata from disk.
        
        Args:
            feature_name: Name of the feature
            version: Specific version (latest if None)
            
        Returns:
            FeatureMetadata instance
        """
        try:
            # Check cache first
            if feature_name in self._metadata_cache:
                versions = self._metadata_cache[feature_name]
                if version is None:
                    # Get latest version
                    if versions:
                        version = max(versions.keys())
                    else:
                        raise FeatureRegistryError(f"No versions found for {feature_name}")
                
                if version in versions:
                    return versions[version]
            
            # Load from disk
            if version is None:
                # Find latest version from files
                version_files = list(self.version_dir.glob(f'{feature_name}_*.json'))
                if not version_files:
                    raise FeatureRegistryError(f"No metadata found for {feature_name}")
                
                # Get latest version
                versions = [f.stem.replace(f'{feature_name}_', '') for f in version_files]
                version = max(versions)
            
            version_file = self.version_dir / f"{feature_name}_{version}.json"
            if not version_file.exists():
                raise FeatureRegistryError(f"Version {version} not found for {feature_name}")
            
            with open(version_file, 'r') as f:
                data = json.load(f)
                metadata = FeatureMetadata(**data)
            
            # Update cache
            if feature_name not in self._metadata_cache:
                self._metadata_cache[feature_name] = {}
            self._metadata_cache[feature_name][version] = metadata
            
            return metadata
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to load metadata: {e}")
    
    def validate_metadata(self, metadata: FeatureMetadata) -> bool:
        """
        Validate feature metadata.
        
        Args:
            metadata: Feature metadata to validate
            
        Returns:
            True if valid
            
        Raises:
            FeatureValidationError: If validation fails
        """
        try:
            # Basic validation
            if not metadata.feature_name:
                raise FeatureValidationError("Feature name is required")
            
            if not metadata.version:
                raise FeatureValidationError("Version is required")
            
            if not metadata.source_columns:
                raise FeatureValidationError("Source columns are required")
            
            if not metadata.formula:
                raise FeatureValidationError("Formula is required")
            
            # Validate version format (semantic versioning)
            version_parts = metadata.version.split('.')
            if len(version_parts) != 3:
                raise FeatureValidationError(
                    f"Version '{metadata.version}' must follow semantic versioning (e.g., 1.0.0)"
                )
            
            for part in version_parts:
                if not part.isdigit():
                    raise FeatureValidationError(
                        f"Version part '{part}' must be numeric"
                    )
            
            # Validate feature type
            if metadata.feature_type not in FeatureType:
                raise FeatureValidationError(
                    f"Invalid feature type: {metadata.feature_type}"
                )
            
            # Validate data type
            valid_dtypes = ['int64', 'float64', 'object', 'category', 'datetime64', 'bool']
            if metadata.data_type not in valid_dtypes:
                logger.warning(f"Unusual data type: {metadata.data_type}")
            
            # Validate dependencies exist
            if metadata.dependencies:
                for dep in metadata.dependencies:
                    if dep == metadata.feature_name:
                        raise FeatureValidationError(
                            f"Feature {metadata.feature_name} cannot depend on itself"
                        )
            
            return True
            
        except FeatureValidationError:
            raise
        except Exception as e:
            raise FeatureValidationError(f"Metadata validation failed: {e}")
    
    def update_metadata(
        self,
        feature_name: str,
        version: str,
        updates: Dict[str, Any]
    ) -> FeatureMetadata:
        """
        Update existing metadata.
        
        Args:
            feature_name: Name of the feature
            version: Version to update
            updates: Dictionary of updates
            
        Returns:
            Updated FeatureMetadata
        """
        try:
            # Load existing metadata
            metadata = self.load_metadata(feature_name, version)
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            
            # Update modification timestamp
            metadata.last_modified = datetime.now()
            
            # Add to transformation history
            metadata.transformation_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'updated',
                'updates': updates
            })
            
            # Regenerate checksum
            metadata.checksum = self._generate_checksum(metadata)
            
            # Save updated metadata
            self.save_metadata(metadata)
            
            logger.info(f"Updated metadata for {feature_name} version {version}")
            return metadata
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to update metadata: {e}")
    
    def get_metadata_history(
        self,
        feature_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get version history for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            List of version history entries
        """
        try:
            history = []
            
            if feature_name in self._version_history:
                for version in self._version_history[feature_name]:
                    metadata = self.load_metadata(feature_name, version)
                    history.append({
                        'version': version,
                        'created_date': metadata.created_date,
                        'last_modified': metadata.last_modified,
                        'owner': metadata.owner,
                        'is_active': metadata.is_active,
                        'checksum': metadata.checksum,
                        'transformation_history': metadata.transformation_history
                    })
            
            return sorted(history, key=lambda x: x['created_date'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get metadata history: {e}")
            return []
    
    def get_metadata_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about metadata.
        
        Returns:
            Dictionary with metadata statistics
        """
        stats = {
            'total_features': len(self._metadata_cache),
            'feature_types': {},
            'total_versions': 0,
            'active_features': 0,
            'inactive_features': 0,
            'most_recent_update': None,
            'owners': set(),
            'tags': set()
        }
        
        for feature_name, versions in self._metadata_cache.items():
            for version, metadata in versions.items():
                stats['total_versions'] += 1
                
                # Count feature types
                feature_type = metadata.feature_type.value
                stats['feature_types'][feature_type] = stats['feature_types'].get(feature_type, 0) + 1
                
                # Count active/inactive
                if metadata.is_active:
                    stats['active_features'] += 1
                else:
                    stats['inactive_features'] += 1
                
                # Track owners
                stats['owners'].add(metadata.owner)
                
                # Track tags
                stats['tags'].update(metadata.tags)
                
                # Track most recent update
                if (stats['most_recent_update'] is None or 
                    metadata.last_modified > stats['most_recent_update']):
                    stats['most_recent_update'] = metadata.last_modified
        
        # Convert sets to lists for JSON serialization
        stats['owners'] = list(stats['owners'])
        stats['tags'] = list(stats['tags'])
        
        return stats
    
    def generate_dependency_graph(
        self,
        feature_name: str,
        depth: int = 3
    ) -> Dict[str, Any]:
        """
        Generate dependency graph for a feature.
        
        Args:
            feature_name: Name of the feature
            depth: Maximum depth to traverse
            
        Returns:
            Dependency graph as nested dictionary
        """
        def _build_graph(name: str, current_depth: int) -> Dict[str, Any]:
            if current_depth > depth:
                return {'_truncated': True}
            
            try:
                metadata = self.load_metadata(name)
                graph = {
                    'feature': name,
                    'version': metadata.version,
                    'type': metadata.feature_type,
                    'dependencies': {}
                }
                
                for dep in metadata.dependencies:
                    graph['dependencies'][dep] = _build_graph(dep, current_depth + 1)
                
                return graph
                
            except Exception as e:
                logger.warning(f"Failed to build graph for {name}: {e}")
                return {'_error': str(e)}
        
        return _build_graph(feature_name, 0)
    
    def export_metadata(
        self,
        format: str = 'json',
        feature_names: Optional[List[str]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Export metadata to various formats.
        
        Args:
            format: Export format ('json', 'yaml', 'pandas')
            feature_names: List of feature names to export (all if None)
            
        Returns:
            Exported metadata
        """
        try:
            features_to_export = feature_names or list(self._metadata_cache.keys())
            
            export_data = {}
            for feature_name in features_to_export:
                if feature_name in self._metadata_cache:
                    # Get latest version
                    versions = self._metadata_cache[feature_name]
                    latest_version = max(versions.keys())
                    metadata = versions[latest_version]
                    export_data[feature_name] = metadata.model_dump()
            
            if format == 'json':
                return json.dumps(export_data, indent=2, default=str)
            
            elif format == 'yaml':
                return yaml.dump(export_data, default_flow_style=False)
            
            elif format == 'pandas':
                # Convert to DataFrame
                rows = []
                for feature_name, data in export_data.items():
                    metadata = FeatureMetadata(**data)
                    row = {
                        'feature_name': metadata.feature_name,
                        'version': metadata.version,
                        'type': metadata.feature_type,
                        'owner': metadata.owner,
                        'created_date': metadata.created_date,
                        'description': metadata.description,
                        'formula': metadata.formula,
                        'source_columns': ', '.join(metadata.source_columns),
                        'dependencies': ', '.join(metadata.dependencies),
                        'tags': ', '.join(metadata.tags),
                        'is_active': metadata.is_active
                    }
                    rows.append(row)
                
                return pd.DataFrame(rows)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            raise FeatureRegistryError(f"Failed to export metadata: {e}")
    
    def import_metadata(
        self,
        data: Union[str, Dict],
        format: str = 'json'
    ) -> int:
        """
        Import metadata from various formats.
        
        Args:
            data: Metadata data to import
            format: Import format ('json', 'yaml')
            
        Returns:
            Number of features imported
        """
        try:
            if format == 'json':
                if isinstance(data, str):
                    import_data = json.loads(data)
                else:
                    import_data = data
            
            elif format == 'yaml':
                if isinstance(data, str):
                    import_data = yaml.safe_load(data)
                else:
                    import_data = data
            
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            count = 0
            for feature_name, feature_data in import_data.items():
                try:
                    metadata = FeatureMetadata(**feature_data)
                    self.save_metadata(metadata)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to import {feature_name}: {e}")
            
            logger.info(f"Imported {count} features from {format}")
            return count
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to import metadata: {e}")
    
    def create_metadata_snapshot(
        self,
        snapshot_name: str,
        feature_names: Optional[List[str]] = None
    ) -> Path:
        """
        Create a snapshot of current metadata.
        
        Args:
            snapshot_name: Name for the snapshot
            feature_names: List of features to include (all if None)
            
        Returns:
            Path to snapshot file
        """
        try:
            snapshot_data = {
                'snapshot_name': snapshot_name,
                'timestamp': datetime.now().isoformat(),
                'total_features': len(self._metadata_cache),
                'features': {}
            }
            
            features_to_include = feature_names or list(self._metadata_cache.keys())
            
            for feature_name in features_to_include:
                if feature_name in self._metadata_cache:
                    versions = self._metadata_cache[feature_name]
                    snapshot_data['features'][feature_name] = {}
                    for version, metadata in versions.items():
                        snapshot_data['features'][feature_name][version] = metadata.model_dump()
            
            # Save snapshot
            snapshot_file = self.snapshot_dir / f"{snapshot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2, default=str)
            
            logger.info(f"Created metadata snapshot: {snapshot_file}")
            return snapshot_file
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to create metadata snapshot: {e}")
    
    def restore_metadata_snapshot(
        self,
        snapshot_path: Path,
        overwrite: bool = False
    ) -> int:
        """
        Restore metadata from a snapshot.
        
        Args:
            snapshot_path: Path to snapshot file
            overwrite: Whether to overwrite existing metadata
            
        Returns:
            Number of features restored
        """
        try:
            with open(snapshot_path, 'r') as f:
                snapshot_data = json.load(f)
            
            count = 0
            for feature_name, versions in snapshot_data['features'].items():
                for version, metadata_data in versions.items():
                    try:
                        metadata = FeatureMetadata(**metadata_data)
                        
                        # Check if metadata exists
                        existing = None
                        try:
                            existing = self.load_metadata(feature_name, version)
                        except:
                            pass
                        
                        if existing and not overwrite:
                            logger.warning(f"Skipping {feature_name} version {version} (already exists)")
                            continue
                        
                        self.save_metadata(metadata)
                        count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to restore {feature_name} version {version}: {e}")
            
            logger.info(f"Restored {count} features from snapshot")
            return count
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to restore metadata snapshot: {e}")
    
    def compare_metadata_versions(
        self,
        feature_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a feature.
        
        Args:
            feature_name: Name of the feature
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        try:
            meta1 = self.load_metadata(feature_name, version1)
            meta2 = self.load_metadata(feature_name, version2)
            
            comparison = {
                'feature_name': feature_name,
                'version1': version1,
                'version2': version2,
                'differences': {},
                'similarities': {}
            }
            
            # Compare fields
            fields = [
                'description', 'formula', 'data_type', 'feature_type',
                'owner', 'source_columns', 'dependencies', 'tags',
                'validation_rules', 'is_active'
            ]
            
            for field in fields:
                val1 = getattr(meta1, field)
                val2 = getattr(meta2, field)
                
                if val1 == val2:
                    comparison['similarities'][field] = val1
                else:
                    comparison['differences'][field] = {
                        'version1': val1,
                        'version2': val2
                    }
            
            # Compare checksums
            comparison['checksum_match'] = meta1.checksum == meta2.checksum
            
            # Compare transformation histories
            comparison['transformations1'] = len(meta1.transformation_history)
            comparison['transformations2'] = len(meta2.transformation_history)
            
            return comparison
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to compare metadata versions: {e}")
    
    def cleanup_old_versions(
        self,
        feature_name: str,
        keep_versions: int = 5
    ) -> List[str]:
        """
        Clean up old versions of a feature.
        
        Args:
            feature_name: Name of the feature
            keep_versions: Number of recent versions to keep
            
        Returns:
            List of removed versions
        """
        try:
            if feature_name not in self._version_history:
                return []
            
            versions = sorted(self._version_history[feature_name])
            if len(versions) <= keep_versions:
                return []
            
            # Keep most recent versions
            to_remove = versions[:-keep_versions]
            removed = []
            
            for version in to_remove:
                # Remove version file
                version_file = self.version_dir / f"{feature_name}_{version}.json"
                if version_file.exists():
                    version_file.unlink()
                    removed.append(version)
                
                # Remove from cache
                if (feature_name in self._metadata_cache and 
                    version in self._metadata_cache[feature_name]):
                    del self._metadata_cache[feature_name][version]
                
                # Remove from version history
                if version in self._version_history[feature_name]:
                    self._version_history[feature_name].remove(version)
            
            logger.info(f"Removed {len(removed)} old versions for {feature_name}")
            return removed
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to cleanup old versions: {e}")
    
    def get_feature_lineage(
        self,
        feature_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get complete feature lineage including sources and transformations.
        
        Args:
            feature_name: Name of the feature
            version: Specific version (latest if None)
            
        Returns:
            Feature lineage information
        """
        try:
            metadata = self.load_metadata(feature_name, version)
            
            lineage = {
                'feature_name': metadata.feature_name,
                'version': metadata.version,
                'sources': {
                    'columns': metadata.source_columns,
                    'dependencies': metadata.dependencies
                },
                'transformations': metadata.transformation_history,
                'formula': metadata.formula,
                'created_date': metadata.created_date,
                'last_modified': metadata.last_modified,
                'type': metadata.feature_type,
                'validation': metadata.validation_rules
            }
            
            # Recursively get lineage for dependencies
            if metadata.dependencies:
                lineage['dependency_lineage'] = {}
                for dep in metadata.dependencies:
                    try:
                        dep_metadata = self.load_metadata(dep)
                        lineage['dependency_lineage'][dep] = {
                            'version': dep_metadata.version,
                            'type': dep_metadata.feature_type,
                            'created_date': dep_metadata.created_date
                        }
                    except Exception as e:
                        logger.warning(f"Failed to get lineage for dependency {dep}: {e}")
            
            return lineage
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to get feature lineage: {e}")
    
    def get_feature_impact_analysis(
        self,
        feature_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze impact of a feature on other features.
        
        Args:
            feature_name: Name of the feature
            version: Specific version (latest if None)
            
        Returns:
            Impact analysis results
        """
        try:
            metadata = self.load_metadata(feature_name, version)
            
            impact = {
                'feature': feature_name,
                'version': metadata.version,
                'direct_dependents': [],
                'indirect_dependents': [],
                'total_impact': 0
            }
            
            # Find features that depend on this feature
            for feature, versions in self._metadata_cache.items():
                for ver, meta in versions.items():
                    if feature != feature_name and meta.is_active:
                        if feature_name in meta.dependencies:
                            impact['direct_dependents'].append({
                                'feature': feature,
                                'version': ver,
                                'type': meta.feature_type
                            })
                        else:
                            # Check indirect dependencies
                            for dep in meta.dependencies:
                                try:
                                    dep_meta = self.load_metadata(dep)
                                    if feature_name in dep_meta.dependencies:
                                        impact['indirect_dependents'].append({
                                            'feature': feature,
                                            'version': ver,
                                            'path': f"{feature} -> {dep} -> {feature_name}"
                                        })
                                except:
                                    pass
            
            impact['direct_dependents'] = sorted(
                impact['direct_dependents'], 
                key=lambda x: x['feature']
            )
            impact['indirect_dependents'] = sorted(
                impact['indirect_dependents'], 
                key=lambda x: x['feature']
            )
            impact['total_impact'] = (
                len(impact['direct_dependents']) + 
                len(impact['indirect_dependents'])
            )
            
            return impact
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to analyze feature impact: {e}")