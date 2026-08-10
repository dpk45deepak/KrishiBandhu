# app/data/feature_engineering/feature_registry.py
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import pandas as pd
from loguru import logger
from pydantic import ValidationError

from app.data.feature_engineering.models import (
    FeatureMetadata, FeatureSet, FeatureDefinition
)
from app.data.feature_engineering.exceptions import (
    FeatureRegistryError, FeatureNotFoundError, FeatureVersionError
)


class FeatureRegistry:
    """
    Enterprise feature registry for managing feature metadata.
    
    Implements versioning, dependency tracking, and feature discovery.
    """
    
    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, FeatureMetadata]] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load existing registry from disk."""
        try:
            registry_file = self.registry_path / "registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    for feature_name, versions in data.items():
                        self._cache[feature_name] = {}
                        for version, metadata in versions.items():
                            try:
                                self._cache[feature_name][version] = FeatureMetadata(
                                    **metadata
                                )
                            except ValidationError as e:
                                logger.warning(f"Invalid metadata for {feature_name}: {e}")
            logger.info(f"Loaded registry with {len(self._cache)} features")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._cache = {}
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            registry_file = self.registry_path / "registry.json"
            data = {}
            for feature_name, versions in self._cache.items():
                data[feature_name] = {}
                for version, metadata in versions.items():
                    data[feature_name][version] = metadata.model_dump()
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Registry saved successfully")
        except Exception as e:
            raise FeatureRegistryError(f"Failed to save registry: {e}")
    
    def register_feature(self, metadata: FeatureMetadata) -> str:
        """
        Register a new feature version in the registry.
        
        Args:
            metadata: Feature metadata to register
            
        Returns:
            Feature version string
        """
        try:
            if metadata.feature_name not in self._cache:
                self._cache[metadata.feature_name] = {}
            
            # Generate checksum if not provided
            if metadata.checksum is None:
                metadata.checksum = self._generate_checksum(metadata)
            
            # Check if version already exists
            if metadata.version in self._cache[metadata.feature_name]:
                raise FeatureVersionError(
                    f"Version {metadata.version} already exists for feature {metadata.feature_name}"
                )
            
            self._cache[metadata.feature_name][metadata.version] = metadata
            self._save_registry()
            logger.info(f"Registered feature {metadata.feature_name} version {metadata.version}")
            return metadata.version
            
        except Exception as e:
            raise FeatureRegistryError(f"Failed to register feature: {e}")
    
    def get_feature(self, feature_name: str, version: Optional[str] = None) -> FeatureMetadata:
        """
        Get feature metadata by name and optional version.
        
        Args:
            feature_name: Name of the feature
            version: Specific version (latest if None)
            
        Returns:
            Feature metadata
        """
        if feature_name not in self._cache:
            raise FeatureNotFoundError(f"Feature {feature_name} not found in registry")
        
        if version is None:
            # Get latest version
            versions = list(self._cache[feature_name].keys())
            if not versions:
                raise FeatureNotFoundError(f"No versions found for feature {feature_name}")
            version = max(versions)
        
        if version not in self._cache[feature_name]:
            raise FeatureNotFoundError(f"Version {version} not found for feature {feature_name}")
        
        return self._cache[feature_name][version]
    
    def get_feature_set(self, feature_names: List[str], version: Optional[str] = None) -> FeatureSet:
        """Get multiple features as a feature set."""
        features = []
        for name in feature_names:
            features.append(self.get_feature(name, version))
        
        return FeatureSet(
            dataset_name="feature_set",
            features=features,
            version=version or "latest",
            schema={},
            statistics={},
            checksum=self._generate_feature_set_checksum(features)
        )
    
    def list_features(self, feature_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all features with optional type filtering."""
        result = []
        for feature_name, versions in self._cache.items():
            latest_version = max(versions.keys())
            metadata = versions[latest_version]
            
            if feature_type and metadata.feature_type != feature_type:
                continue
            
            result.append({
                "name": feature_name,
                "type": metadata.feature_type,
                "latest_version": latest_version,
                "created_date": metadata.created_date,
                "owner": metadata.owner,
                "tags": metadata.tags
            })
        
        return sorted(result, key=lambda x: x["name"])
    
    def get_dependencies(self, feature_name: str) -> Set[str]:
        """Get all dependencies for a feature."""
        metadata = self.get_feature(feature_name)
        dependencies = set(metadata.dependencies)
        
        # Recursively get dependencies
        for dep in metadata.dependencies:
            try:
                dependencies.update(self.get_dependencies(dep))
            except FeatureNotFoundError:
                logger.warning(f"Dependency {dep} not found for {feature_name}")
        
        return dependencies
    
    def update_metadata(self, feature_name: str, version: str, updates: Dict[str, Any]) -> None:
        """Update metadata for a specific feature version."""
        metadata = self.get_feature(feature_name, version)
        
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        metadata.last_modified = datetime.now()
        self._save_registry()
        logger.info(f"Updated metadata for {feature_name} version {version}")
    
    def deactivate_feature(self, feature_name: str, version: Optional[str] = None) -> None:
        """Deactivate a feature version or all versions."""
        if version:
            metadata = self.get_feature(feature_name, version)
            metadata.is_active = False
        else:
            for version in self._cache.get(feature_name, {}):
                self._cache[feature_name][version].is_active = False
        
        self._save_registry()
        logger.info(f"Deactivated feature {feature_name}")
    
    def _generate_checksum(self, metadata: FeatureMetadata) -> str:
        """Generate checksum for feature metadata."""
        data = f"{metadata.feature_name}{metadata.version}{metadata.formula}{'-'.join(metadata.source_columns)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _generate_feature_set_checksum(self, features: List[FeatureMetadata]) -> str:
        """Generate checksum for a feature set."""
        data = "".join([f"{f.feature_name}{f.version}" for f in sorted(features, key=lambda x: x.feature_name)])
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def search_features(self, query: str) -> List[Dict[str, Any]]:
        """Search features by name, description, or tags."""
        results = []
        query_lower = query.lower()
        
        for feature_name, versions in self._cache.items():
            latest_version = max(versions.keys())
            metadata = versions[latest_version]
            
            if (query_lower in feature_name.lower() or 
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                results.append({
                    "name": feature_name,
                    "description": metadata.description,
                    "type": metadata.feature_type,
                    "version": latest_version,
                    "tags": metadata.tags
                })
        
        return results