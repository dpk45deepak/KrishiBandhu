# app/services/feature_store/service.py
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pandas as pd

from app.config import settings
from app.logger import get_logger
from app.services.feature_store.models import (
    FeatureDefinition,
    FeatureGroup,
    FeatureSource,
    FeatureStats,
    FeatureType,
    FeatureVector,
    FeatureLineage,
    FeatureStoreConfig,
)
from app.utils.decorators import timed

logger = get_logger(__name__)


class FeatureStoreService:
    """Feature Store service for managing ML features.
    
    Consumes existing modules:
    - data.FeatureEngineering: feature transformation pipelines
    - data.Standardizer: feature standardization
    - data.Profiler: feature statistics
    - config: storage paths
    - logger: structured logging
    
    Provides:
    - Feature registration and discovery
    - Online/offline feature serving
    - Feature statistics and drift monitoring
    - Feature lineage tracking
    """
    
    def __init__(self):
        self._feature_groups: Dict[str, FeatureGroup] = {}
        self._online_store: Dict[str, Dict[str, FeatureVector]] = {}  # group_name -> entity_id -> vector
        self._config = FeatureStoreConfig()
    
    @timed
    async def create_feature_group(
        self,
        name: str,
        description: str,
        features: List[FeatureDefinition],
        dataset_id: Optional[str] = None,
        entity_key: str = "id",
        user_id: Optional[str] = None,
    ) -> FeatureGroup:
        """Register a new feature group."""
        
        # Validate feature names are unique
        feature_names = [f.name for f in features]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Duplicate feature names in group")
        
        group = FeatureGroup(
            id=uuid4(),
            name=name,
            description=description,
            features=features,
            dataset_id=dataset_id,
            entity_key=entity_key,
            created_by=user_id,
        )
        
        self._feature_groups[name] = group
        
        logger.info(
            f"Feature group created: {name} with {len(features)} features "
            f"(entity_key={entity_key})"
        )
        return group
    
    @timed
    async def ingest_features(
        self,
        group_name: str,
        dataframe: pd.DataFrame,
        entity_column: str,
        timestamp_column: Optional[str] = None,
        pipeline_run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest feature values from a DataFrame into the feature store.
        
        Uses existing data.Standardizer and data.Profiler modules.
        """
        from app.data import Standardizer, Profiler
        
        group = self._get_group(group_name)
        
        # Standardize features using existing module
        standardizer = Standardizer()
        df = await standardizer.standardize_dataframe(
            dataframe,
            columns=[f.name for f in group.features if f.name in dataframe.columns],
        )
        
        # Compute statistics using existing Profiler
        profiler = Profiler()
        profile_result = await profiler.profile_dataframe(df)
        
        # Build feature statistics
        feature_stats = {}
        for col in profile_result.columns:
            feature_stats[col.name] = {
                "count": col.count,
                "mean": col.mean,
                "std": col.std,
                "min": col.min,
                "max": col.max,
                "null_count": col.null_count,
                "null_percentage": col.null_percentage,
            }
        
        # Update group statistics
        group.statistics = FeatureStats(
            row_count=len(df),
            feature_stats=feature_stats,
            missing_counts={
                col.name: col.null_count
                for col in profile_result.columns
            },
            correlation_matrix=profile_result.correlations,
        )
        
        # Update lineage
        group.lineage = FeatureLineage(
            source_dataset_id=group.dataset_id,
            pipeline_run_id=pipeline_run_id,
            transformations=[
                {"type": "ingest", "rows": len(df), "columns": list(df.columns)}
            ],
        )
        
        # Load into online store if enabled
        if self._config.online_store_enabled:
            online_vectors = {}
            for _, row in df.iterrows():
                entity_id = str(row[entity_column])
                feature_values = {
                    f.name: row[f.name]
                    for f in group.features
                    if f.name in df.columns
                }
                
                vector = FeatureVector(
                    entity_id=entity_id,
                    feature_values=feature_values,
                    timestamp=row.get(timestamp_column) if timestamp_column else None,
                )
                online_vectors[entity_id] = vector
            
            self._online_store[group_name] = online_vectors
            logger.info(f"Online store updated: {group_name} ({len(online_vectors)} entities)")
        
        group.updated_at = datetime.now(timezone.utc)
        
        return {
            "group_name": group_name,
            "rows_ingested": len(df),
            "features_stored": len(feature_stats),
            "online_entities": len(self._online_store.get(group_name, {})),
            "statistics": {k: {"mean": v.get("mean"), "std": v.get("std")} 
                          for k, v in feature_stats.items()},
        }
    
    @timed
    async def get_online_features(
        self, group_name: str, entity_ids: List[str]
    ) -> Dict[str, FeatureVector]:
        """Retrieve features from online store for given entity IDs."""
        group = self._get_group(group_name)
        store = self._online_store.get(group_name, {})
        
        result = {}
        missing = []
        
        for entity_id in entity_ids:
            vector = store.get(entity_id)
            if vector:
                result[entity_id] = vector
            else:
                missing.append(entity_id)
        
        if missing:
            # Try offline store or compute on demand
            if self._config.compute_on_demand:
                logger.info(f"Computing features on demand for {len(missing)} entities")
                computed = await self._compute_features_on_demand(group_name, missing)
                result.update(computed)
            else:
                logger.warning(f"Missing features for {len(missing)} entities in {group_name}")
        
        return result
    
    @timed
    async def get_feature_history(
        self, group_name: str, entity_id: str, start_time: datetime, end_time: datetime
    ) -> List[FeatureVector]:
        """Get historical feature values for an entity (offline store)."""
        # Offline retrieval from stored feature data
        group = self._get_group(group_name)
        
        # This would query a time-series database or feature store backend
        logger.info(
            f"Retrieving feature history: {group_name}/{entity_id} "
            f"from {start_time.isoformat()} to {end_time.isoformat()}"
        )
        
        # Placeholder - in production this queries the offline store
        return []
    
    async def get_feature_group(self, group_name: str) -> Optional[FeatureGroup]:
        """Get feature group definition."""
        return self._feature_groups.get(group_name)
    
    async def list_feature_groups(
        self, dataset_id: Optional[str] = None
    ) -> List[FeatureGroup]:
        """List all feature groups."""
        groups = list(self._feature_groups.values())
        if dataset_id:
            groups = [g for g in groups if g.dataset_id == dataset_id]
        return groups
    
    @timed
    async def compute_feature_stats(self, group_name: str) -> FeatureStats:
        """Recompute statistics for a feature group."""
        from app.data import Profiler
        
        group = self._get_group(group_name)
        
        # Get feature data from online store
        store = self._online_store.get(group_name, {})
        if not store:
            raise ValueError(f"No data in online store for group: {group_name}")
        
        # Convert to DataFrame
        rows = []
        for entity_id, vector in store.items():
            row = {"entity_id": entity_id, **vector.feature_values}
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        profiler = Profiler()
        profile = await profiler.profile_dataframe(df)
        
        stats = FeatureStats(
            row_count=len(df),
            feature_stats={
                col.name: {
                    "mean": col.mean,
                    "std": col.std,
                    "min": col.min,
                    "max": col.max,
                }
                for col in profile.columns
            },
            missing_counts={
                col.name: col.null_count
                for col in profile.columns
            },
        )
        
        group.statistics = stats
        return stats
    
    async def _compute_features_on_demand(
        self, group_name: str, entity_ids: List[str]
    ) -> Dict[str, FeatureVector]:
        """Compute features on demand for missing entities."""
        from app.data import FeatureEngineering
        
        group = self._get_group(group_name)
        
        # Use existing FeatureEngineering module
        fe = FeatureEngineering()
        
        # Build transformation pipeline from feature definitions
        transformations = {
            f.name: f.transformation
            for f in group.features
            if f.transformation
        }
        
        if transformations:
            result = await fe.transform_entities(
                entity_ids=entity_ids,
                transformations=transformations,
                source_columns={
                    f.name: f.source_columns
                    for f in group.features
                    if f.source == FeatureSource.DERIVED
                },
            )
            
            vectors = {}
            for entity_id, values in result.items():
                vectors[entity_id] = FeatureVector(
                    entity_id=entity_id,
                    feature_values=values,
                )
            
            # Cache in online store
            self._online_store.setdefault(group_name, {}).update(vectors)
            
            return vectors
        
        return {}
    
    async def delete_feature_group(self, group_name: str) -> bool:
        """Delete a feature group."""
        if group_name not in self._feature_groups:
            return False
        
        del self._feature_groups[group_name]
        self._online_store.pop(group_name, None)
        
        logger.info(f"Feature group deleted: {group_name}")
        return True
    
    def _get_group(self, group_name: str) -> FeatureGroup:
        """Get group or raise."""
        group = self._feature_groups.get(group_name)
        if group is None:
            raise ValueError(f"Feature group not found: {group_name}")
        return group