# app/services/feature_store/router.py
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.logger import get_logger
from app.services.api.dependencies import get_current_user, require_permission
from app.services.auth.models import Permission
from app.services.feature_store.models import (
    FeatureDefinition,
    FeatureGroup,
    FeatureVector,
)
from app.services.feature_store.service import FeatureStoreService

logger = get_logger(__name__)
router = APIRouter(prefix="/features", tags=["Feature Store"])

_feature_store = FeatureStoreService()


def get_feature_store() -> FeatureStoreService:
    return _feature_store


@router.post("/groups", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_feature_group(
    name: str,
    description: str,
    features: List[FeatureDefinition],
    dataset_id: Optional[str] = None,
    entity_key: str = "id",
    current_user: dict = Depends(require_permission(Permission.FEATURE_WRITE.value)),
    service: FeatureStoreService = Depends(get_feature_store),
):
    """Create a new feature group."""
    try:
        group = await service.create_feature_group(
            name=name,
            description=description,
            features=features,
            dataset_id=dataset_id,
            entity_key=entity_key,
            user_id=current_user["sub"],
        )
        return {
            "id": str(group.id),
            "name": group.name,
            "feature_count": len(group.features),
            "entity_key": group.entity_key,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/groups", response_model=List[dict])
async def list_feature_groups(
    dataset_id: Optional[str] = Query(None),
    current_user: dict = Depends(require_permission(Permission.FEATURE_READ.value)),
    service: FeatureStoreService = Depends(get_feature_store),
):
    """List all feature groups."""
    groups = await service.list_feature_groups(dataset_id=dataset_id)
    return [
        {
            "id": str(g.id),
            "name": g.name,
            "description": g.description,
            "feature_count": len(g.features),
            "entity_key": g.entity_key,
            "version": g.version,
            "status": g.status,
            "row_count": g.statistics.row_count if g.statistics else None,
            "created_at": g.created_at.isoformat(),
        }
        for g in groups
    ]


@router.get("/groups/{group_name}", response_model=dict)
async def get_feature_group(
    group_name: str,
    current_user: dict = Depends(require_permission(Permission.FEATURE_READ.value)),
    service: FeatureStoreService = Depends(get_feature_store),
):
    """Get feature group details."""
    group = await service.get_feature_group(group_name)
    if group is None:
        raise HTTPException(status_code=404, detail="Feature group not found")
    
    return {
        "id": str(group.id),
        "name": group.name,
        "description": group.description,
        "entity_key": group.entity_key,
        "version": group.version,
        "status": group.status,
        "features": [
            {
                "name": f.name,
                "dtype": f.dtype.value,
                "source": f.source.value,
                "description": f.description,
                "tags": f.tags,
            }
            for f in group.features
        ],
        "statistics": {
            "row_count": group.statistics.row_count,
            "feature_stats": group.statistics.feature_stats,
        } if group.statistics else None,
        "lineage": {
            "source_dataset_id": group.lineage.source_dataset_id,
            "pipeline_run_id": group.lineage.pipeline_run_id,
        } if group.lineage else None,
        "created_at": group.created_at.isoformat(),
        "updated_at": group.updated_at.isoformat(),
    }


@router.get("/groups/{group_name}/features", response_model=Dict[str, dict])
async def get_online_features(
    group_name: str,
    entity_ids: List[str] = Query(...),
    current_user: dict = Depends(require_permission(Permission.FEATURE_READ.value)),
    service: FeatureStoreService = Depends(get_feature_store),
):
    """Get features for specific entities from online store."""
    try:
        vectors = await service.get_online_features(group_name, entity_ids)
        return {
            entity_id: {
                "features": vector.feature_values,
                "timestamp": vector.timestamp.isoformat() if vector.timestamp else None,
            }
            for entity_id, vector in vectors.items()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/groups/{group_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_feature_group(
    group_name: str,
    current_user: dict = Depends(require_permission(Permission.FEATURE_WRITE.value)),
    service: FeatureStoreService = Depends(get_feature_store),
):
    """Delete a feature group."""
    deleted = await service.delete_feature_group(group_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Feature group not found")