# app/services/datasets/router.py
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from app.logger import get_logger
from app.services.api.dependencies import get_current_user, require_permission
from app.services.auth.models import Permission
from app.services.datasets.models import (
    CleaningConfig,
    DatasetCreate,
    DatasetProfile,
    DatasetResponse,
    DatasetStatus,
    DatasetVersion,
    StandardizationConfig,
    ValidationReport,
    ValidationRule,
)
from app.services.datasets.service import DatasetService

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets", tags=["Datasets"])

# Service singleton
_dataset_service = DatasetService()


def get_dataset_service() -> DatasetService:
    return _dataset_service


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset: DatasetCreate,
    current_user: dict = Depends(require_permission(Permission.DATASET_WRITE.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """Create a new dataset entry."""
    return await service.create_dataset(dataset, user_id=current_user["sub"])


@router.post("/{dataset_id}/upload", response_model=DatasetVersion)
async def upload_dataset(
    dataset_id: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(require_permission(Permission.DATASET_WRITE.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """Upload dataset file and trigger scanning."""
    content = await file.read()
    return await service.upload_and_scan(dataset_id, content, file.filename)


@router.get("", response_model=List[DatasetResponse])
async def list_datasets(
    status: Optional[DatasetStatus] = Query(None),
    tags: Optional[List[str]] = Query(None),
    current_user: dict = Depends(require_permission(Permission.DATASET_READ.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """List all datasets with optional filtering."""
    return await service.list_datasets(status=status, tags=tags)


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    current_user: dict = Depends(require_permission(Permission.DATASET_READ.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """Get dataset details."""
    dataset = await service.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.post("/{dataset_id}/profile", response_model=DatasetProfile)
async def profile_dataset(
    dataset_id: str,
    current_user: dict = Depends(require_permission(Permission.DATASET_READ.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """Generate statistical profile for a dataset."""
    try:
        return await service.profile_dataset(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/validate", response_model=ValidationReport)
async def validate_dataset(
    dataset_id: str,
    rules: Optional[List[ValidationRule]] = None,
    current_user: dict = Depends(require_permission(Permission.DATASET_READ.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """Validate dataset against rules."""
    try:
        return await service.validate_dataset(dataset_id, rules)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/clean", response_model=DatasetVersion)
async def clean_dataset(
    dataset_id: str,
    config: CleaningConfig,
    current_user: dict = Depends(require_permission(Permission.DATASET_WRITE.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """Clean dataset and create new version."""
    try:
        return await service.clean_dataset(dataset_id, config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/standardize", response_model=DatasetVersion)
async def standardize_dataset(
    dataset_id: str,
    config: StandardizationConfig,
    current_user: dict = Depends(require_permission(Permission.DATASET_WRITE.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """Standardize dataset and create new version."""
    try:
        return await service.standardize_dataset(dataset_id, config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dataset_id}/versions", response_model=List[DatasetVersion])
async def list_versions(
    dataset_id: str,
    current_user: dict = Depends(require_permission(Permission.DATASET_READ.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """List all versions of a dataset."""
    return await service.get_versions(dataset_id)


@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    version: Optional[int] = Query(None),
    current_user: dict = Depends(require_permission(Permission.DATASET_READ.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """Download dataset file."""
    try:
        content, filename = await service.download_dataset(dataset_id, version)
        return StreamingResponse(
            iter([content]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    current_user: dict = Depends(require_permission(Permission.DATASET_DELETE.value)),
    service: DatasetService = Depends(get_dataset_service),
):
    """Delete dataset and all versions."""
    deleted = await service.delete_dataset(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")