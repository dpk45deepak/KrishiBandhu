# app/services/inference/router.py
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.logger import get_logger
from app.services.api.dependencies import get_current_user, require_permission
from app.services.auth.models import Permission
from app.services.inference.models import (
    BatchInferenceRequest,
    EndpointStatus,
    InferenceConfig,
    InferenceRequest,
    InferenceResponse,
    InferenceJob,
)
from app.services.inference.service import InferenceService

logger = get_logger(__name__)
router = APIRouter(prefix="/inference", tags=["Inference"])

_inference_service = InferenceService()


def get_inference_service() -> InferenceService:
    return _inference_service


@router.post("/endpoints", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_endpoint(
    model_id: str,
    model_version: int,
    name: str,
    config: Optional[InferenceConfig] = None,
    current_user: dict = Depends(require_permission(Permission.MODEL_DEPLOY.value)),
    service: InferenceService = Depends(get_inference_service),
):
    """Create an inference endpoint for a deployed model."""
    try:
        endpoint = await service.create_endpoint(
            model_id=model_id,
            model_version=model_version,
            name=name,
            config=config,
        )
        return {
            "id": str(endpoint.id),
            "name": endpoint.name,
            "endpoint_path": endpoint.endpoint_path,
            "status": endpoint.status.value,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/endpoints", response_model=List[dict])
async def list_endpoints(
    status: Optional[EndpointStatus] = Query(None),
    current_user: dict = Depends(require_permission(Permission.MODEL_READ.value)),
    service: InferenceService = Depends(get_inference_service),
):
    """List all inference endpoints."""
    endpoints = await service.list_endpoints(status=status)
    return [
        {
            "id": str(e.id),
            "name": e.name,
            "model_id": str(e.model_id),
            "model_version": e.model_version,
            "endpoint_path": e.endpoint_path,
            "status": e.status.value,
            "request_count": e.request_count,
            "error_count": e.error_count,
            "avg_latency_ms": e.avg_latency_ms,
        }
        for e in endpoints
    ]


@router.get("/endpoints/{name}", response_model=dict)
async def get_endpoint(
    name: str,
    current_user: dict = Depends(require_permission(Permission.MODEL_READ.value)),
    service: InferenceService = Depends(get_inference_service),
):
    """Get endpoint details."""
    endpoint = await service.get_endpoint(name)
    if endpoint is None:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    
    return {
        "id": str(endpoint.id),
        "name": endpoint.name,
        "model_id": str(endpoint.model_id),
        "model_version": endpoint.model_version,
        "endpoint_path": endpoint.endpoint_path,
        "status": endpoint.status.value,
        "config": {
            "batch_size": endpoint.config.batch_size,
            "timeout_ms": endpoint.config.timeout_ms,
            "cache_enabled": endpoint.config.cache_enabled,
            "explain_predictions": endpoint.config.explain_predictions,
        },
        "metrics": {
            "request_count": endpoint.request_count,
            "error_count": endpoint.error_count,
            "avg_latency_ms": endpoint.avg_latency_ms,
            "last_request_at": endpoint.last_request_at.isoformat() if endpoint.last_request_at else None,
        },
        "deployed_at": endpoint.deployed_at.isoformat(),
    }


@router.post("/endpoints/{name}/predict", response_model=InferenceResponse)
async def predict(
    name: str,
    request: InferenceRequest,
    current_user: dict = Depends(require_permission(Permission.PREDICT_EXECUTE.value)),
    service: InferenceService = Depends(get_inference_service),
):
    """Make predictions through an inference endpoint."""
    try:
        return await service.predict(name, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/endpoints/{name}/batch", response_model=dict)
async def batch_predict(
    name: str,
    request: BatchInferenceRequest,
    current_user: dict = Depends(require_permission(Permission.PREDICT_EXECUTE.value)),
    service: InferenceService = Depends(get_inference_service),
):
    """Start a batch inference job."""
    try:
        job = await service.batch_predict(name, request, user_id=current_user["sub"])
        return {
            "job_id": str(job.id),
            "status": job.status,
            "endpoint": name,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/jobs", response_model=List[dict])
async def list_jobs(
    endpoint_name: Optional[str] = Query(None),
    current_user: dict = Depends(require_permission(Permission.PREDICT_READ.value)),
    service: InferenceService = Depends(get_inference_service),
):
    """List inference jobs."""
    jobs = await service.list_jobs(endpoint_name=endpoint_name)
    return [
        {
            "id": str(j.id),
            "endpoint_id": str(j.endpoint_id),
            "status": j.status,
            "total_instances": j.total_instances,
            "processed_instances": j.processed_instances,
            "output_path": j.output_path,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
        }
        for j in jobs
    ]


@router.get("/jobs/{job_id}", response_model=dict)
async def get_job(
    job_id: str,
    current_user: dict = Depends(require_permission(Permission.PREDICT_READ.value)),
    service: InferenceService = Depends(get_inference_service),
):
    """Get inference job status."""
    job = await service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "id": str(job.id),
        "status": job.status,
        "total_instances": job.total_instances,
        "processed_instances": job.processed_instances,
        "progress": job.processed_instances / max(job.total_instances, 1) * 100,
        "output_path": job.output_path,
        "errors": job.errors,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.patch("/endpoints/{name}/config", response_model=dict)
async def update_config(
    name: str,
    config: InferenceConfig,
    current_user: dict = Depends(require_permission(Permission.MODEL_DEPLOY.value)),
    service: InferenceService = Depends(get_inference_service),
):
    """Update endpoint configuration."""
    try:
        endpoint = await service.update_endpoint_config(name, config)
        return {"message": "Config updated", "name": name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/endpoints/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_endpoint(
    name: str,
    current_user: dict = Depends(require_permission(Permission.MODEL_DEPLOY.value)),
    service: InferenceService = Depends(get_inference_service),
):
    """Delete an inference endpoint."""
    deleted = await service.delete_endpoint(name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Endpoint not found")