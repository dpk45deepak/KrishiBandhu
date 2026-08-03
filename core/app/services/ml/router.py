# app/services/ml/router.py
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, status

from app.logger import get_logger
from app.services.api.dependencies import get_current_user, require_permission
from app.services.auth.models import Permission
from app.services.ml.models import (
    ModelCreate,
    ModelResponse,
    ModelType,
    ModelStatus,
    TrainingConfig,
    TrainingJob,
    TrainingStatus,
    EvaluationReport,
    TuningConfig,
    TuningResult,
    PredictionRequest,
    PredictionResponse,
    ExplainabilityReport,
)
from app.services.ml.service import MLService

logger = get_logger(__name__)
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

_ml_service = MLService()


def get_ml_service() -> MLService:
    return _ml_service


# ============ Model Registry ============

@router.post("/models", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def register_model(
    model_data: ModelCreate,
    current_user: dict = Depends(require_permission(Permission.MODEL_TRAIN.value)),
    service: MLService = Depends(get_ml_service),
):
    """Register a new model."""
    return await service.register_model(model_data, user_id=current_user["sub"])


@router.get("/models", response_model=List[ModelResponse])
async def list_models(
    model_type: Optional[ModelType] = Query(None),
    status: Optional[ModelStatus] = Query(None),
    tags: Optional[List[str]] = Query(None),
    current_user: dict = Depends(require_permission(Permission.MODEL_READ.value)),
    service: MLService = Depends(get_ml_service),
):
    """List all models with optional filtering."""
    return await service.list_models(model_type=model_type, status=status, tags=tags)


@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    current_user: dict = Depends(require_permission(Permission.MODEL_READ.value)),
    service: MLService = Depends(get_ml_service),
):
    """Get model details."""
    model = await service.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    current_user: dict = Depends(require_permission(Permission.MODEL_DEPLOY.value)),
    service: MLService = Depends(get_ml_service),
):
    """Delete a model."""
    deleted = await service.delete_model(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")


# ============ Training ============

@router.post("/models/{model_id}/train", response_model=TrainingJob)
async def train_model(
    model_id: str,
    config: TrainingConfig,
    dataset_id: Optional[str] = None,
    current_user: dict = Depends(require_permission(Permission.MODEL_TRAIN.value)),
    service: MLService = Depends(get_ml_service),
):
    """Train a model."""
    try:
        # Resolve dataset path from dataset service
        dataset_path = f"data/{dataset_id or model_id}/latest"
        
        return await service.train_model(
            model_id=model_id,
            config=config,
            dataset_path=dataset_path,
            user_id=current_user["sub"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-jobs", response_model=List[TrainingJob])
async def list_training_jobs(
    model_id: Optional[str] = Query(None),
    status: Optional[TrainingStatus] = Query(None),
    current_user: dict = Depends(require_permission(Permission.MODEL_READ.value)),
    service: MLService = Depends(get_ml_service),
):
    """List training jobs."""
    return await service.list_training_jobs(model_id=model_id, status=status)


# ============ Evaluation ============

@router.post("/models/{model_id}/evaluate", response_model=EvaluationReport)
async def evaluate_model(
    model_id: str,
    test_data_path: str,
    target_column: str,
    current_user: dict = Depends(require_permission(Permission.MODEL_READ.value)),
    service: MLService = Depends(get_ml_service),
):
    """Evaluate a trained model."""
    try:
        return await service.evaluate_model(
            model_id=model_id,
            test_data_path=test_data_path,
            target_column=target_column,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============ Hyperparameter Tuning ============

@router.post("/models/{model_id}/tune", response_model=TuningResult)
async def tune_model(
    model_id: str,
    config: TuningConfig,
    dataset_path: str,
    target_column: str,
    current_user: dict = Depends(require_permission(Permission.MODEL_TRAIN.value)),
    service: MLService = Depends(get_ml_service),
):
    """Tune model hyperparameters."""
    try:
        return await service.tune_model(
            model_id=model_id,
            config=config,
            dataset_path=dataset_path,
            target_column=target_column,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============ Predictions ============

@router.post("/models/{model_id}/predict", response_model=PredictionResponse)
async def predict(
    model_id: str,
    request: PredictionRequest,
    current_user: dict = Depends(require_permission(Permission.PREDICT_EXECUTE.value)),
    service: MLService = Depends(get_ml_service),
):
    """Make predictions using a trained model."""
    try:
        return await service.predict(model_id, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict", response_model=PredictionResponse)
async def predict_batch(
    request: PredictionRequest,
    model_id: str = Query(...),
    current_user: dict = Depends(require_permission(Permission.PREDICT_EXECUTE.value)),
    service: MLService = Depends(get_ml_service),
):
    """Make batch predictions (query parameter for model_id)."""
    return await predict(model_id, request, current_user, service)


# ============ Explainability ============

@router.get("/models/{model_id}/explain", response_model=ExplainabilityReport)
async def explain_model(
    model_id: str,
    data_path: Optional[str] = Query(None),
    current_user: dict = Depends(require_permission(Permission.MODEL_READ.value)),
    service: MLService = Depends(get_ml_service),
):
    """Generate model explainability report."""
    try:
        return await service.explain_model(model_id, data_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============ Deployment ============

@router.post("/models/{model_id}/deploy", response_model=dict)
async def deploy_model(
    model_id: str,
    deployment_config: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(require_permission(Permission.MODEL_DEPLOY.value)),
    service: MLService = Depends(get_ml_service),
):
    """Deploy a model version."""
    try:
        version = await service.deploy_model(model_id, deployment_config)
        return {
            "message": "Model deployed",
            "model_id": model_id,
            "version": version.version,
            "deployed": version.deployed,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))