# app/services/ml/service.py
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.config import settings
from app.logger import get_logger
from app.services.ml.models import (
    EvaluationReport,
    ExplainabilityReport,
    ModelCreate,
    ModelResponse,
    ModelStatus,
    ModelType,
    ModelVersion,
    PredictionRequest,
    PredictionResponse,
    TrainingConfig,
    TrainingJob,
    TrainingStatus,
    TuningConfig,
    TuningResult,
)
from app.utils.decorators import timed

logger = get_logger(__name__)


class MLService:
    """Machine Learning service.
    
    Consumes existing modules:
    - ml.classification: classification model training
    - ml.regression: regression model training
    - ml.evaluation: model evaluation metrics
    - ml.tuning: hyperparameter optimization
    - ml.explainability: model interpretability
    - config: model storage paths
    - logger: structured logging
    """
    
    def __init__(self):
        self._models: Dict[str, ModelResponse] = {}
        self._training_jobs: Dict[str, TrainingJob] = {}
        self._models_path = Path(settings.MODELS_STORAGE_PATH)
        self._models_path.mkdir(parents=True, exist_ok=True)
    
    @timed
    async def register_model(
        self, model_data: ModelCreate, user_id: str
    ) -> ModelResponse:
        """Register a new model in the registry."""
        model_id = uuid4()
        
        model = ModelResponse(
            id=model_id,
            name=model_data.name,
            model_type=model_data.model_type,
            description=model_data.description,
            status=ModelStatus.DRAFT,
            tags=model_data.tags,
            framework=model_data.framework,
            metadata=model_data.metadata,
            created_by=user_id,
        )
        
        self._models[str(model_id)] = model
        
        logger.info(f"Model registered: {model.name} ({model.model_type.value})")
        return model
    
    @timed
    async def train_model(
        self,
        model_id: str,
        config: TrainingConfig,
        dataset_path: str,
        user_id: str,
    ) -> TrainingJob:
        """Start model training using existing ml modules."""
        from app.ml import MLPlatform
        
        model = self._get_model(model_id)
        model.status = ModelStatus.TRAINING
        
        # Create training job
        job = TrainingJob(
            id=uuid4(),
            model_id=model.id,
            model_name=model.name,
            config=config,
            dataset_path=dataset_path,
            status=TrainingStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            created_by=user_id,
        )
        
        self._training_jobs[str(job.id)] = job
        model.training_jobs.append(job)
        
        try:
            platform = MLPlatform()
            
            # Route to correct training module based on model type
            if model.model_type == ModelType.CLASSIFICATION:
                result = await platform.classification.train(
                    dataset_path,
                    target_column=config.target_column,
                    feature_columns=config.feature_columns,
                    exclude_columns=config.exclude_columns,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    algorithm=config.algorithm,
                    hyperparameters=config.hyperparameters,
                    class_weight=config.class_weight,
                    cv_folds=config.cv_folds,
                    metric=config.metric,
                )
            elif model.model_type == ModelType.REGRESSION:
                result = await platform.regression.train(
                    dataset_path,
                    target_column=config.target_column,
                    feature_columns=config.feature_columns,
                    exclude_columns=config.exclude_columns,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    algorithm=config.algorithm,
                    hyperparameters=config.hyperparameters,
                    cv_folds=config.cv_folds,
                    metric=config.metric,
                )
            else:
                raise ValueError(f"Unsupported model type for training: {model.model_type.value}")
            
            # Update job with results
            job.status = TrainingStatus.COMPLETED
            job.metrics = result.metrics
            job.model_path = result.model_path
            job.artifacts = result.artifacts or {}
            job.completed_at = datetime.now(timezone.utc)
            job.duration_seconds = (job.completed_at - job.started_at).total_seconds()
            
            # Create model version
            version = ModelVersion(
                version=len(model.versions) + 1,
                model_path=result.model_path,
                metrics=result.metrics,
                training_config=config,
                artifacts=result.artifacts or {},
            )
            model.versions.append(version)
            model.current_version = version
            model.status = ModelStatus.TRAINED
            model.updated_at = datetime.now(timezone.utc)
            
            logger.info(
                f"Model trained: {model.name} v{version.version} - "
                f"metrics: {result.metrics}"
            )
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            model.status = ModelStatus.FAILED
            logger.error(f"Training failed for {model.name}: {e}")
            raise
        
        return job
    
    @timed
    async def evaluate_model(
        self,
        model_id: str,
        test_data_path: str,
        target_column: str,
    ) -> EvaluationReport:
        """Evaluate a trained model using existing ml.evaluation."""
        from app.ml import MLPlatform
        
        model = self._get_model(model_id)
        version = model.current_version
        
        if version is None:
            raise ValueError("Model has no trained version to evaluate")
        
        model.status = ModelStatus.EVALUATING
        
        platform = MLPlatform()
        result = await platform.evaluation.evaluate(
            model_path=version.model_path,
            test_data_path=test_data_path,
            target_column=target_column,
        )
        
        report = EvaluationReport(
            model_id=model.id,
            model_version=version.version,
            metrics=result.metrics,
            confusion_matrix=getattr(result, 'confusion_matrix', None),
            classification_report=getattr(result, 'classification_report', None),
            roc_auc=getattr(result, 'roc_auc', None),
            precision_recall=getattr(result, 'precision_recall', None),
            residual_plot=getattr(result, 'residual_plot', None),
            feature_importance=getattr(result, 'feature_importance', None),
            cross_validation_scores=getattr(result, 'cross_validation_scores', None),
            test_metrics=getattr(result, 'test_metrics', None),
            overfit_analysis=getattr(result, 'overfit_analysis', None),
        )
        
        model.evaluation_report = report
        model.status = ModelStatus.READY
        model.updated_at = datetime.now(timezone.utc)
        
        logger.info(f"Model evaluated: {model.name} - metrics: {report.metrics}")
        return report
    
    @timed
    async def tune_model(
        self,
        model_id: str,
        config: TuningConfig,
        dataset_path: str,
        target_column: str,
    ) -> TuningResult:
        """Tune hyperparameters using existing ml.tuning."""
        from app.ml import MLPlatform
        
        model = self._get_model(model_id)
        version = model.current_version
        
        if version is None:
            raise ValueError("Model has no trained version to tune")
        
        model.status = ModelStatus.TUNING
        
        platform = MLPlatform()
        result = await platform.tuning.tune(
            model_path=version.model_path,
            dataset_path=dataset_path,
            target_column=target_column,
            param_grid=config.param_grid,
            tuning_method=config.tuning_method,
            n_iter=config.n_iter,
            cv_folds=config.cv_folds,
            metric=config.metric,
            early_stopping_rounds=config.early_stopping_rounds,
            n_jobs=config.n_jobs,
        )
        
        tuning_result = TuningResult(
            best_params=result.best_params,
            best_score=result.best_score,
            all_results=result.all_results,
            tuning_time_seconds=result.tuning_time_seconds,
            param_importance=getattr(result, 'param_importance', None),
            convergence_plot=getattr(result, 'convergence_plot', None),
            tuned_model_path=result.tuned_model_path,
        )
        
        # Create new version with tuned model
        if result.tuned_model_path:
            new_version = ModelVersion(
                version=len(model.versions) + 1,
                model_path=result.tuned_model_path,
                metrics={"best_score": result.best_score, **result.best_params},
                training_config=version.training_config,
                artifacts=version.artifacts,
            )
            model.versions.append(new_version)
            model.current_version = new_version
        
        model.status = ModelStatus.READY
        model.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            f"Model tuned: {model.name} - best_score={result.best_score:.4f}"
        )
        return tuning_result
    
    @timed
    async def predict(
        self, model_id: str, request: PredictionRequest
    ) -> PredictionResponse:
        """Make predictions using a trained model."""
        import time
        
        model = self._get_model(model_id)
        version = model.current_version
        
        if version is None:
            raise ValueError("Model has no trained version for prediction")
        
        start = time.perf_counter()
        
        from app.ml import MLPlatform
        platform = MLPlatform()
        
        result = await platform.predict(
            model_path=version.model_path,
            data=request.data,
            return_probabilities=request.return_probabilities,
            threshold=request.threshold,
        )
        
        prediction_time = (time.perf_counter() - start) * 1000
        
        response = PredictionResponse(
            predictions=result.predictions,
            probabilities=result.probabilities if request.return_probabilities else None,
            model_id=model.id,
            model_version=version.version,
            prediction_time_ms=prediction_time,
        )
        
        # Add explanations if requested
        if request.explain:
            explanations = await self._explain_predictions(
                model_id, request.data, result.predictions
            )
            response.explanations = explanations
        
        logger.debug(
            f"Prediction made: {model.name} - {len(request.data)} samples "
            f"in {prediction_time:.1f}ms"
        )
        
        return response
    
    @timed
    async def explain_model(
        self, model_id: str, data_path: Optional[str] = None
    ) -> ExplainabilityReport:
        """Generate model explainability report using existing ml.explainability."""
        from app.ml import MLPlatform
        
        model = self._get_model(model_id)
        version = model.current_version
        
        if version is None:
            raise ValueError("Model has no trained version to explain")
        
        platform = MLPlatform()
        result = await platform.explainability.explain(
            model_path=version.model_path,
            data_path=data_path,
        )
        
        report = ExplainabilityReport(
            feature_importance=result.feature_importance,
            shap_values=getattr(result, 'shap_values', None),
            lime_explanations=getattr(result, 'lime_explanations', None),
            partial_dependence=getattr(result, 'partial_dependence', None),
            global_importance_plot=getattr(result, 'global_importance_plot', None),
            summary_plot=getattr(result, 'summary_plot', None),
        )
        
        logger.info(f"Explainability report generated for: {model.name}")
        return report
    
    async def _explain_predictions(
        self, model_id: str, data: List[Dict[str, Any]], predictions: List[Any]
    ) -> List[Dict[str, Any]]:
        """Generate per-prediction explanations."""
        from app.ml import MLPlatform
        
        model = self._get_model(model_id)
        version = model.current_version
        
        platform = MLPlatform()
        explanations = await platform.explainability.explain_predictions(
            model_path=version.model_path,
            instances=data,
            predictions=predictions,
        )
        
        return explanations
    
    async def get_model(self, model_id: str) -> Optional[ModelResponse]:
        """Get model by ID."""
        return self._models.get(model_id)
    
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelResponse]:
        """List models with optional filtering."""
        models = list(self._models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        if status:
            models = [m for m in models if m.status == status]
        if tags:
            models = [m for m in models if any(t in m.tags for t in tags)]
        
        return models
    
    async def list_training_jobs(
        self, model_id: Optional[str] = None, status: Optional[TrainingStatus] = None
    ) -> List[TrainingJob]:
        """List training jobs."""
        jobs = list(self._training_jobs.values())
        
        if model_id:
            jobs = [j for j in jobs if str(j.model_id) == model_id]
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def deploy_model(self, model_id: str, deployment_config: Dict[str, Any] = None) -> ModelVersion:
        """Mark a model version as deployed."""
        model = self._get_model(model_id)
        version = model.current_version
        
        if version is None:
            raise ValueError("No version to deploy")
        
        version.deployed = True
        version.deployment_info = deployment_config or {}
        model.status = ModelStatus.DEPLOYED
        model.updated_at = datetime.now(timezone.utc)
        
        logger.info(f"Model deployed: {model.name} v{version.version}")
        return version
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model and its artifacts."""
        if model_id not in self._models:
            return False
        
        model = self._models.pop(model_id)
        
        # Clean up training jobs
        jobs_to_remove = [
            jid for jid, job in self._training_jobs.items()
            if str(job.model_id) == model_id
        ]
        for jid in jobs_to_remove:
            del self._training_jobs[jid]
        
        logger.info(f"Model deleted: {model.name}")
        return True
    
    def _get_model(self, model_id: str) -> ModelResponse:
        """Get model or raise."""
        model = self._models.get(model_id)
        if model is None:
            raise ValueError(f"Model not found: {model_id}")
        return model