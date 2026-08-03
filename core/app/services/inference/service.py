# app/services/inference/service.py
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.config import settings
from app.logger import get_logger
from app.services.inference.models import (
    BatchInferenceRequest,
    EndpointStatus,
    InferenceConfig,
    InferenceJob,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    ModelEndpoint,
)
from app.utils.decorators import timed

logger = get_logger(__name__)


class InferenceService:
    """Model inference serving service.
    
    Consumes:
    - ml.MLPlatform: model loading and prediction
    - ml.explainability: prediction explanations
    - feature_store: feature retrieval for inference
    - config: serving configuration
    - logger: structured logging
    
    Provides:
    - Real-time inference endpoints
    - Batch inference jobs
    - Model endpoint management
    - Inference caching
    - Performance monitoring
    """
    
    def __init__(self):
        self._endpoints: Dict[str, ModelEndpoint] = {}
        self._jobs: Dict[str, InferenceJob] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}  # cache_key -> {result, expires_at}
        self._model_cache: Dict[str, Any] = {}  # model_id -> loaded model object
    
    @timed
    async def create_endpoint(
        self, model_id: str, model_version: int, name: str, config: Optional[InferenceConfig] = None
    ) -> ModelEndpoint:
        """Create a new inference endpoint for a deployed model."""
        endpoint_id = uuid4()
        
        endpoint = ModelEndpoint(
            id=endpoint_id,
            name=name,
            model_id=UUID(model_id),
            model_version=model_version,
            endpoint_path=f"/inference/{name}",
            config=config or InferenceConfig(),
        )
        
        # Pre-load model
        await self._load_model(model_id, model_version)
        
        self._endpoints[name] = endpoint
        
        logger.info(
            f"Inference endpoint created: {name} -> model={model_id} v{model_version}"
        )
        return endpoint
    
    @timed
    async def predict(
        self,
        endpoint_name: str,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Serve a prediction request through an endpoint."""
        endpoint = self._get_endpoint(endpoint_name)
        config = request.config_override or endpoint.config
        
        start_time = time.perf_counter()
        errors = []
        
        try:
            # Check cache
            if config.cache_enabled:
                cache_key = self._build_cache_key(
                    str(endpoint.model_id), request.instances
                )
                cached = self._get_from_cache(cache_key)
                if cached:
                    logger.debug(f"Cache hit for endpoint: {endpoint_name}")
                    cached["latency_ms"] = (time.perf_counter() - start_time) * 1000
                    return InferenceResponse(**cached)
            
            # Load model
            model = await self._load_model(
                str(endpoint.model_id), endpoint.model_version
            )
            
            # Preprocess
            instances = request.instances
            if config.preprocess:
                instances = await self._preprocess(instances, config.preprocess)
            
            # Predict
            from app.ml import MLPlatform
            platform = MLPlatform()
            
            result = await platform.predict(
                model=model,
                data=instances,
                return_probabilities=config.return_probabilities,
                threshold=config.threshold,
                batch_size=config.batch_size,
            )
            
            # Postprocess
            predictions = result.predictions
            if config.postprocess:
                predictions = await self._postprocess(predictions, config.postprocess)
            
            # Generate explanations
            explanations = None
            if config.explain_predictions:
                explanations = await platform.explainability.explain_predictions(
                    model=model,
                    instances=instances,
                    predictions=predictions,
                )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            response = InferenceResponse(
                predictions=predictions,
                probabilities=result.probabilities if config.return_probabilities else None,
                explanations=explanations,
                status=InferenceStatus.SUCCESS,
                model_id=endpoint.model_id,
                model_version=endpoint.model_version,
                latency_ms=latency_ms,
                metadata={
                    "endpoint": endpoint_name,
                    "cached": False,
                    "batch_size": len(instances),
                },
            )
            
            # Update cache
            if config.cache_enabled:
                cache_key = self._build_cache_key(str(endpoint.model_id), instances)
                self._set_cache(cache_key, response, config.cache_ttl_seconds)
            
            # Update endpoint metrics
            endpoint.request_count += 1
            endpoint.last_request_at = datetime.now(timezone.utc)
            endpoint.avg_latency_ms = (
                endpoint.avg_latency_ms * (endpoint.request_count - 1) + latency_ms
            ) / endpoint.request_count
            
            logger.debug(
                f"Prediction served: {endpoint_name} - "
                f"{len(instances)} instances in {latency_ms:.1f}ms"
            )
            
            return response
            
        except asyncio.TimeoutError:
            endpoint.error_count += 1
            logger.error(f"Prediction timeout for {endpoint_name}")
            return InferenceResponse(
                predictions=[],
                status=InferenceStatus.TIMEOUT,
                model_id=endpoint.model_id,
                model_version=endpoint.model_version,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                errors=[{"type": "timeout", "message": f"Request exceeded {config.timeout_ms}ms"}],
            )
            
        except Exception as e:
            endpoint.error_count += 1
            logger.exception(f"Prediction failed for {endpoint_name}: {e}")
            errors.append({"type": type(e).__name__, "message": str(e)})
            
            return InferenceResponse(
                predictions=[],
                status=InferenceStatus.FAILED,
                model_id=endpoint.model_id,
                model_version=endpoint.model_version,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                errors=errors,
            )
    
    @timed
    async def batch_predict(
        self,
        endpoint_name: str,
        request: BatchInferenceRequest,
        user_id: str,
    ) -> InferenceJob:
        """Start a batch inference job."""
        endpoint = self._get_endpoint(endpoint_name)
        
        job = InferenceJob(
            id=uuid4(),
            endpoint_id=endpoint.id,
            status="queued",
            created_by=user_id,
        )
        
        self._jobs[str(job.id)] = job
        
        # Start background processing
        asyncio.create_task(self._process_batch_job(job, endpoint, request))
        
        logger.info(f"Batch inference job queued: {job.id} ({endpoint_name})")
        return job
    
    async def _process_batch_job(
        self,
        job: InferenceJob,
        endpoint: ModelEndpoint,
        request: BatchInferenceRequest,
    ):
        """Process a batch inference job in the background."""
        import pandas as pd
        
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        
        try:
            # Read dataset
            df = pd.read_parquet(request.dataset_path)
            job.total_instances = len(df)
            
            config = request.config_override or endpoint.config
            
            # Process in chunks
            all_predictions = []
            for chunk_start in range(0, len(df), request.chunk_size):
                chunk = df.iloc[chunk_start:chunk_start + request.chunk_size]
                instances = chunk.to_dict("records")
                
                inf_request = InferenceRequest(
                    instances=instances,
                    config_override=config,
                )
                
                response = await self.predict(endpoint.name, inf_request)
                
                if response.status == InferenceStatus.SUCCESS:
                    all_predictions.extend(response.predictions)
                else:
                    job.errors.extend(response.errors or [])
                
                job.processed_instances += len(instances)
            
            # Save results
            output_path = request.output_path or f"predictions/{job.id}/results.parquet"
            result_df = df.copy()
            result_df["prediction"] = all_predictions
            
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_parquet(output_path)
            
            job.output_path = output_path
            job.status = "completed"
            
            logger.info(
                f"Batch inference completed: {job.id} - "
                f"{job.total_instances} instances"
            )
            
        except Exception as e:
            job.status = "failed"
            job.errors.append({"type": type(e).__name__, "message": str(e)})
            logger.exception(f"Batch inference failed: {job.id}")
        
        finally:
            job.completed_at = datetime.now(timezone.utc)
    
    async def get_endpoint(self, name: str) -> Optional[ModelEndpoint]:
        """Get endpoint by name."""
        return self._endpoints.get(name)
    
    async def list_endpoints(
        self, status: Optional[EndpointStatus] = None
    ) -> List[ModelEndpoint]:
        """List all inference endpoints."""
        endpoints = list(self._endpoints.values())
        if status:
            endpoints = [e for e in endpoints if e.status == status]
        return endpoints
    
    async def get_job(self, job_id: str) -> Optional[InferenceJob]:
        """Get batch inference job status."""
        return self._jobs.get(job_id)
    
    async def list_jobs(
        self, endpoint_name: Optional[str] = None
    ) -> List[InferenceJob]:
        """List inference jobs."""
        jobs = list(self._jobs.values())
        if endpoint_name:
            endpoint = self._endpoints.get(endpoint_name)
            if endpoint:
                jobs = [j for j in jobs if j.endpoint_id == endpoint.id]
        return jobs
    
    async def update_endpoint_config(
        self, name: str, config: InferenceConfig
    ) -> ModelEndpoint:
        """Update endpoint configuration."""
        endpoint = self._get_endpoint(name)
        endpoint.config = config
        logger.info(f"Endpoint config updated: {name}")
        return endpoint
    
    async def delete_endpoint(self, name: str) -> bool:
        """Delete an endpoint."""
        if name not in self._endpoints:
            return False
        
        endpoint = self._endpoints.pop(name)
        endpoint.status = EndpointStatus.INACTIVE
        
        # Clear model from cache if no other endpoints use it
        model_id = str(endpoint.model_id)
        still_used = any(
            str(e.model_id) == model_id
            for e in self._endpoints.values()
        )
        if not still_used:
            self._model_cache.pop(model_id, None)
        
        logger.info(f"Endpoint deleted: {name}")
        return True
    
    async def _load_model(self, model_id: str, version: int) -> Any:
        """Load a model into memory, with caching."""
        cache_key = f"{model_id}:v{version}"
        
        if cache_key not in self._model_cache:
            from app.ml import MLPlatform
            platform = MLPlatform()
            model = await platform.load_model(model_id, version)
            self._model_cache[cache_key] = model
            logger.info(f"Model loaded: {cache_key}")
        
        return self._model_cache[cache_key]
    
    async def _preprocess(
        self, instances: List[Dict[str, Any]], preprocess_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply preprocessing transformations."""
        # Use existing Standardizer or custom preprocessing
        from app.data import Standardizer
        standardizer = Standardizer()
        
        import pandas as pd
        df = pd.DataFrame(instances)
        df = await standardizer.standardize_dataframe(df, **preprocess_config)
        return df.to_dict("records")
    
    async def _postprocess(
        self, predictions: List[Any], postprocess_config: Dict[str, Any]
    ) -> List[Any]:
        """Apply postprocessing transformations."""
        # Custom postprocessing logic
        return predictions
    
    def _build_cache_key(self, model_id: str, instances: List[Dict[str, Any]]) -> str:
        """Build a deterministic cache key."""
        content = json.dumps({"model_id": model_id, "instances": instances}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve from cache if not expired."""
        entry = self._cache.get(cache_key)
        if entry and entry["expires_at"] > time.time():
            return entry["result"]
        elif entry:
            del self._cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, response: InferenceResponse, ttl_seconds: int):
        """Store result in cache."""
        self._cache[cache_key] = {
            "result": {
                "predictions": response.predictions,
                "probabilities": response.probabilities,
                "explanations": response.explanations,
                "status": response.status.value,
                "model_id": response.model_id,
                "model_version": response.model_version,
            },
            "expires_at": time.time() + ttl_seconds,
        }
        # Prune old cache entries
        self._prune_cache()
    
    def _prune_cache(self):
        """Remove expired cache entries."""
        now = time.time()
        expired = [
            k for k, v in self._cache.items()
            if v["expires_at"] <= now
        ]
        for k in expired:
            del self._cache[k]
    
    def _get_endpoint(self, name: str) -> ModelEndpoint:
        """Get endpoint or raise."""
        endpoint = self._endpoints.get(name)
        if endpoint is None:
            raise ValueError(f"Endpoint not found: {name}")
        if endpoint.status != EndpointStatus.ACTIVE:
            raise ValueError(f"Endpoint not active: {name} ({endpoint.status.value})")
        return endpoint