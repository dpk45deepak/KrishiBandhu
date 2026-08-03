# app/cli/client.py
"""
API Client for CLI - wraps all service calls.

This is the single point of communication between CLI and services.
All CLI commands use this client instead of calling services directly.
"""
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


class APIClient:
    """HTTP client for the AgriMind API.
    
    Consumes: config (settings), logger
    Used by: All CLI commands
    """
    
    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None):
        self.base_url = base_url or settings.API_BASE_URL
        self.token = token
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers=self._build_headers(),
        )
    
    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate and store token."""
        response = await self._client.post("/api/auth/token", data={
            "username": username,
            "password": password,
        })
        response.raise_for_status()
        data = response.json()
        self.token = data["access_token"]
        self._client.headers["Authorization"] = f"Bearer {self.token}"
        return data
    
    # ============ Health ============
    
    async def health_check(self) -> Dict[str, Any]:
        response = await self._client.get("/api/health")
        response.raise_for_status()
        return response.json()
    
    # ============ Datasets ============
    
    async def create_dataset(self, name: str, description: str = "", **kwargs) -> Dict[str, Any]:
        response = await self._client.post("/api/datasets", json={
            "name": name,
            "description": description,
            **kwargs,
        })
        response.raise_for_status()
        return response.json()
    
    async def list_datasets(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {}
        if status:
            params["status"] = status
        response = await self._client.get("/api/datasets", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        response = await self._client.get(f"/api/datasets/{dataset_id}")
        response.raise_for_status()
        return response.json()
    
    async def upload_dataset(self, dataset_id: str, file_path: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=120.0) as client:
            with open(file_path, "rb") as f:
                response = await client.post(
                    f"/api/datasets/{dataset_id}/upload",
                    files={"file": f},
                    headers=self._build_headers(),
                )
            response.raise_for_status()
            return response.json()
    
    async def profile_dataset(self, dataset_id: str) -> Dict[str, Any]:
        response = await self._client.post(f"/api/datasets/{dataset_id}/profile")
        response.raise_for_status()
        return response.json()
    
    async def validate_dataset(self, dataset_id: str, rules: Optional[List[Dict]] = None) -> Dict[str, Any]:
        response = await self._client.post(
            f"/api/datasets/{dataset_id}/validate",
            json=rules or [],
        )
        response.raise_for_status()
        return response.json()
    
    async def clean_dataset(self, dataset_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post(
            f"/api/datasets/{dataset_id}/clean",
            json=config,
        )
        response.raise_for_status()
        return response.json()
    
    async def download_dataset(self, dataset_id: str, output_path: str, version: Optional[int] = None):
        params = {}
        if version:
            params["version"] = version
        async with httpx.AsyncClient(base_url=self.base_url, timeout=120.0) as client:
            response = await client.get(
                f"/api/datasets/{dataset_id}/download",
                params=params,
                headers=self._build_headers(),
            )
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        response = await self._client.delete(f"/api/datasets/{dataset_id}")
        return response.status_code == 204
    
    # ============ Pipelines ============
    
    async def create_pipeline(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post("/api/pipeline", json={
            "name": name,
            "config": config,
        })
        response.raise_for_status()
        return response.json()
    
    async def list_pipelines(self) -> List[Dict[str, Any]]:
        response = await self._client.get("/api/pipeline")
        response.raise_for_status()
        return response.json()
    
    async def run_pipeline(self, pipeline_id: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        response = await self._client.post(
            f"/api/pipeline/{pipeline_id}/run",
            json=params or {},
        )
        response.raise_for_status()
        return response.json()
    
    async def get_pipeline_run(self, pipeline_id: str, run_id: str) -> Dict[str, Any]:
        response = await self._client.get(f"/api/pipeline/{pipeline_id}/runs/{run_id}")
        response.raise_for_status()
        return response.json()
    
    # ============ ML ============
    
    async def register_model(self, name: str, model_type: str, **kwargs) -> Dict[str, Any]:
        response = await self._client.post("/api/ml/models", json={
            "name": name,
            "model_type": model_type,
            **kwargs,
        })
        response.raise_for_status()
        return response.json()
    
    async def list_models(self) -> List[Dict[str, Any]]:
        response = await self._client.get("/api/ml/models")
        response.raise_for_status()
        return response.json()
    
    async def train_model(self, model_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        response = await self._client.post(
            f"/api/ml/models/{model_id}/train",
            json=config,
        )
        response.raise_for_status()
        return response.json()
    
    async def evaluate_model(self, model_id: str, test_path: str, target: str) -> Dict[str, Any]:
        response = await self._client.post(
            f"/api/ml/models/{model_id}/evaluate",
            params={"test_data_path": test_path, "target_column": target},
        )
        response.raise_for_status()
        return response.json()
    
    async def predict(self, model_id: str, instances: List[Dict]) -> Dict[str, Any]:
        response = await self._client.post(
            f"/api/ml/models/{model_id}/predict",
            json={"instances": instances},
        )
        response.raise_for_status()
        return response.json()
    
    async def deploy_model(self, model_id: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        response = await self._client.post(
            f"/api/ml/models/{model_id}/deploy",
            json=config or {},
        )
        response.raise_for_status()
        return response.json()
    
    # ============ Feature Store ============
    
    async def create_feature_group(
        self, name: str, features: List[Dict], **kwargs
    ) -> Dict[str, Any]:
        response = await self._client.post(
            "/api/features/groups",
            params={"name": name, "features": features, **kwargs},
        )
        response.raise_for_status()
        return response.json()
    
    async def get_online_features(self, group: str, entity_ids: List[str]) -> Dict[str, Any]:
        response = await self._client.get(
            f"/api/features/groups/{group}/features",
            params={"entity_ids": entity_ids},
        )
        response.raise_for_status()
        return response.json()
    
    # ============ Inference ============
    
    async def create_endpoint(self, model_id: str, version: int, name: str) -> Dict[str, Any]:
        response = await self._client.post(
            "/api/inference/endpoints",
            params={"model_id": model_id, "model_version": version, "name": name},
        )
        response.raise_for_status()
        return response.json()
    
    async def inference_predict(self, endpoint: str, instances: List[Dict]) -> Dict[str, Any]:
        response = await self._client.post(
            f"/api/inference/endpoints/{endpoint}/predict",
            json={"instances": instances},
        )
        response.raise_for_status()
        return response.json()
    
    # ============ Reports ============
    
    async def generate_report(self, report_type: str, **kwargs) -> Dict[str, Any]:
        response = await self._client.post("/api/reports/generate", json={
            "report_type": report_type,
            **kwargs,
        })
        response.raise_for_status()
        return response.json()
    
    # ============ Monitoring ============
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        response = await self._client.get("/api/monitoring/system")
        response.raise_for_status()
        return response.json()
    
    async def get_metric_summary(self, metric_type: str, window: int = 3600) -> Dict[str, Any]:
        response = await self._client.get(
            f"/api/monitoring/metrics/{metric_type}",
            params={"window_seconds": window},
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        await self._client.aclose()