# app/services/ml/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    TUNING = "tuning"
    READY = "ready"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    target_column: str
    feature_columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cv_folds: int = 5
    metric: str = "auto"  # "auto", "accuracy", "f1", "rmse", "mae", "r2"
    algorithm: Optional[str] = None  # Auto-select if not specified
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    class_weight: Optional[str] = None  # "balanced" for classification
    early_stopping: bool = False
    feature_selection: bool = False
    preprocessing: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """A training job instance."""
    id: UUID
    model_id: UUID
    model_name: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.QUEUED
    dataset_path: Optional[str] = None
    feature_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    model_path: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EvaluationReport:
    """Model evaluation results from existing ml.evaluation."""
    model_id: UUID
    model_version: int
    metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    roc_auc: Optional[float] = None
    precision_recall: Optional[Dict[str, float]] = None
    residual_plot: Optional[str] = None  # Path to plot
    feature_importance: Optional[Dict[str, float]] = None
    cross_validation_scores: Optional[List[float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    overfit_analysis: Optional[Dict[str, Any]] = None
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TuningConfig:
    """Hyperparameter tuning configuration."""
    param_grid: Dict[str, List[Any]]
    tuning_method: str = "grid"  # "grid", "random", "bayesian"
    n_iter: int = 100
    cv_folds: int = 5
    metric: str = "auto"
    early_stopping_rounds: Optional[int] = None
    n_jobs: int = -1


@dataclass
class TuningResult:
    """Hyperparameter tuning results from existing ml.tuning."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    tuning_time_seconds: float
    param_importance: Optional[Dict[str, float]] = None
    convergence_plot: Optional[str] = None
    tuned_model_path: Optional[str] = None


@dataclass
class PredictionRequest:
    """Request for model prediction."""
    data: List[Dict[str, Any]]  # List of feature dictionaries
    return_probabilities: bool = False
    threshold: Optional[float] = None  # For classification
    explain: bool = False


@dataclass
class PredictionResponse:
    """Model prediction response."""
    predictions: List[Any]
    probabilities: Optional[List[List[float]]] = None
    model_id: UUID
    model_version: int
    prediction_time_ms: float
    explanations: Optional[List[Dict[str, Any]]] = None


@dataclass
class ExplainabilityReport:
    """Model explainability from existing ml.explainability."""
    feature_importance: Dict[str, float]
    shap_values: Optional[Dict[str, Any]] = None
    lime_explanations: Optional[List[Dict[str, Any]]] = None
    partial_dependence: Optional[Dict[str, Any]] = None
    global_importance_plot: Optional[str] = None
    summary_plot: Optional[str] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModelVersion:
    """A specific version of a model."""
    version: int
    model_path: str
    metrics: Dict[str, float]
    training_config: TrainingConfig
    artifacts: Dict[str, str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deployed: bool = False
    deployment_info: Optional[Dict[str, Any]] = None


@dataclass
class ModelCreate:
    """Request to create/register a model."""
    name: str
    model_type: ModelType
    description: str = ""
    tags: List[str] = field(default_factory=list)
    framework: str = "auto"  # "sklearn", "xgboost", "lightgbm", "pytorch", "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Model information returned by API."""
    id: UUID
    name: str
    model_type: ModelType
    description: str
    status: ModelStatus
    tags: List[str]
    framework: str
    metadata: Dict[str, Any]
    current_version: Optional[ModelVersion] = None
    versions: List[ModelVersion] = field(default_factory=list)
    training_jobs: List[TrainingJob] = field(default_factory=list)
    evaluation_report: Optional[EvaluationReport] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None


@dataclass
class ModelRegistryEntry:
    """Summary entry for model registry."""
    id: UUID
    name: str
    model_type: ModelType
    status: ModelStatus
    current_version: Optional[int] = None
    latest_metrics: Optional[Dict[str, float]] = None
    deployed: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))