# app/services/pipeline/executors.py
"""
Pre-built stage executors that connect to other services.

Each executor consumes existing modules and services.
These are registered with PipelineEngine on import.
"""
from typing import Any, Dict

from app.logger import get_logger
from app.services.pipeline.engine import PipelineEngine
from app.services.pipeline.models import PipelineRun, PipelineStage, StageType

logger = get_logger(__name__)


async def execute_scan(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Scan stage - uses existing data.Scanner."""
    from app.data import Scanner
    
    file_path = context.get("file_path") or stage.config.params.get("file_path")
    scanner = Scanner()
    result = await scanner.scan(file_path)
    
    stage.metrics = {
        "row_count": result.row_count,
        "column_count": result.column_count,
        "size_bytes": result.size_bytes,
        "format": result.detected_format,
    }
    
    return {"scan_result": result, "row_count": result.row_count}


async def execute_profile(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Profile stage - uses dataset service's profiler integration."""
    from app.data import Profiler
    
    file_path = context.get("file_path") or stage.config.params.get("file_path")
    profiler = Profiler()
    result = await profiler.profile(file_path)
    
    stage.metrics = {
        "row_count": result.row_count,
        "column_count": result.column_count,
        "missing_percentage": sum(
            c.null_percentage for c in result.columns
        ) / max(len(result.columns), 1),
    }
    
    return {"profile_result": result}


async def execute_validate(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Validate stage - uses dataset service's validator integration."""
    from app.data import Validator
    
    file_path = context.get("file_path") or stage.config.params.get("file_path")
    rules = stage.config.params.get("rules", [])
    
    validator = Validator()
    result = await validator.validate(file_path, rules)
    
    stage.metrics = {
        "is_valid": result.is_valid,
        "error_count": sum(1 for i in result.issues if i.severity == "error"),
        "warning_count": sum(1 for i in result.issues if i.severity == "warning"),
    }
    
    return {"validation_result": result}


async def execute_clean(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Clean stage - uses dataset service's cleaner integration."""
    from app.data import Cleaner
    
    file_path = context.get("file_path") or stage.config.params.get("file_path")
    config = stage.config.params.get("cleaning_config", {})
    
    cleaner = Cleaner()
    cleaned_path = await cleaner.clean(file_path, **config)
    
    stage.artifacts["cleaned_data"] = cleaned_path
    
    return {"file_path": cleaned_path, "cleaned_path": cleaned_path}


async def execute_feature_engineer(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Feature engineering stage - uses existing data.FeatureEngineering."""
    from app.data import FeatureEngineering
    
    file_path = context.get("file_path")
    config = stage.config.params.get("feature_config", {})
    
    fe = FeatureEngineering()
    result = await fe.engineer(file_path, **config)
    
    stage.metrics = {
        "original_features": result.original_count,
        "engineered_features": result.engineered_count,
    }
    stage.artifacts["feature_matrix"] = result.output_path
    
    return {
        "feature_matrix_path": result.output_path,
        "feature_names": result.feature_names,
        "file_path": result.output_path,
    }


async def execute_feature_store(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Feature store stage - will connect to feature_store service."""
    # Will be fully integrated when we build feature_store service
    logger.info(f"Feature store stage: storing features from {context.get('feature_matrix_path')}")
    
    stage.metrics = {"features_stored": 0}
    stage.artifacts["feature_group"] = "pending"
    
    return {"feature_group": "pending"}


async def execute_train(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Train stage - uses existing ml.classification/regression modules."""
    from app.ml import MLPlatform
    
    ml_config = stage.config.params.get("ml_config", {})
    feature_path = context.get("feature_matrix_path") or context.get("file_path")
    target_column = ml_config.get("target_column")
    model_type = ml_config.get("model_type", "classification")
    
    platform = MLPlatform()
    
    if model_type == "classification":
        result = await platform.classification.train(feature_path, target_column, **ml_config)
    else:
        result = await platform.regression.train(feature_path, target_column, **ml_config)
    
    stage.metrics = result.metrics
    stage.artifacts["model"] = result.model_path
    stage.artifacts["model_metadata"] = result.metadata_path
    
    return {
        "model_path": result.model_path,
        "model_id": result.model_id,
        "metrics": result.metrics,
    }


async def execute_evaluate(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Evaluate stage - uses existing ml.evaluation."""
    from app.ml import MLPlatform
    
    model_path = context.get("model_path")
    test_data = context.get("file_path")
    target_column = stage.config.params.get("target_column")
    
    platform = MLPlatform()
    result = await platform.evaluation.evaluate(model_path, test_data, target_column)
    
    stage.metrics = result.metrics
    
    return {"evaluation_metrics": result.metrics}


async def execute_tune(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Hyperparameter tuning stage - uses existing ml.tuning."""
    from app.ml import MLPlatform
    
    model_path = context.get("model_path")
    param_grid = stage.config.params.get("param_grid", {})
    
    platform = MLPlatform()
    result = await platform.tuning.tune(model_path, param_grid)
    
    stage.metrics = {"best_params": result.best_params, "best_score": result.best_score}
    stage.artifacts["tuned_model"] = result.tuned_model_path
    
    return {
        "model_path": result.tuned_model_path,
        "best_params": result.best_params,
    }


async def execute_explain(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Explainability stage - uses existing ml.explainability."""
    from app.ml import MLPlatform
    
    model_path = context.get("model_path")
    data_path = context.get("file_path")
    
    platform = MLPlatform()
    result = await platform.explainability.explain(model_path, data_path)
    
    stage.artifacts["explanation"] = result.output_path
    stage.metrics = {"feature_importance": result.feature_importance}
    
    return {"explanation_path": result.output_path}


async def execute_report(stage: PipelineStage, context: Dict[str, Any], run: PipelineRun) -> Dict[str, Any]:
    """Report generation stage - will connect to reports service."""
    # Will integrate with reports service
    logger.info(f"Report stage: generating report with context keys: {list(context.keys())}")
    
    stage.artifacts["report"] = "pending"
    
    return {"report_path": "pending"}


# Register all executors
def register_executors():
    """Register all stage executors with the PipelineEngine."""
    PipelineEngine.register_executor(StageType.SCAN, execute_scan)
    PipelineEngine.register_executor(StageType.PROFILE, execute_profile)
    PipelineEngine.register_executor(StageType.VALIDATE, execute_validate)
    PipelineEngine.register_executor(StageType.CLEAN, execute_clean)
    PipelineEngine.register_executor(StageType.STANDARDIZE, execute_clean)  # Reuse for now
    PipelineEngine.register_executor(StageType.FEATURE_ENGINEER, execute_feature_engineer)
    PipelineEngine.register_executor(StageType.FEATURE_STORE, execute_feature_store)
    PipelineEngine.register_executor(StageType.TRAIN, execute_train)
    PipelineEngine.register_executor(StageType.EVALUATE, execute_evaluate)
    PipelineEngine.register_executor(StageType.TUNE, execute_tune)
    PipelineEngine.register_executor(StageType.EXPLAIN, execute_explain)
    PipelineEngine.register_executor(StageType.REPORT, execute_report)
    
    logger.info("Pipeline executors registered successfully")


# Auto-register on import
register_executors()