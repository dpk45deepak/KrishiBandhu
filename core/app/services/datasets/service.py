# app/services/datasets/service.py
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import pandas as pd

from app.config import settings
from app.logger import get_logger
from app.services.datasets.models import (
    CleaningConfig,
    ColumnProfile,
    DatasetCreate,
    DatasetFormat,
    DatasetProfile,
    DatasetResponse,
    DatasetStatus,
    DatasetVersion,
    StandardizationConfig,
    ValidationIssue,
    ValidationReport,
    ValidationRule,
)
from app.utils.decorators import timed

logger = get_logger(__name__)


class DatasetService:
    """Dataset management service.
    
    Consumes existing modules:
    - data.Scanner: file scanning and format detection
    - data.Profiler: statistical profiling
    - data.Validator: data validation rules engine
    - data.Cleaner: data cleaning operations
    - data.Standardizer: data standardization
    - data.Versioning: dataset version management
    - config: storage paths and settings
    - logger: structured logging
    """
    
    def __init__(self):
        self._datasets: Dict[str, DatasetResponse] = {}
        self._versions: Dict[str, List[DatasetVersion]] = {}
        self._storage_path = Path(settings.DATA_STORAGE_PATH)
        self._storage_path.mkdir(parents=True, exist_ok=True)
    
    @timed
    async def create_dataset(
        self, dataset: DatasetCreate, user_id: str
    ) -> DatasetResponse:
        """Register a new dataset. Uses data.Scanner for initial format detection."""
        dataset_id = uuid4()
        
        response = DatasetResponse(
            id=dataset_id,
            name=dataset.name,
            description=dataset.description,
            format=dataset.format,
            status=DatasetStatus.UPLOADING,
            tags=dataset.tags,
            metadata=dataset.metadata,
            created_by=user_id,
        )
        
        self._datasets[str(dataset_id)] = response
        self._versions[str(dataset_id)] = []
        
        logger.info(f"Dataset created: {dataset.name} (id={dataset_id}) by user={user_id}")
        return response
    
    @timed
    async def upload_and_scan(
        self, dataset_id: str, file_content: bytes, filename: str
    ) -> DatasetVersion:
        """Upload dataset file and scan it using existing Scanner module."""
        from app.data import Scanner
        
        dataset = self._get_dataset(dataset_id)
        dataset.status = DatasetStatus.SCANNING
        
        # Save uploaded file
        file_path = self._get_version_path(dataset_id, filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(file_content)
        
        # Use existing Scanner module
        scanner = Scanner()
        scan_result = await scanner.scan(str(file_path))
        
        # Detect format if unknown
        if dataset.format == DatasetFormat.UNKNOWN:
            dataset.format = DatasetFormat(scan_result.detected_format)
        
        # Create version
        checksum = hashlib.sha256(file_content).hexdigest()
        version = DatasetVersion(
            version_id=uuid4(),
            dataset_id=UUID(dataset_id),
            version_number=len(self._versions[dataset_id]) + 1,
            file_path=str(file_path),
            row_count=scan_result.row_count,
            column_count=scan_result.column_count,
            size_bytes=scan_result.size_bytes,
            checksum=checksum,
        )
        
        self._versions[dataset_id].append(version)
        dataset.current_version = version
        
        logger.info(
            f"Dataset uploaded: {dataset.name} v{version.version_number} "
            f"({version.row_count} rows, {version.column_count} cols)"
        )
        
        return version
    
    @timed
    async def profile_dataset(self, dataset_id: str) -> DatasetProfile:
        """Generate statistical profile using existing Profiler module."""
        from app.data import Profiler
        
        dataset = self._get_dataset(dataset_id)
        dataset.status = DatasetStatus.PROFILING
        
        version = dataset.current_version
        if version is None:
            raise ValueError("No dataset version to profile")
        
        # Use existing Profiler module
        profiler = Profiler()
        profile_result = await profiler.profile(version.file_path)
        
        # Convert to our model format
        columns = []
        for col in profile_result.columns:
            columns.append(ColumnProfile(
                name=col.name,
                dtype=col.dtype,
                count=col.count,
                null_count=col.null_count,
                null_percentage=col.null_percentage,
                unique_count=col.unique_count,
                mean=col.mean,
                std=col.std,
                min=col.min,
                max=col.max,
                quantiles=col.quantiles or {},
                top_values=col.top_values or [],
                histogram=col.histogram,
            ))
        
        profile = DatasetProfile(
            row_count=profile_result.row_count,
            column_count=profile_result.column_count,
            total_size_bytes=profile_result.total_size_bytes,
            columns=columns,
            correlations=profile_result.correlations,
            missing_patterns=profile_result.missing_patterns,
        )
        
        dataset.profile = profile
        logger.info(f"Dataset profiled: {dataset.name} - {len(columns)} columns")
        
        return profile
    
    @timed
    async def validate_dataset(
        self, dataset_id: str, rules: Optional[List[ValidationRule]] = None
    ) -> ValidationReport:
        """Validate dataset using existing Validator module."""
        from app.data import Validator
        
        dataset = self._get_dataset(dataset_id)
        dataset.status = DatasetStatus.VALIDATING
        
        version = dataset.current_version
        if version is None:
            raise ValueError("No dataset version to validate")
        
        # Use existing Validator module
        validator = Validator()
        
        # Convert our rules to validator format
        validator_rules = []
        if rules:
            for rule in rules:
                validator_rules.append({
                    "column": rule.column,
                    "rule_type": rule.rule_type,
                    "params": rule.params,
                    "severity": rule.severity,
                })
        
        result = await validator.validate(version.file_path, validator_rules)
        
        # Convert to our model
        issues = []
        for issue in result.issues:
            issues.append(ValidationIssue(
                rule=ValidationRule(
                    column=issue.column,
                    rule_type=issue.rule_type,
                    params=issue.params,
                    severity=issue.severity,
                ),
                row_indices=issue.row_indices,
                affected_count=issue.affected_count,
                message=issue.message,
            ))
        
        report = ValidationReport(
            is_valid=result.is_valid,
            total_rows=result.total_rows,
            rules_count=len(validator_rules),
            issues=issues,
            error_count=sum(1 for i in issues if i.rule.severity == "error"),
            warning_count=sum(1 for i in issues if i.rule.severity == "warning"),
            summary=result.summary,
        )
        
        logger.info(
            f"Dataset validated: {dataset.name} - "
            f"{'valid' if report.is_valid else 'invalid'} "
            f"({report.error_count} errors, {report.warning_count} warnings)"
        )
        
        return report
    
    @timed
    async def clean_dataset(
        self, dataset_id: str, config: CleaningConfig
    ) -> DatasetVersion:
        """Clean dataset using existing Cleaner module. Creates new version."""
        from app.data import Cleaner
        
        dataset = self._get_dataset(dataset_id)
        dataset.status = DatasetStatus.CLEANING
        
        version = dataset.current_version
        if version is None:
            raise ValueError("No dataset version to clean")
        
        # Use existing Cleaner module
        cleaner = Cleaner()
        cleaned_path = await cleaner.clean(
            version.file_path,
            null_strategy=config.handle_nulls,
            drop_duplicates=config.drop_duplicates,
            outlier_method=config.outlier_method,
            outlier_columns=config.outlier_columns,
            text_cleaning=config.text_cleaning,
            custom_transforms=config.custom_transforms,
        )
        
        # Create new version with cleaned data
        new_version = DatasetVersion(
            version_id=uuid4(),
            dataset_id=UUID(dataset_id),
            version_number=len(self._versions[dataset_id]) + 1,
            file_path=cleaned_path,
            row_count=version.row_count,  # Will be updated on next scan
            column_count=version.column_count,
            size_bytes=os.path.getsize(cleaned_path),
            checksum=hashlib.sha256(Path(cleaned_path).read_bytes()).hexdigest(),
            parent_version=version.version_id,
            changelog=f"Cleaned with config: {config}",
        )
        
        self._versions[dataset_id].append(new_version)
        dataset.current_version = new_version
        dataset.status = DatasetStatus.READY
        
        logger.info(f"Dataset cleaned: {dataset.name} -> v{new_version.version_number}")
        return new_version
    
    @timed
    async def standardize_dataset(
        self, dataset_id: str, config: StandardizationConfig
    ) -> DatasetVersion:
        """Standardize dataset using existing Standardizer module."""
        from app.data import Standardizer
        
        dataset = self._get_dataset(dataset_id)
        version = dataset.current_version
        if version is None:
            raise ValueError("No dataset version to standardize")
        
        # Use existing Standardizer module
        standardizer = Standardizer()
        standardized_path = await standardizer.standardize(
            version.file_path,
            date_columns=config.date_columns,
            numeric_scaling=config.numeric_scaling,
            categorical_encoding=config.categorical_encoding,
            text_normalization=config.text_normalization,
            coordinate_system=config.coordinate_system,
        )
        
        new_version = DatasetVersion(
            version_id=uuid4(),
            dataset_id=UUID(dataset_id),
            version_number=len(self._versions[dataset_id]) + 1,
            file_path=standardized_path,
            row_count=version.row_count,
            column_count=version.column_count,
            size_bytes=os.path.getsize(standardized_path),
            checksum=hashlib.sha256(Path(standardized_path).read_bytes()).hexdigest(),
            parent_version=version.version_id,
            changelog=f"Standardized with config: {config}",
        )
        
        self._versions[dataset_id].append(new_version)
        dataset.current_version = new_version
        
        logger.info(f"Dataset standardized: {dataset.name} -> v{new_version.version_number}")
        return new_version
    
    async def get_dataset(self, dataset_id: str) -> Optional[DatasetResponse]:
        """Get dataset by ID."""
        return self._datasets.get(dataset_id)
    
    async def list_datasets(
        self, status: Optional[DatasetStatus] = None, tags: Optional[List[str]] = None
    ) -> List[DatasetResponse]:
        """List datasets with optional filtering."""
        results = list(self._datasets.values())
        
        if status:
            results = [d for d in results if d.status == status]
        
        if tags:
            results = [d for d in results if any(t in d.tags for t in tags)]
        
        return results
    
    async def get_versions(self, dataset_id: str) -> List[DatasetVersion]:
        """Get all versions of a dataset."""
        self._get_dataset(dataset_id)
        return self._versions.get(dataset_id, [])
    
    async def get_version(self, dataset_id: str, version_number: int) -> Optional[DatasetVersion]:
        """Get a specific version."""
        versions = self._versions.get(dataset_id, [])
        for v in versions:
            if v.version_number == version_number:
                return v
        return None
    
    async def download_dataset(self, dataset_id: str, version_number: Optional[int] = None) -> Tuple[bytes, str]:
        """Download dataset file."""
        dataset = self._get_dataset(dataset_id)
        
        if version_number:
            version = await self.get_version(dataset_id, version_number)
        else:
            version = dataset.current_version
        
        if version is None:
            raise ValueError("No version available")
        
        file_path = Path(version.file_path)
        content = file_path.read_bytes()
        filename = f"{dataset.name}_v{version.version_number}.{dataset.format.value}"
        
        return content, filename
    
    @timed
    async def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset and all versions."""
        if dataset_id not in self._datasets:
            return False
        
        dataset = self._datasets[dataset_id]
        
        # Remove version files
        for version in self._versions.get(dataset_id, []):
            try:
                Path(version.file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete version file: {e}")
        
        del self._datasets[dataset_id]
        self._versions.pop(dataset_id, None)
        
        logger.info(f"Dataset deleted: {dataset.name}")
        return True
    
    def _get_dataset(self, dataset_id: str) -> DatasetResponse:
        """Get dataset or raise."""
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset not found: {dataset_id}")
        return dataset
    
    def _get_version_path(self, dataset_id: str, filename: str) -> Path:
        """Get storage path for a dataset version."""
        return self._storage_path / dataset_id / filename