# app/data/feature_engineering/models.py
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATE = "date"
    TIME = "time"
    INTERACTION = "interaction"
    AGGREGATED = "aggregated"
    ROLLING = "rolling"
    STATISTICAL = "statistical"
    POLYNOMIAL = "polynomial"
    DOMAIN = "domain"


class EncodingType(str, Enum):
    LABEL = "label"
    ORDINAL = "ordinal"
    ONE_HOT = "one_hot"
    FREQUENCY = "frequency"
    TARGET = "target"
    HASH = "hash"
    BINARY = "binary"


class ScalingType(str, Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    LOG = "log"
    POWER = "power"
    QUANTILE = "quantile"
    NORMALIZATION = "normalization"


class SelectionMethod(str, Enum):
    VARIANCE_THRESHOLD = "variance_threshold"
    CORRELATION_THRESHOLD = "correlation_threshold"
    MUTUAL_INFORMATION = "mutual_information"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    RECURSIVE_ELIMINATION = "recursive_elimination"
    TREE_IMPORTANCE = "tree_importance"


class FeatureMetadata(BaseModel):
    """Complete metadata for a generated feature."""
    
    feature_name: str
    description: str
    formula: str
    data_type: str
    feature_type: FeatureType
    owner: str
    version: str
    created_date: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    dependencies: List[str] = Field(default_factory=list)
    source_columns: List[str] = Field(default_factory=list)
    transformation_history: List[Dict[str, Any]] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_active: bool = True
    checksum: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FeatureDefinition(BaseModel):
    """Definition for a feature to be generated."""
    
    name: str
    description: str
    feature_type: FeatureType
    source_columns: List[str]
    transformation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('source_columns')
    def validate_source_columns(cls, v):
        if not v:
            raise ValueError("Source columns cannot be empty")
        return v


class FeatureSet(BaseModel):
    """Collection of features with metadata."""
    
    dataset_name: str
    features: List[FeatureMetadata]
    generated_at: datetime = Field(default_factory=datetime.now)
    version: str
    schema: Dict[str, Any]
    statistics: Dict[str, Any]
    checksum: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FeatureStoreConfig(BaseModel):
    """Configuration for feature store operations."""
    
    offline_store_path: str
    online_store_path: Optional[str] = None
    metadata_path: str
    registry_path: str
    enable_versioning: bool = True
    enable_checksum: bool = True
    compression: Optional[str] = "gzip"
    partition_columns: List[str] = Field(default_factory=list)
    

# Add these fields to FeatureMetadata in models.py

class FeatureMetadata(BaseModel):
    """Complete metadata for a generated feature."""
    
    feature_name: str
    description: str
    formula: str
    data_type: str
    feature_type: FeatureType
    owner: str
    version: str
    created_date: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    dependencies: List[str] = Field(default_factory=list)
    source_columns: List[str] = Field(default_factory=list)
    transformation_history: List[Dict[str, Any]] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_active: bool = True
    checksum: Optional[str] = None
    
    # Additional fields for enhanced metadata tracking
    business_context: Optional[str] = None
    data_quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    usage_statistics: Dict[str, Any] = Field(default_factory=dict)
    retention_policy: Optional[Dict[str, Any]] = None
    security_classification: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }