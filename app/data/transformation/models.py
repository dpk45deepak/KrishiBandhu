"""Pydantic models for schema definitions and metadata."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import hashlib


class DataType(str, Enum):
    """Supported data types in the standardized schema."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    CATEGORY = "category"
    JSON = "json"


class UnitCategory(str, Enum):
    """Unit categories for conversion tracking."""
    TEMPERATURE = "temperature"
    LENGTH = "length"
    AREA = "area"
    WEIGHT = "weight"
    VOLUME = "volume"
    NONE = "none"


class ColumnDefinition(BaseModel):
    """Definition of a column in the schema."""
    name: str
    data_type: DataType
    description: Optional[str] = None
    unit: Optional[str] = None
    unit_category: Optional[UnitCategory] = None
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    is_primary_key: bool = False
    is_target: bool = False
    required: bool = True
    default: Optional[Any] = None
    
    class Config:
        use_enum_values = True


class SchemaMetadata(BaseModel):
    """Complete schema definition for a dataset."""
    schema_name: str
    schema_version: str
    description: str
    source: str
    owner: str
    license: str
    columns: List[ColumnDefinition]
    primary_key: Optional[List[str]] = None
    target_column: Optional[str] = None
    update_frequency: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    @validator('schema_version')
    def validate_version(cls, v):
        """Validate semantic versioning format."""
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError("Schema version must follow semantic versioning (X.Y.Z)")
        return v


class DatasetRegistry(BaseModel):
    """Registry entry for a processed dataset."""
    dataset_id: str
    source: str
    schema_name: str
    schema_version: str
    file_path: str
    checksum: str
    rows: int
    columns: int
    quality_score: float
    missing_percentage: float
    owner: str
    license: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: Optional[datetime] = None
    metadata_path: Optional[str] = None
    
    def compute_checksum(self, content: bytes) -> str:
        """Compute SHA-256 checksum of content."""
        return hashlib.sha256(content).hexdigest()


class StandardizationReport(BaseModel):
    """Detailed report of standardization operations."""
    source_file: str
    output_file: str
    schema_name: str
    schema_version: str
    total_rows: int
    total_columns: int
    
    column_mappings: Dict[str, str] = Field(default_factory=dict)
    unit_conversions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    category_mappings: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    text_normalizations: Dict[str, int] = Field(default_factory=dict)
    date_standardizations: Dict[str, int] = Field(default_factory=dict)
    
    missing_values_handled: int = 0
    transformations_applied: List[str] = Field(default_factory=list)
    errors_encountered: List[str] = Field(default_factory=list)
    
    processing_time: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ColumnMappingRule(BaseModel):
    """Rule for mapping source column to standardized name."""
    source_pattern: str
    target_name: str
    priority: int = 1
    is_regex: bool = False
    case_sensitive: bool = False


class ConversionRule(BaseModel):
    """Rule for unit or category conversion."""
    source_value: Any
    target_value: Any
    source_unit: Optional[str] = None
    target_unit: Optional[str] = None
    conversion_factor: Optional[float] = None
    conversion_function: Optional[str] = None


class TransformationConfig(BaseModel):
    """Configuration for a transformation step."""
    step_name: str
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    order: int = 0