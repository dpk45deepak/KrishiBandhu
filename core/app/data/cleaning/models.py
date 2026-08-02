"""Pydantic models for cleaning configuration and metadata."""

from typing import Dict, List, Optional, Any, Literal, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, ConfigDict
import pandas as pd


class MissingValueStrategy(str, Enum):
    """Strategies for handling missing values."""
    DROP_ROW = "drop_row"
    DROP_COLUMN = "drop_column"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    CUSTOM = "custom"


class OutlierStrategy(str, Enum):
    """Strategies for handling outliers."""
    IQR = "iqr"
    ZSCORE = "zscore"
    WINSORIZE = "winsorize"
    CLIP = "clip"
    REMOVE = "remove"
    KEEP = "keep"


class UnitConversion(BaseModel):
    """Unit conversion configuration."""
    from_unit: str
    to_unit: str
    conversion_factor: Optional[float] = None
    conversion_function: Optional[str] = None


class ColumnMapping(BaseModel):
    """Column name mapping configuration."""
    original: str
    target: str
    aliases: List[str] = Field(default_factory=list)


class CleaningStep(BaseModel):
    """Configuration for a single cleaning step."""
    enabled: bool = True
    strategy: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    columns: Optional[List[str]] = None
    priority: int = 0


class CleaningConfig(BaseModel):
    """Main cleaning configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # General settings
    dataset_name: Optional[str] = None
    enabled: bool = True
    preserve_original: bool = True
    
    # Column operations
    column_standardization: CleaningStep = Field(
        default_factory=lambda: CleaningStep(
            enabled=True,
            strategy="standardize",
            params={"case": "lower", "replace_spaces": "_"}
        )
    )
    column_aliases: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Missing values
    missing_values: Dict[str, CleaningStep] = Field(default_factory=dict)
    global_missing_strategy: Optional[MissingValueStrategy] = None
    missing_value_threshold: float = Field(default=0.5, ge=0, le=1)
    
    # Data types
    data_type_conversion: CleaningStep = Field(
        default_factory=lambda: CleaningStep(
            enabled=True,
            params={"infer_objects": True, "convert_numeric": True}
        )
    )
    
    # Text cleaning
    text_cleaning: CleaningStep = Field(
        default_factory=lambda: CleaningStep(
            enabled=True,
            params={
                "trim": True,
                "remove_whitespace": True,
                "case_normalization": "lower",
                "remove_special_chars": True,
                "remove_invalid_chars": True,
                "null_string_conversion": True,
            }
        )
    )
    
    # Duplicate handling
    duplicate_removal: CleaningStep = Field(
        default_factory=lambda: CleaningStep(
            enabled=True,
            strategy="remove_duplicates",
            params={"keep": "first", "subset": None}
        )
    )
    
    # Outlier handling
    outlier_handling: Dict[str, CleaningStep] = Field(default_factory=dict)
    global_outlier_strategy: Optional[OutlierStrategy] = None
    outlier_threshold: float = Field(default=3.0, ge=0)
    
    # Unit standardization
    unit_conversion: Dict[str, CleaningStep] = Field(default_factory=dict)
    unit_mappings: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)
    
    # Additional operations
    remove_empty_rows: CleaningStep = Field(
        default_factory=lambda: CleaningStep(
            enabled=True,
            params={"how": "all", "threshold": 0.5}
        )
    )
    remove_empty_columns: CleaningStep = Field(
        default_factory=lambda: CleaningStep(
            enabled=True,
            params={"threshold": 0.9}
        )
    )
    
    # Encoding fixes
    encoding_fixes: CleaningStep = Field(
        default_factory=lambda: CleaningStep(
            enabled=True,
            params={"encoding": "utf-8", "errors": "replace"}
        )
    )
    
    # Custom operations
    custom_operations: List[CleaningStep] = Field(default_factory=list)
    
    @validator('global_missing_strategy')
    def validate_missing_strategy(cls, v):
        if v and v not in MissingValueStrategy.__members__.values():
            raise ValueError(f"Invalid missing value strategy: {v}")
        return v
    
    @validator('global_outlier_strategy')
    def validate_outlier_strategy(cls, v):
        if v and v not in OutlierStrategy.__members__.values():
            raise ValueError(f"Invalid outlier strategy: {v}")
        return v
    
    def get_enabled_steps(self) -> List[CleaningStep]:
        """Get all enabled cleaning steps sorted by priority."""
        steps = []
        
        # Core cleaning steps with priorities
        core_steps = [
            (0, "column_standardization", self.column_standardization),
            (10, "data_type_conversion", self.data_type_conversion),
            (20, "text_cleaning", self.text_cleaning),
            (30, "duplicate_removal", self.duplicate_removal),
            (40, "remove_empty_rows", self.remove_empty_rows),
            (50, "remove_empty_columns", self.remove_empty_columns),
            (60, "encoding_fixes", self.encoding_fixes),
        ]
        
        for priority, name, step in core_steps:
            if step.enabled:
                step.priority = priority
                steps.append(step)
        
        # Missing value handling
        for col, step in self.missing_values.items():
            if step.enabled:
                step.priority = 100
                steps.append(step)
        
        # Outlier handling
        for col, step in self.outlier_handling.items():
            if step.enabled:
                step.priority = 110
                steps.append(step)
        
        # Unit conversion
        for col, step in self.unit_conversion.items():
            if step.enabled:
                step.priority = 120
                steps.append(step)
        
        # Custom operations
        for step in self.custom_operations:
            if step.enabled:
                steps.append(step)
        
        return sorted(steps, key=lambda x: x.priority)


class CleaningStatistics(BaseModel):
    """Statistics for a cleaning operation."""
    operation: str
    column: Optional[str] = None
    rows_before: int
    rows_after: int
    changes_made: int
    execution_time_ms: float
    details: Dict[str, Any] = Field(default_factory=dict)


class CleaningMetadata(BaseModel):
    """Complete metadata for a cleaning operation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    dataset_name: str
    source_path: str
    output_path: str
    
    rows_before: int
    rows_after: int
    columns_before: List[str]
    columns_after: List[str]
    
    operations: List[CleaningStatistics] = Field(default_factory=list)
    
    missing_values_fixed: Dict[str, int] = Field(default_factory=dict)
    duplicates_removed: int = 0
    outliers_handled: Dict[str, int] = Field(default_factory=dict)
    datatype_changes: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    
    start_time: datetime
    end_time: datetime
    execution_time_seconds: float
    
    config: CleaningConfig
    validation_status: Optional[str] = None
    
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)