# app/data/eda/models.py
"""
Pydantic models for EDA platform.
Enforces type safety and data validation throughout the pipeline.
"""
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import polars as pl
import numpy as np


class DataType(str, Enum):
    """Data type classifications"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    GEOSPATIAL = "geospatial"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class MissingnessPattern(BaseModel):
    """Missing data pattern analysis"""
    column: str
    missing_count: int
    missing_percentage: float
    missing_type: str  # 'MCAR', 'MAR', 'MNAR'
    pattern_with: Optional[List[str]] = None


class StatisticalSummary(BaseModel):
    """Comprehensive statistical summary"""
    count: int
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Union[int, float, str]] = None
    variance: Optional[float] = None
    std_dev: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    unique_count: Optional[int] = None
    null_count: Optional[int] = None
    null_percentage: Optional[float] = None


class CorrelationMatrix(BaseModel):
    """Correlation analysis results"""
    columns: List[str]
    correlation_values: List[List[float]]
    correlation_type: str  # 'pearson', 'spearman', 'kendall'
    p_values: Optional[List[List[float]]] = None
    significant_correlations: List[Dict[str, Any]]


class PCAInfo(BaseModel):
    """PCA analysis results"""
    explained_variance: List[float]
    cumulative_variance: List[float]
    components: List[List[float]]
    n_components: int
    feature_contributions: Dict[str, List[float]]


class ClusterInfo(BaseModel):
    """Clustering analysis results"""
    n_clusters: int
    cluster_sizes: List[int]
    silhouette_score: float
    cluster_stats: Dict[str, Any]
    optimal_clusters: Optional[int] = None


class FeatureQualityScore(BaseModel):
    """Quality assessment for each feature"""
    column: str
    data_type: DataType
    completeness_score: float  # 0-1
    uniqueness_score: float  # 0-1
    consistency_score: float  # 0-1
    relevance_score: float  # 0-1
    overall_score: float  # 0-1
    issues: List[str]
    recommendations: List[str]


class MLReadinessReport(BaseModel):
    """Machine Learning readiness assessment"""
    is_classification_ready: bool
    is_regression_ready: bool
    target_candidates: List[str]
    feature_count: int
    feature_types: Dict[DataType, int]
    sample_size: int
    missing_handling_needed: bool
    scaling_needed: bool
    encoding_needed: bool
    outlier_presence: bool
    class_balance: Optional[Dict[str, float]]
    recommendations: List[str]


class EDAReport(BaseModel):
    """Master EDA report container"""
    dataset_name: str
    generation_timestamp: datetime
    dataset_overview: Dict[str, Any]
    feature_summaries: Dict[str, StatisticalSummary]
    quality_scores: Dict[str, FeatureQualityScore]
    correlation_matrix: Optional[CorrelationMatrix] = None
    pca_results: Optional[PCAInfo] = None
    clustering_results: Optional[ClusterInfo] = None
    missingness_patterns: List[MissingnessPattern]
    outliers_detected: Dict[str, int]
    ml_readiness: MLReadinessReport
    visualizations: Dict[str, str]  # Paths to generated figures
    recommendations: List[str]
    executive_summary: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class VisualizationConfig(BaseModel):
    """Configuration for visualizations"""
    fig_width: int = 800
    fig_height: int = 600
    template: str = "plotly_white"
    color_continuous_scale: str = "Viridis"
    color_discrete_sequence: List[str] = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", 
        "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"
    ]
    font_size: int = 12
    title_font_size: int = 16
    show_grid: bool = True
    save_formats: List[str] = ["png", "svg", "html"]
    
    @validator('fig_width', 'fig_height')
    def validate_dimensions(cls, v):
        if v < 100:
            raise ValueError("Dimensions must be at least 100 pixels")
        return v


class EDAAnalysisConfig(BaseModel):
    """Master EDA configuration"""
    enable_statistics: bool = True
    enable_correlation: bool = True
    enable_pca: bool = True
    enable_clustering: bool = True
    enable_geospatial: bool = True
    enable_quality_analysis: bool = True
    
    max_categories_to_plot: int = 20
    correlation_threshold: float = 0.5
    pca_components: int = 5
    max_clusters: int = 10
    outlier_threshold: float = 1.5  # IQR multiplier
    
    visualization: VisualizationConfig = VisualizationConfig()
    numerical_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    geospatial_features: Optional[List[str]] = None
    target_feature: Optional[str] = None
    ignore_features: List[str] = Field(default_factory=list)