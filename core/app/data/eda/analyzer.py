# app/data/eda/analyzer.py
"""
Core analysis pipeline orchestrator.
Coordinates all analysis modules and manages the EDA workflow.
"""
import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
from loguru import logger
from app.data.eda.models import (
    EDAReport, DataType, FeatureQualityScore, 
    MLReadinessReport, EDAAnalysisConfig
)
from app.data.eda.statistics import StatisticsEngine
from app.data.eda.correlation import CorrelationEngine
from app.data.eda.visualizer import VisualizationEngine
from app.data.eda.distributions import DistributionAnalyzer
from app.data.eda.categorical import CategoricalAnalyzer
from app.data.eda.pca import PCAAnalyzer
from app.data.eda.clustering import ClusteringAnalyzer
from app.data.eda.geospatial import GeospatialAnalyzer
from app.data.eda.quality import QualityAnalyzer


class EDAAnalyzer:
    """Main EDA analyzer orchestrator"""
    
    def __init__(
        self,
        df: pl.DataFrame,
        config: Optional[EDAAnalysisConfig] = None,
        dataset_name: Optional[str] = None
    ):
        self.df = df
        self.config = config or EDAAnalysisConfig()
        self.dataset_name = dataset_name or "unnamed_dataset"
        self.logger = logger.bind(module="eda_analyzer")
        
        # Initialize engines
        self.stats_engine = StatisticsEngine(df, self.config.dict())
        self.corr_engine = CorrelationEngine(df, self.config.dict())
        self.viz_engine = VisualizationEngine(self.config.visualization)
        self.dist_analyzer = DistributionAnalyzer(df, self.config.dict())
        self.cat_analyzer = CategoricalAnalyzer(df, self.config.dict())
        self.pca_analyzer = PCAAnalyzer(df, self.config.dict())
        self.cluster_analyzer = ClusteringAnalyzer(df, self.config.dict())
        self.geo_analyzer = GeospatialAnalyzer(df, self.config.dict())
        self.quality_analyzer = QualityAnalyzer(df, self.config.dict())
        
        # Cache
        self._analysis_cache = {}
        self._figure_cache = {}
        
    def analyze(self) -> EDAReport:
        """
        Run complete EDA analysis pipeline.
        
        Returns:
            Complete EDAReport
        """
        self.logger.info(f"Starting EDA for dataset: {self.dataset_name}")
        start_time = datetime.now()
        
        try:
            # 1. Dataset Overview
            overview = self._analyze_overview()
            self._analysis_cache['overview'] = overview
            
            # 2. Feature Summaries
            feature_summaries = self._analyze_features()
            self._analysis_cache['feature_summaries'] = feature_summaries
            
            # 3. Quality Analysis
            quality_scores = self._analyze_quality()
            self._analysis_cache['quality_scores'] = quality_scores
            
            # 4. Missingness Patterns
            missing_patterns = self._analyze_missingness()
            self._analysis_cache['missing_patterns'] = missing_patterns
            
            # 5. Correlation Analysis
            correlation = self._analyze_correlation()
            self._analysis_cache['correlation'] = correlation
            
            # 6. Advanced Analysis (PCA, Clustering)
            pca_results = self._analyze_pca() if self.config.enable_pca else None
            clustering_results = self._analyze_clustering() if self.config.enable_clustering else None
            
            # 7. ML Readiness
            ml_readiness = self._analyze_ml_readiness(feature_summaries)
            
            # 8. Outliers Detection
            outliers = self._detect_outliers()
            
            # 9. Generate Recommendations
            recommendations = self._generate_recommendations(
                quality_scores, missing_patterns, correlation, ml_readiness
            )
            
            # 10. Executive Summary
            executive_summary = self._generate_executive_summary(
                overview, feature_summaries, ml_readiness, recommendations
            )
            
            # Build report
            report = EDAReport(
                dataset_name=self.dataset_name,
                generation_timestamp=datetime.now(),
                dataset_overview=overview,
                feature_summaries=feature_summaries,
                quality_scores=quality_scores,
                correlation_matrix=correlation,
                pca_results=pca_results,
                clustering_results=clustering_results,
                missingness_patterns=missing_patterns,
                outliers_detected=outliers,
                ml_readiness=ml_readiness,
                visualizations={},  # Will be filled during report generation
                recommendations=recommendations,
                executive_summary=executive_summary
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"EDA completed in {elapsed:.2f} seconds")
            
            return report
            
        except Exception as e:
            self.logger.error(f"EDA analysis failed: {e}")
            raise
    
    def _analyze_overview(self) -> Dict[str, Any]:
        """Analyze dataset overview."""
        return {
            "num_rows": self.df.height,
            "num_columns": self.df.width,
            "num_numerical": len(self._get_numerical_columns()),
            "num_categorical": len(self._get_categorical_columns()),
            "num_datetime": len(self._get_datetime_columns()),
            "memory_usage": self.df.estimated_size('mb'),
            "duplicate_rows": self.df.select(pl.all()).is_duplicated().sum(),
            "total_missing": self.df.select(pl.all().is_null().sum().sum()).item()
        }
    
    def _analyze_features(self) -> Dict[str, Any]:
        """Analyze all features."""
        summaries = {}
        
        for col in self.df.columns:
            dtype = self._infer_data_type(col)
            summary = self.stats_engine.compute_summary_statistics(col, dtype)
            summaries[col] = summary
            
        return summaries
    
    def _analyze_quality(self) -> Dict[str, FeatureQualityScore]:
        """Analyze data quality."""
        return self.quality_analyzer.analyze_quality()
    
    def _analyze_missingness(self):
        """Analyze missing data patterns."""
        return self.stats_engine.detect_missingness_patterns(self.df.columns)
    
    def _analyze_correlation(self):
        """Analyze correlations."""
        numerical_cols = self._get_numerical_columns()
        if len(numerical_cols) > 1:
            return self.corr_engine.compute_correlation_matrix(numerical_cols)
        return None
    
    def _analyze_pca(self):
        """Perform PCA analysis."""
        numerical_cols = self._get_numerical_columns()
        if len(numerical_cols) >= 3:
            return self.pca_analyzer.analyze_pca(numerical_cols)
        return None
    
    def _analyze_clustering(self):
        """Perform clustering analysis."""
        numerical_cols = self._get_numerical_columns()
        if len(numerical_cols) >= 2:
            return self.cluster_analyzer.analyze_clustering(numerical_cols)
        return None
    
    def _detect_outliers(self) -> Dict[str, int]:
        """Detect outliers in numerical columns."""
        outliers = {}
        for col in self._get_numerical_columns():
            _, count = self.stats_engine.detect_outliers_iqr(col)
            if count > 0:
                outliers[col] = count
        return outliers
    
    def _analyze_ml_readiness(self, feature_summaries) -> MLReadinessReport:
        """Assess ML readiness."""
        return self.quality_analyzer.assess_ml_readiness(feature_summaries)
    
    def _generate_recommendations(self, quality_scores, missing_patterns, 
                                  correlation, ml_readiness) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Quality recommendations
        for col, score in quality_scores.items():
            if score.overall_score < 0.6:
                recommendations.append(f"Consider improving quality of '{col}': {', '.join(score.recommendations)}")
                
        # Missing data recommendations
        for pattern in missing_patterns:
            if pattern.missing_percentage > 20:
                recommendations.append(
                    f"High missingness ({pattern.missing_percentage:.1f}%) in '{pattern.column}'. "
                    f"Consider imputation or removal."
                )
                
        # Correlation recommendations
        if correlation and correlation.significant_correlations:
            high_corr = [c for c in correlation.significant_correlations 
                        if abs(c['correlation']) > 0.8]
            if high_corr:
                features = set()
                for corr in high_corr[:3]:
                    features.add(corr['feature1'])
                    features.add(corr['feature2'])
                recommendations.append(
                    f"High correlation detected between features: {', '.join(features)}. "
                    f"Consider feature selection to reduce multicollinearity."
                )
                
        # ML readiness recommendations
        if ml_readiness.recommendations:
            recommendations.extend(ml_readiness.recommendations)
            
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_executive_summary(self, overview, feature_summaries, 
                                    ml_readiness, recommendations) -> str:
        """Generate executive summary."""
        summary_parts = []
        
        # Dataset overview
        summary_parts.append(
            f"Dataset '{self.dataset_name}' contains {overview['num_rows']} rows "
            f"and {overview['num_columns']} columns."
        )
        
        # Data quality
        avg_quality = np.mean([score.overall_score for score in feature_summaries.values()])
        summary_parts.append(
            f"Overall data quality score: {avg_quality:.2f}/1.0. "
            f"{'Good' if avg_quality > 0.7 else 'Needs improvement'}."
        )
        
        # ML readiness
        readiness = "Ready" if ml_readiness.is_regression_ready or ml_readiness.is_classification_ready else "Not ready"
        summary_parts.append(
            f"ML readiness assessment: {readiness}."
        )
        
        # Key recommendations
        if recommendations:
            summary_parts.append(f"Key recommendations: {recommendations[0]}")
            if len(recommendations) > 1:
                summary_parts.append(f"Also consider: {recommendations[1]}")
                
        return " ".join(summary_parts)
    
    def _infer_data_type(self, column: str) -> DataType:
        """Infer data type of a column."""
        series = self.df[column]
        
        # Check datetime
        if pl.Series.is_dtype(series.dtype, pl.Datetime):
            return DataType.DATETIME
            
        # Check numerical
        if pl.Series.is_dtype(series.dtype, [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]):
            if series.n_unique() <= 2:
                return DataType.BOOLEAN
            return DataType.NUMERICAL
            
        # Check categorical
        if series.n_unique() / len(series) < 0.5:
            return DataType.CATEGORICAL
            
        # Check geospatial (simplified)
        if column.lower() in ['latitude', 'longitude', 'lat', 'lon']:
            return DataType.GEOSPATIAL
            
        return DataType.UNKNOWN
    
    def _get_numerical_columns(self) -> List[str]:
        """Get numerical columns."""
        return [col for col in self.df.columns 
                if self._infer_data_type(col) == DataType.NUMERICAL]
    
    def _get_categorical_columns(self) -> List[str]:
        """Get categorical columns."""
        return [col for col in self.df.columns 
                if self._infer_data_type(col) in [DataType.CATEGORICAL, DataType.BOOLEAN]]
    
    def _get_datetime_columns(self) -> List[str]:
        """Get datetime columns."""
        return [col for col in self.df.columns 
                if self._infer_data_type(col) == DataType.DATETIME]