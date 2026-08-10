# app/data/eda/statistics.py
"""
Statistical analysis module.
Computes descriptive statistics, tests, and distribution metrics.
"""
import polars as pl
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from scipy.stats import skew, kurtosis, normaltest
from loguru import logger
from app.data.eda.models import StatisticalSummary, DataType, MissingnessPattern


class StatisticsEngine:
    """Core statistical computation engine"""
    
    def __init__(self, df: pl.DataFrame, config: Dict[str, Any]):
        self.df = df
        self.config = config
        self.logger = logger.bind(module="statistics")
        
    def compute_summary_statistics(
        self, 
        column: str, 
        dtype: DataType
    ) -> StatisticalSummary:
        """
        Compute comprehensive statistics for a column.
        
        Args:
            column: Column name
            dtype: Data type classification
            
        Returns:
            StatisticalSummary with computed values
        """
        try:
            series = self.df[column]
            null_mask = series.is_null()
            non_null = series.filter(~null_mask)
            
            if len(non_null) == 0:
                return StatisticalSummary(
                    count=len(series),
                    null_count=null_mask.sum(),
                    null_percentage=100.0,
                    unique_count=0
                )
            
            summary = StatisticalSummary(
                count=len(series),
                null_count=null_mask.sum(),
                null_percentage=(null_mask.sum() / len(series)) * 100,
                unique_count=non_null.n_unique()
            )
            
            if dtype == DataType.NUMERICAL:
                numeric_data = non_null.to_numpy()
                summary.mean = float(np.mean(numeric_data))
                summary.median = float(np.median(numeric_data))
                summary.variance = float(np.var(numeric_data, ddof=1))
                summary.std_dev = float(np.std(numeric_data, ddof=1))
                summary.skewness = float(skew(numeric_data, nan_policy='omit'))
                summary.kurtosis = float(kurtosis(numeric_data, nan_policy='omit'))
                summary.min = float(np.min(numeric_data))
                summary.max = float(np.max(numeric_data))
                summary.q1 = float(np.percentile(numeric_data, 25))
                summary.q3 = float(np.percentile(numeric_data, 75))
                summary.iqr = summary.q3 - summary.q1
                
                # Compute mode(s)
                if len(numeric_data) > 0:
                    mode_result = stats.mode(numeric_data, keepdims=False)
                    if mode_result.count > 0:
                        summary.mode = float(mode_result.mode)
                
            elif dtype == DataType.CATEGORICAL:
                value_counts = non_null.value_counts().to_dict()
                if value_counts:
                    # Find most common category
                    max_count = 0
                    mode_value = None
                    for val, cnt in value_counts.items():
                        if cnt > max_count:
                            max_count = cnt
                            mode_value = val
                    summary.mode = mode_value
                    
                    # Compute category proportions
                    summary.unique_count = len(value_counts)
                    
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to compute statistics for {column}: {e}")
            raise
    
    def detect_missingness_patterns(
        self,
        columns: List[str]
    ) -> List[MissingnessPattern]:
        """
        Detect patterns in missing data.
        
        Args:
            columns: Columns to analyze
            
        Returns:
            List of MissingnessPattern objects
        """
        patterns = []
        
        for col in columns:
            missing_mask = self.df[col].is_null()
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                continue
                
            missing_percentage = (missing_count / len(self.df)) * 100
            
            # Detect missingness type
            missing_type = self._determine_missingness_type(col, missing_mask)
            
            # Find patterns with other columns
            pattern_with = self._find_missing_patterns(col, missing_mask)
            
            patterns.append(
                MissingnessPattern(
                    column=col,
                    missing_count=int(missing_count),
                    missing_percentage=float(missing_percentage),
                    missing_type=missing_type,
                    pattern_with=pattern_with
                )
            )
            
        return patterns
    
    def _determine_missingness_type(
        self,
        column: str,
        missing_mask: pl.Series
    ) -> str:
        """Determine if missingness is MCAR, MAR, or MNAR"""
        # Simplified detection - can be enhanced with more sophisticated methods
        
        # Check correlation with other columns
        numeric_cols = self.df.select(pl.col(pl.Float64)).columns
        correlations = []
        
        for other_col in numeric_cols:
            if other_col == column:
                continue
            try:
                # Convert missing mask to float
                mask_float = missing_mask.cast(pl.Float64)
                corr = mask_float.corr(self.df[other_col])
                if corr is not None and abs(corr) > 0.3:
                    correlations.append(abs(corr))
            except:
                pass
        
        if len(correlations) == 0:
            return "MCAR"  # Missing Completely At Random
        elif len(correlations) > 0 and max(correlations) > 0.5:
            return "MAR"  # Missing At Random
        else:
            return "MNAR"  # Missing Not At Random
            
    def _find_missing_patterns(
        self,
        column: str,
        missing_mask: pl.Series
    ) -> Optional[List[str]]:
        """Find columns that have similar missing patterns"""
        pattern_with = []
        
        for other_col in self.df.columns:
            if other_col == column:
                continue
                
            other_missing = self.df[other_col].is_null()
            similarity = (missing_mask == other_missing).mean()
            
            if similarity > 0.8:  # 80% similar missing pattern
                pattern_with.append(other_col)
                
        return pattern_with if pattern_with else None
    
    def detect_outliers_iqr(
        self,
        column: str,
        threshold: float = 1.5
    ) -> Tuple[np.ndarray, int]:
        """
        Detect outliers using IQR method.
        
        Returns:
            Tuple of (outlier_mask, count)
        """
        data = self.df[column].drop_nulls().to_numpy()
        
        if len(data) == 0:
            return np.array([]), 0
            
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        
        return outliers, len(outliers)
    
    def compute_feature_importance_preview(
        self,
        target: str,
        features: List[str]
    ) -> Dict[str, float]:
        """
        Compute preliminary feature importance using mutual information.
        
        Args:
            target: Target column name
            features: List of feature columns
            
        Returns:
            Dictionary mapping feature to importance score
        """
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            # Prepare data
            y = self.df[target].drop_nulls().to_numpy()
            
            feature_importances = {}
            
            for feature in features:
                X = self.df[feature].drop_nulls().to_numpy().reshape(-1, 1)
                
                # Align indices
                valid_indices = ~(self.df[target].is_null() | self.df[feature].is_null())
                X_aligned = self.df[feature].filter(valid_indices).to_numpy().reshape(-1, 1)
                y_aligned = self.df[target].filter(valid_indices).to_numpy()
                
                if len(X_aligned) == 0:
                    feature_importances[feature] = 0.0
                    continue
                
                # Determine if regression or classification
                if len(np.unique(y_aligned)) <= 10:  # Classification
                    mi = mutual_info_classif(X_aligned, y_aligned, random_state=42)[0]
                else:  # Regression
                    mi = mutual_info_regression(X_aligned, y_aligned, random_state=42)[0]
                    
                feature_importances[feature] = float(mi)
            
            # Normalize to 0-1
            max_val = max(feature_importances.values()) if feature_importances else 1
            for feature in feature_importances:
                feature_importances[feature] /= max_val
                
            return feature_importances
            
        except Exception as e:
            self.logger.error(f"Failed to compute feature importance: {e}")
            return {}