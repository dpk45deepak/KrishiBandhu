"""Statistical computations for cleaning operations."""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger


class CleaningStatistics:
    """Computes and manages statistics for cleaning operations."""
    
    def __init__(self):
        self.stats: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
    
    def compute_dataset_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive statistics for a dataset."""
        stats = {
            "shape": data.shape,
            "columns": len(data.columns),
            "rows": len(data),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "null_counts": data.isna().sum().to_dict(),
            "null_percentages": (data.isna().sum() / len(data) * 100).to_dict(),
            "dtypes": data.dtypes.astype(str).to_dict(),
        }
        
        # Numeric statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric"] = {
                "columns": numeric_cols.tolist(),
                "min": data[numeric_cols].min().to_dict(),
                "max": data[numeric_cols].max().to_dict(),
                "mean": data[numeric_cols].mean().to_dict(),
                "median": data[numeric_cols].median().to_dict(),
                "std": data[numeric_cols].std().to_dict(),
                "skew": data[numeric_cols].skew().to_dict(),
                "kurtosis": data[numeric_cols].kurtosis().to_dict(),
            }
        
        # Categorical statistics
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            stats["categorical"] = {
                "columns": cat_cols.tolist(),
                "unique_counts": data[cat_cols].nunique().to_dict(),
                "most_frequent": {
                    col: data[col].mode()[0] if len(data[col].mode()) > 0 else None
                    for col in cat_cols
                }
            }
        
        # Date statistics
        date_cols = data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            stats["datetime"] = {
                "columns": date_cols.tolist(),
                "min": data[date_cols].min().to_dict(),
                "max": data[date_cols].max().to_dict(),
                "range": {
                    col: (data[col].max() - data[col].min()).days
                    for col in date_cols
                }
            }
        
        # Duplicate statistics
        stats["duplicates"] = {
            "duplicated_rows": data.duplicated().sum(),
            "duplicate_percentage": (data.duplicated().sum() / len(data)) * 100,
        }
        
        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            stats["correlation"] = data[numeric_cols].corr().to_dict()
        
        self.stats = stats
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "stats": stats.copy()
        })
        
        return stats
    
    def compute_delta(self, before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, Any]:
        """Compute changes between two datasets."""
        delta = {
            "rows_before": len(before),
            "rows_after": len(after),
            "rows_removed": len(before) - len(after),
            "columns_before": len(before.columns),
            "columns_after": len(after.columns),
            "columns_added": len(set(after.columns) - set(before.columns)),
            "columns_removed": len(set(before.columns) - set(after.columns)),
            "columns_changed": 0,
        }
        
        # Check for changed columns
        common_columns = set(before.columns) & set(after.columns)
        for col in common_columns:
            if not before[col].equals(after[col]):
                delta["columns_changed"] += 1
                
                # Track specific changes
                if col not in delta.get("column_changes", {}):
                    delta["column_changes"] = {}
                
                before_null = before[col].isna().sum()
                after_null = after[col].isna().sum()
                delta["column_changes"][col] = {
                    "nulls_fixed": before_null - after_null,
                    "unique_values_before": before[col].nunique(),
                    "unique_values_after": after[col].nunique(),
                }
        
        return delta
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of cleaning statistics."""
        if not self.stats:
            return {"status": "No statistics computed"}
        
        return {
            "dataset_shape": self.stats.get("shape"),
            "null_total": sum(self.stats.get("null_counts", {}).values()),
            "duplicates_total": self.stats.get("duplicates", {}).get("duplicated_rows", 0),
            "numeric_columns": len(self.stats.get("numeric", {}).get("columns", [])),
            "categorical_columns": len(self.stats.get("categorical", {}).get("columns", [])),
            "datetime_columns": len(self.stats.get("datetime", {}).get("columns", [])),
        }
    
    def log_statistics(self) -> None:
        """Log current statistics."""
        summary = self.get_summary()
        logger.info(f"Dataset Statistics Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")


class QualityMetrics:
    """Computes quality metrics for datasets."""
    
    @staticmethod
    def compute_data_quality(data: pd.DataFrame) -> Dict[str, float]:
        """Compute various data quality metrics."""
        metrics = {}
        
        # Completeness
        completeness = 1 - (data.isna().sum().sum() / (data.shape[0] * data.shape[1]))
        metrics["completeness"] = completeness
        
        # Uniqueness (for each column)
        uniqueness = data.nunique().mean() / data.shape[0] if data.shape[0] > 0 else 0
        metrics["uniqueness"] = uniqueness
        
        # Consistency (based on data types)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Check for negative values in columns that should be positive
            positive_cols = [col for col in numeric_cols if (data[col] >= 0).all()]
            consistency = len(positive_cols) / len(numeric_cols) if len(numeric_cols) > 0 else 1.0
            metrics["consistency"] = consistency
        else:
            metrics["consistency"] = 1.0
        
        # Validity (based on value ranges)
        if len(numeric_cols) > 0:
            valid_percentages = []
            for col in numeric_cols:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                valid = ((data[col] >= lower) & (data[col] <= upper)).mean()
                valid_percentages.append(valid)
            metrics["validity"] = np.mean(valid_percentages) if valid_percentages else 1.0
        else:
            metrics["validity"] = 1.0
        
        # Overall quality score
        metrics["quality_score"] = np.mean([
            metrics["completeness"],
            metrics["uniqueness"],
            metrics["consistency"],
            metrics["validity"]
        ])
        
        return metrics
    
    @staticmethod
    def compute_column_quality(series: pd.Series) -> Dict[str, float]:
        """Compute quality metrics for a single column."""
        metrics = {}
        
        # Completeness
        metrics["completeness"] = 1 - (series.isna().sum() / len(series)) if len(series) > 0 else 0
        
        # Uniqueness
        if len(series) > 0:
            metrics["uniqueness"] = series.nunique() / len(series)
        else:
            metrics["uniqueness"] = 0
        
        # Validity based on outliers (for numeric columns)
        if pd.api.types.is_numeric_dtype(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            metrics["validity"] = ((series >= lower) & (series <= upper)).mean()
        else:
            metrics["validity"] = 1.0
        
        # Column quality score
        metrics["quality_score"] = np.mean([
            metrics["completeness"],
            metrics["uniqueness"],
            metrics["validity"]
        ])
        
        return metrics