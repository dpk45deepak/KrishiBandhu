# app/data/feature_engineering/statistics.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from scipy import stats
from scipy.stats import skew, kurtosis, zscore
from loguru import logger

from app.data.feature_engineering.exceptions import FeatureGenerationError


class StatisticalFeatureGenerator:
    """
    Generate statistical features from data.
    
    Computes various statistical measures for feature engineering.
    """
    
    def __init__(self):
        self.computed_statistics: Dict[str, Dict] = {}
    
    def compute_statistic(
        self,
        series: pd.Series,
        stat_type: str,
        **kwargs
    ) -> Union[float, pd.Series]:
        """
        Compute a statistical feature.
        
        Args:
            series: Input series
            stat_type: Type of statistic to compute
            **kwargs: Additional parameters
            
        Returns:
            Computed statistic
        """
        try:
            if stat_type == 'mean':
                return series.mean(**kwargs)
            elif stat_type == 'median':
                return series.median(**kwargs)
            elif stat_type == 'std':
                return series.std(**kwargs)
            elif stat_type == 'var':
                return series.var(**kwargs)
            elif stat_type == 'skew':
                return skew(series.dropna(), **kwargs)
            elif stat_type == 'kurtosis':
                return kurtosis(series.dropna(), **kwargs)
            elif stat_type == 'zscore':
                return zscore(series, **kwargs)
            elif stat_type == 'iqr':
                return series.quantile(0.75) - series.quantile(0.25)
            elif stat_type == 'mad':
                return (series - series.mean()).abs().mean()
            elif stat_type == 'range':
                return series.max() - series.min()
            elif stat_type == 'mode':
                return series.mode()[0] if not series.mode().empty else np.nan
            elif stat_type == 'quantile':
                q = kwargs.get('q', 0.5)
                return series.quantile(q)
            elif stat_type == 'cv':  # Coefficient of variation
                mean = series.mean()
                if mean == 0:
                    return np.nan
                return series.std() / mean
            else:
                raise ValueError(f"Unsupported statistic type: {stat_type}")
                
        except Exception as e:
            raise FeatureGenerationError(f"Failed to compute statistic: {e}")
    
    def compute_multi_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        statistics: List[str]
    ) -> pd.DataFrame:
        """
        Compute multiple statistics for multiple columns.
        
        Args:
            df: Input dataframe
            columns: Columns to compute statistics for
            statistics: List of statistics to compute
            
        Returns:
            Dataframe with computed statistics
        """
        try:
            result = {}
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                result[col] = {}
                for stat_type in statistics:
                    try:
                        value = self.compute_statistic(df[col], stat_type)
                        result[col][stat_type] = value
                    except Exception as e:
                        logger.warning(f"Failed to compute {stat_type} for {col}: {e}")
                        result[col][stat_type] = np.nan
            
            # Create DataFrame
            stats_df = pd.DataFrame(result).T
            
            # Store computed statistics
            self.computed_statistics['multi_statistics'] = {
                'columns': columns,
                'statistics': statistics,
                'results': stats_df
            }
            
            return stats_df
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to compute multi-statistics: {e}")
    
    def compute_rolling_statistics(
        self,
        series: pd.Series,
        window: int,
        statistics: List[str],
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute rolling statistics for a series.
        
        Args:
            series: Input series
            window: Rolling window size
            statistics: List of statistics to compute
            min_periods: Minimum number of observations in window
            
        Returns:
            Dataframe with rolling statistics
        """
        try:
            min_periods = min_periods or window
            rolling = series.rolling(window=window, min_periods=min_periods)
            
            result = pd.DataFrame()
            
            for stat_type in statistics:
                if stat_type == 'mean':
                    result[f'rolling_mean_{window}'] = rolling.mean()
                elif stat_type == 'std':
                    result[f'rolling_std_{window}'] = rolling.std()
                elif stat_type == 'min':
                    result[f'rolling_min_{window}'] = rolling.min()
                elif stat_type == 'max':
                    result[f'rolling_max_{window}'] = rolling.max()
                elif stat_type == 'quantile':
                    result[f'rolling_quantile_0.5_{window}'] = rolling.quantile(0.5)
                    result[f'rolling_quantile_0.75_{window}'] = rolling.quantile(0.75)
                    result[f'rolling_quantile_0.25_{window}'] = rolling.quantile(0.25)
                elif stat_type == 'skew':
                    result[f'rolling_skew_{window}'] = rolling.apply(skew)
                elif stat_type == 'kurtosis':
                    result[f'rolling_kurtosis_{window}'] = rolling.apply(kurtosis)
                else:
                    logger.warning(f"Unsupported rolling statistic: {stat_type}")
            
            # Store computed statistics
            self.computed_statistics[f'rolling_statistics_{window}'] = {
                'window': window,
                'statistics': statistics,
                'min_periods': min_periods,
                'results': result
            }
            
            return result
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to compute rolling statistics: {e}")
    
    def compute_correlation_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute correlation-based features.
        
        Args:
            df: Input dataframe
            columns: Columns to compute correlations for
            target_col: Target column for correlation with target
            
        Returns:
            Dataframe with correlation features
        """
        try:
            if target_col and target_col in df.columns:
                # Correlations with target
                correlations = df[columns].corrwith(df[target_col])
                correlation_df = pd.DataFrame({
                    'feature': correlations.index,
                    'target_correlation': correlations.values
                })
            else:
                # Pairwise correlations
                correlation_df = df[columns].corr()
            
            # Store computed statistics
            self.computed_statistics['correlations'] = {
                'target_col': target_col,
                'columns': columns,
                'results': correlation_df
            }
            
            return correlation_df
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to compute correlation features: {e}")
    
    def compute_distribution_stats(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_quantiles: int = 10
    ) -> pd.DataFrame:
        """
        Compute distribution statistics for columns.
        
        Args:
            df: Input dataframe
            columns: Columns to analyze
            n_quantiles: Number of quantile bins
            
        Returns:
            Dataframe with distribution statistics
        """
        try:
            distribution_stats = {}
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                series = df[col].dropna()
                n = len(series)
                
                distribution_stats[col] = {
                    'n': n,
                    'n_missing': len(df) - n,
                    'unique_values': series.nunique(),
                    'min': series.min(),
                    'max': series.max(),
                    'mean': series.mean(),
                    'median': series.median(),
                    'std': series.std(),
                    'skew': skew(series),
                    'kurtosis': kurtosis(series),
                    'q1': series.quantile(0.25),
                    'q3': series.quantile(0.75),
                    'iqr': series.quantile(0.75) - series.quantile(0.25),
                    'range': series.max() - series.min(),
                    'zero_count': (series == 0).sum(),
                    'negative_count': (series < 0).sum(),
                }
                
                # Add quantiles
                quantiles = series.quantile(np.linspace(0, 1, n_quantiles))
                for q, value in quantiles.items():
                    distribution_stats[col][f'q_{q:.1f}'] = value
            
            # Create DataFrame
            stats_df = pd.DataFrame(distribution_stats).T
            
            # Store computed statistics
            self.computed_statistics['distribution_stats'] = {
                'columns': columns,
                'n_quantiles': n_quantiles,
                'results': stats_df
            }
            
            return stats_df
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to compute distribution stats: {e}")
    
    def compute_outlier_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Compute outlier-based features.
        
        Args:
            df: Input dataframe
            columns: Columns to analyze
            method: Outlier detection method ('iqr', 'zscore', 'quantile')
            threshold: Threshold for outlier detection
            
        Returns:
            Dataframe with outlier features
        """
        try:
            outlier_features = pd.DataFrame(index=df.index)
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                series = df[col]
                
                if method == 'iqr':
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    outliers = (series < lower_bound) | (series > upper_bound)
                    outlier_features[f'{col}_outlier'] = outliers.astype(int)
                    outlier_features[f'{col}_lower_bound'] = lower_bound
                    outlier_features[f'{col}_upper_bound'] = upper_bound
                    
                elif method == 'zscore':
                    z_scores = np.abs(zscore(series))
                    outliers = z_scores > threshold
                    outlier_features[f'{col}_zscore'] = z_scores
                    outlier_features[f'{col}_outlier'] = outliers.astype(int)
                    
                elif method == 'quantile':
                    lower_bound = series.quantile(threshold)
                    upper_bound = series.quantile(1 - threshold)
                    outliers = (series < lower_bound) | (series > upper_bound)
                    outlier_features[f'{col}_outlier'] = outliers.astype(int)
            
            # Store computed statistics
            self.computed_statistics['outlier_features'] = {
                'columns': columns,
                'method': method,
                'threshold': threshold,
                'results': outlier_features
            }
            
            return outlier_features
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to compute outlier features: {e}")
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get summary of computed statistics."""
        return self.computed_statistics