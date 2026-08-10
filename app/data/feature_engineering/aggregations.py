# app/data/feature_engineering/aggregations.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from loguru import logger

from app.data.feature_engineering.exceptions import FeatureGenerationError


class FeatureAggregator:
    """
    Generate aggregated and rolling features.
    
    Supports various aggregation functions and window operations.
    """
    
    def __init__(self):
        self.aggregations: List[Dict] = []
    
    def aggregate_by_group(
        self,
        df: pd.DataFrame,
        group_col: str,
        agg_col: str,
        agg_func: Union[str, Callable],
        **kwargs
    ) -> pd.Series:
        """
        Aggregate feature by group.
        
        Args:
            df: Input dataframe
            group_col: Column to group by
            agg_col: Column to aggregate
            agg_func: Aggregation function
            **kwargs: Additional parameters
            
        Returns:
            Aggregated series
        """
        try:
            # Calculate aggregation
            if isinstance(agg_func, str):
                grouped = df.groupby(group_col)[agg_col].agg(agg_func)
            else:
                grouped = df.groupby(group_col)[agg_col].apply(agg_func)
            
            # Map back to original rows
            result = df[group_col].map(grouped)
            
            # Store aggregation info
            self.aggregations.append({
                'type': 'group_aggregation',
                'group_col': group_col,
                'agg_col': agg_col,
                'agg_func': str(agg_func),
                'parameters': kwargs
            })
            
            return result
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to aggregate by group: {e}")
    
    def rolling_aggregation(
        self,
        series: pd.Series,
        window: int,
        agg_func: Union[str, Callable],
        min_periods: Optional[int] = None,
        **kwargs
    ) -> pd.Series:
        """
        Apply rolling aggregation to a series.
        
        Args:
            series: Input series
            window: Rolling window size
            agg_func: Aggregation function
            min_periods: Minimum number of observations in window
            **kwargs: Additional parameters
            
        Returns:
            Rolling aggregated series
        """
        try:
            min_periods = min_periods or window
            
            if isinstance(agg_func, str):
                rolling = series.rolling(window=window, min_periods=min_periods)
                result = getattr(rolling, agg_func)()
            else:
                rolling = series.rolling(window=window, min_periods=min_periods)
                result = rolling.apply(agg_func)
            
            # Store aggregation info
            self.aggregations.append({
                'type': 'rolling_aggregation',
                'window': window,
                'agg_func': str(agg_func),
                'min_periods': min_periods,
                'parameters': kwargs
            })
            
            return result
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to apply rolling aggregation: {e}")
    
    def time_based_aggregation(
        self,
        df: pd.DataFrame,
        time_col: str,
        agg_col: str,
        time_period: str,
        agg_func: Union[str, Callable],
        **kwargs
    ) -> pd.DataFrame:
        """
        Aggregate features based on time periods.
        
        Args:
            df: Input dataframe
            time_col: Time column
            agg_col: Column to aggregate
            time_period: 'day', 'week', 'month', 'quarter', 'year'
            agg_func: Aggregation function
            **kwargs: Additional parameters
            
        Returns:
            Time-based aggregated features
        """
        try:
            # Ensure time column is datetime
            df_time = df.copy()
            df_time[time_col] = pd.to_datetime(df_time[time_col])
            
            # Set time column as index
            df_time = df_time.set_index(time_col)
            
            # Resample based on time period
            if time_period == 'day':
                resampled = df_time.resample('D')
            elif time_period == 'week':
                resampled = df_time.resample('W')
            elif time_period == 'month':
                resampled = df_time.resample('M')
            elif time_period == 'quarter':
                resampled = df_time.resample('Q')
            elif time_period == 'year':
                resampled = df_time.resample('Y')
            else:
                raise ValueError(f"Unsupported time period: {time_period}")
            
            # Apply aggregation
            if isinstance(agg_func, str):
                aggregated = resampled[agg_col].agg(agg_func)
            else:
                aggregated = resampled[agg_col].apply(agg_func)
            
            # Store aggregation info
            self.aggregations.append({
                'type': 'time_based_aggregation',
                'time_col': time_col,
                'agg_col': agg_col,
                'time_period': time_period,
                'agg_func': str(agg_func),
                'parameters': kwargs
            })
            
            return aggregated
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to apply time-based aggregation: {e}")
    
    def weighted_aggregation(
        self,
        df: pd.DataFrame,
        col: str,
        weights: Union[pd.Series, np.ndarray],
        agg_func: str = 'sum',
        **kwargs
    ) -> pd.Series:
        """
        Apply weighted aggregation to a column.
        
        Args:
            df: Input dataframe
            col: Column to aggregate
            weights: Weight vector
            agg_func: Aggregation function ('sum', 'mean', 'weighted_mean')
            **kwargs: Additional parameters
            
        Returns:
            Weighted aggregated series
        """
        try:
            if agg_func == 'sum':
                result = (df[col] * weights).sum()
            elif agg_func == 'mean':
                result = (df[col] * weights).sum() / weights.sum()
            elif agg_func == 'weighted_mean':
                result = (df[col] * weights).sum() / weights.sum()
            else:
                raise ValueError(f"Unsupported aggregation function: {agg_func}")
            
            # Store aggregation info
            self.aggregations.append({
                'type': 'weighted_aggregation',
                'col': col,
                'agg_func': agg_func,
                'parameters': kwargs
            })
            
            return result
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to apply weighted aggregation: {e}")
    
    def cumulative_aggregation(
        self,
        series: pd.Series,
        agg_func: str = 'sum'
    ) -> pd.Series:
        """
        Apply cumulative aggregation to a series.
        
        Args:
            series: Input series
            agg_func: 'sum', 'mean', 'min', 'max', 'prod'
            
        Returns:
            Cumulative aggregated series
        """
        try:
            if agg_func == 'sum':
                result = series.cumsum()
            elif agg_func == 'mean':
                result = series.expanding().mean()
            elif agg_func == 'min':
                result = series.expanding().min()
            elif agg_func == 'max':
                result = series.expanding().max()
            elif agg_func == 'prod':
                result = series.expanding().prod()
            else:
                raise ValueError(f"Unsupported aggregation function: {agg_func}")
            
            # Store aggregation info
            self.aggregations.append({
                'type': 'cumulative_aggregation',
                'agg_func': agg_func
            })
            
            return result
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to apply cumulative aggregation: {e}")
    
    def multiple_aggregations(
        self,
        df: pd.DataFrame,
        group_col: str,
        agg_cols: List[str],
        agg_funcs: List[Union[str, Callable]]
    ) -> pd.DataFrame:
        """
        Apply multiple aggregations to multiple columns.
        
        Args:
            df: Input dataframe
            group_col: Column to group by
            agg_cols: Columns to aggregate
            agg_funcs: Aggregation functions
            
        Returns:
            Dataframe with multiple aggregated features
        """
        try:
            result_df = pd.DataFrame()
            
            for agg_col in agg_cols:
                for agg_func in agg_funcs:
                    if agg_col in df.columns:
                        aggregated = self.aggregate_by_group(
                            df, group_col, agg_col, agg_func
                        )
                        col_name = f"{agg_col}_{str(agg_func) if callable(agg_func) else agg_func}"
                        result_df[col_name] = aggregated
            
            return result_df
            
        except Exception as e:
            raise FeatureGenerationError(f"Failed to apply multiple aggregations: {e}")
    
    def get_aggregation_summary(self) -> pd.DataFrame:
        """Get summary of generated aggregations."""
        if not self.aggregations:
            return pd.DataFrame()
        
        return pd.DataFrame(self.aggregations)