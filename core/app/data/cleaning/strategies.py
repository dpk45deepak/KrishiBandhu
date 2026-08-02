"""Strategy implementations for cleaning operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from .exceptions import StrategyError
from .models import MissingValueStrategy, OutlierStrategy


class BaseStrategy(ABC):
    """Base class for all cleaning strategies."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.name = self.__class__.__name__
    
    @abstractmethod
    def apply(self, data: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
        """Apply the strategy to the data."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the strategy application."""
        return {
            "strategy": self.name,
            "params": self.params,
            "affected_rows": 0,
            "affected_columns": 0,
        }


class MissingValueHandler(BaseStrategy):
    """Handles missing values in datasets."""
    
    def __init__(self, strategy: MissingValueStrategy, **kwargs):
        super().__init__(strategy=strategy, **kwargs)
        self.strategy = strategy
        self._affected_rows = 0
        self._affected_columns = 0
    
    def apply(self, data: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
        """Apply missing value handling strategy."""
        df = data.copy()
        
        if column:
            # Handle single column
            if column not in df.columns:
                raise StrategyError(f"Column '{column}' not found in dataset")
            df[column] = self._handle_column(df[column])
            self._affected_rows = df[column].isna().sum()
            self._affected_columns = 1
        else:
            # Handle all columns with missing values
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = self._handle_column(df[col])
                    self._affected_columns += 1
            
            self._affected_rows = df.isna().sum().sum()
        
        return df
    
    def _handle_column(self, series: pd.Series) -> pd.Series:
        """Handle missing values in a single column."""
        if self.strategy == MissingValueStrategy.DROP_ROW:
            return series.dropna()
        
        elif self.strategy == MissingValueStrategy.DROP_COLUMN:
            return pd.Series([], dtype=series.dtype)
        
        elif self.strategy == MissingValueStrategy.MEAN:
            if pd.api.types.is_numeric_dtype(series):
                return series.fillna(series.mean())
            else:
                logger.warning(f"Cannot use mean for non-numeric column: {series.name}")
                return series
        
        elif self.strategy == MissingValueStrategy.MEDIAN:
            if pd.api.types.is_numeric_dtype(series):
                return series.fillna(series.median())
            else:
                logger.warning(f"Cannot use median for non-numeric column: {series.name}")
                return series
        
        elif self.strategy == MissingValueStrategy.MODE:
            mode_value = series.mode()
            if not mode_value.empty:
                return series.fillna(mode_value[0])
            return series
        
        elif self.strategy == MissingValueStrategy.CONSTANT:
            fill_value = self.params.get("constant_value", 0)
            return series.fillna(fill_value)
        
        elif self.strategy == MissingValueStrategy.FORWARD_FILL:
            return series.fillna(method='ffill')
        
        elif self.strategy == MissingValueStrategy.BACKWARD_FILL:
            return series.fillna(method='bfill')
        
        elif self.strategy == MissingValueStrategy.INTERPOLATE:
            if pd.api.types.is_numeric_dtype(series):
                return series.interpolate(method='linear', limit_direction='both')
            else:
                logger.warning(f"Cannot interpolate non-numeric column: {series.name}")
                return series
        
        elif self.strategy == MissingValueStrategy.CUSTOM:
            custom_func = self.params.get("custom_function")
            if custom_func and callable(custom_func):
                return series.map(custom_func)
            else:
                raise StrategyError("Custom function not provided or not callable")
        
        else:
            raise StrategyError(f"Unknown missing value strategy: {self.strategy}")
    
    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "strategy_type": "missing_values",
            "strategy": self.strategy.value,
            "affected_rows": self._affected_rows,
            "affected_columns": self._affected_columns,
        })
        return metadata


class OutlierHandler(BaseStrategy):
    """Handles outliers in datasets."""
    
    def __init__(self, strategy: OutlierStrategy, threshold: float = 3.0, **kwargs):
        super().__init__(strategy=strategy, threshold=threshold, **kwargs)
        self.strategy = strategy
        self.threshold = threshold
        self._affected_rows = 0
        self._affected_columns = 0
    
    def apply(self, data: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
        """Apply outlier handling strategy."""
        df = data.copy()
        
        if column:
            # Handle single column
            if column not in df.columns:
                raise StrategyError(f"Column '{column}' not found in dataset")
            df[column] = self._handle_outliers(df[column])
            self._affected_columns = 1
        else:
            # Handle all numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = self._handle_outliers(df[col])
                self._affected_columns += 1
        
        self._affected_rows = len(df) - len(data)
        return df
    
    def _handle_outliers(self, series: pd.Series) -> pd.Series:
        """Handle outliers in a single column."""
        if not pd.api.types.is_numeric_dtype(series):
            logger.warning(f"Cannot handle outliers for non-numeric column: {series.name}")
            return series
        
        # Identify outliers
        if self.strategy == OutlierStrategy.IQR:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
        
        elif self.strategy == OutlierStrategy.ZSCORE:
            z_scores = np.abs(stats.zscore(series.dropna()))
            threshold = self.params.get("threshold", self.threshold)
            outliers = pd.Series(False, index=series.index)
            outliers.loc[series.dropna().index[z_scores > threshold]] = True
        
        else:
            raise StrategyError(f"Unknown outlier strategy: {self.strategy}")
        
        if outliers.sum() == 0:
            return series
        
        # Handle outliers based on strategy
        if self.strategy == OutlierStrategy.REMOVE:
            return series[~outliers]
        
        elif self.strategy == OutlierStrategy.CLIP:
            if self.strategy == OutlierStrategy.IQR:
                return series.clip(lower=lower_bound, upper=upper_bound)
            elif self.strategy == OutlierStrategy.ZSCORE:
                return series.clip(
                    lower=series.mean() - threshold * series.std(),
                    upper=series.mean() + threshold * series.std()
                )
        
        elif self.strategy == OutlierStrategy.WINSORIZE:
            if self.strategy == OutlierStrategy.IQR:
                series.loc[series < lower_bound] = lower_bound
                series.loc[series > upper_bound] = upper_bound
                return series
            elif self.strategy == OutlierStrategy.ZSCORE:
                lower = series.mean() - threshold * series.std()
                upper = series.mean() + threshold * series.std()
                series.loc[series < lower] = lower
                series.loc[series > upper] = upper
                return series
        
        elif self.strategy == OutlierStrategy.KEEP:
            return series
        
        else:
            raise StrategyError(f"Unknown outlier handling: {self.strategy}")
    
    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "strategy_type": "outliers",
            "strategy": self.strategy.value,
            "threshold": self.threshold,
            "affected_rows": self._affected_rows,
            "affected_columns": self._affected_columns,
        })
        return metadata


class DuplicateHandler(BaseStrategy):
    """Handles duplicate rows in datasets."""
    
    def __init__(self, keep: str = "first", subset: Optional[List[str]] = None, **kwargs):
        super().__init__(keep=keep, subset=subset, **kwargs)
        self.keep = keep
        self.subset = subset
        self._duplicates_removed = 0
    
    def apply(self, data: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
        """Remove duplicate rows."""
        df = data.copy()
        before_count = len(df)
        
        if column:
            # Remove duplicates based on specific column
            df = df.drop_duplicates(subset=[column], keep=self.keep)
        else:
            # Remove duplicates based on all columns or specified subset
            df = df.drop_duplicates(subset=self.subset, keep=self.keep)
        
        self._duplicates_removed = before_count - len(df)
        return df
    
    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "strategy_type": "duplicates",
            "duplicates_removed": self._duplicates_removed,
            "keep": self.keep,
            "subset": self.subset,
        })
        return metadata


class EmptyRowHandler(BaseStrategy):
    """Handles empty rows in datasets."""
    
    def __init__(self, how: str = "all", threshold: float = 0.5, **kwargs):
        super().__init__(how=how, threshold=threshold, **kwargs)
        self.how = how
        self.threshold = threshold
        self._rows_removed = 0
    
    def apply(self, data: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
        """Remove empty rows."""
        df = data.copy()
        before_count = len(df)
        
        if self.how == "all":
            df = df.dropna(how="all")
        elif self.how == "any":
            df = df.dropna(how="any")
        elif self.how == "threshold":
            # Remove rows with more than threshold% missing values
            missing_percentage = df.isna().mean(axis=1)
            df = df[missing_percentage < self.threshold]
        else:
            raise StrategyError(f"Unknown empty row handling: {self.how}")
        
        self._rows_removed = before_count - len(df)
        return df
    
    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "strategy_type": "empty_rows",
            "rows_removed": self._rows_removed,
            "how": self.how,
            "threshold": self.threshold,
        })
        return metadata


class EmptyColumnHandler(BaseStrategy):
    """Handles empty columns in datasets."""
    
    def __init__(self, threshold: float = 0.9, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold
        self._columns_removed = 0
    
    def apply(self, data: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
        """Remove empty columns."""
        df = data.copy()

        if column:
            # Check specific column
            if column in df.columns:
                missing_percentage = df[column].isna().mean()
                if missing_percentage >= self.threshold:
                    df = df.drop(columns=[column])
                    self._columns_removed = 1
        else:
            # Check all columns
            missing_percentage = df.isna().mean()
            columns_to_remove = missing_percentage[missing_percentage >= self.threshold].index
            if len(columns_to_remove) > 0:
                df = df.drop(columns=columns_to_remove)
                self._columns_removed = len(columns_to_remove)

        return df

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "strategy_type": "empty_columns",
            "columns_removed": self._columns_removed,
            "threshold": self.threshold,
        })
        return metadata