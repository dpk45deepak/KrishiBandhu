# app/data/feature_engineering/feature_transformer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer,
    Normalizer
)
from sklearn.compose import ColumnTransformer
from loguru import logger

from app.data.feature_engineering.models import ScalingType
from app.data.feature_engineering.exceptions import TransformationError


class FeatureTransformer:
    """
    Enterprise feature transformer implementing various numerical transformations.
    
    Supports multiple scaling strategies with fit/transform capabilities.
    """
    
    def __init__(self):
        self.scalers: Dict[str, object] = {}
        self.transformer: Optional[ColumnTransformer] = None
        self.fitted: bool = False
        self.transformation_params: Dict[str, Dict] = {}
    
    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        scaling_type: ScalingType,
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply scaling to specified columns.
        
        Args:
            df: Input dataframe
            columns: Columns to scale
            scaling_type: Type of scaling to apply
            **kwargs: Additional parameters for the scaler
            
        Returns:
            Scaled dataframe
        """
        try:
            result_df = df.copy()
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                scaler = self._create_scaler(scaling_type, **kwargs)
                data = df[col].values.reshape(-1, 1)
                
                # Handle NaN values
                mask = ~np.isnan(data)
                if not mask.all():
                    logger.warning(f"NaN values found in {col}, filling with median")
                    data = np.nan_to_num(data, nan=np.nanmedian(data[mask]))
                
                transformed = scaler.fit_transform(data)
                result_df[f"{col}_scaled"] = transformed.flatten()
                
                # Store parameters for inverse transform
                self.transformation_params[col] = {
                    "type": scaling_type,
                    "scaler": scaler,
                    "params": kwargs
                }
            
            self.fitted = True
            logger.info(f"Scaled {len(columns)} columns using {scaling_type}")
            return result_df
            
        except Exception as e:
            raise TransformationError(f"Failed to scale features: {e}")
    
    def _create_scaler(self, scaling_type: ScalingType, **kwargs) -> object:
        """Create appropriate scaler based on type."""
        scalers = {
            ScalingType.STANDARD: StandardScaler,
            ScalingType.MINMAX: MinMaxScaler,
            ScalingType.ROBUST: RobustScaler,
            ScalingType.POWER: PowerTransformer,
            ScalingType.QUANTILE: QuantileTransformer,
            ScalingType.NORMALIZATION: Normalizer
        }
        
        scaler_class = scalers.get(scaling_type)
        if not scaler_class:
            raise ValueError(f"Unknown scaling type: {scaling_type}")
        
        return scaler_class(**kwargs)
    
    def apply_log_transform(
        self,
        df: pd.DataFrame,
        columns: List[str],
        base: float = np.e,
        shift: float = 1.0
    ) -> pd.DataFrame:
        """Apply log transformation to columns."""
        result_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Apply log transform with shift to handle zeros
            transformed = np.log(df[col] + shift) / np.log(base)
            result_df[f"{col}_log"] = transformed
            
            self.transformation_params[col] = {
                "type": "log",
                "base": base,
                "shift": shift
            }
        
        logger.info(f"Applied log transform to {len(columns)} columns")
        return result_df
    
    def apply_power_transform(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'yeo-johnson'
    ) -> pd.DataFrame:
        """Apply power transformation."""
        result_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            pt = PowerTransformer(method=method)
            data = df[col].values.reshape(-1, 1)
            
            # Handle NaN values
            data = np.nan_to_num(data, nan=np.nanmedian(data[~np.isnan(data)]))
            
            transformed = pt.fit_transform(data)
            result_df[f"{col}_power"] = transformed.flatten()
            
            self.transformation_params[col] = {
                "type": "power",
                "method": method,
                "transformer": pt
            }
        
        logger.info(f"Applied power transform to {len(columns)} columns")
        return result_df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        norm: str = 'l2'
    ) -> pd.DataFrame:
        """Normalize features."""
        result_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            scaler = Normalizer(norm=norm)
            data = df[col].values.reshape(1, -1)
            transformed = scaler.fit_transform(data)
            result_df[f"{col}_normalized"] = transformed.flatten()
            
            self.transformation_params[col] = {
                "type": "normalization",
                "norm": norm
            }
        
        logger.info(f"Normalized {len(columns)} columns")
        return result_df
    
    def get_transformation_params(self, column: str) -> Optional[Dict]:
        """Get transformation parameters for a column."""
        return self.transformation_params.get(column)
    
    def inverse_transform(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Inverse transform scaled features (when possible)."""
        result_df = df.copy()
        
        for col in columns:
            if col not in self.transformation_params:
                continue
            
            params = self.transformation_params[col]
            if "scaler" in params:
                data = df[col].values.reshape(-1, 1)
                transformed = params["scaler"].inverse_transform(data)
                result_df[f"{col}_inverse"] = transformed.flatten()
        
        return result_df
    
    def batch_transform(
        self,
        df: pd.DataFrame,
        transformations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply multiple transformations in batch."""
        result_df = df.copy()
        
        for transform in transformations:
            method = transform.get("method")
            columns = transform.get("columns", [])
            
            if method == "scale":
                result_df = self.scale_features(
                    result_df, columns, transform.get("scaling_type"), 
                    **transform.get("params", {})
                )
            elif method == "log":
                result_df = self.apply_log_transform(
                    result_df, columns, 
                    transform.get("base", np.e),
                    transform.get("shift", 1.0)
                )
            elif method == "power":
                result_df = self.apply_power_transform(
                    result_df, columns,
                    transform.get("method", "yeo-johnson")
                )
            elif method == "normalize":
                result_df = self.normalize_features(
                    result_df, columns,
                    transform.get("norm", "l2")
                )
        
        return result_df