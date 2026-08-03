"""
Preprocessing utilities for the ML framework.
"""

from typing import Optional, Union, Any, Callable
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pydantic import BaseModel, Field
from loguru import logger

from .exceptions import PreprocessingError


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing pipelines."""
    
    # Scaling options
    numeric_scaler: str = Field(default='standard')  # standard, minmax, robust, none
    scaling_columns: Optional[list[str]] = None
    
    # Encoding options
    categorical_encoder: str = Field(default='onehot')  # onehot, label, none
    encoding_columns: Optional[list[str]] = None
    
    # Feature transformations
    polynomial_features: bool = False
    polynomial_degree: int = Field(default=2, ge=1, le=5)
    interaction_only: bool = False
    
    # Feature selection
    variance_threshold: Optional[float] = None
    correlation_threshold: Optional[float] = None


class FeatureTransformer:
    """
    Handles feature transformations and preprocessing.
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._preprocessor: Optional[Pipeline] = None
        self._feature_names: Optional[list[str]] = None
        
    def fit(self, X: pd.DataFrame) -> 'FeatureTransformer':
        """
        Fit the preprocessor on the data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Self
        """
        try:
            # Identify numeric and categorical columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # If specific columns are specified, use them
            if self.config.scaling_columns:
                numeric_cols = [c for c in self.config.scaling_columns if c in numeric_cols]
            if self.config.encoding_columns:
                categorical_cols = [c for c in self.config.encoding_columns if c in categorical_cols]
            
            # Build transformers
            transformers = []
            
            # Numeric preprocessing
            if numeric_cols:
                scaler = self._get_scaler()
                transformers.append(('numeric', scaler, numeric_cols))
            
            # Categorical preprocessing
            if categorical_cols:
                encoder = self._get_encoder()
                transformers.append(('categorical', encoder, categorical_cols))
            
            # Create column transformer
            if transformers:
                self._preprocessor = Pipeline([
                    ('preprocessor', ColumnTransformer(transformers)),
                ])
                
                # Fit the preprocessor
                self._preprocessor.fit(X)
                
                # Store feature names
                self._feature_names = self._get_feature_names(X, numeric_cols, categorical_cols)
                
                logger.info(
                    f"Fit preprocessor: {len(numeric_cols)} numeric, "
                    f"{len(categorical_cols)} categorical features"
                )
            else:
                self._preprocessor = None
                self._feature_names = X.columns.tolist()
                logger.warning("No preprocessing steps configured")
            
            return self
            
        except Exception as e:
            raise PreprocessingError(f"Failed to fit preprocessor: {str(e)}") from e
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform the data using the fitted preprocessor.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed feature array
        """
        if self._preprocessor is None:
            return X.values if hasattr(X, 'values') else np.array(X)
        
        try:
            return self._preprocessor.transform(X)
        except Exception as e:
            raise PreprocessingError(f"Failed to transform data: {str(e)}") from e
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform the data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed feature array
        """
        self.fit(X)
        return self.transform(X)
    
    def _get_scaler(self):
        """Get the appropriate scaler based on configuration."""
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'none': None
        }
        scaler = scaler_map.get(self.config.numeric_scaler)
        if scaler is None:
            return 'passthrough'
        return scaler
    
    def _get_encoder(self):
        """Get the appropriate encoder based on configuration."""
        encoder_map = {
            'onehot': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'label': LabelEncoder(),
            'none': None
        }
        encoder = encoder_map.get(self.config.categorical_encoder)
        if encoder is None:
            return 'passthrough'
        return encoder
    
    def _get_feature_names(
        self,
        X: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str]
    ) -> list[str]:
        """Get feature names after transformation."""
        feature_names = []
        
        # Get numeric feature names
        feature_names.extend(numeric_cols)
        
        # Get categorical feature names
        if categorical_cols and self.config.categorical_encoder == 'onehot':
            encoder = self._preprocessor.named_steps['preprocessor'].transformers_[1][1]
            if hasattr(encoder, 'get_feature_names_out'):
                encoder_names = encoder.get_feature_names_out(categorical_cols)
                feature_names.extend(encoder_names)
            else:
                feature_names.extend(categorical_cols)
        elif categorical_cols:
            feature_names.extend(categorical_cols)
        
        return feature_names
    
    @property
    def feature_names(self) -> list[str]:
        """Get the feature names after transformation."""
        if self._feature_names is None:
            raise PreprocessingError("Preprocessor not fitted")
        return self._feature_names
    
    @property
    def n_features(self) -> int:
        """Get the number of features after transformation."""
        if self._feature_names is None:
            raise PreprocessingError("Preprocessor not fitted")
        return len(self._feature_names)
    
    def get_params(self) -> dict[str, Any]:
        """Get preprocessor parameters."""
        return {
            'numeric_scaler': self.config.numeric_scaler,
            'categorical_encoder': self.config.categorical_encoder,
            'poly_features': self.config.polynomial_features,
            'poly_degree': self.config.polynomial_degree,
            'feature_count': self.n_features if self._feature_names else 0
        }