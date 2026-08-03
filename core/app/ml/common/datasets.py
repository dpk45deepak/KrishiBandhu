"""
Dataset management for the ML framework.
"""

from typing import Optional, Union, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from .exceptions import DatasetError, ValidationError
from .utils import validate_dataframe, ensure_directory


class DatasetConfig(BaseModel):
    """Configuration for dataset loading."""
    
    file_path: Union[str, Path]
    target_column: str
    feature_columns: Optional[list[str]] = None
    test_size: float = Field(gt=0.0, lt=1.0, default=0.2)
    random_seed: int = 42
    shuffle: bool = True
    stratify: Optional[str] = None
    
    # Preprocessing options
    handle_missing: str = Field(default="drop")  # drop, fill, none
    fill_value: Optional[Any] = None
    normalize: bool = False
    standardize: bool = False
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: Union[str, Path]) -> Path:
        """Validate that the file path exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        return path
    
    @field_validator('handle_missing')
    @classmethod
    def validate_handle_missing(cls, v: str) -> str:
        """Validate missing value handling strategy."""
        allowed = ['drop', 'fill', 'none']
        if v not in allowed:
            raise ValueError(f"handle_missing must be one of {allowed}")
        return v


class DatasetLoader:
    """Handles loading and preprocessing of datasets."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self._data: Optional[pd.DataFrame] = None
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
        
    def load(self) -> 'DatasetLoader':
        """Load the dataset from file."""
        try:
            file_path = Path(self.config.file_path)
            ext = file_path.suffix.lower()
            
            if ext == '.csv':
                self._data = pd.read_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                self._data = pd.read_excel(file_path)
            elif ext == '.parquet':
                self._data = pd.read_parquet(file_path)
            elif ext == '.json':
                self._data = pd.read_json(file_path)
            else:
                raise DatasetError(f"Unsupported file format: {ext}")
            
            logger.info(f"Loaded dataset with {len(self._data)} rows and {len(self._data.columns)} columns")
            return self
            
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {str(e)}") from e
    
    def validate(self) -> 'DatasetLoader':
        """Validate the dataset."""
        if self._data is None:
            raise DatasetError("Dataset not loaded")
        
        # Check if target column exists
        if self.config.target_column not in self._data.columns:
            raise ValidationError(
                f"Target column '{self.config.target_column}' not found in dataset"
            )
        
        # Validate feature columns if specified
        if self.config.feature_columns:
            missing = set(self.config.feature_columns) - set(self._data.columns)
            if missing:
                raise ValidationError(f"Missing feature columns: {missing}")
        
        # Validate data types
        try:
            validate_dataframe(
                self._data,
                required_columns=[self.config.target_column] + (self.config.feature_columns or [])
            )
        except Exception as e:
            raise ValidationError(f"Data validation failed: {str(e)}") from e
        
        return self
    
    def preprocess(self) -> 'DatasetLoader':
        """Preprocess the dataset."""
        if self._data is None:
            raise DatasetError("Dataset not loaded")
        
        df = self._data.copy()
        
        # Handle missing values
        if self.config.handle_missing == 'drop':
            df = df.dropna()
        elif self.config.handle_missing == 'fill':
            if self.config.fill_value is not None:
                df = df.fillna(self.config.fill_value)
            else:
                # Fill numeric with mean, categorical with mode
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64']:
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        # Separate features and target
        feature_cols = self.config.feature_columns or [
            col for col in df.columns if col != self.config.target_column
        ]
        
        self._X = df[feature_cols].copy()
        self._y = df[self.config.target_column].copy()
        
        # Normalize or standardize
        if self.config.normalize:
            self._X = (self._X - self._X.min()) / (self._X.max() - self._X.min() + 1e-10)
        
        if self.config.standardize:
            self._X = (self._X - self._X.mean()) / (self._X.std() + 1e-10)
        
        logger.info(f"Preprocessed data: {len(self._X)} samples, {len(self._X.columns)} features")
        return self
    
    def split(
        self
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data into train and test sets.
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        if self._X is None or self._y is None:
            raise DatasetError("Data not preprocessed")
        
        from sklearn.model_selection import train_test_split
        
        stratify = None
        if self.config.stratify and self.config.stratify in self._X.columns:
            stratify = self._X[self.config.stratify]
        elif self.config.stratify == 'target':
            stratify = self._y
        
        X_train, X_test, y_train, y_test = train_test_split(
            self._X,
            self._y,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            shuffle=self.config.shuffle,
            stratify=stratify
        )
        
        logger.info(
            f"Split data: Train {len(X_train)} samples, Test {len(X_test)} samples"
        )
        
        return X_train, y_train, X_test, y_test
    
    def get_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get the preprocessed features and target."""
        if self._X is None or self._y is None:
            raise DatasetError("Data not preprocessed")
        return self._X, self._y
    
    @property
    def feature_names(self) -> list[str]:
        """Get feature names."""
        if self._X is None:
            raise DatasetError("Data not preprocessed")
        return self._X.columns.tolist()
    
    @property
    def n_features(self) -> int:
        """Get number of features."""
        if self._X is None:
            raise DatasetError("Data not preprocessed")
        return self._X.shape[1]
    
    @property
    def n_samples(self) -> int:
        """Get number of samples."""
        if self._X is None:
            raise DatasetError("Data not preprocessed")
        return self._X.shape[0]
    
    def get_summary(self) -> dict[str, Any]:
        """Get dataset summary."""
        if self._data is None:
            raise DatasetError("Dataset not loaded")
        
        return {
            'total_samples': len(self._data),
            'n_features': len(self._data.columns) - 1,
            'target_column': self.config.target_column,
            'missing_values': self._data.isnull().sum().to_dict(),
            'data_types': self._data.dtypes.astype(str).to_dict(),
            'feature_names': self.feature_names if self._X is not None else []
        }