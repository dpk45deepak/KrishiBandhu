"""Data transformers for various cleaning operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Pattern
import re
import unicodedata
import pandas as pd
import numpy as np
from loguru import logger

from .exceptions import TransformationError, UnitConversionError
from .models import ColumnMapping, UnitConversion


class BaseTransformer(ABC):
    """Base class for all data transformers."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.name = self.__class__.__name__
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the transformation."""
        return {
            "transformer": self.name,
            "params": self.params,
            "affected_rows": 0,
            "affected_columns": 0,
        }


class ColumnNameStandardizer(BaseTransformer):
    """Standardizes column names according to configuration."""
    
    def __init__(self, 
                 case: str = "lower",
                 replace_spaces: str = "_",
                 remove_special: bool = True,
                 alias_mapping: Optional[Dict[str, List[str]]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.case = case
        self.replace_spaces = replace_spaces
        self.remove_special = remove_special
        self.alias_mapping = alias_mapping or {}
        self._mapping: Dict[str, str] = {}
        self._affected_columns = 0
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        df = data.copy()
        original_columns = df.columns.tolist()
        
        # Create mapping
        self._mapping = self._create_mapping(original_columns)
        
        # Apply mapping
        if self._mapping:
            df = df.rename(columns=self._mapping)
            self._affected_columns = len(self._mapping)
        
        return df
    
    def _create_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Create column name mapping based on configuration."""
        mapping = {}
        
        # First, apply alias mapping
        for target, aliases in self.alias_mapping.items():
            for col in columns:
                if col in aliases or col.lower() in [a.lower() for a in aliases]:
                    mapping[col] = target
        
        # Then apply standard transformations to remaining columns
        for col in columns:
            if col not in mapping:  # Skip if already mapped
                new_name = col
                
                # Case transformation
                if self.case == "lower":
                    new_name = new_name.lower()
                elif self.case == "upper":
                    new_name = new_name.upper()
                elif self.case == "title":
                    new_name = new_name.title()
                
                # Replace spaces
                if self.replace_spaces:
                    new_name = new_name.replace(" ", self.replace_spaces)
                    new_name = re.sub(rf'{re.escape(self.replace_spaces)}+', self.replace_spaces, new_name)
                
                # Remove special characters
                if self.remove_special:
                    new_name = re.sub(r'[^a-zA-Z0-9_]', '', new_name)
                
                # Remove leading/trailing underscores
                new_name = new_name.strip('_')
                
                if new_name and new_name != col:
                    mapping[col] = new_name
        
        return mapping
    
    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "transformer_type": "column_standardization",
            "mapping": self._mapping,
            "affected_columns": self._affected_columns,
        })
        return metadata


class UnitConverter(BaseTransformer):
    """Converts units in datasets."""
    
    # Built-in unit conversions
    UNIT_MAPPINGS = {
        "length": {
            "cm": {"meter": 0.01, "mm": 10},
            "meter": {"cm": 100, "mm": 1000},
            "mm": {"cm": 0.1, "meter": 0.001},
        },
        "weight": {
            "kg": {"gram": 1000},
            "gram": {"kg": 0.001},
        },
        "area": {
            "hectare": {"acre": 2.47105},
            "acre": {"hectare": 0.404686},
        },
        "temperature": {
            "celsius": {"fahrenheit": lambda x: x * 9/5 + 32, "kelvin": lambda x: x + 273.15},
            "fahrenheit": {"celsius": lambda x: (x - 32) * 5/9, "kelvin": lambda x: (x - 32) * 5/9 + 273.15},
            "kelvin": {"celsius": lambda x: x - 273.15, "fahrenheit": lambda x: (x - 273.15) * 9/5 + 32},
        }
    }
    
    def __init__(self, 
                 conversion_mappings: Optional[Dict[str, List[Dict[str, str]]]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.conversion_mappings = conversion_mappings or {}
        self._conversions_performed: List[Dict[str, Any]] = []
        self._affected_columns = 0
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply unit conversions to the data."""
        df = data.copy()
        
        for column, mappings in self.conversion_mappings.items():
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found for unit conversion")
                continue
            
            for mapping in mappings:
                from_unit = mapping.get("from_unit")
                to_unit = mapping.get("to_unit")
                
                if not from_unit or not to_unit:
                    continue
                
                try:
                    df[column] = self._convert_column(df[column], from_unit, to_unit)
                    self._conversions_performed.append({
                        "column": column,
                        "from": from_unit,
                        "to": to_unit,
                    })
                    self._affected_columns += 1
                except Exception as e:
                    logger.error(f"Failed to convert {column} from {from_unit} to {to_unit}: {e}")
        
        return df
    
    def _convert_column(self, series: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
        """Convert a single column's units."""
        if not pd.api.types.is_numeric_dtype(series):
            raise UnitConversionError(f"Column '{series.name}' must be numeric for unit conversion")
        
        # Check if it's a temperature conversion (special handling)
        if self._is_temperature_unit(from_unit) and self._is_temperature_unit(to_unit):
            return self._convert_temperature(series, from_unit, to_unit)
        
        # Try to find conversion in mappings
        conversion_factor = self._get_conversion_factor(from_unit, to_unit)
        if conversion_factor is None:
            raise UnitConversionError(f"Unknown conversion from {from_unit} to {to_unit}")
        
        return series * conversion_factor
    
    def _is_temperature_unit(self, unit: str) -> bool:
        """Check if a unit is a temperature unit."""
        return unit.lower() in ['celsius', 'fahrenheit', 'kelvin']
    
    def _convert_temperature(self, series: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
        """Convert temperature values."""
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit == to_unit:
            return series
        
        # Get conversion function
        conversion_func = self.UNIT_MAPPINGS.get("temperature", {}).get(from_unit, {}).get(to_unit)
        if conversion_func is None:
            raise UnitConversionError(f"Unknown temperature conversion from {from_unit} to {to_unit}")
        
        if callable(conversion_func):
            return series.apply(conversion_func)
        else:
            raise UnitConversionError(f"Invalid temperature conversion function")
    
    def _get_conversion_factor(self, from_unit: str, to_unit: str) -> Optional[float]:
        """Get conversion factor between units."""
        # Check all unit categories
        for category, units in self.UNIT_MAPPINGS.items():
            if from_unit in units and to_unit in units[from_unit]:
                conversion = units[from_unit][to_unit]
                if isinstance(conversion, (int, float)):
                    return conversion
        return None
    
    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "transformer_type": "unit_conversion",
            "conversions_performed": self._conversions_performed,
            "affected_columns": self._affected_columns,
        })
        return metadata


class DataTypeConverter(BaseTransformer):
    """Converts data types according to configuration."""
    
    def __init__(self,
                 infer_objects: bool = True,
                 convert_numeric: bool = True,
                 convert_dates: bool = True,
                 custom_mappings: Optional[Dict[str, str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.infer_objects = infer_objects
        self.convert_numeric = convert_numeric
        self.convert_dates = convert_dates
        self.custom_mappings = custom_mappings or {}
        self._type_changes: Dict[str, Dict[str, str]] = {}
        self._affected_columns = 0
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types."""
        df = data.copy()
        
        # Apply custom mappings first
        for column, target_type in self.custom_mappings.items():
            if column in df.columns:
                try:
                    df[column] = df[column].astype(target_type)
                    self._track_type_change(column, df[column].dtype, target_type)
                    self._affected_columns += 1
                except Exception as e:
                    logger.warning(f"Failed to convert {column} to {target_type}: {e}")
        
        # Infer object types
        if self.infer_objects:
            df = df.infer_objects()
        
        # Convert numeric types
        if self.convert_numeric:
            numeric_columns = df.select_dtypes(include=['object']).columns
            for col in numeric_columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                    self._track_type_change(col, df[col].dtype, 'numeric')
                    self._affected_columns += 1
                except Exception:
                    pass
        
        # Convert date types
        if self.convert_dates:
            date_columns = df.select_dtypes(include=['object']).columns
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                    self._track_type_change(col, df[col].dtype, 'datetime')
                    self._affected_columns += 1
                except Exception:
                    pass
        
        return df
    
    def _track_type_change(self, column: str, old_type: str, new_type: str):
        """Track type changes for metadata."""
        if column not in self._type_changes:
            self._type_changes[column] = {}
        self._type_changes[column]['old'] = str(old_type)
        self._type_changes[column]['new'] = str(new_type)
    
    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "transformer_type": "data_type_conversion",
            "type_changes": self._type_changes,
            "affected_columns": self._affected_columns,
        })
        return metadata


class TextCleaner(BaseTransformer):
    """Cleans text data."""
    
    def __init__(self,
                 trim: bool = True,
                 remove_whitespace: bool = True,
                 case_normalization: Optional[str] = None,
                 remove_special_chars: bool = True,
                 remove_invalid_chars: bool = True,
                 null_string_conversion: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.trim = trim
        self.remove_whitespace = remove_whitespace
        self.case_normalization = case_normalization
        self.remove_special_chars = remove_special_chars
        self.remove_invalid_chars = remove_invalid_chars
        self.null_string_conversion = null_string_conversion
        self._affected_rows = 0
        self._affected_columns = 0
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean text data."""
        df = data.copy()
        
        # Get text columns
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            self._affected_columns += 1
            original = df[col].copy()
            
            # Apply cleaning operations
            df[col] = self._clean_column(df[col])
            
            # Track affected rows
            if self._affected_rows == 0:
                changed = (df[col] != original).sum()
                self._affected_rows = changed
        
        return df
    
    def _clean_column(self, series: pd.Series) -> pd.Series:
        """Clean a single text column."""
        # Convert to string if needed
        if not pd.api.types.is_string_dtype(series):
            series = series.astype(str)
        
        # Null string conversion
        if self.null_string_conversion:
            series = series.replace(['nan', 'None', 'NULL', 'null', ''], np.nan)
        
        # Clean non-null values
        mask = series.notna()
        if mask.any():
            cleaned = series[mask].copy()
            
            # Remove invalid characters
            if self.remove_invalid_chars:
                cleaned = cleaned.apply(lambda x: ''.join(
                    c for c in x if unicodedata.category(c)[0] not in ['C', 'Z']
                ))
            
            # Trim whitespace
            if self.trim:
                cleaned = cleaned.str.strip()
            
            # Remove extra whitespace
            if self.remove_whitespace:
                cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
            
            # Case normalization
            if self.case_normalization:
                if self.case_normalization == 'lower':
                    cleaned = cleaned.str.lower()
                elif self.case_normalization == 'upper':
                    cleaned = cleaned.str.upper()
                elif self.case_normalization == 'title':
                    cleaned = cleaned.str.title()
            
            # Remove special characters (keep alphanumeric and spaces)
            if self.remove_special_chars:
                cleaned = cleaned.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            
            series[mask] = cleaned
        
        return series
    
    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "transformer_type": "text_cleaning",
            "affected_rows": self._affected_rows,
            "affected_columns": self._affected_columns,
            "operations": {
                "trim": self.trim,
                "remove_whitespace": self.remove_whitespace,
                "case_normalization": self.case_normalization,
                "remove_special_chars": self.remove_special_chars,
                "remove_invalid_chars": self.remove_invalid_chars,
                "null_string_conversion": self.null_string_conversion,
            }
        })
        return metadata


class EncoderFixer(BaseTransformer):
    """Fixes encoding issues in text data."""
    
    def __init__(self, encoding: str = "utf-8", errors: str = "replace", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.errors = errors
        self._affected_rows = 0
        self._affected_columns = 0
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix encoding issues."""
        df = data.copy()
        
        text_columns = df.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            self._affected_columns += 1
            original = df[col].copy()
            
            try:
                df[col] = df[col].apply(self._fix_encoding)
                changed = (df[col] != original).sum()
                self._affected_rows += changed
            except Exception as e:
                logger.warning(f"Failed to fix encoding for column {col}: {e}")
        
        return df
    
    def _fix_encoding(self, value: Any) -> str:
        """Fix encoding for a single value."""
        if not isinstance(value, str):
            return value
        
        try:
            # Try to encode and decode
            return value.encode(self.encoding, errors=self.errors).decode(self.encoding)
        except Exception:
            return value
    
    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "transformer_type": "encoding_fix",
            "affected_rows": self._affected_rows,
            "affected_columns": self._affected_columns,
            "encoding": self.encoding,
            "errors": self.errors,
        })
        return metadata