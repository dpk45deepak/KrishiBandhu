# app/data/feature_engineering/encoders.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from collections import Counter
import hashlib
from loguru import logger

from app.data.feature_engineering.models import EncodingType
from app.data.feature_engineering.exceptions import EncodingError


class FeatureEncoder:
    """
    Enterprise feature encoder for categorical variables.
    
    Implements multiple encoding strategies with consistent API.
    """
    
    def __init__(self):
        self.encoders: Dict[str, Any] = {}
        self.encoding_maps: Dict[str, Dict] = {}
        self.fitted: bool = False
    
    def encode_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        encoding_type: EncodingType,
        **kwargs
    ) -> pd.DataFrame:
        """
        Encode categorical features using specified strategy.
        
        Args:
            df: Input dataframe
            columns: Columns to encode
            encoding_type: Type of encoding to apply
            **kwargs: Additional parameters for the encoder
            
        Returns:
            Encoded dataframe
        """
        try:
            result_df = df.copy()
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                if encoding_type == EncodingType.LABEL:
                    result_df = self._label_encode(result_df, col, **kwargs)
                elif encoding_type == EncodingType.ORDINAL:
                    result_df = self._ordinal_encode(result_df, col, **kwargs)
                elif encoding_type == EncodingType.ONE_HOT:
                    result_df = self._one_hot_encode(result_df, col, **kwargs)
                elif encoding_type == EncodingType.FREQUENCY:
                    result_df = self._frequency_encode(result_df, col, **kwargs)
                elif encoding_type == EncodingType.HASH:
                    result_df = self._hash_encode(result_df, col, **kwargs)
                elif encoding_type == EncodingType.BINARY:
                    result_df = self._binary_encode(result_df, col, **kwargs)
                else:
                    raise ValueError(f"Unsupported encoding type: {encoding_type}")
                
                self.encoders[col] = encoding_type
            
            self.fitted = True
            logger.info(f"Encoded {len(columns)} columns using {encoding_type}")
            return result_df
            
        except Exception as e:
            raise EncodingError(f"Failed to encode features: {e}")
    
    def _label_encode(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """Apply label encoding."""
        le = LabelEncoder()
        # Handle NaN values
        data = df[column].fillna('missing').astype(str)
        encoded = le.fit_transform(data)
        df[f"{column}_label"] = encoded
        self.encoding_maps[column] = {
            "type": "label",
            "classes": dict(zip(le.classes_, le.transform(le.classes_)))
        }
        return df
    
    def _ordinal_encode(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """Apply ordinal encoding with custom ordering."""
        categories = kwargs.get('categories', None)
        if categories is None:
            # Use sklearn OrdinalEncoder with automatic categories
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            data = df[[column]].fillna('missing').astype(str)
            encoded = oe.fit_transform(data)
            df[f"{column}_ordinal"] = encoded.flatten()
            self.encoding_maps[column] = {
                "type": "ordinal",
                "categories": oe.categories_
            }
        else:
            # Custom ordering
            mapping = {cat: i for i, cat in enumerate(categories)}
            df[f"{column}_ordinal"] = df[column].map(mapping).fillna(-1)
            self.encoding_maps[column] = {
                "type": "ordinal",
                "mapping": mapping
            }
        return df
    
    def _one_hot_encode(
        self, 
        df: pd.DataFrame, 
        column: str, 
        **kwargs
    ) -> pd.DataFrame:
        """Apply one-hot encoding."""
        max_categories = kwargs.get('max_categories', 20)
        
        # Check cardinality
        unique_count = df[column].nunique()
        if unique_count > max_categories:
            logger.warning(
                f"Column {column} has {unique_count} unique values, "
                f"which exceeds max_categories ({max_categories}). "
                f"Consider using other encoding methods."
            )
        
        # One-hot encode
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        data = df[[column]].fillna('missing').astype(str)
        encoded = ohe.fit_transform(data)
        
        # Create column names
        feature_names = ohe.get_feature_names_out([column])
        encoded_df = pd.DataFrame(
            encoded, 
            columns=feature_names,
            index=df.index
        )
        
        # Add encoded columns to dataframe
        for col_name in feature_names:
            df[col_name] = encoded_df[col_name]
        
        self.encoding_maps[column] = {
            "type": "one_hot",
            "feature_names": list(feature_names),
            "categories": ohe.categories_
        }
        
        return df
    
    def _frequency_encode(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """Apply frequency encoding."""
        freq_map = df[column].value_counts(normalize=True).to_dict()
        df[f"{column}_freq"] = df[column].map(freq_map).fillna(0)
        
        self.encoding_maps[column] = {
            "type": "frequency",
            "freq_map": freq_map
        }
        
        return df
    
    def _hash_encode(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """Apply hash encoding."""
        n_components = kwargs.get('n_components', 8)
        salt = kwargs.get('salt', '')
        
        def hash_value(val: str) -> int:
            hash_obj = hashlib.md5(f"{salt}{val}".encode())
            return int(hash_obj.hexdigest(), 16) % (2 ** n_components)
        
        df[f"{column}_hash"] = df[column].fillna('missing').astype(str).apply(hash_value)
        
        self.encoding_maps[column] = {
            "type": "hash",
            "n_components": n_components,
            "salt": salt
        }
        
        return df
    
    def _binary_encode(self, df: pd.DataFrame, column: str, **kwargs) -> pd.DataFrame:
        """Apply binary encoding."""
        unique_values = df[column].unique()
        mapping = {val: bin(i)[2:].zfill(len(bin(len(unique_values)))-2) 
                   for i, val in enumerate(unique_values)}
        
        # Create binary columns
        max_bits = max(len(mapping[val]) for val in mapping)
        for i in range(max_bits):
            df[f"{column}_bin_{i}"] = df[column].map(
                lambda x: int(mapping.get(x, '0' * max_bits)[i]) if len(mapping.get(x, '0' * max_bits)) > i else 0
            )
        
        self.encoding_maps[column] = {
            "type": "binary",
            "mapping": mapping,
            "max_bits": max_bits
        }
        
        return df
    
    def get_encoding_info(self, column: str) -> Optional[Dict]:
        """Get encoding information for a column."""
        return self.encoding_maps.get(column)
    
    def transform_with_encoder(
        self,
        df: pd.DataFrame,
        column: str,
        encoder: Any,
        **kwargs
    ) -> pd.DataFrame:
        """Transform using a pre-fitted encoder."""
        if encoder not in self.encoders:
            raise ValueError(f"Encoder for {column} not fitted")
        
        return self.encode_features(df, [column], self.encoders[column], **kwargs)
    
    def batch_encode(
        self,
        df: pd.DataFrame,
        encodings: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply multiple encodings in batch."""
        result_df = df.copy()
        
        for encoding in encodings:
            columns = encoding.get("columns", [])
            encoding_type = encoding.get("encoding_type")
            
            if not columns or not encoding_type:
                continue
            
            result_df = self.encode_features(
                result_df,
                columns,
                encoding_type,
                **encoding.get("params", {})
            )
        
        return result_df