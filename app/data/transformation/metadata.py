"""Metadata generation for processed datasets."""

import pandas as pd
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .models import DatasetRegistry, SchemaMetadata
from .exceptions import MetadataGenerationError

logger = logging.getLogger(__name__)


class MetadataGenerator:
    """Generates metadata for processed datasets."""
    
    def __init__(self, registry_manager=None):
        """
        Initialize metadata generator.
        
        Args:
            registry_manager: RegistryManager instance for schema lookups
        """
        self.registry_manager = registry_manager
    
    def generate_metadata(self, df: pd.DataFrame, source: str, 
                         schema_name: str, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate metadata for a dataset.
        
        Args:
            df: DataFrame to generate metadata for
            source: Source identifier
            schema_name: Schema name
            dataset_id: Optional dataset ID
            
        Returns:
            Dictionary of metadata
        """
        try:
            # Load schema if available
            schema = None
            if self.registry_manager:
                try:
                    schema = self.registry_manager.load_schema(schema_name)
                except Exception as e:
                    logger.warning(f"Failed to load schema {schema_name}: {e}")
            
            metadata = {
                'dataset_id': dataset_id or self._generate_dataset_id(),
                'source': source,
                'schema_name': schema_name,
                'schema_version': schema.schema_version if schema else 'unknown',
                'rows': len(df),
                'columns': len(df.columns),
                'column_info': self._get_column_info(df),
                'statistics': self._get_statistics(df),
                'quality': self._get_quality_metrics(df),
                'timestamp': datetime.utcnow().isoformat(),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            if schema:
                metadata['schema_columns'] = [col.name for col in schema.columns]
                metadata['primary_key'] = schema.primary_key
                metadata['target_column'] = schema.target_column
            
            return metadata
            
        except Exception as e:
            raise MetadataGenerationError(f"Failed to generate metadata: {e}")
    
    def _get_column_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed column information."""
        column_info = {}
        
        for col in df.columns:
            info = {
                'data_type': str(df[col].dtype),
                'null_count': int(df[col].isna().sum()),
                'null_percentage': float(df[col].isna().sum() / len(df) * 100),
                'unique_count': int(df[col].nunique()),
            }
            
            # Add numeric statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                info.update({
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                    'median': float(df[col].median()) if not df[col].isna().all() else None,
                })
            
            # Add categorical info
            if pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
                info['top_values'] = df[col].value_counts().head(5).to_dict()
            
            column_info[col] = info
        
        return column_info
    
    def _get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get overall dataset statistics."""
        return {
            'total_cells': len(df) * len(df.columns),
            'total_nulls': int(df.isna().sum().sum()),
            'null_percentage': float(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
            'duplicate_rows': int(df.duplicated().sum()),
            'numeric_columns': len(df.select_dtypes(include=['number']).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
        }
    
    def _get_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        metrics = {
            'completeness': 0.0,
            'uniqueness': 0.0,
            'consistency': 1.0,
            'overall_score': 0.0
        }
        
        # Completeness (non-null percentage)
        total_cells = len(df) * len(df.columns)
        non_null = total_cells - int(df.isna().sum().sum())
        metrics['completeness'] = non_null / total_cells if total_cells > 0 else 0
        
        # Uniqueness (based on primary key or row uniqueness)
        if len(df) > 0:
            duplicates = df.duplicated().sum()
            metrics['uniqueness'] = 1 - (duplicates / len(df))
        
        # Overall score (weighted average)
        metrics['overall_score'] = (
            metrics['completeness'] * 0.4 +
            metrics['uniqueness'] * 0.3 +
            metrics['consistency'] * 0.3
        )
        
        return metrics
    
    def _generate_dataset_id(self) -> str:
        """Generate a unique dataset ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"dataset_{timestamp}"
    
    def create_registry_entry(self, df: pd.DataFrame, file_path: str, 
                             schema_name: str, source: str, owner: str = 'Unknown') -> DatasetRegistry:
        """
        Create a registry entry for a dataset.
        
        Args:
            df: DataFrame
            file_path: Path to saved file
            schema_name: Schema name
            source: Source identifier
            owner: Dataset owner
            
        Returns:
            DatasetRegistry object
        """
        try:
            # Load schema
            schema = None
            if self.registry_manager:
                try:
                    schema = self.registry_manager.load_schema(schema_name)
                except Exception as e:
                    logger.warning(f"Failed to load schema {schema_name}: {e}")
            
            # Generate checksum
            with open(file_path, 'rb') as f:
                content = f.read()
                checksum = hashlib.sha256(content).hexdigest()
            
            # Get quality metrics
            quality_metrics = self._get_quality_metrics(df)
            
            registry = DatasetRegistry(
                dataset_id=self._generate_dataset_id(),
                source=source,
                schema_name=schema_name,
                schema_version=schema.schema_version if schema else 'unknown',
                file_path=str(file_path),
                checksum=checksum,
                rows=len(df),
                columns=len(df.columns),
                quality_score=quality_metrics['overall_score'],
                missing_percentage=df.isna().sum().sum() / (len(df) * len(df.columns)) * 100,
                owner=owner,
                license=schema.license if schema else 'Unknown',
                created_at=datetime.utcnow()
            )
            
            return registry
            
        except Exception as e:
            raise MetadataGenerationError(f"Failed to create registry entry: {e}")
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: Path) -> None:
        """Save metadata to JSON file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Metadata saved to {output_path}")
            
        except Exception as e:
            raise MetadataGenerationError(f"Failed to save metadata: {e}")