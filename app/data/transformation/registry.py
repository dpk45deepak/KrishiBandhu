"""Dataset registry management."""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import hashlib

from .models import DatasetRegistry, SchemaMetadata
from .exceptions import RegistryError, SchemaNotFoundError
from .schema_loader import SchemaLoader

logger = logging.getLogger(__name__)


class RegistryManager:
    """Manages dataset registrations and schema versions."""
    
    def __init__(self, registry_path: str):
        """
        Initialize the registry manager.
        
        Args:
            registry_path: Path to registry storage directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.registry_path / "registry.json"
        self._registry_data: Dict[str, DatasetRegistry] = {}
        self._schema_loader = SchemaLoader(str(self.registry_path / ".." / "schemas"))
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry data from disk."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, 'r') as f:
                    data = json.load(f)
                    self._registry_data = {
                        k: DatasetRegistry(**v) 
                        for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._registry_data)} registry entries")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self._registry_data = {}
    
    def _save_registry(self) -> None:
        """Save registry data to disk."""
        try:
            data = {
                k: v.dict() 
                for k, v in self._registry_data.items()
            }
            with open(self._registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved registry with {len(self._registry_data)} entries")
        except Exception as e:
            raise RegistryError(f"Failed to save registry: {e}")
    
    def register_dataset(self, registry_entry: DatasetRegistry) -> str:
        """
        Register a dataset in the registry.
        
        Args:
            registry_entry: Registry entry to add
            
        Returns:
            Dataset ID
        """
        try:
            # Validate schema exists
            self._schema_loader.load_schema(registry_entry.schema_name)
            
            # Check if dataset already exists
            if registry_entry.dataset_id in self._registry_data:
                logger.warning(f"Dataset {registry_entry.dataset_id} already registered, updating...")
            
            self._registry_data[registry_entry.dataset_id] = registry_entry
            self._save_registry()
            logger.info(f"Registered dataset: {registry_entry.dataset_id}")
            return registry_entry.dataset_id
            
        except Exception as e:
            raise RegistryError(f"Failed to register dataset: {e}")
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetRegistry]:
        """Get dataset registry entry by ID."""
        return self._registry_data.get(dataset_id)
    
    def load_schema(self, schema_name: str) -> SchemaMetadata:
        """
        Load schema definition by name.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            SchemaMetadata object
        """
        return self._schema_loader.load_schema(schema_name)
    
    def compare_versions(self, schema_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a schema.
        
        Args:
            schema_name: Schema name
            version1: First version
            version2: Second version
            
        Returns:
            Dictionary of differences
        """
        schema1 = self._schema_loader.load_schema(schema_name, version1)
        schema2 = self._schema_loader.load_schema(schema_name, version2)
        
        differences = {
            'columns_added': [],
            'columns_removed': [],
            'columns_modified': [],
            'metadata_changed': []
        }
        
        # Compare columns
        col1_names = {col.name for col in schema1.columns}
        col2_names = {col.name for col in schema2.columns}
        
        differences['columns_added'] = list(col2_names - col1_names)
        differences['columns_removed'] = list(col1_names - col2_names)
        
        # Check modified columns
        col1_map = {col.name: col for col in schema1.columns}
        col2_map = {col.name: col for col in schema2.columns}
        
        for col_name in col1_names & col2_names:
            if col1_map[col_name] != col2_map[col_name]:
                differences['columns_modified'].append(col_name)
        
        return differences
    
    def validate_against_registry(self, dataset_path: str, schema_name: str) -> Dict[str, Any]:
        """
        Validate a dataset against its schema in the registry.
        
        Args:
            dataset_path: Path to dataset file
            schema_name: Schema name to validate against
            
        Returns:
            Validation results
        """
        import pandas as pd
        
        try:
            # Load schema
            schema = self._schema_loader.load_schema(schema_name)
            
            # Load dataset
            df = pd.read_csv(dataset_path)
            
            results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'schema': schema_name,
                'dataset_rows': len(df)
            }
            
            # Check required columns
            required_cols = [col.name for col in schema.columns if col.required]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                results['valid'] = False
                results['errors'].append(f"Missing required columns: {missing_cols}")
            
            # Check data types and values
            for col_def in schema.columns:
                if col_def.name not in df.columns:
                    continue
                    
                col_data = df[col_def.name]
                
                # Type validation
                try:
                    if col_def.data_type.value == "integer":
                        pd.to_numeric(col_data, errors='raise')
                    elif col_def.data_type.value == "float":
                        pd.to_numeric(col_data, errors='raise')
                    elif col_def.data_type.value == "date":
                        pd.to_datetime(col_data, errors='raise')
                except Exception as e:
                    results['valid'] = False
                    results['errors'].append(f"Column {col_def.name}: type validation failed - {e}")
                
                # Value range validation
                if col_def.min_value is not None:
                    invalid = col_data[col_data < col_def.min_value]
                    if not invalid.empty:
                        results['warnings'].append(
                            f"Column {col_def.name}: {len(invalid)} values below min {col_def.min_value}"
                        )
                
                if col_def.max_value is not None:
                    invalid = col_data[col_data > col_def.max_value]
                    if not invalid.empty:
                        results['warnings'].append(
                            f"Column {col_def.name}: {len(invalid)} values above max {col_def.max_value}"
                        )
            
            return results
            
        except Exception as e:
            raise RegistryError(f"Validation failed: {e}")
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all registered datasets."""
        return [
            {
                'dataset_id': entry.dataset_id,
                'schema_name': entry.schema_name,
                'rows': entry.rows,
                'columns': entry.columns,
                'quality_score': entry.quality_score,
                'created_at': entry.created_at.isoformat() if entry.created_at else None
            }
            for entry in self._registry_data.values()
        ]
    
    def update_registry(self, dataset_id: str, updates: Dict[str, Any]) -> None:
        """
        Update existing registry entry.
        
        Args:
            dataset_id: Dataset ID to update
            updates: Dictionary of updates
        """
        if dataset_id not in self._registry_data:
            raise RegistryError(f"Dataset {dataset_id} not found in registry")
        
        current = self._registry_data[dataset_id]
        for key, value in updates.items():
            if hasattr(current, key):
                setattr(current, key, value)
        
        current.last_updated = datetime.utcnow()
        self._save_registry()
        logger.info(f"Updated registry entry: {dataset_id}")