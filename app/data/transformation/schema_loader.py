"""Schema loading and parsing from YAML files."""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from .models import SchemaMetadata, ColumnDefinition, DataType, UnitCategory
from .exceptions import SchemaNotFoundError, SchemaValidationError

logger = logging.getLogger(__name__)


class SchemaLoader:
    """Loads and parses schema definitions from YAML files."""
    
    def __init__(self, schemas_path: str):
        """
        Initialize schema loader.
        
        Args:
            schemas_path: Path to schemas directory
        """
        self.schemas_path = Path(schemas_path)
        self._schema_cache: Dict[str, SchemaMetadata] = {}
        
        if not self.schemas_path.exists():
            logger.warning(f"Schemas directory {schemas_path} does not exist")
    
    def load_schema(self, schema_name: str, version: Optional[str] = None) -> SchemaMetadata:
        """
        Load schema by name and optional version.
        
        Args:
            schema_name: Name of schema (e.g., 'crop')
            version: Specific version (default: latest)
            
        Returns:
            SchemaMetadata object
            
        Raises:
            SchemaNotFoundError: If schema file not found
            SchemaValidationError: If schema validation fails
        """
        cache_key = f"{schema_name}:{version or 'latest'}"
        
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]
        
        # Find schema file
        schema_files = list(self.schemas_path.glob(f"{schema_name}*.yaml"))
        
        if not schema_files:
            raise SchemaNotFoundError(f"Schema '{schema_name}' not found in {self.schemas_path}")
        
        # If version specified, find matching file
        if version:
            target_file = self.schemas_path / f"{schema_name}-{version}.yaml"
            if target_file.exists():
                schema_files = [target_file]
            else:
                raise SchemaNotFoundError(f"Schema '{schema_name}' version {version} not found")
        
        # Use latest (last modified) if multiple found
        schema_file = max(schema_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(schema_file, 'r') as f:
                data = yaml.safe_load(f)
            
            schema = self._parse_schema(data)
            self._schema_cache[cache_key] = schema
            logger.info(f"Loaded schema: {schema.schema_name} v{schema.schema_version}")
            return schema
            
        except yaml.YAMLError as e:
            raise SchemaValidationError(f"Invalid YAML in {schema_file}: {e}")
        except Exception as e:
            raise SchemaValidationError(f"Failed to parse schema: {e}")
    
    def _parse_schema(self, data: Dict[str, Any]) -> SchemaMetadata:
        """Parse raw YAML data into SchemaMetadata."""
        try:
            # Parse columns
            columns = []
            for col_data in data.get('columns', []):
                col_def = ColumnDefinition(
                    name=col_data['name'],
                    data_type=DataType(col_data['data_type']),
                    description=col_data.get('description'),
                    unit=col_data.get('unit'),
                    unit_category=UnitCategory(col_data.get('unit_category', 'none')) 
                        if col_data.get('unit_category') else None,
                    allowed_values=col_data.get('allowed_values'),
                    min_value=col_data.get('min_value'),
                    max_value=col_data.get('max_value'),
                    is_primary_key=col_data.get('is_primary_key', False),
                    is_target=col_data.get('is_target', False),
                    required=col_data.get('required', True),
                    default=col_data.get('default')
                )
                columns.append(col_def)
            
            # Parse primary key
            primary_key = data.get('primary_key')
            if not primary_key and columns:
                # Auto-detect primary key
                primary_key = [col.name for col in columns if col.is_primary_key]
            
            # Parse target column
            target_column = data.get('target_column')
            if not target_column:
                # Auto-detect target
                target_cols = [col.name for col in columns if col.is_target]
                target_column = target_cols[0] if target_cols else None
            
            schema = SchemaMetadata(
                schema_name=data['schema_name'],
                schema_version=data['schema_version'],
                description=data.get('description', ''),
                source=data['source'],
                owner=data.get('owner', 'Unknown'),
                license=data.get('license', 'Unknown'),
                columns=columns,
                primary_key=primary_key,
                target_column=target_column,
                update_frequency=data.get('update_frequency'),
                created_at=data.get('created_at'),
                updated_at=data.get('updated_at')
            )
            
            return schema
            
        except KeyError as e:
            raise SchemaValidationError(f"Missing required field in schema: {e}")
        except Exception as e:
            raise SchemaValidationError(f"Failed to parse schema: {e}")
    
    def list_schemas(self) -> List[Dict[str, str]]:
        """List all available schemas."""
        schemas = []
        for file in self.schemas_path.glob("*.yaml"):
            try:
                with open(file, 'r') as f:
                    data = yaml.safe_load(f)
                    schemas.append({
                        'name': data.get('schema_name', file.stem),
                        'version': data.get('schema_version', 'unknown'),
                        'file': file.name
                    })
            except Exception as e:
                logger.error(f"Failed to read schema file {file}: {e}")
        
        return schemas
    
    def get_schema_path(self, schema_name: str) -> Optional[Path]:
        """Get file path for a schema."""
        files = list(self.schemas_path.glob(f"{schema_name}*.yaml"))
        return files[0] if files else None